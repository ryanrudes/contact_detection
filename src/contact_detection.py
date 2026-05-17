from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree

try:
    from .silence_detection import (
        QuietDetectionConfig,
        QuietSignalType,
        VectorQuietMode,
        clean_mask_by_time,
        detect_quiet_intervals,
        intervals_from_mask,
        local_polynomial_derivative,
        score_hysteresis_mask,
    )
except ImportError:  # pragma: no cover - supports direct script-style imports
    from silence_detection import (
        QuietDetectionConfig,
        QuietSignalType,
        VectorQuietMode,
        clean_mask_by_time,
        detect_quiet_intervals,
        intervals_from_mask,
        local_polynomial_derivative,
        score_hysteresis_mask,
    )


@dataclass(frozen=True)
class SupportDetectionConfig:
    plane_residual_tolerance: float = 0.025
    coplanarity_ratio_threshold: float = 0.85
    heightmap_cell_size: float = 0.10
    heightmap_quantile: float = 0.20
    min_support_points: int = 20
    max_bootstrap_iterations: int = 3
    convergence_tolerance: float = 0.01
    ransac_iterations: int = 128
    random_seed: int = 17
    up_axis: int = 2


@dataclass(frozen=True)
class ContactDetectionConfig:
    quiet_config: QuietDetectionConfig = field(
        default_factory=lambda: QuietDetectionConfig(
            signal_type=QuietSignalType.VECTOR_POSITION,
            vector_mode=VectorQuietMode.EUCLIDEAN,
            min_activity_on_eps=0.003,
            min_activity_off_eps=0.010,
            min_spread_on_eps=0.0,
            min_spread_off_eps=0.0,
        )
    )
    support_config: SupportDetectionConfig = field(default_factory=SupportDetectionConfig)
    feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "quiet_activity": 0.75,
            "quiet_spread": 0.50,
            "clearance": 1.50,
            "normal_speed": 0.75,
            "tangential_speed": 1.25,
            "angular_speed": 0.25,
            "reference_force": 1.00,
        }
    )
    clearance_scale: float = 0.03
    normal_speed_scale: float = 0.08
    tangential_speed_scale: float = 0.08
    angular_speed_scale: float = 1.00
    score_on_threshold: float = 0.60
    score_off_threshold: float = 0.40
    min_contact_time: float = 0.18
    max_gap_time: float = 0.10
    min_blip_time: float = 0.15
    moving_support_mode: bool = False


@dataclass
class ContactDetectionResult:
    intervals: list[tuple[float, float]]
    mask: np.ndarray
    scores: np.ndarray
    support_ids: np.ndarray
    features: dict[str, np.ndarray]
    support_models: list["SupportModel"]
    debug: dict[str, object]


class SupportModel(Protocol):
    name: str

    def clearance(self, points: np.ndarray) -> np.ndarray:
        ...

    def normals_at(self, points: np.ndarray) -> np.ndarray:
        ...


@dataclass
class PlaneSupportModel:
    normal: np.ndarray
    origin: np.ndarray
    residuals: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    name: str = "plane"

    @classmethod
    def fit(cls, points: np.ndarray, config: SupportDetectionConfig) -> "PlaneSupportModel":
        points = _validate_point_cloud(points)
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a plane.")

        rng = np.random.default_rng(config.random_seed)
        best_inliers = None
        best_count = -1

        if len(points) >= 3:
            for _ in range(config.ransac_iterations):
                sample_idx = rng.choice(len(points), size=3, replace=False)
                p0, p1, p2 = points[sample_idx]
                normal = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(normal)
                if norm <= 1e-12:
                    continue
                normal = normal / norm
                normal = _orient_normal_up(normal, config.up_axis)
                residuals = np.abs((points - p0[None, :]) @ normal)
                inliers = residuals <= config.plane_residual_tolerance
                count = int(np.sum(inliers))
                if count > best_count:
                    best_count = count
                    best_inliers = inliers

        if best_inliers is None or np.sum(best_inliers) < 3:
            best_inliers = np.ones(len(points), dtype=bool)

        origin, normal = _fit_plane_svd(points[best_inliers], config.up_axis)
        residuals = (points - origin[None, :]) @ normal
        inlier_mask = np.abs(residuals) <= config.plane_residual_tolerance
        return cls(normal=normal, origin=origin, residuals=residuals, inlier_mask=inlier_mask)

    def clearance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return (points - self.origin) @ self.normal

    def normals_at(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return np.broadcast_to(self.normal, points.shape).copy()


@dataclass
class HeightmapSupportModel:
    cell_size: float
    cells_xy: np.ndarray
    heights: np.ndarray
    up_axis: int = 2
    name: str = "heightmap"
    _tree: cKDTree | None = field(default=None, init=False, repr=False)

    @classmethod
    def fit(cls, points: np.ndarray, config: SupportDetectionConfig) -> "HeightmapSupportModel":
        points = _validate_point_cloud(points)
        if len(points) < 1:
            raise ValueError("Need at least 1 point to fit a heightmap.")

        horizontal_axes = _horizontal_axes(config.up_axis)
        xy = points[:, horizontal_axes]
        z = points[:, config.up_axis]
        cell_size = float(config.heightmap_cell_size)
        if cell_size <= 0.0:
            raise ValueError("heightmap_cell_size must be positive.")

        cell_ids = np.floor(xy / cell_size).astype(int)
        buckets: dict[tuple[int, int], list[float]] = {}
        for cell_id, height in zip(cell_ids, z):
            buckets.setdefault((int(cell_id[0]), int(cell_id[1])), []).append(float(height))

        cells = []
        heights = []
        for cell_id, values in sorted(buckets.items()):
            center = (np.asarray(cell_id, dtype=float) + 0.5) * cell_size
            cells.append(center)
            heights.append(float(np.quantile(values, config.heightmap_quantile)))

        model = cls(
            cell_size=cell_size,
            cells_xy=np.asarray(cells, dtype=float),
            heights=np.asarray(heights, dtype=float),
            up_axis=config.up_axis,
        )
        model._tree = cKDTree(model.cells_xy)
        return model

    def _height_at_xy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if self._tree is None:
            self._tree = cKDTree(self.cells_xy)
        _, idx = self._tree.query(xy, k=1)
        return self.heights[idx]

    def clearance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        horizontal_axes = _horizontal_axes(self.up_axis)
        support_height = self._height_at_xy(points[:, horizontal_axes])
        return points[:, self.up_axis] - support_height

    def normals_at(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        normals = np.zeros_like(points, dtype=float)
        normals[:, self.up_axis] = 1.0
        return normals


def detect_contact_intervals(
    t,
    points,
    config: ContactDetectionConfig | None = None,
    orientations=None,
    supports=None,
    reference_force=None,
) -> ContactDetectionResult:
    config = config or ContactDetectionConfig()
    t = np.asarray(t, dtype=float)
    points = _validate_points_over_time(points)

    if config.moving_support_mode and supports is None:
        raise NotImplementedError("moving_support_mode requires explicit support poses/models in this v1 API.")
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if len(t) != points.shape[0]:
        raise ValueError("t and points must have the same number of frames.")
    if len(t) < 3:
        raise ValueError("Need at least 3 frames.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")

    velocities = local_polynomial_derivative(
        points.reshape(len(t), -1),
        t,
        window_time=config.quiet_config.derivative_window_time,
        degree=config.quiet_config.derivative_poly_degree,
    ).reshape(points.shape)

    quiet_masks, quiet_debug = _detect_point_quietness(t, points, config.quiet_config)
    current_point_mask = quiet_masks.copy()
    support_models = _normalize_supports(supports)

    iteration_debug = []
    last_frame_mask = None
    final_features = {}
    final_point_scores = np.zeros(points.shape[:2], dtype=float)
    final_support_models = support_models.copy()

    max_iterations = 1 if support_models else config.support_config.max_bootstrap_iterations
    for iteration in range(max_iterations):
        if not support_models:
            support_model = fit_support_model_from_candidates(points, current_point_mask, config.support_config)
            support_models = [support_model]
        else:
            support_model = support_models[0]

        features = compute_support_relative_features(
            points,
            velocities,
            support_model,
            quiet_debug,
            config,
            t=t,
            orientations=orientations,
            reference_force=reference_force,
        )
        point_scores = score_contact_features(features, config)
        frame_scores = np.max(point_scores, axis=1)
        frame_mask = score_hysteresis_mask(
            frame_scores,
            config.score_on_threshold,
            config.score_off_threshold,
        )
        frame_mask = clean_mask_by_time(
            t,
            frame_mask,
            max_gap_time=config.max_gap_time,
            min_blip_time=config.min_blip_time,
        )

        high_confidence = point_scores >= max(config.score_on_threshold, 0.75)
        current_point_mask = high_confidence & frame_mask[:, None]

        changed = None
        if last_frame_mask is not None:
            changed = float(np.mean(frame_mask != last_frame_mask))
            if changed <= config.support_config.convergence_tolerance:
                final_features = features
                final_point_scores = point_scores
                last_frame_mask = frame_mask
                final_support_models = [support_model]
                iteration_debug.append({"iteration": iteration, "changed": changed, "support": support_model.name})
                break

        final_features = features
        final_point_scores = point_scores
        last_frame_mask = frame_mask
        final_support_models = [support_model]
        iteration_debug.append({"iteration": iteration, "changed": changed, "support": support_model.name})

        if supports is None and iteration < max_iterations - 1:
            support_models = []

    if last_frame_mask is None:
        last_frame_mask = np.zeros(len(t), dtype=bool)

    intervals = intervals_from_mask(t, last_frame_mask, min_duration=config.min_contact_time)
    interval_mask = np.zeros_like(last_frame_mask)
    for start, end in intervals:
        interval_mask |= (t >= start) & (t <= end)

    frame_scores = np.max(final_point_scores, axis=1)
    support_ids = np.where(interval_mask, 0, -1).astype(int)
    final_features = dict(final_features)
    final_features["point_scores"] = final_point_scores
    final_features["frame_scores"] = frame_scores
    final_features["quiet_mask"] = quiet_masks

    return ContactDetectionResult(
        intervals=intervals,
        mask=interval_mask,
        scores=frame_scores,
        support_ids=support_ids,
        features=final_features,
        support_models=final_support_models,
        debug={
            "quiet_debug": quiet_debug,
            "iterations": iteration_debug,
            "config": config,
        },
    )


def fit_support_model_from_candidates(
    points: np.ndarray,
    point_mask: np.ndarray,
    config: SupportDetectionConfig,
) -> SupportModel:
    candidate_points = points[np.asarray(point_mask, dtype=bool)]
    if len(candidate_points) < config.min_support_points:
        fallback_count = min(max(config.min_support_points, 3), points.shape[0] * points.shape[1])
        flattened = points.reshape(-1, 3)
        order = np.argsort(flattened[:, config.up_axis])
        candidate_points = flattened[order[:fallback_count]]

    plane = PlaneSupportModel.fit(candidate_points, config)
    residuals = np.abs(plane.clearance(candidate_points))
    coplanar_ratio = float(np.mean(residuals <= config.plane_residual_tolerance))
    if coplanar_ratio >= config.coplanarity_ratio_threshold:
        return plane

    return HeightmapSupportModel.fit(candidate_points, config)


def compute_support_relative_features(
    points: np.ndarray,
    velocities: np.ndarray,
    support_model: SupportModel,
    quiet_debug: list[dict[str, object]],
    config: ContactDetectionConfig,
    t: np.ndarray | None = None,
    orientations=None,
    reference_force=None,
) -> dict[str, np.ndarray]:
    n_frames, n_points, _ = points.shape
    flat_points = points.reshape(-1, 3)
    flat_velocities = velocities.reshape(-1, 3)

    clearance = support_model.clearance(flat_points).reshape(n_frames, n_points)
    normals = support_model.normals_at(flat_points)
    normal_velocity = np.sum(flat_velocities * normals, axis=1).reshape(n_frames, n_points)
    tangent_velocity = flat_velocities - np.sum(flat_velocities * normals, axis=1)[:, None] * normals
    tangential_speed = np.linalg.norm(tangent_velocity, axis=1).reshape(n_frames, n_points)

    quiet_activity = np.column_stack([debug["activity"] for debug in quiet_debug])
    quiet_spread = np.column_stack([debug["spread"] for debug in quiet_debug])
    activity_scale = np.asarray([debug["activity_off_eps"] for debug in quiet_debug], dtype=float)
    spread_scale = np.asarray([debug["spread_off_eps"] for debug in quiet_debug], dtype=float)
    activity_scale = np.where(activity_scale > 1e-12, activity_scale, 1.0)
    spread_scale = np.where(spread_scale > 1e-12, spread_scale, 1.0)

    angular_speed = np.zeros((n_frames, n_points), dtype=float)
    if orientations is not None:
        if t is None:
            raise ValueError("t is required when orientations are provided.")
        angular_speed = _estimate_orientation_activity(
            orientations,
            t,
            n_points,
            n_frames,
            scalar_last=config.quiet_config.quaternion_scalar_last,
        )

    reference_contact = None
    if reference_force is not None:
        reference_contact = _normalize_reference_force(reference_force, n_frames, n_points)

    return {
        "clearance": clearance,
        "abs_clearance": np.abs(clearance),
        "normal_speed": np.abs(normal_velocity),
        "signed_normal_velocity": normal_velocity,
        "tangential_speed": tangential_speed,
        "quiet_activity": quiet_activity,
        "quiet_spread": quiet_spread,
        "quiet_activity_scale": activity_scale[None, :],
        "quiet_spread_scale": spread_scale[None, :],
        "angular_speed": angular_speed,
        "reference_contact": reference_contact,
    }


def score_contact_features(features: dict[str, np.ndarray], config: ContactDetectionConfig) -> np.ndarray:
    weights = config.feature_weights

    penalty = np.zeros_like(features["abs_clearance"], dtype=float)
    penalty += weights.get("clearance", 0.0) * _squared_ratio(features["abs_clearance"], config.clearance_scale)
    penalty += weights.get("normal_speed", 0.0) * _squared_ratio(features["normal_speed"], config.normal_speed_scale)
    penalty += weights.get("tangential_speed", 0.0) * _squared_ratio(features["tangential_speed"], config.tangential_speed_scale)
    penalty += weights.get("quiet_activity", 0.0) * _squared_ratio(
        features["quiet_activity"],
        features["quiet_activity_scale"],
    )
    penalty += weights.get("quiet_spread", 0.0) * _squared_ratio(
        features["quiet_spread"],
        features["quiet_spread_scale"],
    )
    penalty += weights.get("angular_speed", 0.0) * _squared_ratio(features["angular_speed"], config.angular_speed_scale)

    scores = np.exp(-0.5 * penalty)
    reference_contact = features.get("reference_contact")
    if reference_contact is not None:
        ref_weight = weights.get("reference_force", 0.0)
        scores = (scores + ref_weight * reference_contact) / (1.0 + ref_weight)

    return np.clip(scores, 0.0, 1.0)


def _detect_point_quietness(t, points, quiet_config: QuietDetectionConfig):
    quiet_config = QuietDetectionConfig(
        signal_type=QuietSignalType.VECTOR_POSITION,
        vector_mode=quiet_config.vector_mode,
        contact_min_time=quiet_config.contact_min_time,
        max_gap_time=quiet_config.max_gap_time,
        min_blip_time=quiet_config.min_blip_time,
        pos_smooth_time=quiet_config.pos_smooth_time,
        vel_smooth_time=quiet_config.vel_smooth_time,
        quiet_window_time=quiet_config.quiet_window_time,
        spread_window_time=quiet_config.spread_window_time,
        derivative_window_time=quiet_config.derivative_window_time,
        derivative_poly_degree=quiet_config.derivative_poly_degree,
        activity_on_eps=quiet_config.activity_on_eps,
        activity_off_eps=quiet_config.activity_off_eps,
        spread_on_eps=quiet_config.spread_on_eps,
        spread_off_eps=quiet_config.spread_off_eps,
        min_activity_on_eps=quiet_config.min_activity_on_eps,
        min_activity_off_eps=quiet_config.min_activity_off_eps,
        min_spread_on_eps=quiet_config.min_spread_on_eps,
        min_spread_off_eps=quiet_config.min_spread_off_eps,
        use_time_gaussian_smoothing=quiet_config.use_time_gaussian_smoothing,
        use_time_windows=quiet_config.use_time_windows,
        quaternion_scalar_last=quiet_config.quaternion_scalar_last,
    )

    n_points = points.shape[1]
    masks = []
    debug = []
    for point_idx in range(n_points):
        _, mask, point_debug = detect_quiet_intervals(t, points[:, point_idx, :], config=quiet_config)
        masks.append(mask)
        debug.append(point_debug)
    return np.column_stack(masks), debug


def _estimate_orientation_activity(orientations, t, n_points, n_frames, scalar_last=True):
    try:
        from .silence_detection import quaternion_angular_speed, quaternion_standardize_xyzw
    except ImportError:  # pragma: no cover
        from silence_detection import quaternion_angular_speed, quaternion_standardize_xyzw

    orientations = np.asarray(orientations, dtype=float)
    if orientations.shape == (n_frames, 4):
        orientations = orientations[:, None, :]
    if orientations.shape != (n_frames, n_points, 4):
        raise ValueError("orientations must have shape (N, 4) or (N, K, 4).")

    angular_speed = np.zeros((n_frames, n_points), dtype=float)
    for point_idx in range(n_points):
        q_xyzw = quaternion_standardize_xyzw(orientations[:, point_idx, :], scalar_last=scalar_last)
        angular_speed[:, point_idx] = quaternion_angular_speed(
            q_xyzw,
            t,
        )
    return angular_speed


def _normalize_reference_force(reference_force, n_frames, n_points):
    ref = np.asarray(reference_force, dtype=float)
    if ref.shape == (n_frames,):
        ref = ref[:, None]
    if ref.shape == (n_frames, 1):
        ref = np.repeat(ref, n_points, axis=1)
    if ref.shape != (n_frames, n_points):
        raise ValueError("reference_force must have shape (N,), (N, 1), or (N, K).")

    finite = ref[np.isfinite(ref)]
    if finite.size == 0:
        return np.zeros((n_frames, n_points), dtype=float)
    high = np.percentile(finite, 95)
    if high <= 1e-12:
        return np.zeros((n_frames, n_points), dtype=float)
    return np.clip(ref / high, 0.0, 1.0)


def _normalize_supports(supports):
    if supports is None:
        return []
    if isinstance(supports, (PlaneSupportModel, HeightmapSupportModel)):
        return [supports]
    return list(supports)


def _squared_ratio(value, scale):
    scale = np.asarray(scale, dtype=float)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (value / scale) ** 2


def _validate_points_over_time(points):
    points = np.asarray(points, dtype=float)
    if points.ndim == 2 and points.shape[1] == 3:
        points = points[:, None, :]
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape (N, 3) or (N, K, 3).")
    return points


def _validate_point_cloud(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (M, 3).")
    return points


def _fit_plane_svd(points, up_axis):
    origin = np.mean(points, axis=0)
    centered = points - origin[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = _orient_normal_up(normal, up_axis)
    return origin, normal


def _orient_normal_up(normal, up_axis):
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise ValueError("normal must be non-zero.")
    normal = normal / norm
    if normal[up_axis] < 0.0:
        normal = -normal
    return normal


def _horizontal_axes(up_axis):
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2.")
    return [axis for axis in range(3) if axis != up_axis]
