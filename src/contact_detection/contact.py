"""Support-surface fitting and marker-based contact interval detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .enums import SupportModelType, normalize_enum
from .types import BoolArray, DebugDict, FloatArray, IntervalList

from .quiet import (
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
    """Parameters for fitting ground/support geometry from quiet marker samples."""

    model_type: SupportModelType | str = SupportModelType.AUTO
    plane_residual_tolerance: float = 0.025
    coplanarity_ratio_threshold: float = 0.85
    heightmap_cell_size: float = 0.10
    heightmap_quantile: float = 0.20
    local_heightmap_radius: float = 0.25
    local_heightmap_quantile: float = 0.20
    local_heightmap_min_neighbors: int = 3
    min_support_points: int = 20
    max_bootstrap_iterations: int = 3
    convergence_tolerance: float = 0.01
    ransac_iterations: int = 128
    random_seed: int = 17
    up_axis: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_type", normalize_enum(self.model_type, SupportModelType))


@dataclass(frozen=True)
class ContactDetectionConfig:
    """Thresholds and feature weights for contact scoring and mask cleanup."""

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
    contact_offset_max: float | None = None
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
    """Output of frame-level contact detection on a marker point cloud."""

    intervals: IntervalList
    mask: BoolArray
    scores: FloatArray
    support_ids: NDArray[np.int_]
    features: dict[str, FloatArray]
    support_models: list[SupportModel]
    debug: DebugDict
    point_mask: BoolArray | None = None


class SupportModel(Protocol):
    """Protocol for geometry that exposes clearance and surface normals."""

    name: str

    def clearance(self, points: ArrayLike) -> FloatArray:
        """Signed distance above the support surface (positive = above)."""
        ...

    def normals_at(self, points: ArrayLike) -> FloatArray:
        """Unit surface normal at each query point."""
        ...


@dataclass
class SupportCandidate:
    """A quiet interval summarized as a single 3D support sample."""

    point: FloatArray
    keypoint_index: int
    start: float
    end: float
    duration: float
    spread: float
    activity: float


@dataclass
class SupportCandidateSet:
    """Collection of support candidates gathered from per-marker quiet intervals."""

    candidates: list[SupportCandidate]

    @property
    def points(self) -> FloatArray:
        """Stack candidate points into an ``(M, 3)`` array."""
        if not self.candidates:
            return np.empty((0, 3), dtype=float)
        return np.asarray([candidate.point for candidate in self.candidates], dtype=float)


@dataclass
class PlaneSupportModel:
    """Planar support surface fit with RANSAC refinement."""

    normal: FloatArray
    origin: FloatArray
    residuals: FloatArray | None = None
    inlier_mask: BoolArray | None = None
    name: str = "plane"

    @classmethod
    def fit(cls, points: ArrayLike, config: SupportDetectionConfig) -> PlaneSupportModel:
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

    def clearance(self, points: ArrayLike) -> FloatArray:
        points = np.asarray(points, dtype=float)
        return (points - self.origin) @ self.normal

    def normals_at(self, points: ArrayLike) -> FloatArray:
        points = np.asarray(points, dtype=float)
        return np.broadcast_to(self.normal, points.shape).copy()


@dataclass
class HeightmapSupportModel:
    """Piecewise height field on a regular XY grid."""

    cell_size: float
    cells_xy: FloatArray
    heights: FloatArray
    up_axis: int = 2
    name: str = "heightmap"
    _tree: cKDTree | None = field(default=None, init=False, repr=False)

    @classmethod
    def fit(cls, points: ArrayLike, config: SupportDetectionConfig) -> HeightmapSupportModel:
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

    def _height_at_xy(self, xy: ArrayLike) -> FloatArray:
        xy = np.asarray(xy, dtype=float)
        if self._tree is None:
            self._tree = cKDTree(self.cells_xy)
        _, idx = self._tree.query(xy, k=1)
        return self.heights[idx]

    def clearance(self, points: ArrayLike) -> FloatArray:
        points = np.asarray(points, dtype=float)
        horizontal_axes = _horizontal_axes(self.up_axis)
        support_height = self._height_at_xy(points[:, horizontal_axes])
        return points[:, self.up_axis] - support_height

    def normals_at(self, points: ArrayLike) -> FloatArray:
        points = np.asarray(points, dtype=float)
        normals = np.zeros_like(points, dtype=float)
        normals[:, self.up_axis] = 1.0
        return normals


@dataclass
class LocalPercentileHeightmap:
    """Neighborhood percentile height field for uneven terrain."""

    points: FloatArray
    radius: float
    quantile: float
    min_neighbors: int
    up_axis: int = 2
    name: str = "local_heightmap"
    _tree: cKDTree | None = field(default=None, init=False, repr=False)

    @classmethod
    def fit(cls, points: ArrayLike, config: SupportDetectionConfig) -> LocalPercentileHeightmap:
        points = _validate_point_cloud(points)
        if len(points) < 1:
            raise ValueError("Need at least 1 point to fit a local heightmap.")
        radius = float(config.local_heightmap_radius)
        if radius <= 0.0:
            raise ValueError("local_heightmap_radius must be positive.")
        model = cls(
            points=points.copy(),
            radius=radius,
            quantile=float(config.local_heightmap_quantile),
            min_neighbors=int(config.local_heightmap_min_neighbors),
            up_axis=config.up_axis,
        )
        model._tree = cKDTree(points[:, _horizontal_axes(config.up_axis)])
        return model

    def _height_at_xy(self, xy: ArrayLike) -> FloatArray:
        xy = np.asarray(xy, dtype=float)
        if self._tree is None:
            self._tree = cKDTree(self.points[:, _horizontal_axes(self.up_axis)])

        heights = np.empty(len(xy), dtype=float)
        support_z = self.points[:, self.up_axis]
        k = min(max(self.min_neighbors, 1), len(self.points))

        for i, query in enumerate(xy):
            neighbor_idx = self._tree.query_ball_point(query, r=self.radius)
            if len(neighbor_idx) < self.min_neighbors:
                _, nearest_idx = self._tree.query(query, k=k)
                neighbor_idx = np.atleast_1d(nearest_idx).astype(int).tolist()
            values = support_z[neighbor_idx]
            if values.size == 0:
                _, nearest_idx = self._tree.query(query, k=1)
                values = np.asarray([support_z[int(nearest_idx)]])
            heights[i] = float(np.quantile(values, self.quantile))

        return heights

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


def find_support_candidates(
    t: ArrayLike,
    points: ArrayLike,
    quiet_config: QuietDetectionConfig | None = None,
    min_duration: float | None = None,
) -> SupportCandidateSet:
    """Find quiet intervals per marker and summarize each as a support candidate.

    Parameters
    ----------
    t:
        Timestamps with shape ``(N,)``.
    points:
        Marker positions with shape ``(N, K, 3)`` or ``(N, 3)``.
    quiet_config:
        Quiet-detection settings applied independently to each marker.
    min_duration:
        Override minimum quiet interval length; defaults to the config value.

    Returns
    -------
    SupportCandidateSet
        One candidate per qualifying quiet interval and marker.
    """

    t = np.asarray(t, dtype=float)
    points = _validate_points_over_time(points)
    quiet_config = quiet_config or QuietDetectionConfig(
        signal_type=QuietSignalType.VECTOR_POSITION,
        vector_mode=VectorQuietMode.EUCLIDEAN,
    )
    if quiet_config.signal_type != QuietSignalType.VECTOR_POSITION:
        quiet_config = QuietDetectionConfig(
            signal_type=QuietSignalType.VECTOR_POSITION,
            vector_mode=quiet_config.vector_mode,
            min_interval_time=quiet_config.min_interval_time,
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

    min_duration = quiet_config.min_interval_time if min_duration is None else min_duration
    candidates: list[SupportCandidate] = []

    for keypoint_index in range(points.shape[1]):
        result = detect_quiet_intervals(t, points[:, keypoint_index, :], config=quiet_config)
        for start, end in result.intervals:
            duration = float(end - start)
            if duration < min_duration:
                continue
            interval_mask = (t >= start) & (t <= end)
            if not interval_mask.any():
                continue
            candidates.append(
                SupportCandidate(
                    point=np.median(points[interval_mask, keypoint_index, :], axis=0),
                    keypoint_index=keypoint_index,
                    start=float(start),
                    end=float(end),
                    duration=duration,
                    spread=float(np.median(result.spread[interval_mask])),
                    activity=float(np.median(result.activity[interval_mask])),
                )
            )

    return SupportCandidateSet(candidates)


def filter_support_candidates(
    candidates: SupportCandidateSet | Sequence[SupportCandidate],
    up_axis: int = 2,
    low_quantile: float = 0.50,
    max_spread: float | None = None,
    min_duration: float | None = None,
) -> SupportCandidateSet:
    """Keep low, quiet support candidates suitable for surface bootstrapping."""

    candidate_list = candidates.candidates if isinstance(candidates, SupportCandidateSet) else list(candidates)
    if not candidate_list:
        return SupportCandidateSet([])

    filtered = candidate_list
    if min_duration is not None:
        filtered = [candidate for candidate in filtered if candidate.duration >= min_duration]
    if max_spread is not None:
        filtered = [candidate for candidate in filtered if candidate.spread <= max_spread]
    if not filtered:
        return SupportCandidateSet([])

    heights = np.asarray([candidate.point[up_axis] for candidate in filtered], dtype=float)
    cutoff = float(np.quantile(heights, low_quantile))
    filtered = [candidate for candidate in filtered if candidate.point[up_axis] <= cutoff]
    return SupportCandidateSet(filtered)


def bootstrap_support_surface(
    t: ArrayLike,
    points: ArrayLike,
    config: SupportDetectionConfig | None = None,
    quiet_config: QuietDetectionConfig | None = None,
) -> tuple[SupportModel, DebugDict]:
    """Estimate a support surface from quiet marker samples.

    Returns the fitted model and a debug dictionary with intermediate candidates.
    """

    config = config or SupportDetectionConfig()
    points = _validate_points_over_time(points)
    candidates = find_support_candidates(t, points, quiet_config=quiet_config)
    filtered = filter_support_candidates(
        candidates,
        up_axis=config.up_axis,
        min_duration=(quiet_config.min_interval_time if quiet_config else None),
    )

    candidate_points = filtered.points
    if len(candidate_points) < config.min_support_points:
        flattened = points.reshape(-1, 3)
        fallback_count = min(max(config.min_support_points, 3), len(flattened))
        order = np.argsort(flattened[:, config.up_axis])
        candidate_points = flattened[order[:fallback_count]]

    support_surface = fit_best_support_surface(candidate_points, config)
    debug = {
        "support_candidates": candidates,
        "filtered_support_candidates": filtered,
        "candidate_points": candidate_points,
        "support_model": support_surface,
    }
    return support_surface, debug


def detect_contact_intervals(
    t: ArrayLike,
    points: ArrayLike,
    config: ContactDetectionConfig | None = None,
    orientations: ArrayLike | None = None,
    supports: SupportModel | Sequence[SupportModel] | None = None,
    reference_force: ArrayLike | None = None,
) -> ContactDetectionResult:
    """Detect contact intervals from marker motion relative to a support surface.

    Parameters
    ----------
    t:
        Timestamps with shape ``(N,)``.
    points:
        Marker positions with shape ``(N, K, 3)`` or ``(N, 3)``.
    config:
        Contact scoring, quiet detection, and support-model settings.
    orientations:
        Optional per-frame quaternions with shape ``(N, 4)`` or ``(N, K, 4)``.
    supports:
        Pre-fit support model(s). When omitted, the surface is bootstrapped
        iteratively from high-confidence contact samples.
    reference_force:
        Optional normalized force proxy with shape ``(N,)``, ``(N, 1)``, or
        ``(N, K)``.

    Returns
    -------
    ContactDetectionResult
        Frame mask, per-point scores, fitted support models, and debug metadata.
    """

    config = config or ContactDetectionConfig()
    t = np.asarray(t, dtype=float)
    points = _validate_points_over_time(points)

    if config.moving_support_mode:
        raise NotImplementedError("moving_support_mode is planned but not implemented yet")
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
    point_contact_mask = (final_point_scores >= config.score_on_threshold) & interval_mask[:, None]
    support_ids = np.where(interval_mask, 0, -1).astype(int)
    final_features = dict(final_features)
    final_features["point_scores"] = final_point_scores
    final_features["point_contact_mask"] = point_contact_mask
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
        point_mask=point_contact_mask,
    )


def fit_support_model_from_candidates(
    points: ArrayLike,
    point_mask: ArrayLike,
    config: SupportDetectionConfig,
) -> SupportModel:
    """Fit a support model from marker samples flagged in ``point_mask``."""

    candidate_points = points[np.asarray(point_mask, dtype=bool)]
    if len(candidate_points) < config.min_support_points:
        fallback_count = min(max(config.min_support_points, 3), points.shape[0] * points.shape[1])
        flattened = points.reshape(-1, 3)
        order = np.argsort(flattened[:, config.up_axis])
        candidate_points = flattened[order[:fallback_count]]

    return fit_best_support_surface(candidate_points, config)


def fit_best_support_surface(points: ArrayLike, config: SupportDetectionConfig) -> SupportModel:
    """Choose plane vs heightmap support geometry from point coplanarity."""

    points = _validate_point_cloud(points)
    model_type = normalize_enum(config.model_type, SupportModelType)
    if model_type == SupportModelType.PLANE:
        return PlaneSupportModel.fit(points, config)
    if model_type == SupportModelType.HEIGHTMAP:
        return HeightmapSupportModel.fit(points, config)
    if model_type == SupportModelType.LOCAL_HEIGHTMAP:
        return LocalPercentileHeightmap.fit(points, config)
    if model_type != SupportModelType.AUTO:
        raise ValueError(f"Unsupported support model_type: {model_type}")

    plane = PlaneSupportModel.fit(points, config)
    residuals = np.abs(plane.clearance(points))
    coplanar_ratio = float(np.mean(residuals <= config.plane_residual_tolerance))
    if coplanar_ratio >= config.coplanarity_ratio_threshold:
        return plane

    return HeightmapSupportModel.fit(points, config)


def compute_support_relative_features(
    points: ArrayLike,
    velocities: ArrayLike,
    support_model: SupportModel,
    quiet_debug: list[DebugDict],
    config: ContactDetectionConfig,
    t: ArrayLike | None = None,
    orientations: ArrayLike | None = None,
    reference_force: ArrayLike | None = None,
) -> dict[str, FloatArray]:
    """Build clearance, speed, quietness, and optional auxiliary contact features."""
    n_frames, n_points, _ = points.shape
    flat_points = points.reshape(-1, 3)
    flat_velocities = velocities.reshape(-1, 3)

    raw_clearance = support_model.clearance(flat_points).reshape(n_frames, n_points)
    normals = support_model.normals_at(flat_points)
    normal_velocity = np.sum(flat_velocities * normals, axis=1).reshape(n_frames, n_points)
    tangent_velocity = flat_velocities - np.sum(flat_velocities * normals, axis=1)[:, None] * normals
    tangential_speed = np.linalg.norm(tangent_velocity, axis=1).reshape(n_frames, n_points)

    quiet_activity = np.column_stack([debug["activity"] for debug in quiet_debug])
    quiet_spread = np.column_stack([debug["spread"] for debug in quiet_debug])
    quiet_mask = np.column_stack([debug["quiet_mask"] for debug in quiet_debug])
    activity_scale = np.asarray([debug["activity_off_eps"] for debug in quiet_debug], dtype=float)
    spread_scale = np.asarray([debug["spread_off_eps"] for debug in quiet_debug], dtype=float)
    activity_scale = np.where(activity_scale > 1e-12, activity_scale, 1.0)
    spread_scale = np.where(spread_scale > 1e-12, spread_scale, 1.0)

    offset_limit = config.contact_offset_max
    if offset_limit is None:
        offset_limit = 1.5 * config.clearance_scale
    contact_offset = np.asarray(
        [
            estimate_contact_offset(raw_clearance[:, point_idx], quiet_mask[:, point_idx], max_abs_offset=offset_limit)
            for point_idx in range(n_points)
        ],
        dtype=float,
    )
    clearance = raw_clearance - contact_offset[None, :]

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
        "raw_clearance": raw_clearance,
        "clearance": clearance,
        "abs_clearance": np.abs(clearance),
        "normal_speed": np.abs(normal_velocity),
        "signed_normal_velocity": normal_velocity,
        "tangential_speed": tangential_speed,
        "quiet_activity": quiet_activity,
        "quiet_spread": quiet_spread,
        "quiet_mask": quiet_mask,
        "quiet_activity_scale": activity_scale[None, :],
        "quiet_spread_scale": spread_scale[None, :],
        "contact_offset": contact_offset,
        "angular_speed": angular_speed,
        "reference_contact": reference_contact,
    }


def score_contact_features(
    features: dict[str, FloatArray],
    config: ContactDetectionConfig,
) -> FloatArray:
    """Map support-relative features to per-marker contact scores in ``[0, 1]``."""
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


def estimate_contact_offset(
    clearance: ArrayLike,
    quiet_mask: ArrayLike,
    quantile: float = 0.20,
    max_abs_offset: float = 0.045,
) -> float:
    """Estimate a per-marker clearance bias from quiet, near-contact samples."""

    clearance = np.asarray(clearance, dtype=float)
    quiet_mask = np.asarray(quiet_mask, dtype=bool)
    candidate_clearance = clearance[quiet_mask & np.isfinite(clearance)]
    if candidate_clearance.size == 0:
        return 0.0

    low_cutoff = np.quantile(candidate_clearance, min(max(quantile, 0.0), 1.0))
    low_clearance = candidate_clearance[candidate_clearance <= low_cutoff]
    if low_clearance.size == 0:
        low_clearance = candidate_clearance

    offset = float(np.median(low_clearance))
    if abs(offset) > max_abs_offset:
        return 0.0
    return offset


def _detect_point_quietness(
    t: ArrayLike,
    points: FloatArray,
    quiet_config: QuietDetectionConfig,
) -> tuple[BoolArray, list[DebugDict]]:
    quiet_config = QuietDetectionConfig(
        signal_type=QuietSignalType.VECTOR_POSITION,
        vector_mode=quiet_config.vector_mode,
        min_interval_time=quiet_config.min_interval_time,
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
        quiet_result = detect_quiet_intervals(t, points[:, point_idx, :], config=quiet_config)
        masks.append(quiet_result.mask)
        debug.append(quiet_result.debug)
    return np.column_stack(masks), debug


def _estimate_orientation_activity(
    orientations: ArrayLike,
    t: ArrayLike,
    n_points: int,
    n_frames: int,
    scalar_last: bool = True,
) -> FloatArray:
    from .quiet import quaternion_angular_speed, quaternion_standardize_xyzw

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


def _normalize_reference_force(
    reference_force: ArrayLike,
    n_frames: int,
    n_points: int,
) -> FloatArray:
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


def _normalize_supports(
    supports: SupportModel | Sequence[SupportModel] | None,
) -> list[SupportModel]:
    if supports is None:
        return []
    if isinstance(supports, (PlaneSupportModel, HeightmapSupportModel, LocalPercentileHeightmap)):
        return [supports]
    if hasattr(supports, "clearance") and hasattr(supports, "normals_at"):
        return [supports]
    return list(supports)


def _squared_ratio(value: ArrayLike, scale: ArrayLike) -> FloatArray:
    scale = np.asarray(scale, dtype=float)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (value / scale) ** 2


def _validate_points_over_time(points: ArrayLike) -> FloatArray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 2 and points.shape[1] == 3:
        points = points[:, None, :]
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape (N, 3) or (N, K, 3).")
    return points


def _validate_point_cloud(points: ArrayLike) -> FloatArray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (M, 3).")
    return points


def fit_plane_svd(points: ArrayLike, up_axis: int = 2) -> tuple[FloatArray, FloatArray]:
    """Fit a plane with SVD and return ``(origin, unit_normal)``."""

    return _fit_plane_svd(_validate_point_cloud(points), up_axis)


def _fit_plane_svd(points: FloatArray, up_axis: int) -> tuple[FloatArray, FloatArray]:
    origin = np.mean(points, axis=0)
    centered = points - origin[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = _orient_normal_up(normal, up_axis)
    return origin, normal


def _orient_normal_up(normal: ArrayLike, up_axis: int) -> FloatArray:
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise ValueError("normal must be non-zero.")
    normal = normal / norm
    if normal[up_axis] < 0.0:
        normal = -normal
    return normal


def _horizontal_axes(up_axis: int) -> list[int]:
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2.")
    return [axis for axis in range(3) if axis != up_axis]
