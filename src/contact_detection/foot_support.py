"""Per-foot air, ground, and skateboard support classification from rigid-body motion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .contact import PlaneSupportModel, SupportDetectionConfig
from .enums import FloorModel, SupportModelType, normalize_enum
from .geometry import (
    ContactPatch,
    RigidBodyContactModel,
    RigidTransform,
    apply_contact_model_offsets,
    compile_contact_models,
    marker_names_for_flat_trajectory,
    marker_names_for_contact_models,
    model_for_marker,
    model_map,
)
from .intervals import clean_mask_by_time, intervals_from_mask
from .quiet import local_polynomial_derivative
from .types import BoolArray, FloatArray, IntervalList

StateArray: TypeAlias = NDArray[np.int8]
FeatureArray: TypeAlias = FloatArray | BoolArray
StateIntervals: TypeAlias = dict[str, IntervalList]


class FootSupportState(IntEnum):
    """Per-frame support state for a tracked foot or contact point.

    Attributes:
        AIR (int): Foot is airborne (no ground or board contact).
        GROUND (int): Foot contacts the calibrated floor support surface.
        SKATEBOARD (int): Foot contacts the skateboard deck (horizontal proximity and low relative motion).
    """

    AIR = 0
    GROUND = 1
    SKATEBOARD = 2


STATE_LABELS: dict[FootSupportState, str] = {
    FootSupportState.AIR: "air",
    FootSupportState.GROUND: "ground",
    FootSupportState.SKATEBOARD: "skateboard",
}

STATE_COLORS: dict[FootSupportState, str] = {
    FootSupportState.AIR: "#d8dbe2",
    FootSupportState.GROUND: "#63a46c",
    FootSupportState.SKATEBOARD: "#4f7fcf",
}


@dataclass(frozen=True)
class FootSupportConfig:
    """Thresholds and body names used for per-foot support classification.

    Floor geometry uses :attr:`FloorModel.PLANE` fit from sole-surface samples on
    provisional then refined **ground-contact** frames. Sole-based trials provide
    :attr:`contact_models`, ``floor_fit_marker_pos``, and ``body_rotations``.

    Attributes:
        foot_names (tuple[str, str]): Rigid-body names for left and right feet.
        board_name (str): Rigid-body name for the skateboard deck.
        up_axis (int): World-axis index treated as vertical (0, 1, or 2).
        floor_model (FloorModel | str): Must be ``plane`` (scalar height model removed).
        provisional_ground_percentile (float): Percentile on sole/body height used to seed ground intervals.
        provisional_ground_height_slack (float): Extra vertical slack (m) when seeding ground intervals.
        floor_fit_refinement_passes (int): Refit plane after updating ground-contact masks (>= 1).
        floor_plane_residual_tolerance (float): Inlier threshold when fitting the floor plane.
        floor_plane_ransac_iterations (int): RANSAC iterations for plane fitting.
        floor_plane_random_seed (int): RNG seed for plane RANSAC.
        min_floor_fit_samples (int): Minimum finite samples required to fit a plane.
        ground_clearance_tolerance (float): Max |clearance| for ground contact (meters).
        ground_speed_tolerance (float): Max foot speed for ground contact (m/s).
        board_horizontal_tolerance (float): Max horizontal foot-board distance (meters).
        board_vertical_tolerance (float): Max |relative height - offset| for board contact.
        board_min_relative_height (float): Lower bound on foot height above board (meters).
        board_max_relative_height (float): Upper bound on foot height above board (meters).
        board_relative_speed_tolerance (float): Max |v_foot - v_board| for moving board (m/s).
        static_board_speed_tolerance (float): Max board speed for static board branch (m/s).
        static_board_foot_speed_tolerance (float): Max foot speed for static board branch (m/s).
        default_board_contact_offset (float): Fallback foot-board vertical offset (meters).
        min_board_offset_samples (int): Min samples to estimate offset from data.
        velocity_window_time (float): Window for polynomial velocity estimation (seconds).
        max_gap_time (float): Max gap to fill in state masks (seconds).
        min_state_time (float): Min duration for a state blip to survive cleaning (seconds).
        contact_models (tuple[RigidBodyContactModel[Any], ...]): Body-local sole/contact models.
        sole_patch_names (Mapping[str, str] | None): Optional body name → patch name mapping.
        floor_fit_marker_names (tuple[str, ...] | None): Marker names for sole samples / floor fit.
    """

    foot_names: tuple[str, str] = ("Left_Shoe", "Right_Shoe")
    board_name: str = "Skateboard"
    up_axis: int = 2
    floor_model: FloorModel | str = FloorModel.PLANE
    provisional_ground_percentile: float = 25.0
    provisional_ground_height_slack: float = 0.02
    floor_fit_refinement_passes: int = 2
    floor_plane_residual_tolerance: float = 0.025
    floor_plane_ransac_iterations: int = 128
    floor_plane_random_seed: int = 17
    min_floor_fit_samples: int = 12
    ground_clearance_tolerance: float = 0.025
    ground_speed_tolerance: float = 0.18
    board_horizontal_tolerance: float = 0.35
    board_vertical_tolerance: float = 0.035
    board_min_relative_height: float = 0.0
    board_max_relative_height: float = 0.14
    board_relative_speed_tolerance: float = 0.12
    static_board_speed_tolerance: float = 0.05
    static_board_foot_speed_tolerance: float = 0.15
    default_board_contact_offset: float = 0.055
    min_board_offset_samples: int = 10
    velocity_window_time: float = 0.08
    max_gap_time: float = 0.10
    min_state_time: float = 0.12
    contact_models: tuple[RigidBodyContactModel[Any], ...] = ()
    sole_patch_names: Mapping[str, str] | None = None
    floor_fit_marker_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        floor_model = normalize_enum(self.floor_model, FloorModel)
        if floor_model != FloorModel.PLANE:
            raise ValueError(
                f"floor_model must be {FloorModel.PLANE.value!r}; "
                f"scalar {FloorModel.HEIGHT.value!r} fitting was removed."
            )
        object.__setattr__(self, "floor_model", floor_model)
        if self.floor_fit_refinement_passes < 1:
            raise ValueError("floor_fit_refinement_passes must be >= 1.")
        contact_models = tuple(self.contact_models or ())
        object.__setattr__(self, "contact_models", contact_models)
        if contact_models and self.floor_fit_marker_names is None:
            object.__setattr__(self, "floor_fit_marker_names", marker_names_for_contact_models(contact_models))


@dataclass
class FootSupportClassification:
    """Output of per-foot support classification.

    Attributes:
        t (FloatArray): Timestamps with shape ``(N,)``.
        states (dict[str, StateArray]): Per-foot :class:`FootSupportState` arrays.
        floor_model (FloorModel): Floor geometry used for ground clearance.
        floor_height (float): Scalar floor height (median reference for plane floors).
        floor_normal (FloatArray | None): Unit normal when ``floor_model`` is plane.
        floor_origin (FloatArray | None): Plane origin when ``floor_model`` is plane.
        board_contact_offsets (dict[str, float]): Estimated foot-board vertical offset per foot.
        features (dict[str, dict[str, FeatureArray]]): Diagnostic traces per foot name.
        intervals (dict[str, StateIntervals]): State intervals keyed by foot then state label.
    """

    t: FloatArray
    states: dict[str, StateArray]
    floor_model: FloorModel
    floor_height: float
    floor_normal: FloatArray | None
    floor_origin: FloatArray | None
    board_contact_offsets: dict[str, float]
    features: dict[str, dict[str, FeatureArray]]
    intervals: dict[str, StateIntervals] = field(default_factory=dict)


def classify_foot_support_states(
    t: ArrayLike,
    body_names: Sequence[str],
    body_pos: ArrayLike,
    config: FootSupportConfig | None = None,
    *,
    floor_fit_marker_pos: ArrayLike | None = None,
    floor_fit_marker_names: Sequence[str] | None = None,
    body_rotations: Mapping[str, FloatArray] | None = None,
) -> FootSupportClassification:
    """Classify each configured foot as air, ground, or skateboard over time.

    The floor plane is fit from sole-surface samples (marker patches with body-local
    offsets) on **ground-contact** frames, refined iteratively, then used with sole
    contact-frame clearance for the final ground mask.

    Args:
        t: Strictly increasing timestamps with shape ``(N,)``.
        body_names: Body names corresponding to axis 1 of ``body_pos``.
        body_pos: Body positions with shape ``(N, B, 3)``.
        config: Classification thresholds and body-name configuration.
        floor_fit_marker_pos: Marker positions with shape ``(N, M, 3)`` for sole patches.
        floor_fit_marker_names: Names for axis 1 of ``floor_fit_marker_pos``.
        body_rotations: Per-body quaternions ``(N, 4)`` for sole offsets and clearance.

    Returns:
        State arrays, intervals, floor diagnostics, board offsets, and per-foot features.

    Raises:
        ValueError: If inputs are malformed or required sole/marker data is missing.
    """

    config = config or FootSupportConfig()
    t = np.asarray(t, dtype=float)
    body_pos = np.asarray(body_pos, dtype=float)
    body_names = list(body_names)

    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if len(t) != len(body_pos):
        raise ValueError("t and body_pos must have the same number of frames.")
    if body_pos.ndim != 3 or body_pos.shape[2] != 3:
        raise ValueError("body_pos must have shape (N, B, 3).")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if config.up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2.")

    foot_indices = [_body_index(body_names, name) for name in config.foot_names]
    board_idx = _body_index(body_names, config.board_name)
    horizontal_axes = [axis for axis in range(3) if axis != config.up_axis]

    sole_patches = _compile_sole_patches(
        config,
        body_names=body_names,
        body_pos=body_pos,
        floor_fit_marker_pos=floor_fit_marker_pos,
        floor_fit_marker_names=floor_fit_marker_names,
        body_rotations=body_rotations,
    )
    velocities = local_polynomial_derivative(
        body_pos.reshape(len(t), -1),
        t,
        window_time=config.velocity_window_time,
        degree=1,
    ).reshape(body_pos.shape)

    board_pos = body_pos[:, board_idx, :]
    board_vel = velocities[:, board_idx, :]
    board_speed = np.linalg.norm(board_vel, axis=1)

    per_foot: dict[str, dict[str, Any]] = {}
    for foot_name, foot_idx in zip(config.foot_names, foot_indices):
        foot_pos = body_pos[:, foot_idx, :]
        foot_vel = velocities[:, foot_idx, :]
        foot_speed = np.linalg.norm(foot_vel, axis=1)
        relative_pos = foot_pos - board_pos
        relative_vel = foot_vel - board_vel
        relative_speed = np.linalg.norm(relative_vel, axis=1)
        horizontal_distance = np.linalg.norm(relative_pos[:, horizontal_axes], axis=1)
        relative_height = relative_pos[:, config.up_axis]
        clearance_point = _clearance_points(
            foot_pos,
            foot_name=foot_name,
            sole_patches=sole_patches,
            body_rotations=body_rotations,
        )
        per_foot[foot_name] = {
            "foot_pos": foot_pos,
            "foot_speed": foot_speed,
            "horizontal_distance": horizontal_distance,
            "relative_height": relative_height,
            "relative_speed": relative_speed,
            "clearance_point": clearance_point,
        }

    board_masks: dict[str, BoolArray] = {}
    board_contact_offsets: dict[str, float] = {}
    for foot_name, data in per_foot.items():
        board_offset = _estimate_board_contact_offset(
            data["horizontal_distance"],
            data["relative_height"],
            data["relative_speed"],
            config,
        )
        board_contact_offsets[foot_name] = board_offset
        board_geometry = (
            (data["horizontal_distance"] <= config.board_horizontal_tolerance)
            & (np.abs(data["relative_height"] - board_offset) <= config.board_vertical_tolerance)
        )
        moving_board_motion = data["relative_speed"] <= config.board_relative_speed_tolerance
        static_board_motion = (
            (board_speed <= config.static_board_speed_tolerance)
            & (data["foot_speed"] <= config.static_board_foot_speed_tolerance)
        )
        skateboard_mask = board_geometry & (moving_board_motion | static_board_motion)
        board_masks[foot_name] = clean_mask_by_time(
            t,
            skateboard_mask,
            max_gap_time=config.max_gap_time,
            min_blip_time=config.min_state_time,
        )

    floor_surface = _FloorSurface(
        model=FloorModel.PLANE,
        height=0.0,
        up_axis=config.up_axis,
        plane=None,
    )
    per_foot_ground_fit: dict[str, BoolArray] = {
        name: np.zeros(len(t), dtype=bool) for name in config.foot_names
    }
    for _pass in range(config.floor_fit_refinement_passes):
        if floor_surface.plane is None:
            per_foot_ground_fit = _provisional_ground_fit_masks(
                t,
                per_foot,
                board_masks,
                config,
            )
        else:
            refined = _refined_ground_fit_masks(
                t,
                per_foot,
                board_masks,
                floor_surface,
                config,
            )
            if any(np.any(mask) for mask in refined.values()):
                per_foot_ground_fit = refined
        ground_fit_mask = np.zeros(len(t), dtype=bool)
        for foot_mask in per_foot_ground_fit.values():
            ground_fit_mask |= foot_mask
        if not np.any(ground_fit_mask):
            raise ValueError("No ground-contact frames available for floor plane fitting.")
        fit_points = _floor_fit_samples_for_ground_masks(
            per_foot_ground_fit,
            config=config,
            body_pos=body_pos,
            foot_indices=foot_indices,
            floor_fit_marker_pos=floor_fit_marker_pos,
            floor_fit_marker_names=floor_fit_marker_names,
            body_rotations=body_rotations,
        )
        floor_surface = _fit_plane_floor(fit_points, config)
    ground_fit_mask = np.zeros(len(t), dtype=bool)
    for foot_mask in per_foot_ground_fit.values():
        ground_fit_mask |= foot_mask

    states: dict[str, StateArray] = {}
    features: dict[str, dict[str, FeatureArray]] = {}
    all_intervals: dict[str, StateIntervals] = {}

    for foot_name, data in per_foot.items():
        clearance_point = data["clearance_point"]
        floor_height_at_sole = floor_surface.height_at(clearance_point)
        ground_clearance = floor_surface.clearance(clearance_point)
        vertical_clearance = clearance_point[:, config.up_axis] - floor_height_at_sole
        skateboard_mask = board_masks[foot_name]

        ground_mask = (
            (np.abs(vertical_clearance) <= config.ground_clearance_tolerance)
            & (data["foot_speed"] <= config.ground_speed_tolerance)
        )
        ground_mask &= ~skateboard_mask
        ground_mask = clean_mask_by_time(
            t,
            ground_mask,
            max_gap_time=config.max_gap_time,
            min_blip_time=config.min_state_time,
        )
        ground_mask &= ~skateboard_mask

        state = np.full(len(t), FootSupportState.AIR, dtype=np.int8)
        state[ground_mask] = FootSupportState.GROUND
        state[skateboard_mask] = FootSupportState.SKATEBOARD
        states[foot_name] = state

        sole_height = clearance_point[:, config.up_axis]
        features[foot_name] = {
            "foot_height": data["foot_pos"][:, config.up_axis],
            "sole_height": sole_height,
            "board_height": board_pos[:, config.up_axis],
            "floor_height_at_foot": floor_height_at_sole,
            "floor_height_at_sole": floor_height_at_sole,
            "ground_clearance": ground_clearance,
            "vertical_ground_clearance": vertical_clearance,
            "horizontal_distance_to_board": data["horizontal_distance"],
            "relative_height_to_board": data["relative_height"],
            "foot_speed": data["foot_speed"],
            "board_speed": board_speed,
            "relative_speed_to_board": data["relative_speed"],
            "ground_mask": ground_mask,
            "ground_fit_mask": ground_fit_mask,
            "skateboard_mask": skateboard_mask,
        }
        all_intervals[foot_name] = intervals_by_state(t, state)

    if config.contact_models and (
        floor_fit_marker_pos is None or body_rotations is None
    ):
        raise ValueError(
            "contact_models require floor_fit_marker_pos and body_rotations."
        )

    return FootSupportClassification(
        t=t,
        states=states,
        floor_model=config.floor_model,
        floor_height=floor_surface.height,
        floor_normal=floor_surface.normal,
        floor_origin=floor_surface.origin,
        board_contact_offsets=board_contact_offsets,
        features=features,
        intervals=all_intervals,
    )


def intervals_by_state(t: ArrayLike, state: ArrayLike) -> StateIntervals:
    """Convert a per-frame state array into intervals grouped by state label."""

    state = np.asarray(state)
    return {
        STATE_LABELS[support_state]: intervals_from_mask(
            t,
            state == int(support_state),
            min_duration=0.0,
        )
        for support_state in FootSupportState
    }


@dataclass(frozen=True)
class _FloorSurface:
    """Planar floor model used by the foot-state classifier."""

    model: FloorModel
    height: float
    up_axis: int
    plane: PlaneSupportModel | None = None

    @property
    def normal(self) -> FloatArray | None:
        if self.plane is None:
            return None
        return self.plane.normal

    @property
    def origin(self) -> FloatArray | None:
        if self.plane is None:
            return None
        return self.plane.origin

    def clearance(self, points: ArrayLike) -> FloatArray:
        points = np.asarray(points, dtype=float)
        if self.plane is not None:
            return self.plane.clearance(points)
        return points[:, self.up_axis] - self.height

    def height_at(self, points: ArrayLike) -> FloatArray:
        points = np.asarray(points, dtype=float)
        if self.plane is None:
            return np.full(len(points), self.height, dtype=float)

        normal = self.plane.normal
        origin = self.plane.origin
        normal_up = normal[self.up_axis]
        if abs(normal_up) <= 1e-12:
            raise ValueError("Fitted floor plane normal is nearly horizontal.")

        horizontal_axes = [axis for axis in range(3) if axis != self.up_axis]
        horizontal_delta = points[:, horizontal_axes] - origin[horizontal_axes]
        horizontal_normal = normal[horizontal_axes]
        return origin[self.up_axis] - (horizontal_delta @ horizontal_normal) / normal_up


def _clearance_points(
    foot_pos: FloatArray,
    *,
    foot_name: str,
    sole_patches: Mapping[str, ContactPatch],
    body_rotations: Mapping[str, FloatArray] | None,
) -> FloatArray:
    if foot_name not in sole_patches or body_rotations is None:
        return foot_pos
    return _sole_contact_points(
        foot_pos,
        body_rotations=body_rotations,
        foot_name=foot_name,
        sole=sole_patches[foot_name],
    )


def _provisional_ground_fit_masks(
    t: FloatArray,
    per_foot: Mapping[str, Mapping[str, Any]],
    board_masks: Mapping[str, BoolArray],
    config: FootSupportConfig,
) -> dict[str, BoolArray]:
    """Seed per-foot ground-contact intervals from low sole/body height."""
    heights: list[FloatArray] = []
    for data in per_foot.values():
        heights.append(np.asarray(data["clearance_point"][:, config.up_axis], dtype=np.float64))
    stacked = np.concatenate(heights)
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        raise ValueError("Cannot seed ground intervals from non-finite sole heights.")
    cutoff = float(
        np.percentile(finite, config.provisional_ground_percentile)
        + config.provisional_ground_height_slack
    )
    masks: dict[str, BoolArray] = {}
    for foot_name, data in per_foot.items():
        sole_z = data["clearance_point"][:, config.up_axis]
        candidate = (
            (sole_z <= cutoff)
            & (data["foot_speed"] <= config.ground_speed_tolerance)
            & ~board_masks[foot_name]
            & np.isfinite(sole_z)
        )
        masks[foot_name] = clean_mask_by_time(
            t,
            candidate,
            max_gap_time=config.max_gap_time,
            min_blip_time=config.min_state_time,
        )
    return masks


def _refined_ground_fit_masks(
    t: FloatArray,
    per_foot: Mapping[str, Mapping[str, Any]],
    board_masks: Mapping[str, BoolArray],
    floor_surface: _FloorSurface,
    config: FootSupportConfig,
) -> dict[str, BoolArray]:
    """Per-foot ground-contact intervals against the current floor plane (for refitting)."""
    masks: dict[str, BoolArray] = {}
    for foot_name, data in per_foot.items():
        sole_z = data["clearance_point"][:, config.up_axis]
        floor_z = floor_surface.height_at(data["clearance_point"])
        vertical_clearance = sole_z - floor_z
        candidate = (
            (np.abs(vertical_clearance) <= config.ground_clearance_tolerance)
            & (data["foot_speed"] <= config.ground_speed_tolerance)
            & ~board_masks[foot_name]
            & np.isfinite(vertical_clearance)
        )
        masks[foot_name] = clean_mask_by_time(
            t,
            candidate,
            max_gap_time=config.max_gap_time,
            min_blip_time=config.min_state_time,
        )
    return masks


def _floor_fit_samples_for_ground_masks(
    per_foot_masks: Mapping[str, BoolArray],
    *,
    config: FootSupportConfig,
    body_pos: FloatArray,
    foot_indices: list[int],
    floor_fit_marker_pos: ArrayLike | None,
    floor_fit_marker_names: Sequence[str] | None,
    body_rotations: Mapping[str, FloatArray] | None,
) -> FloatArray:
    """Collect sole surface samples for each foot only on that foot's ground-contact frames."""
    chunks: list[FloatArray] = []
    contact_models = config.contact_models

    if floor_fit_marker_pos is not None:
        marker_pos = np.asarray(floor_fit_marker_pos, dtype=float)
        if marker_pos.ndim != 3 or marker_pos.shape[2] != 3:
            raise ValueError("floor_fit_marker_pos must have shape (N, M, 3).")
        names = tuple(floor_fit_marker_names or config.floor_fit_marker_names or ())
        if len(names) != marker_pos.shape[1]:
            raise ValueError("floor_fit_marker_names length must match floor_fit_marker_pos.")
        if not contact_models:
            raise ValueError("contact_models are required when floor_fit_marker_pos is set.")
        columns_by_body: dict[str, list[int]] = {}
        for col, name in enumerate(names):
            body = model_for_marker(contact_models, name).body_name
            columns_by_body.setdefault(body, []).append(col)
        for foot_name, foot_mask in per_foot_masks.items():
            indices = np.flatnonzero(foot_mask)
            if len(indices) == 0:
                continue
            cols = columns_by_body.get(foot_name)
            if not cols:
                continue
            samples = marker_pos[np.ix_(indices, cols)].reshape(-1, 3)
            col_names = tuple(names[col] for col in cols)
            flat_names = marker_names_for_flat_trajectory(col_names, len(indices))
            chunks.append(
                apply_contact_model_offsets(
                    samples,
                    flat_names,
                    contact_models,
                    body_rotations=_body_rotations_on_frames(body_rotations, indices),
                )
            )
    else:
        for foot_name, foot_idx in zip(config.foot_names, foot_indices):
            indices = np.flatnonzero(per_foot_masks[foot_name])
            if len(indices) == 0:
                continue
            chunks.append(body_pos[indices, foot_idx, :])

    if not chunks:
        raise ValueError("No sole samples on ground-contact frames for floor plane fitting.")
    return np.vstack(chunks)


def _body_rotations_on_frames(
    body_rotations: Mapping[str, FloatArray] | None,
    frame_indices: NDArray[np.intp],
) -> dict[str, FloatArray] | None:
    if body_rotations is None:
        return None
    return {
        name: np.asarray(quats, dtype=np.float64)[frame_indices]
        for name, quats in body_rotations.items()
    }


def _fit_plane_floor(points: FloatArray, config: FootSupportConfig) -> _FloorSurface:
    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]
    if len(finite_points) < config.min_floor_fit_samples:
        raise ValueError(
            f"Need at least {config.min_floor_fit_samples} finite sole samples on "
            f"ground-contact frames to fit a floor plane; got {len(finite_points)}."
        )
    use_tilted_plane = bool(config.contact_models)
    if use_tilted_plane:
        support_config = SupportDetectionConfig(
            model_type=SupportModelType.PLANE,
            plane_residual_tolerance=config.floor_plane_residual_tolerance,
            ransac_iterations=config.floor_plane_ransac_iterations,
            random_seed=config.floor_plane_random_seed,
            up_axis=config.up_axis,
        )
        plane = PlaneSupportModel.fit(finite_points, support_config)
    else:
        plane = _fit_horizontal_plane(finite_points, config.up_axis)
    height = float(
        np.median(
            _FloorSurface(FloorModel.PLANE, 0.0, config.up_axis, plane).height_at(finite_points)
        )
    )
    return _FloorSurface(model=FloorModel.PLANE, height=height, up_axis=config.up_axis, plane=plane)


def _fit_horizontal_plane(points: FloatArray, up_axis: int) -> PlaneSupportModel:
    """Fit a horizontal plane at the median height of ground-contact samples."""
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    origin = np.zeros(3, dtype=np.float64)
    origin[up_axis] = float(np.median(points[:, up_axis]))
    origin[horizontal_axes[0]] = float(np.median(points[:, horizontal_axes[0]]))
    origin[horizontal_axes[1]] = float(np.median(points[:, horizontal_axes[1]]))
    normal = np.zeros(3, dtype=np.float64)
    normal[up_axis] = 1.0
    return PlaneSupportModel(normal=normal, origin=origin)


def _estimate_board_contact_offset(
    horizontal_distance: FloatArray,
    relative_height: FloatArray,
    relative_speed: FloatArray,
    config: FootSupportConfig,
) -> float:
    candidate_mask = (
        (horizontal_distance <= config.board_horizontal_tolerance)
        & (relative_height >= config.board_min_relative_height)
        & (relative_height <= config.board_max_relative_height)
        & (relative_speed <= config.board_relative_speed_tolerance)
        & np.isfinite(relative_height)
    )
    candidates = relative_height[candidate_mask]
    if candidates.size < config.min_board_offset_samples:
        return config.default_board_contact_offset
    return float(np.median(candidates))


def _compile_sole_patches(
    config: FootSupportConfig,
    *,
    body_names: list[str],
    body_pos: FloatArray,
    floor_fit_marker_pos: ArrayLike | None,
    floor_fit_marker_names: Sequence[str] | None,
    body_rotations: Mapping[str, FloatArray] | None,
) -> dict[str, ContactPatch]:
    if not config.contact_models:
        return {}
    compiled = model_map(config.contact_models)
    if floor_fit_marker_pos is None or body_rotations is None:
        return _sole_patches_from_models(config, compiled)
    marker_pos = np.asarray(floor_fit_marker_pos, dtype=float)
    names = tuple(floor_fit_marker_names or config.floor_fit_marker_names or ())
    marker_trajs = {names[col]: marker_pos[:, col, :] for col in range(len(names))}
    body_positions = {
        body_names[idx]: body_pos[:, idx, :] for idx in range(body_pos.shape[1])
    }
    frame_index = _first_finite_calibration_frame(marker_pos)
    updated = compile_contact_models(
        config.contact_models,
        marker_positions_world=marker_trajs,
        body_positions=body_positions,
        body_quaternions=dict(body_rotations),
        frame_index=frame_index,
    )
    return _sole_patches_from_models(config, model_map(updated))


def _sole_patches_from_models(
    config: FootSupportConfig,
    models: Mapping[str, RigidBodyContactModel[Any]],
) -> dict[str, ContactPatch]:
    out: dict[str, ContactPatch] = {}
    patch_names = dict(config.sole_patch_names or {})
    for foot_name in config.foot_names:
        model = models.get(foot_name)
        if model is None:
            continue
        try:
            out[foot_name] = model.patch(patch_names.get(foot_name))
        except KeyError:
            continue
    return out


def _first_finite_calibration_frame(marker_pos: FloatArray) -> int:
    finite = np.isfinite(marker_pos).all(axis=(1, 2))
    indices = np.flatnonzero(finite)
    if len(indices) == 0:
        return 0
    return int(indices[len(indices) // 2])


def _sole_contact_points(
    foot_pos: FloatArray,
    *,
    body_rotations: Mapping[str, FloatArray],
    foot_name: str,
    sole: ContactPatch,
) -> FloatArray:
    quats = np.asarray(body_rotations[foot_name], dtype=np.float64)
    out = np.empty_like(foot_pos)
    for frame_idx in range(foot_pos.shape[0]):
        rot = Rotation.from_quat(quats[frame_idx])
        view = sole.view(RigidTransform(translation=foot_pos[frame_idx], rotation=rot))
        out[frame_idx] = view.contact_point_world
    return out


def _body_index(body_names: list[str], name: str) -> int:
    try:
        return body_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Body {name!r} not found. Available bodies: {body_names}") from exc
