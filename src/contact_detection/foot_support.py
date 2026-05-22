from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from os import PathLike
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contact import PlaneSupportModel, SupportDetectionConfig
from .intervals import clean_mask_by_time, intervals_from_mask
from .quiet import local_polynomial_derivative
from .types import BoolArray, FloatArray, IntervalList

StateArray: TypeAlias = NDArray[np.int8]
FeatureArray: TypeAlias = FloatArray | BoolArray
StateIntervals: TypeAlias = dict[str, IntervalList]
FloorModelKind: TypeAlias = Literal["height", "plane"]


class FootSupportState(IntEnum):
    """Per-frame support state for a tracked foot or contact point."""

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

    The classifier treats ground and board contact as separate support classes.
    Ground support is estimated from low observed foot heights. Skateboard
    support requires horizontal proximity to the board, plausible relative
    vertical offset, and low foot-board relative motion.
    """

    foot_names: tuple[str, str] = ("Left_Shoe", "Right_Shoe")
    board_name: str = "Skateboard"
    up_axis: int = 2
    floor_model: FloorModelKind = "height"
    floor_low_percentile: float = 15.0
    floor_plane_candidate_percentile: float = 50.0
    floor_plane_residual_tolerance: float = 0.025
    floor_plane_ransac_iterations: int = 128
    floor_plane_random_seed: int = 17
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


@dataclass
class FootSupportClassification:
    """Output of per-foot support classification."""

    t: FloatArray
    states: dict[str, StateArray]
    floor_model: FloorModelKind
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
) -> FootSupportClassification:
    """Classify each configured foot as air, ground, or skateboard over time.

    Parameters
    ----------
    t:
        Strictly increasing timestamps with shape ``(N,)``. The caller may pass
        absolute or trial-relative time; intervals use the same origin.
    body_names:
        Body names corresponding to axis 1 of ``body_pos``.
    body_pos:
        Body positions with shape ``(N, B, 3)``.
    config:
        Optional classification thresholds and body-name configuration.

    Returns
    -------
    FootSupportClassification
        State arrays, state intervals, floor model diagnostics, estimated board
        contact offsets, and diagnostic feature arrays per foot.
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

    foot_points = body_pos[:, foot_indices, :].reshape(-1, 3)
    floor_surface = _fit_floor_surface(foot_points, config)

    velocities = local_polynomial_derivative(
        body_pos.reshape(len(t), -1),
        t,
        window_time=config.velocity_window_time,
        degree=1,
    ).reshape(body_pos.shape)

    board_pos = body_pos[:, board_idx, :]
    board_vel = velocities[:, board_idx, :]
    board_speed = np.linalg.norm(board_vel, axis=1)

    states: dict[str, StateArray] = {}
    board_contact_offsets: dict[str, float] = {}
    features: dict[str, dict[str, FeatureArray]] = {}
    all_intervals: dict[str, StateIntervals] = {}

    for foot_name, foot_idx in zip(config.foot_names, foot_indices):
        foot_pos = body_pos[:, foot_idx, :]
        foot_vel = velocities[:, foot_idx, :]
        foot_speed = np.linalg.norm(foot_vel, axis=1)
        relative_pos = foot_pos - board_pos
        relative_vel = foot_vel - board_vel
        relative_speed = np.linalg.norm(relative_vel, axis=1)
        horizontal_distance = np.linalg.norm(relative_pos[:, horizontal_axes], axis=1)
        relative_height = relative_pos[:, config.up_axis]
        floor_height_at_foot = floor_surface.height_at(foot_pos)
        ground_clearance = floor_surface.clearance(foot_pos)

        board_offset = _estimate_board_contact_offset(
            horizontal_distance,
            relative_height,
            relative_speed,
            config,
        )
        board_contact_offsets[foot_name] = board_offset

        board_geometry = (
            (horizontal_distance <= config.board_horizontal_tolerance)
            & (np.abs(relative_height - board_offset) <= config.board_vertical_tolerance)
        )
        moving_board_motion = relative_speed <= config.board_relative_speed_tolerance
        static_board_motion = (
            (board_speed <= config.static_board_speed_tolerance)
            & (foot_speed <= config.static_board_foot_speed_tolerance)
        )
        skateboard_mask = board_geometry & (moving_board_motion | static_board_motion)
        skateboard_mask = clean_mask_by_time(
            t,
            skateboard_mask,
            max_gap_time=config.max_gap_time,
            min_blip_time=config.min_state_time,
        )

        ground_mask = (
            (np.abs(ground_clearance) <= config.ground_clearance_tolerance)
            & (foot_speed <= config.ground_speed_tolerance)
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

        features[foot_name] = {
            "foot_height": foot_pos[:, config.up_axis],
            "board_height": board_pos[:, config.up_axis],
            "floor_height_at_foot": floor_height_at_foot,
            "ground_clearance": ground_clearance,
            "horizontal_distance_to_board": horizontal_distance,
            "relative_height_to_board": relative_height,
            "foot_speed": foot_speed,
            "board_speed": board_speed,
            "relative_speed_to_board": relative_speed,
            "ground_mask": ground_mask,
            "skateboard_mask": skateboard_mask,
        }
        all_intervals[foot_name] = intervals_by_state(t, state)

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


def load_unified_npz(
    path: str | PathLike[str],
) -> tuple[FloatArray, list[str], FloatArray, np.lib.npyio.NpzFile]:
    """Load Vicon rigid-body data from a unified NPZ file.

    Expected keys are ``t``, ``vicon__body_names``, and ``vicon__body_pos``.
    If a ``valid`` mask is present, invalid frames are removed before returning.
    Returned timestamps are shifted so the first valid frame is time zero.
    """

    data = np.load(path, allow_pickle=True)
    required = {"t", "vicon__body_names", "vicon__body_pos"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing required unified.npz keys: {sorted(missing)}")

    t = data["t"].astype(float)
    body_pos = data["vicon__body_pos"].astype(float)
    body_names = data["vicon__body_names"].tolist()

    if "valid" in data:
        valid = np.asarray(data["valid"], dtype=bool)
        if valid.shape != t.shape:
            raise ValueError("unified.npz valid mask must have shape (N,).")
        t = t[valid]
        body_pos = body_pos[valid]

    if len(t) == 0:
        raise ValueError("unified.npz contains no valid frames.")

    t = t - float(t[0])
    return t, body_names, body_pos, data


@dataclass(frozen=True)
class _FloorSurface:
    """Flat or planar floor model used by the foot-state classifier."""

    model: FloorModelKind
    height: float
    up_axis: int
    plane: PlaneSupportModel | None = None

    @property
    def normal(self) -> FloatArray | None:
        """Plane normal for fitted-plane floors, otherwise ``None``."""

        if self.plane is None:
            return None
        return self.plane.normal

    @property
    def origin(self) -> FloatArray | None:
        """Plane origin for fitted-plane floors, otherwise ``None``."""

        if self.plane is None:
            return None
        return self.plane.origin

    def clearance(self, points: ArrayLike) -> FloatArray:
        """Signed distance or vertical clearance from the floor model."""

        points = np.asarray(points, dtype=float)
        if self.plane is not None:
            return self.plane.clearance(points)
        return points[:, self.up_axis] - self.height

    def height_at(self, points: ArrayLike) -> FloatArray:
        """Floor height at each query point's horizontal location."""

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


def _fit_floor_surface(foot_points: FloatArray, config: FootSupportConfig) -> _FloorSurface:
    """Fit the configured floor model from observed foot rigid-body positions."""

    finite_mask = np.isfinite(foot_points).all(axis=1)
    finite_points = foot_points[finite_mask]
    if len(finite_points) == 0:
        raise ValueError("Cannot estimate floor from non-finite foot positions.")

    if config.floor_model == "height":
        height = _estimate_floor_height(finite_points[:, config.up_axis], config.floor_low_percentile)
        return _FloorSurface(model="height", height=height, up_axis=config.up_axis)

    if config.floor_model == "plane":
        cutoff = np.percentile(finite_points[:, config.up_axis], config.floor_plane_candidate_percentile)
        candidate_points = finite_points[finite_points[:, config.up_axis] <= cutoff]
        if len(candidate_points) < 3:
            candidate_points = finite_points
        if len(candidate_points) < 3:
            raise ValueError("Need at least 3 finite foot positions to fit a floor plane.")
        support_config = SupportDetectionConfig(
            model_type="plane",
            plane_residual_tolerance=config.floor_plane_residual_tolerance,
            ransac_iterations=config.floor_plane_ransac_iterations,
            random_seed=config.floor_plane_random_seed,
            up_axis=config.up_axis,
        )
        plane = PlaneSupportModel.fit(candidate_points, support_config)
        height = float(np.median(_FloorSurface("plane", 0.0, config.up_axis, plane).height_at(finite_points)))
        return _FloorSurface(model="plane", height=height, up_axis=config.up_axis, plane=plane)

    raise ValueError(f"Unsupported floor_model: {config.floor_model!r}")


def _estimate_floor_height(foot_heights: FloatArray, low_percentile: float) -> float:
    """Estimate the floor as the median of low observed foot heights."""

    finite = foot_heights[np.isfinite(foot_heights)]
    if finite.size == 0:
        raise ValueError("Cannot estimate floor height from non-finite foot heights.")
    cutoff = np.percentile(finite, low_percentile)
    low = finite[finite <= cutoff]
    if low.size == 0:
        low = finite
    return float(np.median(low))


def _estimate_board_contact_offset(
    horizontal_distance: FloatArray,
    relative_height: FloatArray,
    relative_speed: FloatArray,
    config: FootSupportConfig,
) -> float:
    """Estimate the foot-board vertical offset from likely board-contact samples."""

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


def _body_index(body_names: list[str], name: str) -> int:
    """Return a named body index or raise a message with available bodies."""

    try:
        return body_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Body {name!r} not found. Available bodies: {body_names}") from exc
