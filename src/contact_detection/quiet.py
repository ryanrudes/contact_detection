"""Quiet-interval detection from scalar, vector, and quaternion motion signals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation as Rotation

from .types import BoolArray, DebugDict, FloatArray, IntervalList

_E = TypeVar("_E", bound=Enum)

CONTACT_MIN_TIME = 0.18      # not 1.0 unless detecting standing
MAX_GAP_TIME = 0.10          # fill tiny holes
MIN_BLIP_TIME = 0.15         # remove tiny false contacts

POS_SMOOTH_TIME = 0.03        # seconds
VEL_SMOOTH_TIME = 0.02        # seconds
QUIET_WINDOW_TIME = 0.10      # seconds
DERIVATIVE_WINDOW_TIME = 0.05  # seconds

# Adaptive hysteresis thresholds are estimated from the signal distribution.
MIN_VZ_ON_EPS = 0.003        # m/s
MIN_VZ_OFF_EPS = 0.010       # m/s
VZ_ON_PERCENTILE = 45.0
VZ_OFF_PERCENTILE = 65.0

Z_RANGE_WINDOW_TIME = 0.12     # seconds
Z_RANGE_ON_EPS = 0.006         # meters
Z_RANGE_OFF_EPS = 0.018        # meters


class QuietSignalType(Enum):
    """Input signal layout accepted by the quiet detector."""

    SCALAR = "scalar"
    POSITION_COMPONENT = "position_component"
    VECTOR_POSITION = "vector_position"
    VECTOR = "vector"
    QUATERNION = "quaternion"


class VectorQuietMode(Enum):
    """How vector signals are reduced to scalar activity and spread metrics."""

    NORM = "norm"
    EUCLIDEAN = "euclidean"
    MAX_COMPONENT = "max_component"
    NORMALIZED_NORM = "normalized_norm"
    MAHALANOBIS = "mahalanobis"


@dataclass(frozen=True)
class QuietDetectionConfig:
    """Smoothing windows, hysteresis thresholds, and signal-type options."""

    signal_type: QuietSignalType = QuietSignalType.SCALAR
    vector_mode: VectorQuietMode = VectorQuietMode.NORM
    min_interval_time: float = CONTACT_MIN_TIME
    contact_min_time: float | None = None
    max_gap_time: float = MAX_GAP_TIME
    min_blip_time: float = MIN_BLIP_TIME
    pos_smooth_time: float = POS_SMOOTH_TIME
    vel_smooth_time: float = VEL_SMOOTH_TIME
    quiet_window_time: float = QUIET_WINDOW_TIME
    spread_window_time: float = Z_RANGE_WINDOW_TIME
    derivative_window_time: float | None = DERIVATIVE_WINDOW_TIME
    derivative_poly_degree: int = 1
    activity_on_eps: float | None = None
    activity_off_eps: float | None = None
    spread_on_eps: float | None = None
    spread_off_eps: float | None = None
    min_activity_on_eps: float = MIN_VZ_ON_EPS
    min_activity_off_eps: float = MIN_VZ_OFF_EPS
    min_spread_on_eps: float = 0.0
    min_spread_off_eps: float = 0.0
    use_time_gaussian_smoothing: bool = True
    use_time_windows: bool = True
    quaternion_scalar_last: bool = True

    @property
    def effective_min_interval_time(self) -> float:
        """Minimum interval length, honoring legacy ``contact_min_time`` override."""

        if self.contact_min_time is not None:
            return self.contact_min_time
        return self.min_interval_time


@dataclass
class QuietDetectionResult:
    """Output of quiet-interval detection on a single motion channel."""

    intervals: IntervalList
    mask: BoolArray
    activity: FloatArray
    spread: FloatArray
    scores: FloatArray | None
    debug: DebugDict

    def __iter__(self) -> Iterator[IntervalList | BoolArray | DebugDict]:
        """Legacy tuple unpacking compatibility: ``intervals``, ``mask``, ``debug``."""
        yield self.intervals
        yield self.mask
        yield self.debug


def near_zero_intervals(
    t: ArrayLike,
    x: ArrayLike,
    T: float,
    eps: float,
) -> IntervalList:
    """Find intervals where ``|x| <= eps`` for at least ``T`` seconds."""
    t = np.asarray(t)
    x = np.asarray(x)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("t and x must both be 1D arrays.")

    if len(t) != len(x):
        raise ValueError("t and x must have the same length.")

    if len(t) == 0:
        return []

    near_zero = np.abs(x) <= eps

    # Pad so runs touching the edges are detected
    padded = np.r_[False, near_zero, False]
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    intervals = []
    for start, end in zip(starts, ends):
        start_time = t[start]
        end_time = t[end - 1]

        duration = end_time - start_time

        if duration >= T:
            intervals.append((start_time, end_time))

    return intervals

def shrink_intervals(intervals: IntervalList, shrink_amount: float) -> IntervalList:
    """Trim each interval inward by ``shrink_amount`` seconds on both ends."""
    new_intervals = []
    for start, end in intervals:
        new_start = start + shrink_amount
        new_end = end - shrink_amount
        if new_start >= new_end:
            continue
        new_intervals.append((new_start, new_end))
    return new_intervals

def intervals_from_mask(
    t: ArrayLike,
    mask: ArrayLike,
    min_duration: float = 0.0,
) -> IntervalList:
    """Convert a boolean mask into ``(start, end)`` time intervals."""
    t = np.asarray(t)
    mask = np.asarray(mask, dtype=bool)

    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    intervals = []
    for start, end in zip(starts, ends):
        start_time = float(t[start])
        end_time = float(t[end - 1])

        if end_time - start_time >= min_duration:
            intervals.append((start_time, end_time))

    return intervals


def odd_window_samples(t: ArrayLike, window_time: float) -> int:
    """Convert a time window to an odd sample count using the median ``dt``."""
    dt_med = np.median(np.diff(t))
    samples = max(3, int(round(window_time / dt_med)))
    if samples % 2 == 0:
        samples += 1
    return samples


def moving_rms(x: ArrayLike, window_samples: int) -> FloatArray:
    x = np.asarray(x, dtype=float)
    return np.sqrt(uniform_filter1d(x * x, size=window_samples, mode="nearest"))


def moving_std(x: ArrayLike, window_samples: int) -> FloatArray:
    x = np.asarray(x, dtype=float)
    mean = uniform_filter1d(x, size=window_samples, mode="nearest")
    mean_sq = uniform_filter1d(x * x, size=window_samples, mode="nearest")
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def moving_range(x: ArrayLike, window_samples: int) -> FloatArray:
    x = np.asarray(x, dtype=float)
    return maximum_filter1d(x, size=window_samples, mode="nearest") - minimum_filter1d(x, size=window_samples, mode="nearest")


def _time_window_slices(t: ArrayLike, window_time: float) -> Iterator[slice]:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if window_time <= 0.0:
        raise ValueError("window_time must be positive.")

    radius = 0.5 * window_time
    left = 0
    right = -1
    n = len(t)

    for i, center in enumerate(t):
        lo = center - radius
        hi = center + radius

        while left < n and t[left] < lo:
            left += 1
        while right + 1 < n and t[right + 1] <= hi:
            right += 1

        if left > right:
            yield slice(i, i + 1)
        else:
            yield slice(left, right + 1)


def time_moving_rms(x: ArrayLike, t: ArrayLike, window_time: float) -> FloatArray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    out = np.empty(len(x), dtype=float)
    for i, sl in enumerate(_time_window_slices(t, window_time)):
        window = x[sl]
        out[i] = np.sqrt(np.mean(window * window))
    return out


def time_moving_std(x: ArrayLike, t: ArrayLike, window_time: float) -> FloatArray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    out = np.empty(len(x), dtype=float)
    for i, sl in enumerate(_time_window_slices(t, window_time)):
        out[i] = np.std(x[sl])
    return out


def time_moving_range(x: ArrayLike, t: ArrayLike, window_time: float) -> FloatArray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    out = np.empty(len(x), dtype=float)
    for i, sl in enumerate(_time_window_slices(t, window_time)):
        window = x[sl]
        out[i] = np.max(window) - np.min(window)
    return out


def time_window_rms(x: ArrayLike, t: ArrayLike, window_time: float) -> FloatArray:
    """Alias for :func:`time_moving_rms` using actual timestamps."""

    return time_moving_rms(x, t, window_time)


def time_window_std(x: ArrayLike, t: ArrayLike, window_time: float) -> FloatArray:
    """Alias for :func:`time_moving_std` using actual timestamps."""

    return time_moving_std(x, t, window_time)


def time_window_range(x: ArrayLike, t: ArrayLike, window_time: float) -> FloatArray:
    """Alias for :func:`time_moving_range` using actual timestamps."""

    return time_moving_range(x, t, window_time)


def ensure_2d_signal(X: ArrayLike) -> tuple[FloatArray, bool]:
    """Promote 1D inputs to ``(N, 1)`` and report whether the input was scalar."""
    X = np.asarray(X, dtype=float)
    scalar_input = X.ndim == 1

    if scalar_input:
        X = X[:, None]

    if X.ndim != 2:
        raise ValueError("X must have shape (N,) or (N, D).")

    return X, scalar_input


def normalize_enum(value: str | _E, enum_cls: type[_E]) -> _E:
    """Coerce a string or enum member to ``enum_cls``."""
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def time_gaussian_smooth(
    X: ArrayLike,
    t: ArrayLike,
    sigma_time: float,
    radius_sigma: float = 3.0,
) -> FloatArray:
    """Smooth a scalar or vector signal using a Gaussian kernel in actual time."""
    X_2d, scalar_input = ensure_2d_signal(X)
    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if len(X_2d) != len(t):
        raise ValueError("X and t must have the same length.")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if len(t) == 0 or sigma_time <= 0.0:
        return X_2d[:, 0].copy() if scalar_input else X_2d.copy()

    out = np.empty_like(X_2d, dtype=float)
    radius = radius_sigma * sigma_time

    left = 0
    right = -1
    n = len(t)

    for i in range(n):
        center = t[i]
        lo = center - radius
        hi = center + radius

        while left < n and t[left] < lo:
            left += 1
        while right + 1 < n and t[right + 1] <= hi:
            right += 1

        idx = np.arange(left, right + 1)
        if idx.size == 0:
            out[i] = X_2d[i]
            continue

        dt = t[idx] - center
        weights = np.exp(-0.5 * (dt / sigma_time) ** 2)
        weight_sum = np.sum(weights)

        if weight_sum <= 0.0:
            out[i] = X_2d[i]
        else:
            out[i] = weights @ X_2d[idx] / weight_sum

    return out[:, 0] if scalar_input else out


def smooth_signal_for_quiet_detector(
    X: ArrayLike,
    t: ArrayLike,
    smooth_time: float,
    use_time_gaussian_smoothing: bool = True,
) -> FloatArray:
    """Apply the configured smoothing stage before quiet feature extraction."""
    X_2d, scalar_input = ensure_2d_signal(X)
    t = np.asarray(t, dtype=float)

    if use_time_gaussian_smoothing:
        out = time_gaussian_smooth(X_2d, t, sigma_time=smooth_time)
    else:
        dt_med = np.median(np.diff(t))
        sigma_samples = smooth_time / dt_med
        out = gaussian_filter1d(X_2d, sigma=sigma_samples, axis=0)

    return out[:, 0] if scalar_input else out


def local_polynomial_derivative(
    X: ArrayLike,
    t: ArrayLike,
    window_time: float | None = DERIVATIVE_WINDOW_TIME,
    degree: int = 1,
) -> FloatArray:
    """Estimate ``dX/dt`` with a local polynomial fit around each timestamp.

    The polynomial is fit in centered time coordinates, so the coefficient of
    the first-order term is the derivative at the center sample. This is more
    stable than finite differences for irregular timestamps.
    """
    X_2d, scalar_input = ensure_2d_signal(X)
    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if len(t) != len(X_2d):
        raise ValueError("X and t must have the same length.")
    if len(t) < 2:
        out = np.zeros_like(X_2d)
        return out[:, 0] if scalar_input else out
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    if window_time is None or window_time <= 0.0:
        out = np.gradient(X_2d, t, axis=0)
        return out[:, 0] if scalar_input else out

    out = np.empty_like(X_2d, dtype=float)
    min_points = degree + 1
    half_window = 0.5 * window_time
    fallback = np.gradient(X_2d, t, axis=0)
    n = len(t)

    for i, center in enumerate(t):
        lo = center - half_window
        hi = center + half_window
        idx = np.where((t >= lo) & (t <= hi))[0]

        if idx.size < min_points:
            order = np.argsort(np.abs(t - center))
            idx = np.sort(order[:min(n, min_points)])

        if idx.size < min_points:
            out[i] = fallback[i]
            continue

        dt = t[idx] - center
        vandermonde = np.vander(dt, N=degree + 1, increasing=True)
        if idx.size > min_points:
            sigma = max(window_time / 4.0, 1e-12)
            weights = np.exp(-0.5 * (dt / sigma) ** 2)
            weighted_a = vandermonde * np.sqrt(weights)[:, None]
            weighted_b = X_2d[idx] * np.sqrt(weights)[:, None]
        else:
            weighted_a = vandermonde
            weighted_b = X_2d[idx]

        try:
            coeffs, *_ = np.linalg.lstsq(weighted_a, weighted_b, rcond=None)
            out[i] = coeffs[1]
        except np.linalg.LinAlgError:
            out[i] = fallback[i]

    return out[:, 0] if scalar_input else out


def moving_component_range(
    X: ArrayLike,
    window_samples: int,
    mode: VectorQuietMode | str = VectorQuietMode.NORM,
) -> FloatArray:
    mode = normalize_enum(mode, VectorQuietMode)
    X_2d, scalar_input = ensure_2d_signal(X)

    component_ranges = []
    for c in range(X_2d.shape[1]):
        component_ranges.append(moving_range(X_2d[:, c], window_samples))
    component_ranges = np.column_stack(component_ranges)

    if scalar_input:
        return component_ranges[:, 0]

    if mode in (VectorQuietMode.NORM, VectorQuietMode.EUCLIDEAN):
        return np.linalg.norm(component_ranges, axis=1)
    if mode == VectorQuietMode.MAX_COMPONENT:
        return np.max(component_ranges, axis=1)
    if mode in (VectorQuietMode.NORMALIZED_NORM, VectorQuietMode.MAHALANOBIS):
        scales = robust_component_scales(X_2d)
        return np.linalg.norm(component_ranges / scales[None, :], axis=1)

    raise ValueError(f"Unsupported vector quiet mode: {mode}")


def time_moving_component_range(
    X: ArrayLike,
    t: ArrayLike,
    window_time: float,
    mode: VectorQuietMode | str = VectorQuietMode.NORM,
) -> FloatArray:
    mode = normalize_enum(mode, VectorQuietMode)
    X_2d, scalar_input = ensure_2d_signal(X)

    component_ranges = []
    for c in range(X_2d.shape[1]):
        component_ranges.append(time_moving_range(X_2d[:, c], t, window_time))
    component_ranges = np.column_stack(component_ranges)

    if scalar_input:
        return component_ranges[:, 0]

    if mode in (VectorQuietMode.NORM, VectorQuietMode.EUCLIDEAN):
        return np.linalg.norm(component_ranges, axis=1)
    if mode == VectorQuietMode.MAX_COMPONENT:
        return np.max(component_ranges, axis=1)
    if mode in (VectorQuietMode.NORMALIZED_NORM, VectorQuietMode.MAHALANOBIS):
        scales = robust_component_scales(X_2d)
        return np.linalg.norm(component_ranges / scales[None, :], axis=1)

    raise ValueError(f"Unsupported vector quiet mode: {mode}")


def time_window_component_range(
    X: ArrayLike,
    t: ArrayLike,
    window_time: float,
    mode: VectorQuietMode | str = VectorQuietMode.NORM,
) -> FloatArray:
    """Alias for :func:`time_moving_component_range` using actual timestamps."""

    return time_moving_component_range(X, t, window_time, mode=mode)


def robust_component_scales(X: ArrayLike) -> FloatArray:
    """Robust per-component scale estimates using MAD with std fallback."""
    X_2d, _ = ensure_2d_signal(X)
    med = np.median(X_2d, axis=0)
    mad = np.median(np.abs(X_2d - med[None, :]), axis=0)
    scales = 1.4826 * mad
    fallback = np.std(X_2d, axis=0)
    scales = np.where(scales > 1e-12, scales, fallback)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return scales


def robust_mahalanobis_energy(X: ArrayLike) -> FloatArray:
    """Per-sample Mahalanobis distance using robust centering and scaling."""
    X_2d, scalar_input = ensure_2d_signal(X)
    if scalar_input:
        scale = robust_scale(X_2d[:, 0])
        if scale <= 1e-12:
            scale = np.std(X_2d[:, 0])
        if scale <= 1e-12:
            scale = 1.0
        return np.abs(X_2d[:, 0] - np.median(X_2d[:, 0])) / scale

    center = np.median(X_2d, axis=0)
    scales = robust_component_scales(X_2d)
    X_scaled = (X_2d - center[None, :]) / scales[None, :]
    cov = np.cov(X_scaled, rowvar=False)
    cov = np.atleast_2d(cov)
    cov.flat[:: cov.shape[0] + 1] += 1e-6
    inv_cov = np.linalg.pinv(cov)
    return np.sqrt(np.einsum("ij,jk,ik->i", X_scaled, inv_cov, X_scaled))


def robust_scale(x: ArrayLike) -> float:
    """MAD-based robust scale for a 1D signal."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def quaternion_standardize_xyzw(
    quat: ArrayLike,
    scalar_last: bool = True,
) -> FloatArray:
    """Normalize quaternions to xyzw layout and enforce hemisphere continuity."""
    quat = np.asarray(quat, dtype=float)

    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("Quaternion signal must have shape (N, 4).")

    if scalar_last:
        q = quat.copy()
    else:
        q = np.column_stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]])

    norms = np.linalg.norm(q, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Quaternion signal contains zero-norm quaternions.")
    q = q / norms[:, None]

    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0.0:
            q[i] = -q[i]

    return q


def quaternion_angular_speed(q_xyzw: ArrayLike, t: ArrayLike) -> FloatArray:
    """Estimate angular speed (rad/s) from a unit quaternion time series."""
    q_xyzw = quaternion_standardize_xyzw(q_xyzw, scalar_last=True)
    t = np.asarray(t, dtype=float)

    if len(q_xyzw) != len(t):
        raise ValueError("Quaternion signal and t must have the same length.")

    speeds = np.zeros(len(t), dtype=float)
    if len(t) < 2:
        return speeds

    rotations = Rotation.from_quat(q_xyzw)
    rel = rotations[:-1].inv() * rotations[1:]
    angles = rel.magnitude()
    dt = np.diff(t)
    pair_speeds = angles / dt

    speeds[0] = pair_speeds[0]
    speeds[-1] = pair_speeds[-1]
    if len(t) > 2:
        speeds[1:-1] = 0.5 * (pair_speeds[:-1] + pair_speeds[1:])

    return speeds


def quaternion_local_spread(
    q_xyzw: ArrayLike,
    t: ArrayLike,
    window_time: float,
    use_rms: bool = True,
) -> FloatArray:
    """Local orientation spread around each timestamp in radians."""
    q_xyzw = quaternion_standardize_xyzw(q_xyzw, scalar_last=True)
    t = np.asarray(t, dtype=float)

    if len(q_xyzw) != len(t):
        raise ValueError("Quaternion signal and t must have the same length.")

    radius = 0.5 * window_time
    rotations = Rotation.from_quat(q_xyzw)
    spread = np.zeros(len(t), dtype=float)

    left = 0
    right = -1
    n = len(t)

    for i in range(n):
        center = t[i]
        lo = center - radius
        hi = center + radius

        while left < n and t[left] < lo:
            left += 1
        while right + 1 < n and t[right + 1] <= hi:
            right += 1

        if left > right:
            spread[i] = 0.0
            continue

        rel = rotations[i].inv() * rotations[left:right + 1]
        angles = rel.magnitude()
        if use_rms:
            spread[i] = np.sqrt(np.mean(angles * angles))
        else:
            spread[i] = np.max(angles)

    return spread


def adaptive_hysteresis_thresholds(activity: ArrayLike) -> tuple[float, float]:
    """Estimate near-zero activity on/off thresholds from the activity distribution.

    Percentile thresholds are more stable than MAD when most of the recording is
    quiet: the lower/middle part of the distribution is treated as likely quiet
    contact, while large bursts are treated as swing/lift/impact.
    """
    activity = np.asarray(activity, dtype=float)
    finite = activity[np.isfinite(activity)]

    if finite.size == 0:
        return MIN_VZ_ON_EPS, MIN_VZ_OFF_EPS

    on_eps = np.percentile(finite, VZ_ON_PERCENTILE)
    off_eps = np.percentile(finite, VZ_OFF_PERCENTILE)

    on_eps = max(MIN_VZ_ON_EPS, float(on_eps))
    off_eps = max(MIN_VZ_OFF_EPS, float(off_eps), 1.25 * on_eps)

    return on_eps, off_eps


def hysteresis_mask(activity: ArrayLike, on_eps: float, off_eps: float) -> BoolArray:
    """Low-activity hysteresis mask (True when activity is below thresholds)."""
    mask = np.zeros_like(activity, dtype=bool)
    in_contact = False

    for i, val in enumerate(activity):
        if in_contact:
            if val >= off_eps:
                in_contact = False
        elif val <= on_eps:
            in_contact = True

        mask[i] = in_contact

    return mask


def score_hysteresis_mask(
    score: ArrayLike,
    on_threshold: float,
    off_threshold: float,
) -> BoolArray:
    """High-score hysteresis mask (True when score stays above thresholds)."""
    score = np.asarray(score, dtype=float)
    mask = np.zeros_like(score, dtype=bool)
    in_contact = False

    for i, val in enumerate(score):
        if in_contact:
            if val <= off_threshold:
                in_contact = False
        elif val >= on_threshold:
            in_contact = True

        mask[i] = in_contact

    return mask


def _run_durations(
    t: ArrayLike,
    mask: ArrayLike,
    value: bool,
) -> list[tuple[int, int, float]]:
    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(t) == 0:
        return []

    value_mask = mask == value
    padded = np.r_[False, value_mask, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    runs = []
    for start, end in zip(starts, ends):
        start_idx = start
        end_idx = end - 1
        duration = float(t[end_idx] - t[start_idx])
        runs.append((start_idx, end_idx, duration))
    return runs


def fill_short_false_runs(t: ArrayLike, mask: ArrayLike, max_gap_time: float) -> BoolArray:
    cleaned = np.asarray(mask, dtype=bool).copy()
    if max_gap_time <= 0.0:
        return cleaned
    for start, end, duration in _run_durations(t, cleaned, False):
        touches_edge = start == 0 or end == len(cleaned) - 1
        if not touches_edge and duration <= max_gap_time:
            cleaned[start:end + 1] = True
    return cleaned


def remove_short_true_runs(t: ArrayLike, mask: ArrayLike, min_duration: float) -> BoolArray:
    cleaned = np.asarray(mask, dtype=bool).copy()
    if min_duration <= 0.0:
        return cleaned
    for start, end, duration in _run_durations(t, cleaned, True):
        if duration < min_duration:
            cleaned[start:end + 1] = False
    return cleaned


def clean_mask_by_time(
    t: ArrayLike,
    mask: ArrayLike,
    max_gap_time: float = MAX_GAP_TIME,
    min_blip_time: float = MIN_BLIP_TIME,
) -> BoolArray:
    """Fill short gaps then remove short quiet blips."""
    cleaned = fill_short_false_runs(t, mask, max_gap_time)
    cleaned = remove_short_true_runs(t, cleaned, min_blip_time)
    return cleaned


def dual_hysteresis_mask(
    activity: ArrayLike,
    activity_on: float,
    activity_off: float,
    z_range: ArrayLike,
    z_range_on: float,
    z_range_off: float,
) -> BoolArray:
    """Hysteresis using both activity and local position spread."""
    mask = np.zeros_like(activity, dtype=bool)
    in_contact = False

    for i, (a, zr) in enumerate(zip(activity, z_range)):
        if in_contact:
            if a >= activity_off or zr >= z_range_off:
                in_contact = False
        else:
            if a <= activity_on and zr <= z_range_on:
                in_contact = True

        mask[i] = in_contact

    return mask


def adaptive_range_thresholds(
    spread: ArrayLike,
    min_on: float = 0.0,
    min_off: float = 0.0,
    on_percentile: float = 45.0,
    off_percentile: float = 65.0,
) -> tuple[float, float]:
    """Estimate spread on/off thresholds from the spread distribution."""
    spread = np.asarray(spread, dtype=float)
    finite = spread[np.isfinite(spread)]

    if finite.size == 0:
        return float(min_on), float(min_off)

    noise_floor = 1e-12
    on_eps = max(float(min_on), float(np.percentile(finite, on_percentile)), noise_floor)
    off_eps = max(float(min_off), float(np.percentile(finite, off_percentile)), 1.25 * on_eps)

    return on_eps, off_eps


def compute_quiet_activity_and_spread(
    t_raw: ArrayLike,
    X_raw: ArrayLike,
    signal_type: QuietSignalType | str = QuietSignalType.SCALAR,
    vector_mode: VectorQuietMode | str = VectorQuietMode.NORM,
    pos_smooth_time: float = POS_SMOOTH_TIME,
    vel_smooth_time: float = VEL_SMOOTH_TIME,
    quiet_window_time: float = QUIET_WINDOW_TIME,
    spread_window_time: float = Z_RANGE_WINDOW_TIME,
    derivative_window_time: float | None = DERIVATIVE_WINDOW_TIME,
    derivative_poly_degree: int = 1,
    use_time_gaussian_smoothing: bool = True,
    use_time_windows: bool = True,
    quaternion_scalar_last: bool = True,
) -> tuple[FloatArray, FloatArray, DebugDict]:
    """Compute smoothed activity and spread features used for quiet detection."""
    signal_type = normalize_enum(signal_type, QuietSignalType)
    vector_mode = normalize_enum(vector_mode, VectorQuietMode)
    t_raw = np.asarray(t_raw, dtype=float)
    X_raw = np.asarray(X_raw, dtype=float)

    if t_raw.ndim != 1:
        raise ValueError("t_raw must be a 1D array.")
    if len(t_raw) < 3:
        raise ValueError("Need at least 3 samples.")
    if np.any(np.diff(t_raw) <= 0):
        raise ValueError("t_raw must be strictly increasing.")
    if len(X_raw) != len(t_raw):
        raise ValueError("X_raw and t_raw must have the same length.")

    activity_window_samples = odd_window_samples(t_raw, quiet_window_time)
    spread_window_samples = odd_window_samples(t_raw, spread_window_time)

    if signal_type in (QuietSignalType.SCALAR, QuietSignalType.POSITION_COMPONENT):
        if X_raw.ndim != 1:
            raise ValueError("Scalar signal types require X_raw with shape (N,).")

        X_smooth = smooth_signal_for_quiet_detector(
            X_raw,
            t_raw,
            smooth_time=pos_smooth_time,
            use_time_gaussian_smoothing=use_time_gaussian_smoothing,
        )
        dX = local_polynomial_derivative(
            X_smooth,
            t_raw,
            window_time=derivative_window_time,
            degree=derivative_poly_degree,
        )
        dX_smooth = smooth_signal_for_quiet_detector(
            dX,
            t_raw,
            smooth_time=vel_smooth_time,
            use_time_gaussian_smoothing=use_time_gaussian_smoothing,
        )

        activity_source = dX_smooth
        if use_time_windows:
            activity = time_moving_rms(activity_source, t_raw, quiet_window_time)
            spread = time_moving_range(X_smooth, t_raw, spread_window_time)
        else:
            activity = moving_rms(activity_source, activity_window_samples)
            spread = moving_range(X_smooth, spread_window_samples)

    elif signal_type in (QuietSignalType.VECTOR_POSITION, QuietSignalType.VECTOR):
        X_2d, _ = ensure_2d_signal(X_raw)
        X_smooth = smooth_signal_for_quiet_detector(
            X_2d,
            t_raw,
            smooth_time=pos_smooth_time,
            use_time_gaussian_smoothing=use_time_gaussian_smoothing,
        )
        dX = local_polynomial_derivative(
            X_smooth,
            t_raw,
            window_time=derivative_window_time,
            degree=derivative_poly_degree,
        )
        dX_smooth = smooth_signal_for_quiet_detector(
            dX,
            t_raw,
            smooth_time=vel_smooth_time,
            use_time_gaussian_smoothing=use_time_gaussian_smoothing,
        )

        if vector_mode in (VectorQuietMode.NORM, VectorQuietMode.EUCLIDEAN):
            activity_source = np.linalg.norm(dX_smooth, axis=1)
        elif vector_mode == VectorQuietMode.MAX_COMPONENT:
            component_activity = np.column_stack([
                (
                    time_moving_rms(dX_smooth[:, c], t_raw, quiet_window_time)
                    if use_time_windows
                    else moving_rms(dX_smooth[:, c], activity_window_samples)
                )
                for c in range(dX_smooth.shape[1])
            ])
            activity_source = None
            activity = np.max(component_activity, axis=1)
        elif vector_mode == VectorQuietMode.NORMALIZED_NORM:
            scales = robust_component_scales(X_smooth)
            activity_source = np.linalg.norm(dX_smooth / scales[None, :], axis=1)
        elif vector_mode == VectorQuietMode.MAHALANOBIS:
            activity_source = robust_mahalanobis_energy(dX_smooth)
        else:
            raise ValueError(f"Unsupported vector quiet mode: {vector_mode}")

        if vector_mode != VectorQuietMode.MAX_COMPONENT:
            if use_time_windows:
                activity = time_moving_rms(activity_source, t_raw, quiet_window_time)
            else:
                activity = moving_rms(activity_source, activity_window_samples)

        if use_time_windows:
            spread = time_moving_component_range(X_smooth, t_raw, spread_window_time, mode=vector_mode)
        else:
            spread = moving_component_range(X_smooth, spread_window_samples, mode=vector_mode)

    elif signal_type == QuietSignalType.QUATERNION:
        q_xyzw = quaternion_standardize_xyzw(X_raw, scalar_last=quaternion_scalar_last)
        # Gaussian-smooth quaternions through rotation vectors in a locally continuous chart.
        rotvec = Rotation.from_quat(q_xyzw).as_rotvec()
        rotvec_smooth = smooth_signal_for_quiet_detector(
            rotvec,
            t_raw,
            smooth_time=pos_smooth_time,
            use_time_gaussian_smoothing=use_time_gaussian_smoothing,
        )
        q_smooth = Rotation.from_rotvec(rotvec_smooth).as_quat()
        q_smooth = quaternion_standardize_xyzw(q_smooth, scalar_last=True)

        angular_speed = quaternion_angular_speed(q_smooth, t_raw)
        angular_speed_smooth = smooth_signal_for_quiet_detector(
            angular_speed,
            t_raw,
            smooth_time=vel_smooth_time,
            use_time_gaussian_smoothing=use_time_gaussian_smoothing,
        )

        activity_source = angular_speed_smooth
        if use_time_windows:
            activity = time_moving_rms(activity_source, t_raw, quiet_window_time)
        else:
            activity = moving_rms(activity_source, activity_window_samples)
        spread = quaternion_local_spread(q_smooth, t_raw, spread_window_time, use_rms=True)

        X_smooth = q_smooth
        dX = angular_speed
        dX_smooth = angular_speed_smooth

    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    debug = {
        "signal_type": signal_type,
        "vector_mode": vector_mode,
        "X_raw": X_raw,
        "X_smooth": X_smooth,
        "dX": dX,
        "dX_smooth": dX_smooth,
        "activity": activity,
        "spread": spread,
        "activity_window_samples": activity_window_samples,
        "spread_window_samples": spread_window_samples,
        "derivative_window_time": derivative_window_time,
        "derivative_poly_degree": derivative_poly_degree,
        "use_time_windows": use_time_windows,
    }

    if "activity_source" in locals() and activity_source is not None:
        debug["activity_source"] = activity_source

    return activity, spread, debug


def detect_quiet_intervals(
    t_raw: ArrayLike,
    X_raw: ArrayLike,
    signal_type: QuietSignalType | str = QuietSignalType.SCALAR,
    vector_mode: VectorQuietMode | str = VectorQuietMode.NORM,
    contact_min_time: float = CONTACT_MIN_TIME,
    max_gap_time: float = MAX_GAP_TIME,
    min_blip_time: float = MIN_BLIP_TIME,
    pos_smooth_time: float = POS_SMOOTH_TIME,
    vel_smooth_time: float = VEL_SMOOTH_TIME,
    quiet_window_time: float = QUIET_WINDOW_TIME,
    spread_window_time: float = Z_RANGE_WINDOW_TIME,
    activity_on_eps: float | None = None,
    activity_off_eps: float | None = None,
    spread_on_eps: float | None = None,
    spread_off_eps: float | None = None,
    min_activity_on_eps: float = MIN_VZ_ON_EPS,
    min_activity_off_eps: float = MIN_VZ_OFF_EPS,
    min_spread_on_eps: float = 0.0,
    min_spread_off_eps: float = 0.0,
    use_time_gaussian_smoothing: bool = True,
    quaternion_scalar_last: bool = True,
    use_time_windows: bool = True,
    derivative_window_time: float | None = DERIVATIVE_WINDOW_TIME,
    derivative_poly_degree: int = 1,
    config: QuietDetectionConfig | None = None,
) -> QuietDetectionResult:
    """Detect intervals where motion is locally quiet in activity and spread."""
    t_raw = np.asarray(t_raw, dtype=float)

    if config is not None:
        if not isinstance(config, QuietDetectionConfig):
            raise TypeError("config must be a QuietDetectionConfig.")
        signal_type = config.signal_type
        vector_mode = config.vector_mode
        contact_min_time = config.effective_min_interval_time
        max_gap_time = config.max_gap_time
        min_blip_time = config.min_blip_time
        pos_smooth_time = config.pos_smooth_time
        vel_smooth_time = config.vel_smooth_time
        quiet_window_time = config.quiet_window_time
        spread_window_time = config.spread_window_time
        activity_on_eps = config.activity_on_eps
        activity_off_eps = config.activity_off_eps
        spread_on_eps = config.spread_on_eps
        spread_off_eps = config.spread_off_eps
        min_activity_on_eps = config.min_activity_on_eps
        min_activity_off_eps = config.min_activity_off_eps
        min_spread_on_eps = config.min_spread_on_eps
        min_spread_off_eps = config.min_spread_off_eps
        use_time_gaussian_smoothing = config.use_time_gaussian_smoothing
        quaternion_scalar_last = config.quaternion_scalar_last
        use_time_windows = config.use_time_windows
        derivative_window_time = config.derivative_window_time
        derivative_poly_degree = config.derivative_poly_degree

    activity, spread, debug = compute_quiet_activity_and_spread(
        t_raw,
        X_raw,
        signal_type=signal_type,
        vector_mode=vector_mode,
        pos_smooth_time=pos_smooth_time,
        vel_smooth_time=vel_smooth_time,
        quiet_window_time=quiet_window_time,
        spread_window_time=spread_window_time,
        derivative_window_time=derivative_window_time,
        derivative_poly_degree=derivative_poly_degree,
        use_time_gaussian_smoothing=use_time_gaussian_smoothing,
        use_time_windows=use_time_windows,
        quaternion_scalar_last=quaternion_scalar_last,
    )

    if activity_on_eps is None or activity_off_eps is None:
        auto_activity_on, auto_activity_off = adaptive_hysteresis_thresholds(activity)
        if activity_on_eps is None:
            activity_on_eps = max(float(min_activity_on_eps), auto_activity_on)
        if activity_off_eps is None:
            activity_off_eps = max(float(min_activity_off_eps), auto_activity_off, 1.25 * activity_on_eps)

    if spread_on_eps is None or spread_off_eps is None:
        auto_spread_on, auto_spread_off = adaptive_range_thresholds(
            spread,
            min_on=min_spread_on_eps,
            min_off=min_spread_off_eps,
        )
        if spread_on_eps is None:
            spread_on_eps = auto_spread_on
        if spread_off_eps is None:
            spread_off_eps = max(auto_spread_off, 1.25 * spread_on_eps)

    quiet_mask = dual_hysteresis_mask(
        activity,
        activity_on_eps,
        activity_off_eps,
        spread,
        spread_on_eps,
        spread_off_eps,
    )

    dt_med = np.median(np.diff(t_raw))
    quiet_mask = clean_mask_by_time(
        t_raw,
        quiet_mask,
        max_gap_time=max_gap_time,
        min_blip_time=min_blip_time,
    )

    intervals = intervals_from_mask(t_raw, quiet_mask, min_duration=contact_min_time)

    debug.update({
        "dt_med": dt_med,
        "use_time_gaussian_smoothing": use_time_gaussian_smoothing,
        "pos_smooth_time": pos_smooth_time,
        "vel_smooth_time": vel_smooth_time,
        "quiet_window_time": quiet_window_time,
        "spread_window_time": spread_window_time,
        "derivative_window_time": derivative_window_time,
        "derivative_poly_degree": derivative_poly_degree,
        "activity_on_eps": activity_on_eps,
        "activity_off_eps": activity_off_eps,
        "spread_on_eps": spread_on_eps,
        "spread_off_eps": spread_off_eps,
        "quiet_mask": quiet_mask,
    })

    return QuietDetectionResult(
        intervals=intervals,
        mask=quiet_mask,
        activity=activity,
        spread=spread,
        scores=None,
        debug=debug,
    )


def detect_z_quiet_intervals(
    t_raw: ArrayLike,
    z_raw: ArrayLike,
    contact_min_time: float = CONTACT_MIN_TIME,
    max_gap_time: float = MAX_GAP_TIME,
    min_blip_time: float = MIN_BLIP_TIME,
    pos_smooth_time: float = POS_SMOOTH_TIME,
    vel_smooth_time: float = VEL_SMOOTH_TIME,
    quiet_window_time: float = QUIET_WINDOW_TIME,
    z_range_window_time: float = Z_RANGE_WINDOW_TIME,
    z_range_on_eps: float = Z_RANGE_ON_EPS,
    z_range_off_eps: float = Z_RANGE_OFF_EPS,
    use_time_gaussian_smoothing: bool = True,
) -> tuple[IntervalList, BoolArray, DebugDict]:
    """Detect intervals where vertical motion is locally quiet (legacy z-only API).

    Compatibility wrapper around :func:`detect_quiet_intervals` for scalar
    position-component signals. Returns ``(intervals, mask, debug)`` rather than
    :class:`QuietDetectionResult`.
    """

    result = detect_quiet_intervals(
        t_raw,
        z_raw,
        signal_type=QuietSignalType.POSITION_COMPONENT,
        vector_mode=VectorQuietMode.NORM,
        contact_min_time=contact_min_time,
        max_gap_time=max_gap_time,
        min_blip_time=min_blip_time,
        pos_smooth_time=pos_smooth_time,
        vel_smooth_time=vel_smooth_time,
        quiet_window_time=quiet_window_time,
        spread_window_time=z_range_window_time,
        spread_on_eps=z_range_on_eps,
        spread_off_eps=z_range_off_eps,
        min_activity_on_eps=MIN_VZ_ON_EPS,
        min_activity_off_eps=MIN_VZ_OFF_EPS,
        use_time_gaussian_smoothing=use_time_gaussian_smoothing,
    )
    intervals = result.intervals
    quiet_mask = result.mask
    debug = dict(result.debug)
    debug.update({
        "z_raw": debug["X_raw"],
        "z_smooth": debug["X_smooth"],
        "vz": debug["dX"],
        "vz_smooth": debug["dX_smooth"],
        "abs_vz": np.abs(debug["dX_smooth"]),
        "vz_rms": debug["activity"],
        "vz_std": moving_std(debug["dX_smooth"], debug["activity_window_samples"]),
        "z_range": debug["spread"],
        "vz_on_eps": debug["activity_on_eps"],
        "vz_off_eps": debug["activity_off_eps"],
        "z_range_on_eps": debug["spread_on_eps"],
        "z_range_off_eps": debug["spread_off_eps"],
    })

    return intervals, quiet_mask, debug


def print_z_quiet_debug_summary(debug: DebugDict) -> None:
    """Print percentile summaries for z-quiet debug dictionaries."""
    activity = debug["activity"]
    z_range = debug["z_range"]

    print(f"vz activity percentiles: p10={np.percentile(activity, 10):.5f}, "
          f"p25={np.percentile(activity, 25):.5f}, "
          f"p50={np.percentile(activity, 50):.5f}, "
          f"p75={np.percentile(activity, 75):.5f}, "
          f"p90={np.percentile(activity, 90):.5f}")
    print(f"adaptive thresholds: on={debug['vz_on_eps']:.5f}, off={debug['vz_off_eps']:.5f}")

    print(f"z range percentiles: p10={np.percentile(z_range, 10):.5f}, "
          f"p25={np.percentile(z_range, 25):.5f}, "
          f"p50={np.percentile(z_range, 50):.5f}, "
          f"p75={np.percentile(z_range, 75):.5f}, "
          f"p90={np.percentile(z_range, 90):.5f}")


def plot_z_quiet_debug(
    t_raw: ArrayLike,
    intervals: IntervalList,
    debug: DebugDict,
    title: str | None = None,
) -> None:
    """Plot z-quiet debug traces and shaded detected intervals."""
    import matplotlib.pyplot as plt

    vz_smooth = debug["vz_smooth"]
    activity = debug["activity"]
    z_range = debug["z_range"]
    quiet_mask = debug["quiet_mask"]
    vz_on_eps = debug["vz_on_eps"]
    vz_off_eps = debug["vz_off_eps"]
    z_range_on_eps = debug["z_range_on_eps"]
    z_range_off_eps = debug["z_range_off_eps"]

    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    axs[0].plot(t_raw, vz_smooth, label="vz_smooth")
    axs[0].axhline(0.0, color="black", linewidth=0.8)
    axs[0].set_ylabel("vz (m/s)")
    axs[0].legend()

    axs[1].plot(t_raw, activity, label="local vz activity")
    axs[1].axhline(vz_on_eps, linestyle="--", linewidth=0.8, label="on threshold")
    axs[1].axhline(vz_off_eps, linestyle="--", linewidth=0.8, label="off threshold")
    axs[1].set_ylabel("activity")
    axs[1].set_ylim(0.0, np.percentile(activity, 95) * 1.15 + 1e-9)
    axs[1].legend()

    axs[2].plot(t_raw, z_range, label="local z range")
    axs[2].axhline(z_range_on_eps, linestyle="--", linewidth=0.8, label="z range on")
    axs[2].axhline(z_range_off_eps, linestyle="--", linewidth=0.8, label="z range off")
    axs[2].set_ylabel("z range (m)")
    axs[2].set_ylim(0.0, np.percentile(z_range, 95) * 1.15 + 1e-9)
    axs[2].legend()

    axs[3].step(t_raw, quiet_mask.astype(float), where="post", label="near-zero mask")
    axs[3].set_ylabel("mask")
    axs[3].set_xlabel("time (s)")
    axs[3].legend()

    for ax in axs:
        for start, end in intervals:
            ax.axvspan(start, end, color="red", alpha=0.1)

    if title is None:
        title = f"adaptive vz detector: on={vz_on_eps:.4f}, off={vz_off_eps:.4f}"

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def load_demo_foot_z(
    filepath: str,
    foot_name: str = "Right_Shoe",
    up_axis: int = 2,
) -> tuple[FloatArray, FloatArray, np.lib.npyio.NpzFile]:
    """Load vertical position for a named foot from a unified NPZ demo file."""
    demo = np.load(filepath, allow_pickle=True)
    body_names = demo["vicon__body_names"].tolist()
    body_pos = demo["vicon__body_pos"]

    foot_idx = body_names.index(foot_name)
    foot_pos = body_pos[:, foot_idx]

    t_raw = demo["t"]
    z_raw = foot_pos[:, up_axis]

    return t_raw, z_raw, demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug z-axis quiet interval detection for a Vicon NPZ file.")
    parser.add_argument("filepath", help="Path to a unified .npz file.")
    parser.add_argument("--foot-name", default="Right_Shoe")
    parser.add_argument("--up-axis", type=int, default=2)
    args = parser.parse_args()

    t_raw, z_raw, _ = load_demo_foot_z(
        args.filepath,
        foot_name=args.foot_name,
        up_axis=args.up_axis,
    )

    intervals, quiet_mask, debug = detect_z_quiet_intervals(t_raw, z_raw)

    print(intervals)
    print_z_quiet_debug_summary(debug)
    plot_z_quiet_debug(t_raw, intervals, debug)
