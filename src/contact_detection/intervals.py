"""Interval utilities: mask conversion, temporal cleaning, and summaries."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import BoolArray, IntervalList


@dataclass
class IntervalSummary:
    """Per-interval statistics for a multivariate time series."""

    start: float
    end: float
    duration: float
    median: NDArray[np.float64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    min: NDArray[np.float64]
    max: NDArray[np.float64]
    n_samples: int


def intervals_from_mask(
    t: ArrayLike,
    mask: ArrayLike,
    min_duration: float = 0.0,
) -> IntervalList:
    """Convert a boolean mask into ``(start, end)`` time intervals.

    Parameters
    ----------
    t:
        Timestamps with shape ``(N,)``.
    mask:
        Boolean mask with shape ``(N,)``.
    min_duration:
        Drop intervals shorter than this many seconds.

    Returns
    -------
    list[tuple[float, float]]
        Contiguous runs where ``mask`` is True, expressed in the same time
        units as ``t``.
    """

    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if t.ndim != 1 or mask.ndim != 1:
        raise ValueError("t and mask must be 1D arrays.")
    if len(t) != len(mask):
        raise ValueError("t and mask must have the same length.")
    if len(t) == 0:
        return []

    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    intervals: IntervalList = []
    for start, end in zip(starts, ends):
        start_time = float(t[start])
        end_time = float(t[end - 1])
        if end_time - start_time >= min_duration:
            intervals.append((start_time, end_time))
    return intervals


def mask_from_intervals(t: ArrayLike, intervals: IntervalList) -> BoolArray:
    """Build a per-sample mask that is True inside any of the given intervals."""

    t = np.asarray(t, dtype=float)
    mask = np.zeros(len(t), dtype=bool)
    for start, end in intervals:
        mask |= (t >= start) & (t <= end)
    return mask


def _run_durations(
    t: ArrayLike,
    mask: ArrayLike,
    value: bool,
) -> list[tuple[int, int, float]]:
    """Return contiguous runs of ``mask == value`` as ``(start_idx, end_idx, duration)``."""

    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(t) == 0:
        return []

    value_mask = mask == value
    padded = np.r_[False, value_mask, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    runs: list[tuple[int, int, float]] = []
    for start, end in zip(starts, ends):
        start_idx = start
        end_idx = end - 1
        duration = float(t[end_idx] - t[start_idx])
        runs.append((start_idx, end_idx, duration))
    return runs


def fill_short_false_runs(t: ArrayLike, mask: ArrayLike, max_gap_time: float) -> BoolArray:
    """Fill brief False gaps surrounded by True runs (hole filling)."""

    cleaned = np.asarray(mask, dtype=bool).copy()
    if max_gap_time <= 0.0:
        return cleaned
    for start, end, duration in _run_durations(t, cleaned, False):
        touches_edge = start == 0 or end == len(cleaned) - 1
        if not touches_edge and duration <= max_gap_time:
            cleaned[start : end + 1] = True
    return cleaned


def remove_short_true_runs(t: ArrayLike, mask: ArrayLike, min_duration: float) -> BoolArray:
    """Remove brief True blips shorter than ``min_duration`` seconds."""

    cleaned = np.asarray(mask, dtype=bool).copy()
    if min_duration <= 0.0:
        return cleaned
    for start, end, duration in _run_durations(t, cleaned, True):
        if duration < min_duration:
            cleaned[start : end + 1] = False
    return cleaned


def clean_mask_by_time(
    t: ArrayLike,
    mask: ArrayLike,
    max_gap_time: float = 0.10,
    min_blip_time: float = 0.15,
) -> BoolArray:
    """Fill short gaps then remove short contact blips."""

    cleaned = fill_short_false_runs(t, mask, max_gap_time)
    return remove_short_true_runs(t, cleaned, min_blip_time)


def summarize_intervals(
    t: ArrayLike,
    X: ArrayLike,
    intervals: IntervalList,
) -> list[IntervalSummary]:
    """Compute descriptive statistics for ``X`` inside each time interval."""

    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    summaries: list[IntervalSummary] = []

    if len(t) != len(X):
        raise ValueError("t and X must have the same number of samples.")

    for start, end in intervals:
        mask = (t >= start) & (t <= end)
        values = X[mask]
        if len(values) == 0:
            continue
        summaries.append(
            IntervalSummary(
                start=float(start),
                end=float(end),
                duration=float(end - start),
                median=np.median(values, axis=0),
                mean=np.mean(values, axis=0),
                std=np.std(values, axis=0),
                min=np.min(values, axis=0),
                max=np.max(values, axis=0),
                n_samples=int(len(values)),
            )
        )
    return summaries
