"""Shared NumPy typing aliases used across contact_detection."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
"""1D or multi-dimensional array of ``float64`` samples."""

BoolArray: TypeAlias = NDArray[np.bool_]
"""Boolean mask or flag array aligned with a time series."""

IntArray: TypeAlias = NDArray[np.int_]
"""Integer index or label array."""

Interval: TypeAlias = tuple[float, float]
"""Closed time interval ``(start, end)`` in seconds."""

IntervalList: TypeAlias = list[Interval]
"""Ordered list of time intervals."""

DebugDict: TypeAlias = dict[str, Any]
"""Arbitrary debug metadata attached to detector results."""
