"""Shared NumPy typing aliases used across contact_detection."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int_]
Interval: TypeAlias = tuple[float, float]
IntervalList: TypeAlias = list[Interval]
DebugDict: TypeAlias = dict[str, Any]
