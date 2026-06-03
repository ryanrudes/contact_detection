"""String-valued enums for detector modes and configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

_E = TypeVar("_E", bound=StrEnum)


def normalize_enum(value: str | _E, enum_cls: type[_E]) -> _E:
    """Coerce an enum member or its string value to ``enum_cls``."""

    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_cls)
        raise ValueError(
            f"{enum_cls.__name__} must be one of {{{allowed}}}; got {value!r}"
        ) from exc


class FloorModel(StrEnum):
    """Floor geometry used by per-foot support classification.

    Attributes:
        HEIGHT (str): Scalar lower-foot-height floor model (``"height"``).
        PLANE (str): Robust plane fit to low foot samples (``"plane"``).
    """

    HEIGHT = "height"
    PLANE = "plane"


class SupportModelType(StrEnum):
    """Support-surface geometry selected during bootstrap fitting.

    Attributes:
        AUTO (str): Pick plane vs heightmap from data (``"auto"``).
        PLANE (str): Single global plane support model.
        HEIGHTMAP (str): Grid heightmap over the capture volume.
        LOCAL_HEIGHTMAP (str): Local percentile heightmap around each candidate.
    """

    AUTO = "auto"
    PLANE = "plane"
    HEIGHTMAP = "heightmap"
    LOCAL_HEIGHTMAP = "local_heightmap"


class QuietSignalType(StrEnum):
    """Input signal layout accepted by the quiet detector.

    Attributes:
        SCALAR (str): One scalar channel per frame.
        POSITION_COMPONENT (str): Single component of a position vector.
        VECTOR_POSITION (str): 3D position samples.
        VECTOR (str): General vector signal (norm used when reducing to scalar).
        QUATERNION (str): Orientation quaternion samples.
    """

    SCALAR = "scalar"
    POSITION_COMPONENT = "position_component"
    VECTOR_POSITION = "vector_position"
    VECTOR = "vector"
    QUATERNION = "quaternion"


class VectorQuietMode(StrEnum):
    """How vector signals are reduced to scalar activity and spread metrics.

    Attributes:
        NORM (str): L2 norm of the vector.
        EUCLIDEAN (str): Same as norm (Euclidean length).
        MAX_COMPONENT (str): Largest absolute component.
        NORMALIZED_NORM (str): Norm divided by a running scale estimate.
        MAHALANOBIS (str): Mahalanobis distance when covariance is available.
    """

    NORM = "norm"
    EUCLIDEAN = "euclidean"
    MAX_COMPONENT = "max_component"
    NORMALIZED_NORM = "normalized_norm"
    MAHALANOBIS = "mahalanobis"
