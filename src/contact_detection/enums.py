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
    """Floor geometry used by per-foot support classification."""

    HEIGHT = "height"
    PLANE = "plane"


class SupportModelType(StrEnum):
    """Support-surface geometry selected during bootstrap fitting."""

    AUTO = "auto"
    PLANE = "plane"
    HEIGHTMAP = "heightmap"
    LOCAL_HEIGHTMAP = "local_heightmap"


class QuietSignalType(StrEnum):
    """Input signal layout accepted by the quiet detector."""

    SCALAR = "scalar"
    POSITION_COMPONENT = "position_component"
    VECTOR_POSITION = "vector_position"
    VECTOR = "vector"
    QUATERNION = "quaternion"


class VectorQuietMode(StrEnum):
    """How vector signals are reduced to scalar activity and spread metrics."""

    NORM = "norm"
    EUCLIDEAN = "euclidean"
    MAX_COMPONENT = "max_component"
    NORMALIZED_NORM = "normalized_norm"
    MAHALANOBIS = "mahalanobis"
