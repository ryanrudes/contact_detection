"""Body-local contact geometry and marker-to-surface calibration.

The canonical representation is intentionally split in two:

* persistent model objects are expressed in a rigid body's local frame;
* world-frame quantities are views induced by a particular body pose.

Contact-frame ``+Z`` is the outward normal, away from the partner surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .types import FloatArray

MarkerT = TypeVar("MarkerT")


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform from a child frame into a parent frame.

    Attributes:
        translation: Child-frame origin in parent coordinates, shape ``(3,)``.
        rotation: Child-frame orientation relative to the parent frame.
    """

    translation: FloatArray
    rotation: Rotation = field(default_factory=Rotation.identity)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation",
            np.asarray(self.translation, dtype=np.float64).reshape(3),
        )

    @classmethod
    def identity(cls) -> RigidTransform:
        """Return the identity transform."""

        return cls(translation=np.zeros(3, dtype=np.float64))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> RigidTransform:
        """Build from a homogeneous ``(4, 4)`` matrix."""

        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError("matrix must have shape (4, 4).")
        return cls(translation=mat[:3, 3], rotation=Rotation.from_matrix(mat[:3, :3]))

    def as_matrix(self) -> FloatArray:
        """Return a homogeneous ``(4, 4)`` matrix."""

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> RigidTransform:
        """Return the inverse transform."""

        rotation_inv = self.rotation.inv()
        return RigidTransform(
            translation=rotation_inv.apply(-self.translation),
            rotation=rotation_inv,
        )

    def compose(self, other: RigidTransform) -> RigidTransform:
        """Return ``self @ other``; ``other`` is applied first."""

        return RigidTransform(
            translation=self.translation + self.rotation.apply(other.translation),
            rotation=self.rotation * other.rotation,
        )

    def transform_points(self, points: ArrayLike) -> FloatArray:
        """Map child-frame points into the parent frame."""

        arr = np.asarray(points, dtype=np.float64)
        flat = arr.reshape(-1, 3)
        transformed = self.rotation.apply(flat) + self.translation
        return np.asarray(transformed.reshape(arr.shape), dtype=np.float64)

    def inverse_transform_points(self, points: ArrayLike) -> FloatArray:
        """Map parent-frame points into the child frame."""

        arr = np.asarray(points, dtype=np.float64)
        flat = arr.reshape(-1, 3)
        transformed = self.rotation.inv().apply(flat - self.translation)
        return np.asarray(transformed.reshape(arr.shape), dtype=np.float64)

    @property
    def z_axis(self) -> FloatArray:
        """Child ``+Z`` axis expressed in parent coordinates."""

        axis = self.rotation.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError("transform has a degenerate +Z axis.")
        return axis / norm

    def rectangle_corners_xy(self, half_width: float, half_length: float) -> FloatArray:
        """Rectangle corners on the child ``z=0`` plane, expressed in parent coordinates."""

        if half_width < 0 or half_length < 0:
            raise ValueError("half_width and half_length must be non-negative.")
        local = np.asarray(
            [
                [-half_width, -half_length, 0.0],
                [half_width, -half_length, 0.0],
                [half_width, half_length, 0.0],
                [-half_width, half_length, 0.0],
            ],
            dtype=np.float64,
        )
        return self.transform_points(local)


@dataclass(frozen=True)
class SignedAxis:
    """A concrete coordinate axis with sign."""

    axis: "CoordinateAxis"
    sign: int = 1

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise ValueError("sign must be either -1 or +1.")

    def __pos__(self) -> SignedAxis:
        return self

    def __neg__(self) -> SignedAxis:
        return SignedAxis(axis=self.axis, sign=-self.sign)

    def vector(self) -> FloatArray:
        """Return this axis as a unit vector."""

        vector = np.zeros(3, dtype=np.float64)
        vector[int(self.axis)] = float(self.sign)
        return vector


class CoordinateAxis(IntEnum):
    """Concrete coordinate dimensions."""

    X = 0
    Y = 1
    Z = 2

    def __pos__(self) -> SignedAxis:
        return SignedAxis(axis=self, sign=1)

    def __neg__(self) -> SignedAxis:
        return SignedAxis(axis=self, sign=-1)


@dataclass(frozen=True)
class SemanticAxisTranslation:
    """Marker-to-surface displacement along a semantic body axis."""

    axis: "SemanticAxis"
    distance: float

    def __mul__(self, scale: float) -> SemanticAxisTranslation:
        return SemanticAxisTranslation(axis=self.axis, distance=float(scale) * self.distance)

    def __rmul__(self, scale: float) -> SemanticAxisTranslation:
        return self * scale

    def __neg__(self) -> SemanticAxisTranslation:
        return -1.0 * self

    def resolve(self, body: RigidBodyContactModel[Any]) -> FloatArray:
        """Return the concrete body-frame displacement."""

        return self.distance * body.axis(self.axis)


class SemanticAxis(StrEnum):
    """Semantic local body directions."""

    RIGHT = "right"
    FORWARD = "forward"
    UP = "up"

    def __pos__(self) -> SemanticAxisTranslation:
        return SemanticAxisTranslation(axis=self, distance=1.0)

    def __neg__(self) -> SemanticAxisTranslation:
        return SemanticAxisTranslation(axis=self, distance=-1.0)

    def __mul__(self, distance: float) -> SemanticAxisTranslation:
        return SemanticAxisTranslation(axis=self, distance=float(distance))

    def __rmul__(self, distance: float) -> SemanticAxisTranslation:
        return self * distance


@dataclass(frozen=True)
class AxisConvention:
    """Mapping from semantic local axes to concrete coordinate directions."""

    axes: Mapping[SemanticAxis, SignedAxis]

    def signed_axis(self, axis: SemanticAxis) -> SignedAxis:
        """Return the signed coordinate axis for ``axis``."""

        try:
            return self.axes[axis]
        except KeyError as exc:
            raise KeyError(f"axis convention does not define {axis.value!r}") from exc

    def vector(self, axis: SemanticAxis) -> FloatArray:
        """Return a semantic axis as a body-frame vector."""

        return self.signed_axis(axis).vector()


Z_UP_AXES = AxisConvention(
    {
        SemanticAxis.RIGHT: -CoordinateAxis.Y,
        SemanticAxis.FORWARD: +CoordinateAxis.X,
        SemanticAxis.UP: +CoordinateAxis.Z,
    }
)
Y_UP_AXES = AxisConvention(
    {
        SemanticAxis.RIGHT: +CoordinateAxis.X,
        SemanticAxis.FORWARD: +CoordinateAxis.Z,
        SemanticAxis.UP: +CoordinateAxis.Y,
    }
)
MUJOCO_AXES = Z_UP_AXES
ISAAC_AXES = Z_UP_AXES


class ContactRegion(ABC):
    """Finite or infinite region on a contact patch's local ``z=0`` plane."""

    @abstractmethod
    def contains(self, xy: ArrayLike) -> bool:
        """Return whether a contact-plane ``(x, y)`` point lies in the region."""


@dataclass(frozen=True)
class InfinitePlaneRegion(ContactRegion):
    """Unbounded tangent plane."""

    def contains(self, xy: ArrayLike) -> bool:
        _ = np.asarray(xy, dtype=np.float64).reshape(2)
        return True


@dataclass(frozen=True)
class RectangularRegion(ContactRegion):
    """Axis-aligned rectangle in the contact tangent plane."""

    half_width: float
    half_length: float

    def __post_init__(self) -> None:
        if self.half_width < 0 or self.half_length < 0:
            raise ValueError("half_width and half_length must be non-negative.")

    @classmethod
    def from_size(cls, width: float, length: float) -> RectangularRegion:
        """Build from full extents."""

        return cls(half_width=float(width) / 2.0, half_length=float(length) / 2.0)

    def contains(self, xy: ArrayLike) -> bool:
        x, y = np.asarray(xy, dtype=np.float64).reshape(2)
        return bool(abs(float(x)) <= self.half_width and abs(float(y)) <= self.half_length)


@dataclass(frozen=True)
class SampleHullRegion(ContactRegion):
    """Convex hull implied by contact-frame sample points."""

    points_contact: FloatArray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_contact, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_contact must have shape (samples, 3).")
        object.__setattr__(self, "points_contact", points)

    def contains(self, xy: ArrayLike) -> bool:
        """Return an axis-aligned hull test over the projected samples."""

        point = np.asarray(xy, dtype=np.float64).reshape(2)
        if self.points_contact.size == 0:
            return False
        mins = np.min(self.points_contact[:, :2], axis=0)
        maxs = np.max(self.points_contact[:, :2], axis=0)
        return bool(np.all(point >= mins) and np.all(point <= maxs))


@dataclass(frozen=True)
class BodyFrameTranslation:
    """Explicit body-frame marker-to-surface displacement."""

    vector_body: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector_body",
            np.asarray(self.vector_body, dtype=np.float64).reshape(3),
        )

    @classmethod
    def zeros(cls) -> BodyFrameTranslation:
        """Return a zero displacement."""

        return cls(np.zeros(3, dtype=np.float64))

    def __mul__(self, scale: float) -> BodyFrameTranslation:
        return BodyFrameTranslation(vector_body=float(scale) * self.vector_body)

    def __rmul__(self, scale: float) -> BodyFrameTranslation:
        return self * scale

    def __neg__(self) -> BodyFrameTranslation:
        return -1.0 * self

    def resolve(self, body: RigidBodyContactModel[Any]) -> FloatArray:
        """Return the concrete body-frame displacement."""

        _ = body
        return self.vector_body


MarkerTranslation = BodyFrameTranslation | SemanticAxisTranslation


@dataclass(frozen=True)
class PatchCalibration(Generic[MarkerT]):
    """Marker-to-surface calibration for one contact patch."""

    marker_translations: Mapping[MarkerT, MarkerTranslation]
    region: ContactRegion | None = None

    def __post_init__(self) -> None:
        if not self.marker_translations:
            raise ValueError("marker_translations must not be empty.")

    @classmethod
    def from_markers(
        cls,
        markers: Sequence[MarkerT],
        *,
        translation: MarkerTranslation | None = None,
        region: ContactRegion | None = None,
    ) -> PatchCalibration[MarkerT]:
        """Build a calibration with the same displacement for each marker."""

        if not markers:
            raise ValueError("markers must not be empty.")
        if len(frozenset(markers)) != len(markers):
            raise ValueError("markers must not contain duplicates.")
        marker_translation = translation or BodyFrameTranslation.zeros()
        return cls(
            marker_translations={marker: marker_translation for marker in markers},
            region=region,
        )

    @property
    def markers(self) -> tuple[MarkerT, ...]:
        """Markers in declared calibration order."""

        return tuple(self.marker_translations)

    @property
    def marker_names(self) -> tuple[str, ...]:
        """String marker names in declared calibration order."""

        return tuple(_marker_name(marker) for marker in self.marker_translations)

    def surface_points(
        self,
        marker_positions_body: Mapping[MarkerT | str, ArrayLike],
        body: RigidBodyContactModel[MarkerT],
    ) -> FloatArray:
        """Return calibrated body-frame surface samples."""

        points: list[np.ndarray] = []
        for marker, translation in self.marker_translations.items():
            marker_key = _marker_lookup_key(marker, marker_positions_body)
            marker_position = np.asarray(marker_positions_body[marker_key], dtype=np.float64).reshape(3)
            points.append(marker_position + translation.resolve(body))
        return np.stack(points, axis=0)

    def build_patch(
        self,
        marker_positions_body: Mapping[MarkerT | str, ArrayLike],
        body: RigidBodyContactModel[MarkerT],
    ) -> ContactPatch:
        """Fit and return a persistent body-local contact patch."""

        samples_body = self.surface_points(marker_positions_body, body)
        frame = fit_patch_frame_from_points(samples_body, up_axis=body.up_axis)
        region = self.region or SampleHullRegion(frame.inverse_transform_points(samples_body))
        return ContactPatch(transform_body_patch=frame, region=region)


@dataclass(frozen=True)
class ContactPatch:
    """Persistent body-local contact surface."""

    transform_body_patch: RigidTransform
    region: ContactRegion = field(default_factory=InfinitePlaneRegion)

    def view(self, transform_world_body: RigidTransform) -> ContactPatchView:
        """Return a pose-dependent world-frame view."""

        return ContactPatchView(patch=self, transform_world_body=transform_world_body)


@dataclass(frozen=True)
class ContactPatchView:
    """World-frame view of a body-local contact patch."""

    patch: ContactPatch
    transform_world_body: RigidTransform

    @property
    def transform_world_patch(self) -> RigidTransform:
        """Patch frame in world coordinates."""

        return self.transform_world_body.compose(self.patch.transform_body_patch)

    @property
    def contact_point_world(self) -> FloatArray:
        """Contact-frame origin in world coordinates."""

        return self.transform_world_patch.translation.copy()

    @property
    def normal_world(self) -> FloatArray:
        """Contact outward normal in world coordinates."""

        return self.transform_world_patch.z_axis

    def clearance_along_normal(self, world_point: ArrayLike) -> float:
        """Signed clearance from the patch origin along the outward normal."""

        delta = np.asarray(world_point, dtype=np.float64).reshape(3) - self.contact_point_world
        return float(delta @ self.normal_world)


@dataclass(frozen=True)
class RigidBodyContactModel(Generic[MarkerT]):
    """Contact model attached to one tracked rigid body."""

    body_name: str
    marker_type: type[MarkerT] | None = None
    axis_convention: AxisConvention = Z_UP_AXES
    patch_calibrations: Mapping[str, PatchCalibration[MarkerT]] = field(default_factory=dict)
    patches: Mapping[str, ContactPatch] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.body_name:
            raise ValueError("body_name must not be empty.")
        if not self.patch_calibrations and not self.patches:
            raise ValueError("RigidBodyContactModel requires patch_calibrations or patches.")

    @property
    def up_axis(self) -> int:
        """Concrete coordinate axis used for semantic ``UP``."""

        return int(self.axis_convention.signed_axis(SemanticAxis.UP).axis)

    @property
    def marker_names(self) -> tuple[str, ...]:
        """All marker names used by the patch calibrations."""

        names: list[str] = []
        for calibration in self.patch_calibrations.values():
            names.extend(calibration.marker_names)
        return tuple(dict.fromkeys(names))

    @property
    def patch_names(self) -> tuple[str, ...]:
        """Patch names available on this model."""

        return tuple(dict.fromkeys((*self.patches.keys(), *self.patch_calibrations.keys())))

    def axis(self, axis: SemanticAxis) -> FloatArray:
        """Return a semantic axis as a body-frame vector."""

        return self.axis_convention.vector(axis)

    def marker_translation(self, marker_name: str) -> MarkerTranslation:
        """Return the marker-to-surface translation for ``marker_name``."""

        for calibration in self.patch_calibrations.values():
            for marker, translation in calibration.marker_translations.items():
                if _marker_name(marker) == marker_name:
                    return translation
        raise KeyError(f"marker {marker_name!r} is not part of body {self.body_name!r}")

    def compile(
        self,
        *,
        marker_positions_world: Mapping[str, ArrayLike],
        body_translation: ArrayLike,
        body_rotation: Rotation,
    ) -> RigidBodyContactModel[MarkerT]:
        """Return a copy with calibrations compiled into body-local patches."""

        body_transform = RigidTransform(
            translation=np.asarray(body_translation, dtype=np.float64).reshape(3),
            rotation=body_rotation,
        )
        world_to_body = body_transform.inverse()
        patches = dict(self.patches)
        for name, calibration in self.patch_calibrations.items():
            marker_positions_body: dict[str, FloatArray] = {}
            for marker_name in calibration.marker_names:
                if marker_name not in marker_positions_world:
                    raise KeyError(f"missing marker {marker_name!r} for body {self.body_name!r}")
                marker_positions_body[marker_name] = world_to_body.transform_points(
                    np.asarray(marker_positions_world[marker_name], dtype=np.float64).reshape(1, 3)
                )[0]
            patches[name] = calibration.build_patch(marker_positions_body, self)
        return RigidBodyContactModel(
            body_name=self.body_name,
            marker_type=self.marker_type,
            axis_convention=self.axis_convention,
            patch_calibrations=self.patch_calibrations,
            patches=patches,
        )

    def view(self, transform_world_body: RigidTransform) -> RigidBodyContactView:
        """Return a pose-dependent world-frame view."""

        return RigidBodyContactView(model=self, transform_world_body=transform_world_body)

    def patch(self, name: str | None = None) -> ContactPatch:
        """Return one compiled patch."""

        patch_name = name or _first_patch_name(self)
        try:
            return self.patches[patch_name]
        except KeyError as exc:
            raise KeyError(f"patch {patch_name!r} is not compiled for body {self.body_name!r}") from exc

    def surface_sample_delta_body(self, marker_name: str) -> FloatArray:
        """Return marker-to-surface displacement in body coordinates."""

        return self.marker_translation(marker_name).resolve(self)


@dataclass(frozen=True)
class RigidBodyContactView(Generic[MarkerT]):
    """World-frame view of a rigid body contact model."""

    model: RigidBodyContactModel[MarkerT]
    transform_world_body: RigidTransform

    def patch(self, name: str | None = None) -> ContactPatchView:
        """Return a patch view."""

        return self.model.patch(name).view(self.transform_world_body)


def fit_patch_frame_from_points(points_body: ArrayLike, *, up_axis: int = 2) -> RigidTransform:
    """Fit a contact frame from body-local surface samples."""

    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2.")
    points = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] < 3:
        raise ValueError("Need at least 3 finite points to fit a contact patch.")
    centroid = np.mean(finite, axis=0)
    centered = finite - centroid
    if float(np.max(np.linalg.norm(centered, axis=1))) <= 1e-9:
        raise ValueError("Points are too colocated to fit a contact patch.")
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("Degenerate contact patch normal.")
    normal /= normal_norm
    if normal[up_axis] < 0.0:
        normal = -normal
    tangent = np.asarray(vh[0], dtype=np.float64)
    tangent -= normal * float(tangent @ normal)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(normal @ reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = np.cross(normal, reference)
        tangent_norm = float(np.linalg.norm(tangent))
    tangent /= tangent_norm
    bitangent = np.cross(normal, tangent)
    return RigidTransform(
        translation=centroid,
        rotation=Rotation.from_matrix(np.column_stack([tangent, bitangent, normal])),
    )


def rotation_matrices_from_quaternions(
    quaternions: ArrayLike,
    *,
    scalar_last: bool = True,
) -> NDArray[np.float64]:
    """Return rotation matrices with shape ``(N, 3, 3)`` from unit quaternions."""

    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4).")
    if not scalar_last:
        q = q[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(norms > 1e-12, norms, 1.0)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def marker_names_for_flat_trajectory(marker_names: Sequence[str], n_frames: int) -> list[str]:
    """Expand marker names to match ``(N, K, 3).reshape(N * K, 3)`` row-major order."""

    if n_frames < 0:
        raise ValueError("n_frames must be non-negative.")
    return [name for _ in range(n_frames) for name in marker_names]


def marker_names_for_contact_models(models: Sequence[RigidBodyContactModel[Any]]) -> tuple[str, ...]:
    """Return unique marker names used by a collection of contact models."""

    names: list[str] = []
    for model in models:
        names.extend(model.marker_names)
    return tuple(dict.fromkeys(names))


def apply_contact_model_offsets(
    points: ArrayLike,
    marker_names: Sequence[str],
    models: Sequence[RigidBodyContactModel[Any]],
    *,
    body_rotations: Mapping[str, FloatArray] | None = None,
    quaternion_scalar_last: bool = True,
) -> FloatArray:
    """Shift marker positions to calibrated surface samples in world coordinates."""

    arr = np.asarray(points, dtype=np.float64)
    if not models:
        return arr
    marker_to_model = _marker_model_index(models)
    if arr.ndim == 2 and arr.shape[1] == 3:
        if len(marker_names) != arr.shape[0]:
            raise ValueError("marker_names length must match points.shape[0].")
        out = arr.copy()
        markers_per_frame = len(marker_names_for_contact_models(models))
        flat_trajectory = markers_per_frame > 0 and len(marker_names) > markers_per_frame
        for row, marker_name in enumerate(marker_names):
            model = marker_to_model[marker_name]
            frame_index = row // markers_per_frame if flat_trajectory else 0
            rotation = _rotation_for_model(
                model,
                frame_index=frame_index,
                body_rotations=body_rotations,
                quaternion_scalar_last=quaternion_scalar_last,
            )
            out[row] += rotation @ model.surface_sample_delta_body(marker_name)
        return out
    if arr.ndim == 3 and arr.shape[2] == 3:
        if len(marker_names) != arr.shape[1]:
            raise ValueError("marker_names length must match points.shape[1].")
        out = arr.copy()
        for frame_index in range(arr.shape[0]):
            for col, marker_name in enumerate(marker_names):
                model = marker_to_model[marker_name]
                rotation = _rotation_for_model(
                    model,
                    frame_index=frame_index,
                    body_rotations=body_rotations,
                    quaternion_scalar_last=quaternion_scalar_last,
                )
                out[frame_index, col] += rotation @ model.surface_sample_delta_body(marker_name)
        return out
    raise ValueError("points must have shape (M, 3) or (N, K, 3).")


def compile_contact_models(
    models: Sequence[RigidBodyContactModel[Any]],
    *,
    marker_positions_world: Mapping[str, FloatArray],
    body_positions: Mapping[str, FloatArray],
    body_quaternions: Mapping[str, FloatArray],
    frame_index: int = 0,
    quaternion_scalar_last: bool = True,
) -> tuple[RigidBodyContactModel[Any], ...]:
    """Compile marker calibrations into body-local patches for each model."""

    compiled: list[RigidBodyContactModel[Any]] = []
    for model in models:
        marker_positions = {
            name: np.asarray(marker_positions_world[name][frame_index], dtype=np.float64)
            for name in model.marker_names
        }
        body_position = np.asarray(body_positions[model.body_name][frame_index], dtype=np.float64)
        body_quat = np.asarray(body_quaternions[model.body_name][frame_index], dtype=np.float64)
        if not quaternion_scalar_last:
            body_quat = body_quat[[1, 2, 3, 0]]
        compiled.append(
            model.compile(
                marker_positions_world=marker_positions,
                body_translation=body_position,
                body_rotation=Rotation.from_quat(body_quat),
            )
        )
    return tuple(compiled)


def model_for_marker(
    models: Sequence[RigidBodyContactModel[Any]],
    marker_name: str,
) -> RigidBodyContactModel[Any]:
    """Return the contact model that owns ``marker_name``."""

    return _marker_model_index(models)[marker_name]


def model_map(models: Sequence[RigidBodyContactModel[Any]]) -> dict[str, RigidBodyContactModel[Any]]:
    """Return contact models keyed by body name."""

    return {model.body_name: model for model in models}


def _marker_name(marker: Any) -> str:
    return str(marker.value) if hasattr(marker, "value") else str(marker)


def _marker_lookup_key(
    marker: Any,
    mapping: Mapping[MarkerT | str, ArrayLike],
) -> MarkerT | str:
    if marker in mapping:
        return marker
    name = _marker_name(marker)
    if name in mapping:
        return name
    raise KeyError(f"missing marker position for {name!r}")


def _first_patch_name(model: RigidBodyContactModel[Any]) -> str:
    names = model.patch_names
    if not names:
        raise KeyError(f"body {model.body_name!r} has no patches")
    return names[0]


def _marker_model_index(
    models: Sequence[RigidBodyContactModel[Any]],
) -> dict[str, RigidBodyContactModel[Any]]:
    index: dict[str, RigidBodyContactModel[Any]] = {}
    for model in models:
        for marker_name in model.marker_names:
            if marker_name in index:
                raise ValueError(f"marker {marker_name!r} appears in multiple contact models.")
            index[marker_name] = model
    return index


def _rotation_for_model(
    model: RigidBodyContactModel[Any],
    *,
    frame_index: int,
    body_rotations: Mapping[str, FloatArray] | None,
    quaternion_scalar_last: bool,
) -> NDArray[np.float64]:
    if body_rotations is None:
        raise ValueError(f"body_rotations required for contact model {model.body_name!r}.")
    try:
        quaternions = body_rotations[model.body_name]
    except KeyError as exc:
        raise KeyError(f"missing rotations for body {model.body_name!r}") from exc
    return rotation_matrices_from_quaternions(
        np.asarray(quaternions, dtype=np.float64),
        scalar_last=quaternion_scalar_last,
    )[frame_index]
