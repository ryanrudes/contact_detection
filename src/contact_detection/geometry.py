"""Canonical contact-surface geometry: body-root to contact-frame transforms.

Convention:
    Each :class:`BodyContactSurface` stores ``T_body_contact`` (:class:`RigidTransform`).
    Contact-frame **+Z** is the **outward** normal (away from the partner surface).
    ``p_body = R @ p_contact + t`` for points in the contact frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .types import FloatArray

MarkerT = TypeVar("MarkerT")


@dataclass(frozen=True)
class RigidTransform:
    """Fixed rigid transform mapping contact-frame points into the body root frame.

    Attributes:
        translation: Contact-frame origin in body coordinates, shape ``(3,)``.
        rotation: Orientation of the contact frame relative to the body (scipy ``Rotation``).
    """

    translation: FloatArray
    rotation: Rotation

    def __post_init__(self) -> None:
        t = np.asarray(self.translation, dtype=np.float64).reshape(3)
        object.__setattr__(self, "translation", t)

    @classmethod
    def identity(cls) -> RigidTransform:
        """Identity transform (contact frame coincident with body root)."""
        return cls(translation=np.zeros(3, dtype=np.float64), rotation=Rotation.identity())

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> RigidTransform:
        """Build from a ``(4, 4)`` homogeneous matrix (body from contact)."""
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError("matrix must have shape (4, 4).")
        return cls(
            translation=mat[:3, 3].copy(),
            rotation=Rotation.from_matrix(mat[:3, :3]),
        )

    def as_matrix(self) -> FloatArray:
        """Return ``(4, 4)`` homogeneous matrix mapping contact → body."""
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self.rotation.as_matrix()
        out[:3, 3] = self.translation
        return out

    def inverse(self) -> RigidTransform:
        """Return the transform mapping body-frame points into the contact frame."""
        rot_inv = self.rotation.inv()
        return RigidTransform(
            translation=rot_inv.apply(-self.translation),
            rotation=rot_inv,
        )

    def compose(self, other: RigidTransform) -> RigidTransform:
        """Compose as ``self @ other`` (apply ``other`` first, then ``self``)."""
        rot = self.rotation * other.rotation
        trans = self.translation + self.rotation.apply(other.translation)
        return RigidTransform(translation=trans, rotation=rot)

    def rectangle_corners_contact(
        self,
        half_width: float,
        half_length: float,
    ) -> FloatArray:
        """Rectangle corners on the contact plane (z=0) expressed in the parent frame."""
        if half_width < 0 or half_length < 0:
            raise ValueError("half_width and half_length must be non-negative.")
        local = np.array(
            [
                [-half_width, -half_length, 0.0],
                [half_width, -half_length, 0.0],
                [half_width, half_length, 0.0],
                [-half_width, half_length, 0.0],
            ],
            dtype=np.float64,
        )
        return self.transform_points(local)

    def transform_points(self, points: ArrayLike) -> FloatArray:
        """Map contact-frame points to body frame, preserving trailing shape ``(..., 3)``."""
        arr = np.asarray(points, dtype=np.float64)
        flat = arr.reshape(-1, 3)
        out = self.rotation.apply(flat) + self.translation
        return np.asarray(out.reshape(arr.shape), dtype=np.float64)

    def inverse_transform_points(self, points: ArrayLike) -> FloatArray:
        """Map body-frame points to the contact frame."""
        arr = np.asarray(points, dtype=np.float64)
        flat = arr.reshape(-1, 3)
        out = self.rotation.inv().apply(flat - self.translation)
        return np.asarray(out.reshape(arr.shape), dtype=np.float64)

    def outward_normal_body(self) -> FloatArray:
        """Unit outward normal (+Z contact axis) expressed in the body frame."""
        normal = self.rotation.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise ValueError("contact frame normal is degenerate.")
        return normal / norm

    def compose_world(self, body_translation: ArrayLike, body_rotation: Rotation) -> RigidTransform:
        """Return ``T_world_contact`` given ``T_world_body``."""
        body = RigidTransform(
            translation=np.asarray(body_translation, dtype=np.float64).reshape(3),
            rotation=body_rotation,
        )
        return body.compose(self)


@dataclass(frozen=True)
class InfinitePlaneRegion:
    """Unbounded tangent plane (partner is a half-space)."""


@dataclass(frozen=True)
class RectangleExtents:
    """Axis-aligned rectangle in the contact tangent plane (contact X/Y).

    Attributes:
        half_width: Half extent along contact +X (meters).
        half_length: Half extent along contact +Y (meters).
    """

    half_width: float
    half_length: float

    def __post_init__(self) -> None:
        if self.half_width < 0 or self.half_length < 0:
            raise ValueError("half_width and half_length must be non-negative.")


@dataclass(frozen=True)
class PointSamplesRegion:
    """Fixed sample points in the contact frame (shape ``(K, 3)``)."""

    points_contact: FloatArray

    def __post_init__(self) -> None:
        pts = np.asarray(self.points_contact, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_contact must have shape (K, 3).")
        object.__setattr__(self, "points_contact", pts)


@dataclass(frozen=True)
class SampleHullRegion:
    """Convex hull of sample points in the contact tangent plane (XY)."""

    points_contact: FloatArray

    def __post_init__(self) -> None:
        pts = np.asarray(self.points_contact, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_contact must have shape (K, 3).")
        object.__setattr__(self, "points_contact", pts)


ContactSurfaceRegion = InfinitePlaneRegion | RectangleExtents | PointSamplesRegion | SampleHullRegion


class ContactFrameMode(StrEnum):
    """How :class:`MarkerAnchoredPatch` derives ``T_body_contact`` at compile time."""

    EXPLICIT = "explicit"
    FIT_PLANE_FROM_SAMPLES = "fit_plane_from_samples"
    FIT_FROM_BODY_AXES = "fit_from_body_axes"


@dataclass(frozen=True)
class ContactFrameSpec:
    """Specification for deriving the contact frame from marker samples.

    Attributes:
        mode: Compile-time frame derivation strategy.
        explicit: Fixed ``T_body_contact`` when ``mode`` is ``explicit``.
        up_axis: World/capture vertical axis used to orient +Z outward (0, 1, or 2).
        body_up_axis: Body axis index for ``fit_from_body_axes`` (0, 1, or 2).
    """

    mode: ContactFrameMode | str = ContactFrameMode.FIT_PLANE_FROM_SAMPLES
    explicit: RigidTransform | None = None
    up_axis: int = 2
    body_up_axis: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ContactFrameMode(self.mode))
        if self.up_axis not in (0, 1, 2) or self.body_up_axis not in (0, 1, 2):
            raise ValueError("up_axis and body_up_axis must be 0, 1, or 2.")
        if self.mode == ContactFrameMode.EXPLICIT and self.explicit is None:
            raise ValueError("explicit transform is required when mode is 'explicit'.")

    @classmethod
    def fit_plane_from_samples(cls, *, up_axis: int = 2) -> ContactFrameSpec:
        """Fit plane to body-local surface samples; +Z is outward normal."""
        return cls(mode=ContactFrameMode.FIT_PLANE_FROM_SAMPLES, up_axis=up_axis)

    @classmethod
    def fit_from_body_axes(cls, *, body_up_axis: int = 2, up_axis: int = 2) -> ContactFrameSpec:
        """Use a body axis as outward normal; origin at sample centroid."""
        return cls(
            mode=ContactFrameMode.FIT_FROM_BODY_AXES,
            body_up_axis=body_up_axis,
            up_axis=up_axis,
        )

    @classmethod
    def with_explicit(cls, transform: RigidTransform) -> ContactFrameSpec:
        """Use a user-supplied ``T_body_contact``."""
        return cls(mode=ContactFrameMode.EXPLICIT, explicit=transform)


@dataclass(frozen=True)
class BodyContactSurface:
    """Canonical contact surface attached to one rigid body.

    Attributes:
        attach_body: Rigid-body name (Vicon subject).
        frame: ``T_body_contact`` — contact frame expressed in body root coordinates.
        region: Tangent-plane extent or sample geometry in the contact frame.
    """

    attach_body: str
    frame: RigidTransform
    region: ContactSurfaceRegion = InfinitePlaneRegion()

    def outward_normal_world(
        self,
        body_translation: ArrayLike,
        body_rotation: Rotation,
    ) -> FloatArray:
        """Outward normal in world coordinates at one time step."""
        world = self.frame.compose_world(body_translation, body_rotation)
        normal = world.rotation.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        norm = float(np.linalg.norm(normal))
        return normal / norm if norm > 1e-12 else normal

    def center_world(
        self,
        body_translation: ArrayLike,
        body_rotation: Rotation,
    ) -> FloatArray:
        """Contact-frame origin in world coordinates."""
        world = self.frame.compose_world(body_translation, body_rotation)
        return world.translation.copy()

    def clearance_along_normal(
        self,
        world_point: ArrayLike,
        body_translation: ArrayLike,
        body_rotation: Rotation,
    ) -> float:
        """Signed clearance along outward normal (positive = separated)."""
        center = self.center_world(body_translation, body_rotation)
        normal = self.outward_normal_world(body_translation, body_rotation)
        delta = np.asarray(world_point, dtype=np.float64).reshape(3) - center
        return float(delta @ normal)


def fit_plane_frame_from_points(
    points_body: ArrayLike,
    *,
    up_axis: int = 2,
) -> RigidTransform:
    """Fit ``T_body_contact`` from body-local sample points (SVD plane + centroid).

    The contact +Z axis is the plane normal, flipped so ``normal[up_axis] >= 0``.
    Contact +X is the principal in-plane axis from SVD.
    """
    pts = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] < 3:
        raise ValueError("Need at least 3 finite points to fit a contact plane frame.")
    centroid = np.mean(finite, axis=0)
    centered = finite - centroid
    if float(np.max(np.linalg.norm(centered, axis=1))) < 1e-9:
        raise ValueError("Points are too colocated to fit a contact plane.")
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("Degenerate plane normal from SVD.")
    normal /= norm
    if normal[up_axis] < 0.0:
        normal = -normal
    tangent = np.asarray(vh[0], dtype=np.float64)
    tangent -= normal * float(tangent @ normal)
    t_norm = float(np.linalg.norm(tangent))
    if t_norm < 1e-12:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(normal @ reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = np.cross(normal, reference)
        t_norm = float(np.linalg.norm(tangent))
    tangent /= t_norm
    bitangent = np.cross(normal, tangent)
    rot_matrix = np.column_stack([tangent, bitangent, normal])
    return RigidTransform(translation=centroid, rotation=Rotation.from_matrix(rot_matrix))


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
    norms = np.where(norms > 1e-12, norms, 1.0)
    q = q / norms
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


def marker_names_for_flat_trajectory(
    marker_names: Sequence[str],
    n_frames: int,
) -> list[str]:
    """Expand marker names to match ``(N, K, 3).reshape(N * K, 3)`` row-major order."""
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative.")
    return [name for _ in range(n_frames) for name in marker_names]


def _body_local_marker_samples(
    marker_positions_world: FloatArray,
    body_translation: FloatArray,
    body_rotation_matrix: NDArray[np.float64],
    sample_offsets_body: Mapping[Any, tuple[float, float, float]],
    marker_order: Sequence[Any],
) -> FloatArray:
    """Return body-local surface sample points for one calibration frame."""
    rot = np.asarray(body_rotation_matrix, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError("body_rotation_matrix must have shape (3, 3).")
    body_t = np.asarray(body_translation, dtype=np.float64).reshape(3)
    pts = np.asarray(marker_positions_world, dtype=np.float64)
    if pts.shape[0] != len(marker_order):
        raise ValueError("marker_positions_world rows must match patch_markers length.")
    samples: list[np.ndarray] = []
    for idx, marker in enumerate(marker_order):
        p_world = pts[idx].reshape(3)
        p_body = rot.T @ (p_world - body_t)
        delta = np.asarray(sample_offsets_body[marker], dtype=np.float64).reshape(3)
        samples.append(p_body + delta)
    return np.stack(samples, axis=0)


def _resolve_marker_key(marker: Any) -> str:
    return str(marker.value) if hasattr(marker, "value") else str(marker)


@dataclass(frozen=True)
class MarkerAnchoredPatch(Generic[MarkerT]):
    """Authoring spec: markers plus body-local offsets to nominal surface samples.

    Attributes:
        patch_markers: Markers defining the patch (column order for floor-fit stacks).
        sample_offsets_body: Body-local displacement from each marker to a surface sample.
        attach_body: Rigid-body name the markers belong to.
        frame_spec: How to derive ``T_body_contact`` at compile time.
        region_spec: Optional region override; defaults to sample hull from compile.
    """

    patch_markers: tuple[MarkerT, ...]
    sample_offsets_body: Mapping[MarkerT, tuple[float, float, float]]
    attach_body: str
    frame_spec: ContactFrameSpec = ContactFrameSpec.fit_plane_from_samples()
    region_spec: ContactSurfaceRegion | None = None

    def __post_init__(self) -> None:
        if not self.patch_markers:
            raise ValueError("patch_markers must not be empty.")
        patch_set = frozenset(self.patch_markers)
        if len(patch_set) != len(self.patch_markers):
            raise ValueError("patch_markers must not contain duplicates.")
        offset_keys = frozenset(self.sample_offsets_body)
        if offset_keys != patch_set:
            missing = patch_set - offset_keys
            extra = offset_keys - patch_set
            raise ValueError(
                f"sample_offsets_body keys must match patch_markers; "
                f"missing={[ _resolve_marker_key(m) for m in missing ]!r}, "
                f"extra={[ _resolve_marker_key(m) for m in extra ]!r}"
            )

    @property
    def marker_names(self) -> tuple[str, ...]:
        """String marker names for NPZ / detection pipelines."""
        return tuple(_resolve_marker_key(m) for m in self.patch_markers)

    def compile(
        self,
        *,
        marker_positions_world: ArrayLike,
        body_translation: ArrayLike,
        body_rotation_matrix: ArrayLike | None = None,
        body_quaternion_xyzw: ArrayLike | None = None,
        quaternion_scalar_last: bool = True,
    ) -> BodyContactSurface:
        """Derive :class:`BodyContactSurface` from one calibration pose.

        Provide either ``body_rotation_matrix`` ``(3, 3)`` or ``body_quaternion_xyzw`` ``(4,)``.
        """
        if body_rotation_matrix is None:
            if body_quaternion_xyzw is None:
                raise ValueError("body_rotation_matrix or body_quaternion_xyzw is required.")
            rot = rotation_matrices_from_quaternions(
                np.asarray(body_quaternion_xyzw, dtype=np.float64).reshape(1, 4),
                scalar_last=quaternion_scalar_last,
            )[0]
        else:
            rot = np.asarray(body_rotation_matrix, dtype=np.float64)

        samples_body = _body_local_marker_samples(
            np.asarray(marker_positions_world, dtype=np.float64),
            np.asarray(body_translation, dtype=np.float64),
            rot,
            self.sample_offsets_body,
            self.patch_markers,
        )

        spec = self.frame_spec
        if spec.mode == ContactFrameMode.EXPLICIT:
            assert spec.explicit is not None
            frame = spec.explicit
        elif spec.mode == ContactFrameMode.FIT_PLANE_FROM_SAMPLES:
            frame = fit_plane_frame_from_points(samples_body, up_axis=spec.up_axis)
        elif spec.mode == ContactFrameMode.FIT_FROM_BODY_AXES:
            centroid = np.mean(samples_body, axis=0)
            axis = np.zeros(3, dtype=np.float64)
            axis[spec.body_up_axis] = 1.0
            if axis[spec.up_axis] < 0:
                axis = -axis
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(axis @ tangent)) > 0.9:
                tangent = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            tangent -= axis * float(tangent @ axis)
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(axis, tangent)
            rot_mat = np.column_stack([tangent, bitangent, axis])
            frame = RigidTransform(translation=centroid, rotation=Rotation.from_matrix(rot_mat))
        else:
            raise ValueError(f"Unsupported ContactFrameMode: {spec.mode!r}")

        region = self.region_spec
        if region is None:
            samples_contact = frame.inverse_transform_points(samples_body)
            region = SampleHullRegion(points_contact=samples_contact)

        return BodyContactSurface(
            attach_body=self.attach_body,
            frame=frame,
            region=region,
        )

    def _marker_index(self, marker_name: str) -> int:
        for index, marker in enumerate(self.patch_markers):
            if _resolve_marker_key(marker) == marker_name:
                return index
        raise KeyError(f"marker {marker_name!r} not in patch")

    def world_delta(
        self,
        marker_name: str,
        *,
        rotation: NDArray[np.float64],
    ) -> FloatArray:
        """World-frame displacement for one marker at a single time."""
        index = self._marker_index(marker_name)
        marker = self.patch_markers[index]
        local = np.asarray(self.sample_offsets_body[marker], dtype=np.float64).reshape(3)
        return np.asarray(rotation, dtype=np.float64) @ local

    def world_sample_positions(
        self,
        marker_positions_world: ArrayLike,
        *,
        body_rotation_matrix: NDArray[np.float64] | None = None,
        body_quaternion_xyzw: ArrayLike | None = None,
        quaternion_scalar_last: bool = True,
    ) -> FloatArray:
        """Apply body-local offsets to world marker positions for one frame, shape ``(K, 3)``."""
        points = np.asarray(marker_positions_world, dtype=np.float64)
        if points.shape != (len(self.patch_markers), 3):
            raise ValueError("marker_positions_world must have shape (len(patch_markers), 3).")
        rotation = body_rotation_matrix
        if rotation is None:
            if body_quaternion_xyzw is None:
                raise ValueError("body_rotation_matrix or body_quaternion_xyzw is required.")
            rotation = rotation_matrices_from_quaternions(
                np.asarray(body_quaternion_xyzw, dtype=np.float64).reshape(1, 4),
                scalar_last=quaternion_scalar_last,
            )[0]
        out = points.copy()
        for index, marker in enumerate(self.patch_markers):
            local = np.asarray(self.sample_offsets_body[marker], dtype=np.float64).reshape(3)
            out[index] += rotation @ local
        return out


def _rotation_for_marker_patch(
    patch: MarkerAnchoredPatch[Any],
    *,
    frame_index: int,
    body_rotations: Mapping[str, FloatArray] | None,
    quaternion_scalar_last: bool,
) -> NDArray[np.float64]:
    if body_rotations is None:
        raise ValueError(
            f"body_rotations required for marker patch on body {patch.attach_body!r}."
        )
    matrices = rotation_matrices_from_quaternions(
        body_rotations[patch.attach_body],
        scalar_last=quaternion_scalar_last,
    )
    return matrices[frame_index]


@dataclass(frozen=True)
class ContactSurfaceSet:
    """One or more contact surfaces / marker patches for detection and floor fit.

    Attributes:
        marker_patches: Authoring patches used to shift marker samples (floor fit).
        body_surfaces: Compiled canonical surfaces keyed by body name.
    """

    marker_patches: tuple[MarkerAnchoredPatch[Any], ...] = ()
    body_surfaces: tuple[tuple[str, BodyContactSurface], ...] = ()

    def __post_init__(self) -> None:
        if not self.marker_patches and not self.body_surfaces:
            raise ValueError("ContactSurfaceSet requires marker_patches and/or body_surfaces.")
        names: list[str] = []
        for patch in self.marker_patches:
            names.extend(patch.marker_names)
        if len(names) != len(frozenset(names)):
            raise ValueError("marker names must be unique across marker_patches.")

    @property
    def marker_names(self) -> tuple[str, ...]:
        """All marker names in patch order."""
        return tuple(name for patch in self.marker_patches for name in patch.marker_names)

    @classmethod
    def from_marker_patches(cls, *patches: MarkerAnchoredPatch[Any]) -> ContactSurfaceSet:
        """Build a set used only for marker-based floor-fit sample shifting."""
        return cls(marker_patches=patches)

    def _marker_patch_index(self) -> dict[str, MarkerAnchoredPatch[Any]]:
        index: dict[str, MarkerAnchoredPatch[Any]] = {}
        for patch in self.marker_patches:
            for name in patch.marker_names:
                index[name] = patch
        return index

    def body_surface_map(self) -> dict[str, BodyContactSurface]:
        """Body surfaces as a mapping."""
        return dict(self.body_surfaces)

    def compile_body_surfaces(
        self,
        *,
        marker_positions_world: Mapping[str, FloatArray],
        body_positions: Mapping[str, FloatArray],
        body_quaternions: Mapping[str, FloatArray],
        frame_index: int = 0,
        quaternion_scalar_last: bool = True,
    ) -> ContactSurfaceSet:
        """Compile each marker patch at ``frame_index`` and return an updated set."""
        compiled = self.body_surface_map()
        for patch in self.marker_patches:
            names = patch.marker_names
            pts = np.stack(
                [np.asarray(marker_positions_world[n][frame_index], dtype=np.float64) for n in names],
                axis=0,
            )
            body_t = np.asarray(body_positions[patch.attach_body][frame_index], dtype=np.float64)
            body_q = np.asarray(body_quaternions[patch.attach_body][frame_index], dtype=np.float64)
            compiled[patch.attach_body] = patch.compile(
                marker_positions_world=pts,
                body_translation=body_t,
                body_quaternion_xyzw=body_q,
                quaternion_scalar_last=quaternion_scalar_last,
            )
        return ContactSurfaceSet(
            marker_patches=self.marker_patches,
            body_surfaces=tuple(compiled.items()),
        )

    def world_sample_points(
        self,
        points: ArrayLike,
        marker_names: Sequence[str],
        *,
        body_rotations: Mapping[str, FloatArray] | None = None,
        quaternion_scalar_last: bool = True,
    ) -> FloatArray:
        """Shift marker positions onto nominal surface samples in the capture frame."""
        arr = np.asarray(points, dtype=np.float64)
        if not self.marker_patches:
            return arr
        name_to_patch = self._marker_patch_index()
        if arr.ndim == 2 and arr.shape[1] == 3:
            if len(marker_names) != arr.shape[0]:
                raise ValueError("marker_names length must match points.shape[0].")
            out = arr.copy()
            markers_per_frame = len(self.marker_names)
            flat_trajectory = markers_per_frame > 0 and len(marker_names) > markers_per_frame
            for row, name in enumerate(marker_names):
                patch = name_to_patch[name]
                frame_index = row // markers_per_frame if flat_trajectory else 0
                rot = _rotation_for_marker_patch(
                    patch,
                    frame_index=frame_index,
                    body_rotations=body_rotations,
                    quaternion_scalar_last=quaternion_scalar_last,
                )
                out[row] += patch.world_delta(name, rotation=rot)
            return out
        if arr.ndim == 3 and arr.shape[2] == 3:
            if len(marker_names) != arr.shape[1]:
                raise ValueError("marker_names length must match points.shape[1].")
            out = arr.copy()
            for frame_idx in range(arr.shape[0]):
                for col, name in enumerate(marker_names):
                    patch = name_to_patch[name]
                    rot = _rotation_for_marker_patch(
                        patch,
                        frame_index=frame_idx,
                        body_rotations=body_rotations,
                        quaternion_scalar_last=quaternion_scalar_last,
                    )
                    out[frame_idx, col] += patch.world_delta(name, rotation=rot)
            return out
        raise ValueError("points must have shape (N, K, 3), (K, 3), or flat (M, 3).")


def apply_contact_surface_set(
    points: ArrayLike,
    marker_names: Sequence[str],
    surface_set: ContactSurfaceSet | None,
    *,
    body_rotations: Mapping[str, FloatArray] | None = None,
    quaternion_scalar_last: bool = True,
) -> FloatArray:
    """Apply a :class:`ContactSurfaceSet` marker offsets when configured."""
    if surface_set is None:
        return np.asarray(points, dtype=np.float64)
    return surface_set.world_sample_points(
        points,
        marker_names,
        body_rotations=body_rotations,
        quaternion_scalar_last=quaternion_scalar_last,
    )
