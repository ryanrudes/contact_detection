"""Marker-based quiet and contact detection for motion-capture time series.

The package fits support surfaces from quiet marker samples, scores contact
intervals relative to those surfaces, and classifies each foot separately as
air, ground, or skateboard. Configuration uses dataclasses and :class:`enum.StrEnum`
values; see :data:`contact_detection.__all__` for the supported public surface.

Modules:
    quiet: Scalar, vector, and quaternion quiet-interval detection.
    contact: Support-surface fitting and marker contact intervals.
    geometry: Body-to-contact-frame surfaces and marker-anchored patches.
    foot_support: Per-foot air / ground / skateboard classification.
    intervals: Boolean mask ↔ interval conversion and summaries.
    enums: Shared string enums and :func:`normalize_enum`.
    debug: Optional Matplotlib diagnostic plots.
"""

from .contact import (
    ContactDetectionConfig,
    ContactDetectionResult,
    HeightmapSupportModel,
    LocalPercentileHeightmap,
    PlaneSupportModel,
    SupportCandidate,
    SupportCandidateSet,
    SupportDetectionConfig,
    SupportModel,
    bootstrap_support_surface,
    compute_support_relative_features,
    detect_contact_intervals,
    estimate_contact_offset,
    filter_support_candidates,
    find_support_candidates,
    fit_best_support_surface,
    fit_plane_svd,
    fit_support_model_from_candidates,
    score_contact_features,
)
from .intervals import IntervalSummary, mask_from_intervals, summarize_intervals
from .geometry import (
    BodyContactSurface,
    ContactFrameMode,
    ContactFrameSpec,
    ContactSurfaceRegion,
    ContactSurfaceSet,
    InfinitePlaneRegion,
    MarkerAnchoredPatch,
    PointSamplesRegion,
    RectangleExtents,
    RigidTransform,
    SampleHullRegion,
    apply_contact_surface_set,
    fit_plane_frame_from_points,
    marker_names_for_flat_trajectory,
    rotation_matrices_from_quaternions,
)
from .enums import (
    FloorModel,
    QuietSignalType,
    SupportModelType,
    VectorQuietMode,
    normalize_enum,
)
from .foot_support import (
    FootSupportClassification,
    FootSupportConfig,
    FootSupportState,
    classify_foot_support_states,
    intervals_by_state,
)
from .quiet import (
    QuietDetectionConfig,
    QuietDetectionResult,
    clean_mask_by_time,
    compute_quiet_activity_and_spread,
    detect_quiet_intervals,
    intervals_from_mask,
    local_polynomial_derivative,
    quaternion_angular_speed,
    quaternion_local_spread,
    quaternion_standardize_xyzw,
    score_hysteresis_mask,
    time_gaussian_smooth,
    time_window_component_range,
    time_window_range,
    time_window_rms,
    time_window_std,
)

__all__ = [
    "BodyContactSurface",
    "ContactDetectionConfig",
    "ContactDetectionResult",
    "ContactFrameMode",
    "ContactFrameSpec",
    "ContactSurfaceRegion",
    "ContactSurfaceSet",
    "FloorModel",
    "FootSupportClassification",
    "FootSupportConfig",
    "FootSupportState",
    "HeightmapSupportModel",
    "InfinitePlaneRegion",
    "IntervalSummary",
    "LocalPercentileHeightmap",
    "MarkerAnchoredPatch",
    "PlaneSupportModel",
    "PointSamplesRegion",
    "QuietDetectionConfig",
    "QuietDetectionResult",
    "QuietSignalType",
    "RectangleExtents",
    "RigidTransform",
    "SampleHullRegion",
    "SupportCandidate",
    "SupportCandidateSet",
    "SupportDetectionConfig",
    "SupportModel",
    "SupportModelType",
    "VectorQuietMode",
    "apply_contact_surface_set",
    "bootstrap_support_surface",
    "classify_foot_support_states",
    "clean_mask_by_time",
    "compute_quiet_activity_and_spread",
    "compute_support_relative_features",
    "detect_contact_intervals",
    "detect_quiet_intervals",
    "estimate_contact_offset",
    "filter_support_candidates",
    "find_support_candidates",
    "fit_best_support_surface",
    "fit_plane_frame_from_points",
    "fit_plane_svd",
    "fit_support_model_from_candidates",
    "intervals_from_mask",
    "intervals_by_state",
    "local_polynomial_derivative",
    "marker_names_for_flat_trajectory",
    "mask_from_intervals",
    "normalize_enum",
    "quaternion_angular_speed",
    "quaternion_local_spread",
    "quaternion_standardize_xyzw",
    "rotation_matrices_from_quaternions",
    "score_contact_features",
    "score_hysteresis_mask",
    "summarize_intervals",
    "time_gaussian_smooth",
    "time_window_component_range",
    "time_window_range",
    "time_window_rms",
    "time_window_std",
]
