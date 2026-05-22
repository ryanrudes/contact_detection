"""Convenience re-exports for support-surface model types and fitters."""

from .contact import (
    HeightmapSupportModel,
    LocalPercentileHeightmap,
    PlaneSupportModel,
    SupportDetectionConfig,
    SupportModel,
    fit_best_support_surface,
    fit_plane_svd,
)

__all__ = [
    "HeightmapSupportModel",
    "LocalPercentileHeightmap",
    "PlaneSupportModel",
    "SupportDetectionConfig",
    "SupportModel",
    "fit_best_support_surface",
    "fit_plane_svd",
]
