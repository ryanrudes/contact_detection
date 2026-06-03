"""Convenience re-exports for support-surface model types and fitters.

This module mirrors the support-surface API from :mod:`contact_detection.contact`
for callers that only need geometry fitting without the full contact pipeline.
"""

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
