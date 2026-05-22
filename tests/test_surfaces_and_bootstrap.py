import unittest

import numpy as np

from src.contact_detection import (
    HeightmapSupportModel,
    LocalPercentileHeightmap,
    PlaneSupportModel,
    QuietDetectionConfig,
    SupportDetectionConfig,
    VectorQuietMode,
    bootstrap_support_surface,
    filter_support_candidates,
    find_support_candidates,
    fit_best_support_surface,
)


class SurfaceModelTests(unittest.TestCase):
    """Tests for plane and heightmap support model fitting."""
    def test_tilted_plane_with_outliers(self):
        x, y = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 8))
        z = 0.2 + 0.08 * x - 0.04 * y
        inliers = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        outliers = np.array([[0.0, 0.0, 1.0], [0.5, -0.5, -0.4], [-0.7, 0.6, 0.8]])
        points = np.vstack([inliers, outliers])

        model = PlaneSupportModel.fit(
            points,
            SupportDetectionConfig(plane_residual_tolerance=0.015, ransac_iterations=256),
        )

        self.assertGreater(float(np.mean(model.inlier_mask)), 0.80)
        self.assertLess(float(np.median(np.abs(model.clearance(inliers)))), 0.005)
        self.assertGreater(model.normal[2], 0.0)

    def test_auto_surface_uses_heightmap_for_non_coplanar_candidates(self):
        x, y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 4))
        z = np.where(x < 0.5, 0.0, 0.25)
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        model = fit_best_support_surface(
            points,
            SupportDetectionConfig(
                plane_residual_tolerance=0.02,
                coplanarity_ratio_threshold=0.90,
                heightmap_cell_size=0.25,
            ),
        )

        self.assertIsInstance(model, HeightmapSupportModel)
        clearance = model.clearance(points)
        self.assertLess(float(np.max(np.abs(clearance))), 1e-12)

    def test_local_percentile_heightmap_positive_clearance(self):
        x, y = np.meshgrid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 6))
        support_points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        query_points = support_points.copy()
        query_points[:, 2] = 0.10

        model = LocalPercentileHeightmap.fit(
            support_points,
            SupportDetectionConfig(model_type="local_heightmap", local_heightmap_radius=0.30),
        )

        np.testing.assert_allclose(model.clearance(query_points), 0.10, atol=1e-12)


class SupportBootstrapTests(unittest.TestCase):
    """Tests for bootstrapping support surfaces from quiet marker samples."""
    def test_support_candidates_from_planted_intervals(self):
        t = np.linspace(0.0, 2.0, 201)
        points = np.zeros((len(t), 1, 3))
        swing = (t > 0.8) & (t < 1.2)
        points[swing, 0, 2] = 0.15 * np.sin((t[swing] - 0.8) / 0.4 * np.pi)

        quiet_config = QuietDetectionConfig(
            vector_mode=VectorQuietMode.EUCLIDEAN,
            activity_on_eps=0.02,
            activity_off_eps=0.04,
            spread_on_eps=0.02,
            spread_off_eps=0.04,
        )

        candidates = find_support_candidates(t, points, quiet_config=quiet_config)
        filtered = filter_support_candidates(candidates, up_axis=2)

        self.assertGreaterEqual(len(candidates.candidates), 2)
        self.assertGreaterEqual(len(filtered.candidates), 1)
        self.assertEqual(filtered.points.shape[1], 3)

    def test_bootstrap_support_surface_returns_plane(self):
        t = np.linspace(0.0, 2.0, 201)
        points = np.zeros((len(t), 2, 3))
        points[:, 1, 0] = 0.2
        swing = (t > 0.8) & (t < 1.2)
        points[swing, :, 2] = 0.10

        model, debug = bootstrap_support_surface(t, points)

        self.assertIsInstance(model, PlaneSupportModel)
        self.assertIn("support_candidates", debug)
        self.assertLess(float(np.max(np.abs(model.clearance(points[~swing].reshape(-1, 3))))), 1e-12)


if __name__ == "__main__":
    unittest.main()
