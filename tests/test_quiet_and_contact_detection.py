import unittest

import numpy as np

from contact_detection import (
    HeightmapSupportModel,
    PlaneSupportModel,
    QuietSignalType,
    detect_contact_intervals,
)
from contact_detection.quiet import (
    compute_quiet_activity_and_spread,
    local_polynomial_derivative,
    time_gaussian_smooth,
)


class QuietDetectionTests(unittest.TestCase):
    """Tests for quiet-detection signal processing utilities."""
    def test_time_gaussian_smooth_handles_irregular_timestamps(self):
        t = np.array([0.0, 0.01, 0.04, 0.11, 0.20])
        x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

        smoothed = time_gaussian_smooth(x, t, sigma_time=0.04)

        self.assertEqual(smoothed.shape, x.shape)
        self.assertLess(smoothed[2], 1.0)
        self.assertGreater(smoothed[2], smoothed[0])

    def test_local_polynomial_derivative_handles_irregular_timestamps(self):
        t = np.array([0.0, 0.02, 0.05, 0.11, 0.18, 0.30])
        x = 2.0 + 3.0 * t

        derivative = local_polynomial_derivative(x, t, window_time=0.20, degree=1)

        np.testing.assert_allclose(derivative[1:-1], 3.0, atol=1e-8)

    def test_quaternion_sign_flips_stay_quiet(self):
        t = np.linspace(0.0, 1.0, 101)
        q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(t), 1))
        q[::2] *= -1.0

        activity, spread, _ = compute_quiet_activity_and_spread(
            t,
            q,
            signal_type=QuietSignalType.QUATERNION,
            quiet_window_time=0.10,
            spread_window_time=0.10,
        )

        self.assertLess(float(np.max(activity)), 1e-8)
        self.assertLess(float(np.max(spread)), 1e-8)

class ContactDetectionTests(unittest.TestCase):
    """Tests for support-surface fitting and contact interval detection."""
    def test_flat_floor_contact_and_swing(self):
        t = np.linspace(0.0, 2.0, 201)
        points = np.zeros((len(t), 1, 3))
        swing = (t > 0.8) & (t < 1.2)
        points[swing, 0, 0] = (t[swing] - 0.8) * 0.5
        points[swing, 0, 2] = 0.10 * np.sin((t[swing] - 0.8) / 0.4 * np.pi)

        result = detect_contact_intervals(t, points)

        self.assertIsInstance(result.support_models[0], PlaneSupportModel)
        self.assertTrue(result.mask[t < 0.5].all())
        self.assertFalse(result.mask[(t > 0.9) & (t < 1.1)].any())
        self.assertTrue(result.mask[t > 1.5].all())

    def test_ramp_uses_plane_support(self):
        t = np.linspace(0.0, 2.0, 201)
        points = np.zeros((len(t), 1, 3))
        points[:, 0, 0] = 0.2
        points[:, 0, 2] = 0.02
        swing = (t > 0.8) & (t < 1.2)
        points[swing, 0, 0] = np.linspace(0.2, 0.8, np.sum(swing))
        points[swing, 0, 2] = 0.10
        points[t >= 1.2, 0, 0] = 0.8
        points[t >= 1.2, 0, 2] = 0.08

        result = detect_contact_intervals(t, points)

        self.assertIsInstance(result.support_models[0], PlaneSupportModel)
        self.assertGreater(result.mask.mean(), 0.55)

    def test_step_like_support_falls_back_to_heightmap(self):
        t = np.linspace(0.0, 3.0, 301)
        points = np.zeros((len(t), 2, 3))
        points[:, :, 1] = np.array([-0.1, 0.1])

        for i, ti in enumerate(t):
            if ti < 0.8:
                x, z = 0.0, 0.0
            elif ti < 1.1:
                x, z = (ti - 0.8) / 0.3 * 0.5, 0.10
            elif ti < 1.9:
                x, z = 0.5, 0.20
            elif ti < 2.2:
                x, z = 0.5 + (ti - 1.9) / 0.3 * 0.5, 0.10
            else:
                x, z = 1.0, 0.0
            points[i, :, 0] = x
            points[i, :, 2] = z

        result = detect_contact_intervals(t, points)

        self.assertIsInstance(result.support_models[0], HeightmapSupportModel)
        self.assertTrue(result.mask[t < 0.5].all())
        self.assertTrue(result.mask[(t > 1.3) & (t < 1.7)].all())
        self.assertTrue(result.mask[t > 2.5].all())

    def test_hovering_above_known_floor_is_not_contact(self):
        t = np.linspace(0.0, 1.0, 101)
        points = np.zeros((len(t), 1, 3))
        points[:, 0, 2] = 0.06
        floor = PlaneSupportModel(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3))

        result = detect_contact_intervals(t, points, supports=floor)

        self.assertEqual(result.intervals, [])
        self.assertFalse(result.mask.any())
        self.assertLess(float(np.max(result.scores)), 0.60)

    def test_tangential_slip_is_penalized(self):
        t = np.linspace(0.0, 1.0, 101)
        points = np.zeros((len(t), 1, 3))
        points[:, 0, 0] = 0.30 * t
        floor = PlaneSupportModel(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3))

        result = detect_contact_intervals(t, points, supports=floor)

        self.assertEqual(result.intervals, [])
        self.assertFalse(result.mask.any())
        self.assertLess(float(np.max(result.scores)), 0.40)


if __name__ == "__main__":
    unittest.main()
