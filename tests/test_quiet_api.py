import unittest

import numpy as np

from contact_detection import QuietDetectionConfig, QuietSignalType, VectorQuietMode
from contact_detection.quiet import detect_quiet_intervals, time_window_range, time_window_rms


class QuietApiTests(unittest.TestCase):
    """Contract tests for the public quiet-detection API."""
    def test_constant_scalar_signal_is_quiet(self):
        t = np.linspace(0.0, 1.0, 101)
        x = np.full_like(t, 2.0)

        result = detect_quiet_intervals(
            t,
            x,
            config=QuietDetectionConfig(signal_type=QuietSignalType.SCALAR),
        )

        self.assertEqual(result.intervals, [(0.0, 1.0)])
        self.assertTrue(result.mask.all())

    def test_moving_circle_with_constant_norm_is_not_quiet(self):
        t = np.linspace(0.0, 1.0, 201)
        angle = 2.0 * np.pi * t
        points = np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(t)])

        result = detect_quiet_intervals(
            t,
            points,
            config=QuietDetectionConfig(
                signal_type=QuietSignalType.VECTOR_POSITION,
                vector_mode=VectorQuietMode.NORM,
                activity_on_eps=0.10,
                activity_off_eps=0.20,
                spread_on_eps=0.05,
                spread_off_eps=0.10,
            ),
        )

        self.assertEqual(result.intervals, [])
        self.assertFalse(result.mask.any())
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)

    def test_time_window_helpers_handle_irregular_timestamps(self):
        t = np.array([0.0, 0.01, 0.04, 0.11, 0.20])
        x = np.array([0.0, 1.0, 1.0, 1.0, 0.0])

        rms = time_window_rms(x, t, window_time=0.08)
        local_range = time_window_range(x, t, window_time=0.08)

        self.assertEqual(rms.shape, x.shape)
        self.assertEqual(local_range.shape, x.shape)
        self.assertGreater(rms[2], 0.0)
        self.assertGreaterEqual(local_range[1], 1.0)

    def test_detect_quiet_intervals_returns_result_object(self):
        t = np.linspace(0.0, 1.0, 101)
        x = np.zeros_like(t)

        result = detect_quiet_intervals(
            t,
            x,
            config=QuietDetectionConfig(signal_type=QuietSignalType.POSITION_COMPONENT),
        )

        self.assertEqual(result.intervals, [(0.0, 1.0)])
        self.assertEqual(result.mask.shape, t.shape)
        self.assertEqual(result.activity.shape, t.shape)
        self.assertEqual(result.spread.shape, t.shape)
        self.assertIn("activity", result.debug)
        self.assertIn("spread", result.debug)


if __name__ == "__main__":
    unittest.main()
