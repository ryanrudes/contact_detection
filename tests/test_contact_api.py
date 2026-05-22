import unittest

import numpy as np

from src.contact_detection import (
    ContactDetectionConfig,
    PlaneSupportModel,
    QuietDetectionConfig,
    QuietSignalType,
    VectorQuietMode,
    detect_contact_intervals,
    detect_quiet_intervals,
)


class ContactApiTests(unittest.TestCase):
    """Contract tests for the public contact-detection API."""
    def test_quiet_hovering_above_support_is_not_contact(self):
        t = np.linspace(0.0, 1.0, 101)
        points = np.zeros((len(t), 1, 3))
        points[:, 0, 2] = 0.06
        support_surface = PlaneSupportModel(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3))

        quiet = detect_quiet_intervals(
            t,
            points[:, 0, :],
            config=QuietDetectionConfig(
                signal_type=QuietSignalType.VECTOR_POSITION,
                vector_mode=VectorQuietMode.EUCLIDEAN,
            ),
        )
        result = detect_contact_intervals(t, points, supports=support_surface)

        self.assertTrue(quiet.mask.all())
        self.assertFalse(result.mask.any())
        self.assertEqual(result.intervals, [])

    def test_moving_support_mode_is_explicitly_unimplemented(self):
        t = np.linspace(0.0, 1.0, 101)
        points = np.zeros((len(t), 1, 3))

        with self.assertRaisesRegex(NotImplementedError, "planned but not implemented"):
            detect_contact_intervals(
                t,
                points,
                config=ContactDetectionConfig(moving_support_mode=True),
            )

    def test_multi_keypoint_point_masks_and_frame_mask(self):
        t = np.linspace(0.0, 1.0, 101)
        points = np.zeros((len(t), 2, 3))
        points[:, 1, 2] = 0.08
        support_surface = PlaneSupportModel(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3))

        result = detect_contact_intervals(t, points, supports=support_surface)

        self.assertEqual(result.mask.shape, t.shape)
        self.assertEqual(result.point_mask.shape, points.shape[:2])
        self.assertTrue(result.point_mask[:, 0].all())
        self.assertFalse(result.point_mask[:, 1].any())
        self.assertTrue(result.mask.all())


if __name__ == "__main__":
    unittest.main()
