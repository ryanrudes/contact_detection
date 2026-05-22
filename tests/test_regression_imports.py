import unittest
import sys
from pathlib import Path

import numpy as np


class ImportCompatibilityTests(unittest.TestCase):
    """Verify legacy and public import paths remain available."""
    def test_legacy_src_contact_detection_import(self):
        from src.contact_detection import ContactDetectionConfig, detect_contact_intervals

        self.assertTrue(callable(detect_contact_intervals))
        self.assertIsNotNone(ContactDetectionConfig())

    def test_legacy_src_silence_detection_import(self):
        from src.silence_detection import QuietDetectionConfig, detect_quiet_intervals

        self.assertTrue(callable(detect_quiet_intervals))
        self.assertIsNotNone(QuietDetectionConfig())

    def test_public_contact_detection_import_with_src_on_path(self):
        src_path = str(Path(__file__).resolve().parents[1] / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        import contact_detection

        self.assertTrue(callable(contact_detection.detect_quiet_intervals))
        self.assertTrue(callable(contact_detection.detect_contact_intervals))
        self.assertIsNotNone(contact_detection.ContactDetectionConfig())


class QuietRegressionTests(unittest.TestCase):
    """Regression tests for quiet-detection result contracts."""
    def test_detect_quiet_intervals_tuple_unpacking_contract(self):
        from src.silence_detection import QuietSignalType, detect_quiet_intervals

        t = np.linspace(0.0, 1.0, 101)
        x = np.zeros_like(t)

        intervals, mask, debug = detect_quiet_intervals(
            t,
            x,
            signal_type=QuietSignalType.POSITION_COMPONENT,
        )

        self.assertEqual(intervals, [(0.0, 1.0)])
        self.assertEqual(mask.shape, t.shape)
        self.assertTrue(mask.all())
        self.assertIn("activity", debug)
        self.assertIn("spread", debug)

    def test_detect_quiet_intervals_result_object_contract(self):
        from src.silence_detection import QuietDetectionConfig, QuietSignalType, detect_quiet_intervals

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
        intervals, mask, debug = result
        self.assertEqual(intervals, result.intervals)
        self.assertIs(mask, result.mask)
        self.assertIs(debug, result.debug)


class ContactRegressionTests(unittest.TestCase):
    """Regression tests for contact-detection result contracts."""
    def test_detect_contact_intervals_frame_mask_contract(self):
        from src.contact_detection import PlaneSupportModel, detect_contact_intervals

        t = np.linspace(0.0, 1.0, 101)
        points = np.zeros((len(t), 1, 3))
        support_surface = PlaneSupportModel(
            normal=np.array([0.0, 0.0, 1.0]),
            origin=np.zeros(3),
        )

        result = detect_contact_intervals(t, points, supports=support_surface)

        self.assertEqual(result.mask.shape, t.shape)
        self.assertEqual(result.scores.shape, t.shape)
        self.assertTrue(result.mask.all())
        self.assertTrue(result.intervals)


if __name__ == "__main__":
    unittest.main()
