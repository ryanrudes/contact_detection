import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from contact_detection import (
    FootSupportConfig,
    FootSupportState,
    classify_foot_support_states,
    load_unified_npz,
)


class FootSupportClassificationTests(unittest.TestCase):
    """Tests for per-foot air/ground/skateboard classification."""
    def test_classifies_ground_skateboard_and_air_per_foot(self):
        t = np.linspace(0.0, 3.0, 301)
        body_names = ["Left_Shoe", "Right_Shoe", "Skateboard"]
        body_pos = np.zeros((len(t), 3, 3), dtype=float)

        board = body_pos[:, 2, :]
        board[:, 0] = 0.1 * t
        board[:, 1] = -1.0
        board[:, 2] = 0.11

        left = body_pos[:, 0, :]
        left[:, 0] = -0.2
        left[:, 1] = -0.75
        left[:, 2] = 0.065

        right = body_pos[:, 1, :]
        right[:, 0] = board[:, 0] + 0.1
        right[:, 1] = board[:, 1] + 0.05
        right[:, 2] = board[:, 2] + 0.055

        air = t > 2.0
        right[air, 2] = 0.35

        classification = classify_foot_support_states(t, body_names, body_pos)

        left_states = classification.states["Left_Shoe"]
        right_states = classification.states["Right_Shoe"]

        self.assertTrue(np.all(left_states[t < 1.0] == FootSupportState.GROUND))
        self.assertTrue(np.all(right_states[(t > 0.5) & (t < 1.5)] == FootSupportState.SKATEBOARD))
        self.assertTrue(np.all(right_states[t > 2.2] == FootSupportState.AIR))
        self.assertIn("skateboard", classification.intervals["Right_Shoe"])

    def test_load_unified_npz_uses_vicon_schema_and_valid_mask(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unified.npz"
            np.savez(
                path,
                t=np.array([10.0, 10.1, 10.2]),
                vicon__body_names=np.array(["Left_Shoe", "Right_Shoe", "Skateboard"], dtype=object),
                vicon__body_pos=np.zeros((3, 3, 3), dtype=float),
                valid=np.array([True, False, True]),
            )

            t, body_names, body_pos, _ = load_unified_npz(path)

        np.testing.assert_allclose(t, [0.0, 0.2])
        self.assertEqual(body_names, ["Left_Shoe", "Right_Shoe", "Skateboard"])
        self.assertEqual(body_pos.shape, (2, 3, 3))

    def test_floor_plane_model_handles_tilted_floor(self):
        t = np.linspace(0.0, 3.0, 301)
        body_names = ["Left_Shoe", "Right_Shoe", "Skateboard"]
        body_pos = np.zeros((len(t), 3, 3), dtype=float)

        x = np.linspace(-1.0, 1.0, len(t))
        floor_z = 0.06 + 0.04 * x

        body_pos[:, 0, 0] = x
        body_pos[:, 0, 1] = -0.5
        body_pos[:, 0, 2] = floor_z

        body_pos[:, 1, 0] = x
        body_pos[:, 1, 1] = -0.25
        body_pos[:, 1, 2] = floor_z

        body_pos[:, 2, 0] = 0.0
        body_pos[:, 2, 1] = -1.0
        body_pos[:, 2, 2] = 0.14

        classification = classify_foot_support_states(
            t,
            body_names,
            body_pos,
            config=FootSupportConfig(
                floor_model="plane",
                ground_speed_tolerance=1.0,
                board_horizontal_tolerance=0.05,
            ),
        )

        self.assertEqual(classification.floor_model, "plane")
        self.assertIsNotNone(classification.floor_normal)
        self.assertIsNotNone(classification.floor_origin)
        self.assertTrue(np.all(classification.states["Left_Shoe"] == FootSupportState.GROUND))
        np.testing.assert_allclose(
            classification.features["Left_Shoe"]["floor_height_at_foot"],
            floor_z,
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
