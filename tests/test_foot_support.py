import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from contact_detection import (
    FootSupportConfig,
    FootSupportState,
    classify_foot_support_states,
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
        left[:, 1] = -0.5
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

    def test_sync_clip_export_respects_valid_mask(self):
        try:
            from motion_sync.synced_dataset import SyncClip
        except ImportError:
            self.skipTest("install twofoot retargeting (with pydantic) to run this test")

        with TemporaryDirectory() as tmpdir:
            demo_dir = Path(tmpdir) / "demo"
            SyncClip(
                time_s=np.array([10.0, 10.1, 10.2]),
                vicon={
                    "body_names": ("Left_Shoe", "Right_Shoe", "Skateboard"),
                    "body_positions": np.zeros((3, 3, 3), dtype=float),
                },
                video={
                    "joints": np.zeros((3, 2, 3)),
                    "transl": np.zeros((3, 3)),
                    "global_orient": np.zeros((3, 3)),
                    "body_pose": np.zeros((3, 63)),
                    "betas": np.zeros((3, 10)),
                },
                metadata={"lag_s": 0.0},
                valid=np.array([True, False, True]),
            ).save(demo_dir)

            clip = SyncClip.load(demo_dir)
            t, body_names, body_pos = clip.export_vicon_bodies()

        np.testing.assert_allclose(t, [0.0, 0.2])
        self.assertEqual(list(body_names), ["Left_Shoe", "Right_Shoe", "Skateboard"])
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

        from contact_detection.geometry import BodyFrameTranslation, PatchCalibration, RigidBodyContactModel

        heel = np.column_stack([x, -0.5 * np.ones_like(t), floor_z])
        toe = np.column_stack([x, -0.45 * np.ones_like(t), floor_z])
        arch = np.column_stack([x, -0.48 * np.ones_like(t), floor_z])
        marker_pos = np.stack([heel, arch, toe], axis=1)
        sole = RigidBodyContactModel(
            body_name="Left_Shoe",
            patch_calibrations={
                "sole": PatchCalibration(
                    marker_translations={
                        "heel": BodyFrameTranslation([0.0, 0.0, 0.0]),
                        "arch": BodyFrameTranslation([0.0, 0.0, 0.0]),
                        "toe": BodyFrameTranslation([0.0, 0.0, 0.0]),
                    }
                )
            },
        )
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(t), 1))
        classification = classify_foot_support_states(
            t,
            body_names,
            body_pos,
            config=FootSupportConfig(
                ground_speed_tolerance=1.0,
                board_horizontal_tolerance=0.05,
                contact_models=(sole,),
                floor_fit_marker_names=("heel", "arch", "toe"),
            ),
            floor_fit_marker_pos=marker_pos,
            body_rotations={"Left_Shoe": quats, "Right_Shoe": quats},
        )

        self.assertEqual(classification.floor_model, "plane")
        self.assertIsNotNone(classification.floor_normal)
        self.assertIsNotNone(classification.floor_origin)
        self.assertTrue(np.all(classification.states["Left_Shoe"] == FootSupportState.GROUND))
        np.testing.assert_allclose(
            classification.features["Left_Shoe"]["floor_height_at_sole"],
            floor_z,
            atol=0.02,
        )


if __name__ == "__main__":
    unittest.main()
