import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from contact_detection import (
    ContactSurfaceSet,
    FootSupportConfig,
    MarkerAnchoredPatch,
    SupportDetectionConfig,
    apply_contact_surface_set,
    bootstrap_support_surface,
    classify_foot_support_states,
    fit_best_support_surface,
    fit_plane_frame_from_points,
    marker_names_for_flat_trajectory,
)
from contact_detection.enums import FloorModel
from contact_detection.geometry import BodyContactSurface, RigidTransform


class RigidTransformTests(unittest.TestCase):
    def test_round_trip_and_outward_normal(self) -> None:
        frame = RigidTransform(
            translation=np.array([0.1, 0.0, -0.02]),
            rotation=Rotation.from_euler("xyz", [0.1, -0.05, 0.2]),
        )
        local = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
        world_body = frame.transform_points(local)
        back = frame.inverse_transform_points(world_body)
        np.testing.assert_allclose(back, local, atol=1e-12)
        normal = frame.outward_normal_body()
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-12)
        np.testing.assert_allclose(normal, frame.rotation.apply([0, 0, 1]), atol=1e-12)

    def test_compose_world(self) -> None:
        body = RigidTransform(
            translation=np.array([1.0, 2.0, 0.5]),
            rotation=Rotation.from_euler("z", 0.3),
        )
        contact = RigidTransform(
            translation=np.array([0.0, 0.0, -0.03]),
            rotation=Rotation.identity(),
        )
        world = contact.compose_world(body.translation, body.rotation)
        pt = world.transform_points(np.array([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(pt[0], [1.0, 2.0, 0.47], atol=1e-12)


class MarkerAnchoredPatchTests(unittest.TestCase):
    def test_body_up_axis_rotates_with_orientation(self) -> None:
        patch = MarkerAnchoredPatch(
            patch_markers=("m1",),
            sample_offsets_body={"m1": (0.0, 0.0, -0.02)},
            attach_body="Shoe",
        )
        points = np.array([[0.0, 0.0, 1.0]])
        upright = np.eye(3)
        tilted = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        up = patch.world_sample_positions(points, body_rotation_matrix=upright)[0]
        side = patch.world_sample_positions(points, body_rotation_matrix=tilted)[0]
        np.testing.assert_allclose(up, [0.0, 0.0, 0.98])
        np.testing.assert_allclose(side, [0.0, 0.02, 1.0], atol=1e-12)

    def test_assembly_flat_trajectory_with_body_rotations(self) -> None:
        patch = MarkerAnchoredPatch(
            patch_markers=("m1", "m2"),
            sample_offsets_body={"m1": (0.0, 0.0, -0.01), "m2": (0.0, 0.0, -0.02)},
            attach_body="Shoe",
        )
        surface_set = ContactSurfaceSet.from_marker_patches(patch)
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))
        out = apply_contact_surface_set(
            points,
            ["m1", "m2"],
            surface_set,
            body_rotations={"Shoe": quats},
        )
        np.testing.assert_allclose(out[0, 2], 0.99)
        np.testing.assert_allclose(out[1, 2], 0.98)

    def test_compile_plane_matches_svd_helper(self) -> None:
        patch = MarkerAnchoredPatch(
            patch_markers=("a", "b", "c"),
            sample_offsets_body={
                "a": (0.0, 0.0, 0.0),
                "b": (0.0, 0.0, 0.0),
                "c": (0.0, 0.0, 0.0),
            },
            attach_body="Shoe",
        )
        marker_world = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ]
        )
        surface = patch.compile(
            marker_positions_world=marker_world,
            body_translation=np.zeros(3),
            body_rotation_matrix=np.eye(3),
        )
        expected = fit_plane_frame_from_points(marker_world, up_axis=2)
        np.testing.assert_allclose(surface.frame.translation, expected.translation, atol=1e-9)
        np.testing.assert_allclose(
            surface.frame.rotation.as_matrix(),
            expected.rotation.as_matrix(),
            atol=1e-9,
        )
        self.assertEqual(surface.attach_body, "Shoe")
        self.assertIsInstance(surface, BodyContactSurface)


class ContactSurfaceApplicationTests(unittest.TestCase):
    def test_flat_trajectory_names(self) -> None:
        names = marker_names_for_flat_trajectory(["a", "b"], n_frames=2)
        self.assertEqual(names, ["a", "b", "a", "b"])

    def test_body_local_offsets_improve_tilted_plane_fit(self) -> None:
        x = np.linspace(-0.5, 0.5, 40)
        y = np.linspace(-0.2, 0.2, 40)
        z_surface = 0.05 + 0.10 * x
        points = np.stack(
            [
                np.column_stack([x, y, z_surface + 0.03]),
                np.column_stack([x, -y, z_surface - 0.02]),
                np.column_stack([x, y * 0.5, z_surface + 0.08]),
            ],
            axis=1,
        )
        raw_plane = fit_best_support_surface(points.reshape(-1, 3), SupportDetectionConfig())
        patch = MarkerAnchoredPatch(
            patch_markers=("heel", "toe", "arch"),
            sample_offsets_body={
                "heel": (0.0, 0.0, -0.03),
                "toe": (0.0, 0.0, 0.02),
                "arch": (0.0, 0.0, -0.08),
            },
            attach_body="Foot",
        )
        surface_set = ContactSurfaceSet.from_marker_patches(patch)
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(x), 1))
        adjusted = apply_contact_surface_set(
            points,
            ["heel", "toe", "arch"],
            surface_set,
            body_rotations={"Foot": quats},
        ).reshape(-1, 3)
        corrected_plane = fit_best_support_surface(adjusted, SupportDetectionConfig())
        surface_samples = np.column_stack([x, y, z_surface])
        raw_err = float(np.median(np.abs(raw_plane.clearance(surface_samples))))
        corrected_err = float(np.median(np.abs(corrected_plane.clearance(surface_samples))))
        self.assertLess(corrected_err, raw_err)
        self.assertLess(corrected_err, 0.005)

    def test_bootstrap_with_marker_patches(self) -> None:
        t = np.linspace(0.0, 2.0, 101)
        x = np.linspace(-0.4, 0.4, len(t))
        z = 0.04 + 0.08 * x
        points = np.stack(
            [
                np.column_stack([x, np.zeros_like(t), z + 0.03]),
                np.column_stack([x, np.zeros_like(t), z - 0.02]),
            ],
            axis=1,
        )
        patch = MarkerAnchoredPatch(
            patch_markers=("heel", "toe"),
            sample_offsets_body={"heel": (0.0, 0.0, -0.03), "toe": (0.0, 0.0, 0.02)},
            attach_body="Foot",
        )
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(t), 1))
        config = SupportDetectionConfig(
            marker_names=("heel", "toe"),
            contact_surface_set=ContactSurfaceSet.from_marker_patches(patch),
            body_rotations={"Foot": quats},
        )
        model, _debug = bootstrap_support_surface(t, points, config=config)
        surface = np.column_stack([x, np.zeros_like(t), z])
        clearance = np.abs(model.clearance(surface))
        self.assertLess(float(np.median(clearance)), 0.01)

    def test_plane_floor_from_offset_markers(self) -> None:
        t = np.linspace(0.0, 2.0, 201)
        x = np.linspace(-0.5, 0.5, len(t))
        z = 0.06 + 0.12 * x
        marker_pos = np.stack(
            [
                np.column_stack([x, -0.1 * np.ones_like(t), z + 0.03]),
                np.column_stack([x, 0.1 * np.ones_like(t), z - 0.02]),
            ],
            axis=1,
        )
        body_names = ["Left_Shoe", "Right_Shoe", "Skateboard"]
        body_pos = np.zeros((len(t), 3, 3), dtype=float)
        body_pos[:, 0, 0] = x
        body_pos[:, 0, 1] = -0.5
        body_pos[:, 0, 2] = z + 0.05
        body_pos[:, 1, 0] = x + 0.2
        body_pos[:, 1, 2] = z + 0.25
        body_pos[:, 2, 0] = x
        body_pos[:, 2, 2] = z + 0.30
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(t), 1))
        body_rotations = {
            "Left_Shoe": quats,
            "Right_Shoe": quats,
            "Skateboard": quats,
        }
        sole = MarkerAnchoredPatch(
            patch_markers=("heel", "arch", "toe"),
            sample_offsets_body={
                "heel": (0.0, 0.0, -0.03),
                "arch": (0.0, 0.0, -0.05),
                "toe": (0.0, 0.0, 0.02),
            },
            attach_body="Left_Shoe",
        )
        marker_pos = np.concatenate(
            [marker_pos, marker_pos[:, :1, :] + np.array([0.0, 0.05, -0.01])],
            axis=1,
        )
        classification = classify_foot_support_states(
            t,
            body_names,
            body_pos,
            config=FootSupportConfig(
                ground_clearance_tolerance=0.04,
                ground_speed_tolerance=1.0,
                contact_surface_set=ContactSurfaceSet.from_marker_patches(sole),
                floor_fit_marker_names=("heel", "arch", "toe"),
            ),
            floor_fit_marker_pos=marker_pos,
            body_rotations=body_rotations,
        )
        self.assertEqual(classification.floor_model, FloorModel.PLANE)
        self.assertIsNotNone(classification.floor_normal)
        mid = len(t) // 2
        query = np.array([0.0, 0.0, z[mid]])
        delta = query - classification.floor_origin
        clearance = float(np.dot(classification.floor_normal, delta))
        self.assertLess(abs(clearance), 0.02)


if __name__ == "__main__":
    unittest.main()
