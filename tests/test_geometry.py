from enum import StrEnum

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from contact_detection import (
    AxisConvention,
    BodyFrameTranslation,
    CoordinateAxis,
    PatchCalibration,
    RigidBodyContactModel,
    RigidTransform,
    SemanticAxis,
    Z_UP_AXES,
    apply_contact_model_offsets,
    compile_contact_models,
    fit_patch_frame_from_points,
    marker_names_for_flat_trajectory,
)


class ShoeMarkers(StrEnum):
    HEEL = "heel"
    ARCH = "arch"
    TOE = "toe"


def test_signed_axis_and_semantic_translation() -> None:
    axes = AxisConvention(
        {
            SemanticAxis.RIGHT: -CoordinateAxis.Y,
            SemanticAxis.FORWARD: +CoordinateAxis.X,
            SemanticAxis.UP: +CoordinateAxis.Z,
        }
    )
    model = RigidBodyContactModel(
        body_name="Shoe",
        axis_convention=axes,
        patch_calibrations={
            "sole": PatchCalibration.from_markers(
                (ShoeMarkers.HEEL, ShoeMarkers.ARCH, ShoeMarkers.TOE),
                translation=-0.03 * SemanticAxis.UP,
            )
        },
    )

    np.testing.assert_allclose(model.axis(SemanticAxis.UP), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(model.surface_sample_delta_body("heel"), [0.0, 0.0, -0.03])


def test_patch_calibration_builds_body_local_patch_view() -> None:
    model = RigidBodyContactModel(
        body_name="Shoe",
        axis_convention=Z_UP_AXES,
        patch_calibrations={
            "sole": PatchCalibration(
                marker_translations={
                    ShoeMarkers.HEEL: BodyFrameTranslation([0.0, 0.0, -0.03]),
                    ShoeMarkers.ARCH: BodyFrameTranslation([0.0, 0.0, -0.05]),
                    ShoeMarkers.TOE: BodyFrameTranslation([0.0, 0.0, -0.02]),
                }
            )
        },
    )
    patch = model.patch_calibrations["sole"].build_patch(
        {
            "heel": np.array([-0.1, 0.0, 0.03]),
            "arch": np.array([0.0, 0.0, 0.05]),
            "toe": np.array([0.1, 0.0, 0.02]),
        },
        model,
    )

    world_body = RigidTransform(
        translation=np.array([1.0, 2.0, 0.5]),
        rotation=Rotation.from_euler("z", 0.25),
    )
    view = patch.view(world_body)
    np.testing.assert_allclose(view.contact_point_world[:2], [1.0, 2.0], atol=0.05)
    np.testing.assert_allclose(np.linalg.norm(view.normal_world), 1.0, atol=1e-12)


def test_compile_contact_model_from_world_markers() -> None:
    model = RigidBodyContactModel(
        body_name="Shoe",
        patch_calibrations={
            "sole": PatchCalibration.from_markers(
                (ShoeMarkers.HEEL, ShoeMarkers.ARCH, ShoeMarkers.TOE)
            )
        },
    )
    marker_world = {
        "heel": np.array([0.0, 0.0, 0.0]),
        "arch": np.array([0.1, 0.0, 0.0]),
        "toe": np.array([0.0, 0.1, 0.0]),
    }
    compiled = model.compile(
        marker_positions_world=marker_world,
        body_translation=np.zeros(3),
        body_rotation=Rotation.identity(),
    )

    patch = compiled.patch("sole")
    expected = fit_patch_frame_from_points(np.stack(tuple(marker_world.values())))
    np.testing.assert_allclose(patch.transform_body_patch.translation, expected.translation)
    np.testing.assert_allclose(
        patch.transform_body_patch.rotation.as_matrix(),
        expected.rotation.as_matrix(),
        atol=1e-9,
    )


def test_apply_contact_model_offsets_rotates_body_local_offsets() -> None:
    model = RigidBodyContactModel(
        body_name="Shoe",
        patch_calibrations={
            "sole": PatchCalibration(
                marker_translations={
                    ShoeMarkers.HEEL: BodyFrameTranslation([0.0, 0.0, -0.03]),
                    ShoeMarkers.ARCH: BodyFrameTranslation([0.0, 0.0, -0.05]),
                    ShoeMarkers.TOE: BodyFrameTranslation([0.0, 0.0, -0.02]),
                }
            )
        },
    )
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (1, 1))
    shifted = apply_contact_model_offsets(
        points,
        ["heel", "arch", "toe"],
        (model,),
        body_rotations={"Shoe": quats},
    )

    np.testing.assert_allclose(shifted[:, 2], [0.97, 0.95, 0.98])


def test_compile_contact_models_requires_missing_marker() -> None:
    model = RigidBodyContactModel(
        body_name="Shoe",
        patch_calibrations={
            "sole": PatchCalibration.from_markers(
                (ShoeMarkers.HEEL, ShoeMarkers.ARCH, ShoeMarkers.TOE)
            )
        },
    )

    with pytest.raises(KeyError, match="toe"):
        compile_contact_models(
            (model,),
            marker_positions_world={
                "heel": np.zeros((1, 3)),
                "arch": np.zeros((1, 3)),
            },
            body_positions={"Shoe": np.zeros((1, 3))},
            body_quaternions={"Shoe": np.array([[0.0, 0.0, 0.0, 1.0]])},
        )


def test_marker_names_for_flat_trajectory() -> None:
    assert marker_names_for_flat_trajectory(["a", "b"], n_frames=2) == ["a", "b", "a", "b"]
