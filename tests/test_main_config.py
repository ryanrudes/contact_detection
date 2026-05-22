import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from main import build_foot_support_config, load_yaml_config


class MainConfigTests(unittest.TestCase):
    """Tests for YAML-driven CLI configuration."""

    def test_load_yaml_config_and_build_foot_support_config(self):
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
data:
  input: data
  output: outputs
plot:
  show: false
foot_support:
  foot_names: [Left_Shoe, Right_Shoe]
  board_name: Skateboard
  floor_model: plane
  floor_plane_candidate_percentile: 40.0
  ground_speed_tolerance: 0.3
""",
                encoding="utf-8",
            )

            config_data = load_yaml_config(config_path)
            config = build_foot_support_config(config_data)

        self.assertEqual(config.foot_names, ("Left_Shoe", "Right_Shoe"))
        self.assertEqual(config.board_name, "Skateboard")
        self.assertEqual(config.floor_model, "plane")
        self.assertEqual(config.floor_plane_candidate_percentile, 40.0)
        self.assertEqual(config.ground_speed_tolerance, 0.3)

    def test_cli_values_override_yaml_config(self):
        args = argparse.Namespace(
            left_name="L",
            right_name=None,
            board_name="Deck",
            floor_model="height",
            floor_low_percentile=None,
            floor_plane_candidate_percentile=None,
            floor_plane_residual_tolerance=0.04,
        )

        config = build_foot_support_config(
            {
                "foot_support": {
                    "foot_names": ["Left_Shoe", "Right_Shoe"],
                    "board_name": "Skateboard",
                    "floor_model": "plane",
                }
            },
            args,
        )

        self.assertEqual(config.foot_names, ("L", "Right_Shoe"))
        self.assertEqual(config.board_name, "Deck")
        self.assertEqual(config.floor_model, "height")
        self.assertEqual(config.floor_plane_residual_tolerance, 0.04)


if __name__ == "__main__":
    unittest.main()
