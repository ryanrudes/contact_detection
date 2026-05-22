from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    from contact_detection import FootSupportConfig, classify_foot_support_states, load_unified_npz
    from contact_detection.debug import plot_foot_support_states
except ImportError:  # pragma: no cover - supports running from repo root without install
    from src.contact_detection import FootSupportConfig, classify_foot_support_states, load_unified_npz
    from src.contact_detection.debug import plot_foot_support_states

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for generating per-foot support-state plots."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config_data = load_yaml_config(Path(args.config))

    input_path = Path(_resolve_config_value(args.input, config_data, "data", "input", "data"))
    output_value = _resolve_config_value(args.output, config_data, "data", "output", "outputs")
    output_arg = Path(output_value) if output_value is not None else None
    show_plot = bool(_resolve_config_value(args.show, config_data, "plot", "show", False))

    input_paths = _find_unified_inputs(input_path)
    config = build_foot_support_config(config_data, args)

    matplotlib_cache = Path(tempfile.gettempdir()) / "event_detection_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    for input_path in input_paths:
        output_path = _output_path_for(input_path, input_paths, output_arg)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t, body_names, body_pos, _ = load_unified_npz(input_path)
        classification = classify_foot_support_states(t, body_names, body_pos, config=config)

        plot_foot_support_states(
            classification,
            output_path=output_path,
            title=f"{input_path.parent.name}: foot support states",
        )

        print(f"\n{input_path}")
        print(f"Wrote {output_path}")
        print(f"Floor model: {classification.floor_model}")
        print(f"Estimated floor reference height: {classification.floor_height:.4f} m")
        if classification.floor_normal is not None and classification.floor_origin is not None:
            print(f"Floor plane normal: {_format_vector(classification.floor_normal)}")
            print(f"Floor plane origin: {_format_vector(classification.floor_origin)}")
        for foot_name, offset in classification.board_contact_offsets.items():
            print(f"{foot_name} board contact offset: {offset:.4f} m")
            for state_name in ("ground", "skateboard", "air"):
                intervals = classification.intervals[foot_name][state_name]
                print(f"  {state_name}: {_format_intervals(intervals)}")

    if show_plot:
        import matplotlib.pyplot as plt

        plt.show()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for config-driven trial plotting."""

    parser = argparse.ArgumentParser(
        description="Plot per-foot support state annotations for unified Vicon/video NPZ data."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config file path. Defaults to configs/config.yaml.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a unified.npz file or a directory containing trial */unified.npz files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path for one input file, or output directory for multiple inputs. "
            "Defaults to outputs/."
        ),
    )
    parser.add_argument("--left-name", default=None)
    parser.add_argument("--right-name", default=None)
    parser.add_argument("--board-name", default=None)
    parser.add_argument(
        "--floor-model",
        choices=("height", "plane"),
        default=None,
        help="Use one scalar floor height or fit a robust floor plane from foot positions.",
    )
    parser.add_argument(
        "--floor-low-percentile",
        type=float,
        default=None,
        help="Low foot-height percentile used by --floor-model height.",
    )
    parser.add_argument(
        "--floor-plane-residual-tolerance",
        type=float,
        default=None,
        help="RANSAC inlier tolerance in meters for --floor-model plane.",
    )
    parser.add_argument(
        "--floor-plane-candidate-percentile",
        type=float,
        default=None,
        help="Foot-height percentile used to select lower-envelope candidates for plane fitting.",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=None,
        help="Show the plot interactively after saving.",
    )
    show_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable interactive plot display, overriding the YAML config.",
    )
    return parser


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML runtime config file.

    Parameters
    ----------
    path:
        YAML file path. Empty files are treated as empty mappings.

    Returns
    -------
    dict[str, Any]
        Parsed top-level config mapping.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(config).__name__}.")
    _validate_top_level_config(config)
    return config


def build_foot_support_config(
    config_data: Mapping[str, Any],
    args: argparse.Namespace | None = None,
) -> FootSupportConfig:
    """Build a :class:`FootSupportConfig` from YAML data and CLI overrides."""

    section = _mapping_section(config_data, "foot_support")
    field_names = {field.name for field in fields(FootSupportConfig)}
    unknown_keys = sorted(set(section) - field_names)
    if unknown_keys:
        raise ValueError(f"Unknown foot_support config keys: {unknown_keys}")

    values = {key: value for key, value in section.items() if key in field_names}

    defaults = FootSupportConfig()
    if "foot_names" in values:
        values["foot_names"] = _coerce_foot_names(values["foot_names"])

    if args is not None:
        current_foot_names = values.get("foot_names", defaults.foot_names)
        left_name = args.left_name if args.left_name is not None else current_foot_names[0]
        right_name = args.right_name if args.right_name is not None else current_foot_names[1]
        values["foot_names"] = (left_name, right_name)

        for cli_name, config_name in (
            ("board_name", "board_name"),
            ("floor_model", "floor_model"),
            ("floor_low_percentile", "floor_low_percentile"),
            ("floor_plane_candidate_percentile", "floor_plane_candidate_percentile"),
            ("floor_plane_residual_tolerance", "floor_plane_residual_tolerance"),
        ):
            cli_value = getattr(args, cli_name, None)
            if cli_value is not None:
                values[config_name] = cli_value

    if values.get("floor_model", defaults.floor_model) not in ("height", "plane"):
        raise ValueError("foot_support.floor_model must be 'height' or 'plane'.")

    return FootSupportConfig(**values)


def _resolve_config_value(
    cli_value: Any,
    config_data: Mapping[str, Any],
    section_name: str,
    key: str,
    default: Any,
) -> Any:
    """Resolve one setting using CLI override, YAML section, top-level key, then default."""

    if cli_value is not None:
        return cli_value
    section = _mapping_section(config_data, section_name)
    if key in section:
        return section[key]
    if key in config_data:
        return config_data[key]
    return default


def _validate_top_level_config(config_data: Mapping[str, Any]) -> None:
    """Reject misspelled top-level YAML sections and scalar keys."""

    allowed = {"data", "plot", "foot_support", "input", "output", "show"}
    unknown_keys = sorted(set(config_data) - allowed)
    if unknown_keys:
        raise ValueError(f"Unknown top-level config keys: {unknown_keys}")


def _mapping_section(config_data: Mapping[str, Any], section_name: str) -> dict[str, Any]:
    """Return a named config section as a plain mapping."""

    section = config_data.get(section_name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section {section_name!r} must be a mapping.")
    return dict(section)


def _coerce_foot_names(value: Any) -> tuple[str, str]:
    """Validate and normalize the two configured foot body names."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("foot_support.foot_names must be a two-item list of strings.")
    names = tuple(value)
    if len(names) != 2 or not all(isinstance(name, str) for name in names):
        raise ValueError("foot_support.foot_names must be a two-item list of strings.")
    return names


def _find_unified_inputs(input_path: Path) -> list[Path]:
    """Return one or more unified NPZ paths from a file or directory input."""

    if input_path.is_dir():
        paths = sorted(input_path.rglob("unified.npz"))
    elif input_path.name == "unified.npz" and input_path.is_file():
        paths = [input_path]
    elif input_path.is_file():
        raise ValueError(f"Expected a unified.npz file, got {input_path}")
    else:
        raise FileNotFoundError(input_path)

    if not paths:
        raise FileNotFoundError(f"No unified.npz files found under {input_path}")
    return paths


def _output_path_for(input_path: Path, input_paths: list[Path], output_arg: Path | None) -> Path:
    """Resolve the output PNG path for a single input trial."""

    default_name = f"{input_path.parent.name}_foot_support_states.png"
    if output_arg is None:
        return Path("outputs") / default_name
    if len(input_paths) == 1 and output_arg.suffix.lower() == ".png":
        return output_arg
    return output_arg / default_name


def _format_intervals(intervals: list[tuple[float, float]]) -> str:
    """Format state intervals for concise terminal output."""

    if not intervals:
        return "<none>"
    return ", ".join(f"{start:.3f}-{end:.3f}" for start, end in intervals)


def _format_vector(values: Iterable[float]) -> str:
    """Format a short numeric vector for terminal output."""

    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


if __name__ == "__main__":
    main()
