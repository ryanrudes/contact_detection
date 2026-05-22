from __future__ import annotations

from os import PathLike
from typing import Any

from numpy.typing import ArrayLike

from .contact import ContactDetectionResult, SupportCandidateSet, SupportModel
from .foot_support import FootSupportClassification
from .quiet import QuietDetectionResult


def plot_quiet_detection(
    t: ArrayLike,
    result: QuietDetectionResult,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot quiet-detection activity, spread, mask, and detected intervals."""

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(t, result.activity, label="activity")
    axs[0].set_ylabel("activity")
    axs[0].legend()

    axs[1].plot(t, result.spread, label="spread")
    axs[1].set_ylabel("spread")
    axs[1].legend()

    axs[2].step(t, result.mask.astype(float), where="post", label="quiet mask")
    axs[2].set_ylabel("mask")
    axs[2].set_xlabel("time (s)")
    axs[2].legend()

    for ax in axs:
        for start, end in result.intervals:
            ax.axvspan(start, end, color="tab:green", alpha=0.12)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axs


def plot_contact_detection(
    t: ArrayLike,
    result: ContactDetectionResult,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot contact scores, support-relative features, and contact intervals."""

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(t, result.scores, label="frame score")
    axs[0].set_ylabel("score")
    axs[0].legend()

    clearance = result.features.get("clearance")
    if clearance is not None:
        axs[1].plot(t, clearance)
    axs[1].set_ylabel("clearance")

    tangential_speed = result.features.get("tangential_speed")
    if tangential_speed is not None:
        axs[2].plot(t, tangential_speed)
    axs[2].set_ylabel("tan speed")

    axs[3].step(t, result.mask.astype(float), where="post", label="contact mask")
    axs[3].set_ylabel("mask")
    axs[3].set_xlabel("time (s)")
    axs[3].legend()

    for ax in axs:
        for start, end in result.intervals:
            ax.axvspan(start, end, color="tab:red", alpha=0.12)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axs


def plot_support_candidates_3d(
    candidates: SupportCandidateSet,
    support_model: SupportModel | None = None,
) -> tuple[Any, Any]:
    """Plot support candidate points in 3D with an optional model title."""

    import matplotlib.pyplot as plt

    points = candidates.points
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label="support candidates")
    if support_model is not None:
        ax.set_title(f"support model: {support_model.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    return fig, ax


def plot_foot_support_states(
    classification: FootSupportClassification,
    output_path: str | PathLike[str] | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot per-foot air, ground, and skateboard states shaded through time."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from .foot_support import FootSupportState, STATE_COLORS, STATE_LABELS

    foot_names = list(classification.states.keys())
    fig, axs = plt.subplots(len(foot_names), 1, figsize=(13, 3.5 * len(foot_names)), sharex=True)
    if len(foot_names) == 1:
        axs = [axs]

    legend_handles = [
        Patch(
            facecolor=STATE_COLORS[state],
            edgecolor="none",
            alpha=0.22,
            label=STATE_LABELS[state],
        )
        for state in FootSupportState
    ]

    for ax, foot_name in zip(axs, foot_names):
        features = classification.features[foot_name]
        t = classification.t

        for support_state in FootSupportState:
            for start, end in classification.intervals[foot_name][STATE_LABELS[support_state]]:
                ax.axvspan(
                    start,
                    end,
                    color=STATE_COLORS[support_state],
                    alpha=0.22,
                    linewidth=0,
                )

        ax.plot(t, features["foot_height"], color="black", linewidth=1.0, label=f"{foot_name} height")
        ax.plot(t, features["board_height"], color="#4f7fcf", linewidth=0.9, linestyle="--", label="board height")
        floor_height_at_foot = features.get("floor_height_at_foot")
        if floor_height_at_foot is None:
            ax.axhline(
                classification.floor_height,
                color="#63a46c",
                linewidth=0.9,
                linestyle=":",
                label=f"estimated floor ({classification.floor_model})",
            )
        else:
            ax.plot(
                t,
                floor_height_at_foot,
                color="#63a46c",
                linewidth=0.9,
                linestyle=":",
                label=f"estimated floor ({classification.floor_model})",
            )
        ax.set_ylabel("z (m)")
        ax.set_title(foot_name)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0], loc="upper right")

    axs[-1].set_xlabel("time (s)")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig, axs
