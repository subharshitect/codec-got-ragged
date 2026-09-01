#!/usr/bin/env python3
"""Plot frame types from extracted frame metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.plot_format import annotate_first_last_frame, apply_three_decimal_ticks
from common.progress import tqdm


COLORS = {
    "I": "#d62728",
    "P": "#1f77b4",
    "B": "#2ca02c",
}


def number(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_summary_value(value: object) -> object:
    parsed = number(value)
    if parsed is None:
        return value
    return f"{parsed:.3f}"


def draw_frame_strip(ax, rows: list[dict[str, str]]) -> None:
    x_values = [number(row.get("display_time_seconds")) for row in rows]

    for pict_type, color in COLORS.items():
        xs = [x for row, x in zip(rows, x_values) if row.get("pict_type") == pict_type and x is not None]
        if xs:
            ax.scatter(xs, [0] * len(xs), s=32, marker="o", color=color, label=pict_type, zorder=3)

    key_xs = [x for row, x in zip(rows, x_values) if row.get("key_frame") == "1" and x is not None]
    if key_xs:
        ax.scatter(
            key_xs,
            [0] * len(key_xs),
            s=95,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="key_frame",
            zorder=4,
        )

    ax.set_yticks([])
    ax.set_ylabel("frames")
    ax.set_xlabel("display_time_seconds")
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.25)
    apply_three_decimal_ticks(ax, x_axis=True, y_axis=False)
    annotate_first_last_frame(ax, rows, x_values, "display_order_index")
    ax.legend(loc="upper right", frameon=True)


def summary_text(summary: dict) -> str:
    gap = summary.get("average_distance_between_i_frames", {})
    video = summary.get("video", {})
    counts = summary.get("frame_type_counts", {})
    return "\n".join(
        [
            f"frames: {summary.get('total_frames')}",
            f"I/P/B: {counts.get('I', 0)} / {counts.get('P', 0)} / {counts.get('B', 0)}",
            f"keyframes: {summary.get('total_keyframes')}",
            f"avg I distance: {format_summary_value(gap.get('frames'))} frames, {format_summary_value(gap.get('seconds'))} sec",
            f"fps: {format_summary_value(video.get('analysis_video_fps'))}",
            f"codec: {video.get('codec_name')} {video.get('profile')} level {video.get('level')}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted", default="outputs/extracted")
    parser.add_argument("--out", default="outputs/extracted/plots")
    args = parser.parse_args()

    extracted_dir = Path(args.extracted)
    out_dir = Path(args.out)
    frames_path = extracted_dir / "frames.csv"
    summary_path = extracted_dir / "summary.json"

    if not frames_path.exists():
        raise SystemExit(f"Missing frames file: {frames_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")

    rows = read_csv(frames_path)
    rows.sort(key=lambda row: number(row.get("display_order_index")) or 0)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 4.5), gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle("Extracted Frame Types", fontsize=14)

    axes[0].axis("off")
    axes[0].text(0.01, 0.95, summary_text(summary), va="top", family="monospace", fontsize=10)
    for _ in tqdm([None], desc="plot extraction frames", unit="plot"):
        draw_frame_strip(axes[1], rows)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "frame_types.png", dpi=160)
    plt.close(fig)

    print(f"Wrote extraction plot to {out_dir}")


if __name__ == "__main__":
    main()
