#!/usr/bin/env python3
"""Plot embedding similarity scores."""

from __future__ import annotations

import argparse
import csv
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


def draw_score_panel(ax, rows: list[dict[str, str]], x_values: list[float | None], column: str, label: str) -> None:
    y_values = [number(row.get(column)) for row in rows]

    valid_x = []
    valid_y = []
    for x_value, y_value in zip(x_values, y_values):
        if x_value is not None and y_value is not None:
            valid_x.append(x_value)
            valid_y.append(y_value)

    if valid_x:
        ax.plot(valid_x, valid_y, color="#555555", linewidth=0.9, alpha=0.55)

    for pict_type, color in COLORS.items():
        xs = []
        ys = []
        for row, x_value, y_value in zip(rows, x_values, y_values):
            if row.get("pict_type") == pict_type and x_value is not None and y_value is not None:
                xs.append(x_value)
                ys.append(y_value)
        if xs:
            ax.scatter(xs, ys, s=24, color=color, label=pict_type, zorder=3)

    ax.set_ylabel(label)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    apply_three_decimal_ticks(ax, x_axis=False, y_axis=True)


def draw_i_markers(ax, rows: list[dict[str, str]], x_values: list[float | None]) -> None:
    for row, x_value in zip(rows, x_values):
        if row.get("pict_type") == "I" and x_value is not None:
            ax.axvline(x_value, color=COLORS["I"], linewidth=0.8, alpha=0.35, zorder=1)


def draw_frame_strip(ax, rows: list[dict[str, str]], x_values: list[float | None]) -> None:
    for pict_type, color in COLORS.items():
        xs = [x for row, x in zip(rows, x_values) if row.get("pict_type") == pict_type and x is not None]
        if xs:
            ax.scatter(xs, [0] * len(xs), s=28, marker="o", color=color, zorder=3)

    key_xs = [x for row, x in zip(rows, x_values) if row.get("key_frame") == "1" and x is not None]
    if key_xs:
        ax.scatter(
            key_xs,
            [0] * len(key_xs),
            s=90,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
        )

    ax.set_yticks([])
    ax.set_ylabel("frames")
    ax.set_xlabel("time_seconds")
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.25)
    apply_three_decimal_ticks(ax, x_axis=True, y_axis=False)
    annotate_first_last_frame(ax, rows, x_values, "order_index")


def plot_order(
    input_path: Path,
    output_path: Path,
    title: str,
    first_column: str,
    second_column: str,
    first_label: str,
    second_label: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = read_csv(input_path)
    rows = [row for row in rows if number(row.get("order_time_seconds")) is not None]
    rows.sort(key=lambda row: number(row.get("order_index")) or 0)
    x_values = [number(row.get("order_time_seconds")) for row in rows]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 0.75]},
    )
    fig.suptitle(title, fontsize=14)

    for _ in tqdm([None], desc=title, unit="plot"):
        draw_score_panel(axes[0], rows, x_values, first_column, first_label)
        draw_score_panel(axes[1], rows, x_values, second_column, second_label)
        draw_i_markers(axes[0], rows, x_values)
        draw_i_markers(axes[1], rows, x_values)
        draw_frame_strip(axes[2], rows, x_values)

    handles = []
    for pict_type, color in COLORS.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                label=pict_type,
            )
        )
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.4,
            label="key_frame",
        )
    )
    axes[0].legend(handles=handles, loc="upper right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="outputs/embedding_delta")
    parser.add_argument("--out", default="outputs/embedding_delta/plots")
    parser.add_argument("--decode-file", default="decode_order_embedding_delta_similarities.csv")
    parser.add_argument("--display-file", default="display_order_embedding_delta_similarities.csv")
    parser.add_argument("--decode-plot", default="decode_order_embedding_delta_similarities.png")
    parser.add_argument("--display-plot", default="display_order_embedding_delta_similarities.png")
    parser.add_argument("--first-column", default="previous_i_anchor_delta_cosine_similarity")
    parser.add_argument("--second-column", default="previous_adjacent_anchor_delta_cosine_similarity")
    parser.add_argument("--first-label", default="cos sim: e(prev I), delta")
    parser.add_argument("--second-label", default="cos sim: e(prev adj), delta")
    parser.add_argument("--title-prefix", default="Embedding Delta Similarities")
    args = parser.parse_args()

    scores_dir = Path(args.scores)
    out_dir = Path(args.out)

    plot_order(
        scores_dir / args.decode_file,
        out_dir / args.decode_plot,
        f"{args.title_prefix} In Decode Order",
        args.first_column,
        args.second_column,
        args.first_label,
        args.second_label,
    )
    plot_order(
        scores_dir / args.display_file,
        out_dir / args.display_plot,
        f"{args.title_prefix} In Display Order",
        args.first_column,
        args.second_column,
        args.first_label,
        args.second_label,
    )

    print(f"Wrote embedding plots to {out_dir}")


if __name__ == "__main__":
    main()
