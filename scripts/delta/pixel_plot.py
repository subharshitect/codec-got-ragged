#!/usr/bin/env python3
"""Plot adjacent-frame deltas and ranks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.plot_format import annotate_first_last_frame, apply_three_decimal_ticks
from common.progress import tqdm
from common.tabular import number, read_csv


COLORS = {
    "I": "#d62728",
    "P": "#1f77b4",
    "B": "#2ca02c",
}

def plot_order(input_path: Path, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    rows = read_csv(input_path)
    rows = [row for row in rows if number(row.get("order_time_seconds")) is not None]
    rows.sort(key=lambda row: number(row["order_index"]) or 0)

    x_values = [number(row["order_time_seconds"]) for row in rows]
    deltas = [number(row.get("pixel_delta_mean_abs")) for row in rows]
    ranks = [number(row.get("delta_rank")) for row in rows]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 0.75]},
    )
    fig.suptitle(title, fontsize=14)

    for _ in tqdm([None], desc=title, unit="plot"):
        draw_panel(axes[0], rows, x_values, deltas, "Mean abs pixel delta", show_rank=False)
        draw_panel(axes[1], rows, x_values, ranks, "Delta rank", show_rank=True)
        draw_i_markers(axes[0], rows, x_values)
        draw_i_markers(axes[1], rows, x_values)
        draw_frame_strip(axes[2], rows, x_values)

    axes[0].grid(True, linewidth=0.4, alpha=0.35)
    axes[1].grid(True, linewidth=0.4, alpha=0.35)
    axes[2].set_xlabel("time_seconds")
    apply_three_decimal_ticks(axes[0], x_axis=False, y_axis=True)
    apply_three_decimal_ticks(axes[1], x_axis=False, y_axis=True)
    apply_three_decimal_ticks(axes[2], x_axis=True, y_axis=False)

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


def draw_panel(ax, rows, x_values, y_values, y_label: str, show_rank: bool) -> None:
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
            ax.scatter(xs, ys, s=22, color=color, label=pict_type, zorder=3)

    key_xs = []
    key_ys = []
    for row, x_value, y_value in zip(rows, x_values, y_values):
        if row.get("key_frame") == "1" and x_value is not None and y_value is not None:
            key_xs.append(x_value)
            key_ys.append(y_value)
    if key_xs:
        ax.scatter(
            key_xs,
            key_ys,
            s=58,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
        )

    if show_rank:
        ax.invert_yaxis()
    ax.set_ylabel(y_label)


def draw_i_markers(ax, rows, x_values) -> None:
    for row, x_value in zip(rows, x_values):
        if row.get("pict_type") == "I" and x_value is not None:
            ax.axvline(x_value, color=COLORS["I"], linewidth=0.8, alpha=0.35, zorder=1)


def draw_frame_strip(ax, rows, x_values) -> None:
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
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.25)
    annotate_first_last_frame(ax, rows, x_values, "order_index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deltas", default="outputs/deltas")
    parser.add_argument("--out", default="outputs/plots")
    args = parser.parse_args()

    deltas_dir = Path(args.deltas)
    out_dir = Path(args.out)

    plot_order(
        deltas_dir / "decode_order_deltas.csv",
        out_dir / "decode_order_deltas.png",
        "Adjacent Deltas In Decode Order",
    )
    plot_order(
        deltas_dir / "display_order_deltas.csv",
        out_dir / "display_order_deltas.png",
        "Adjacent Deltas In Display Order",
    )

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
