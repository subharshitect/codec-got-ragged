#!/usr/bin/env python3
"""Plot SVD reconstruction error against compression ratio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.plot_format import apply_three_decimal_ticks
from common.progress import tqdm
from common.tabular import number, read_csv


VARIANT_TITLES = {
    "embedding_frames": "Frame Embeddings",
    "embedding_delta_previous_i": "Delta From Previous I",
    "embedding_delta_adjacent": "Delta From Adjacent",
}
ORDERS = ["decode", "display"]
VARIANTS = ["embedding_frames", "embedding_delta_previous_i", "embedding_delta_adjacent"]

def draw_panel(
    ax,
    segment_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    order_name: str,
    variant: str,
) -> None:
    import matplotlib.pyplot as plt

    panel_rows = [
        row
        for row in segment_rows
        if row.get("order") == order_name
        and row.get("variant") == variant
        and number(row.get("compression_ratio")) is not None
        and number(row.get("relative_error")) is not None
    ]

    ranks = sorted({int(row["k"]) for row in panel_rows})
    cmap = plt.get_cmap("tab10")
    for color_index, rank in enumerate(ranks):
        rank_rows = [row for row in panel_rows if int(row["k"]) == rank]
        xs = [number(row["compression_ratio"]) for row in rank_rows]
        ys = [number(row["relative_error"]) for row in rank_rows]
        ax.scatter(xs, ys, s=22, alpha=0.7, color=cmap(color_index % 10), label=f"k={rank}")

    aggregate_points = [
        row
        for row in aggregate_rows
        if row.get("order") == order_name
        and row.get("variant") == variant
        and number(row.get("median_compression_ratio")) is not None
        and number(row.get("median_error")) is not None
    ]
    aggregate_points.sort(key=lambda row: int(row["k"]))
    if aggregate_points:
        xs = [number(row["median_compression_ratio"]) for row in aggregate_points]
        ys = [number(row["median_error"]) for row in aggregate_points]
        ax.plot(xs, ys, color="black", linewidth=1.1, marker="x", markersize=6, label="median")

    ax.set_title(f"{order_name}: {VARIANT_TITLES.get(variant, variant)}")
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("relative reconstruction error")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    apply_three_decimal_ticks(ax)
    if ranks or aggregate_points:
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "no valid rows", transform=ax.transAxes, ha="center", va="center")


def plot_all(segment_rows: list[dict[str, str]], aggregate_rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ORDERS), len(VARIANTS), figsize=(16, 8), squeeze=False)
    fig.suptitle("Embedding SVD Error vs Compression", fontsize=14)

    total_panels = len(ORDERS) * len(VARIANTS)
    with tqdm(total=total_panels, desc="embedding SVD plot", unit="panel") as progress:
        for row_index, order_name in enumerate(ORDERS):
            for col_index, variant in enumerate(VARIANTS):
                draw_panel(axes[row_index][col_index], segment_rows, aggregate_rows, order_name, variant)
                progress.update(1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_order(segment_rows: list[dict[str, str]], aggregate_rows: list[dict[str, str]], order_name: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(16, 4.8), squeeze=False)
    fig.suptitle(f"Embedding SVD Error vs Compression: {order_name}", fontsize=14)

    with tqdm(total=len(VARIANTS), desc=f"{order_name} SVD plot", unit="panel") as progress:
        for col_index, variant in enumerate(VARIANTS):
            draw_panel(axes[0][col_index], segment_rows, aggregate_rows, order_name, variant)
            progress.update(1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd", default="outputs/embedding_svd")
    parser.add_argument("--out", default="outputs/embedding_svd/plots")
    args = parser.parse_args()

    svd_dir = Path(args.svd)
    out_dir = Path(args.out)
    segment_path = svd_dir / "segment_svd.csv"
    aggregate_path = svd_dir / "aggregate_svd.csv"

    if not segment_path.exists():
        raise SystemExit(f"Missing SVD segment file: {segment_path}")
    if not aggregate_path.exists():
        raise SystemExit(f"Missing SVD aggregate file: {aggregate_path}")

    segment_rows = read_csv(segment_path)
    aggregate_rows = read_csv(aggregate_path)
    plot_all(segment_rows, aggregate_rows, out_dir / "svd_error_vs_compression.png")
    for order_name in ORDERS:
        plot_order(segment_rows, aggregate_rows, order_name, out_dir / order_name / "svd_error_vs_compression.png")

    print(f"Wrote embedding SVD plots to {out_dir}")


if __name__ == "__main__":
    main()
