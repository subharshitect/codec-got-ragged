#!/usr/bin/env python3
"""Plot SVD target-error rank selection results."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.progress import tqdm


VARIANT_TITLES = {
    "embedding_frames": "Frame Embeddings",
    "embedding_delta_previous_i": "Delta From Previous I",
    "embedding_delta_adjacent": "Delta From Adjacent",
}
ORDERS = ["decode", "display"]
VARIANTS = ["embedding_frames", "embedding_delta_previous_i", "embedding_delta_adjacent"]


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


def marker_size(rank: float | None) -> float:
    if rank is None:
        return 22
    return min(140, 18 + 8 * (rank ** 0.5))


def panel_rows(rows: list[dict[str, str]], order_name: str, variant: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("order") == order_name
        and row.get("variant") == variant
        and row.get("status", "met") == "met"
    ]


def draw_compression_panel(
    ax,
    segment_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    order_name: str,
    variant: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in panel_rows(segment_rows, order_name, variant)
        if number(row.get("compression_ratio")) is not None and number(row.get("achieved_error")) is not None
    ]

    epsilons = sorted({number(row["epsilon"]) for row in rows if number(row.get("epsilon")) is not None})
    cmap = plt.get_cmap("viridis")
    for color_index, epsilon in enumerate(epsilons):
        epsilon_rows = [row for row in rows if number(row["epsilon"]) == epsilon]
        xs = [number(row["compression_ratio"]) for row in epsilon_rows]
        ys = [number(row["achieved_error"]) for row in epsilon_rows]
        sizes = [marker_size(number(row["selected_k"])) for row in epsilon_rows]
        color = cmap(color_index / max(1, len(epsilons) - 1))
        ax.scatter(xs, ys, s=sizes, alpha=0.65, color=color, label=f"eps={epsilon:g}")

    aggregate_points = [
        row
        for row in panel_rows(aggregate_rows, order_name, variant)
        if number(row.get("median_compression_ratio")) is not None
        and number(row.get("median_achieved_error")) is not None
    ]
    aggregate_points.sort(key=lambda row: number(row["epsilon"]) or 0)
    if aggregate_points:
        xs = [number(row["median_compression_ratio"]) for row in aggregate_points]
        ys = [number(row["median_achieved_error"]) for row in aggregate_points]
        ax.plot(xs, ys, color="black", linewidth=1.1, marker="x", markersize=6, label="median")

    ax.set_title(f"{order_name}: {VARIANT_TITLES.get(variant, variant)}")
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("achieved reconstruction error")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    if epsilons or aggregate_points:
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "no valid rows", transform=ax.transAxes, ha="center", va="center")


def draw_rank_panel(
    ax,
    segment_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    order_name: str,
    variant: str,
) -> None:
    rows = [
        row
        for row in panel_rows(segment_rows, order_name, variant)
        if number(row.get("epsilon")) is not None and number(row.get("selected_k")) is not None
    ]

    if rows:
        xs = [number(row["epsilon"]) for row in rows]
        ys = [number(row["selected_k"]) for row in rows]
        ax.scatter(xs, ys, s=22, alpha=0.6, color="#4C78A8", label="segment")

    aggregate_points = [
        row
        for row in panel_rows(aggregate_rows, order_name, variant)
        if number(row.get("epsilon")) is not None and number(row.get("median_selected_k")) is not None
    ]
    aggregate_points.sort(key=lambda row: number(row["epsilon"]) or 0)
    if aggregate_points:
        xs = [number(row["epsilon"]) for row in aggregate_points]
        ys = [number(row["median_selected_k"]) for row in aggregate_points]
        ax.plot(xs, ys, color="black", linewidth=1.1, marker="x", markersize=6, label="median")

    ax.set_title(f"{order_name}: {VARIANT_TITLES.get(variant, variant)}")
    ax.set_xlabel("target error epsilon")
    ax.set_ylabel("selected k")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    if rows or aggregate_points:
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "no valid rows", transform=ax.transAxes, ha="center", va="center")


def plot_grid(
    segment_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    panel_kind: str,
    orders: list[str],
) -> None:
    import matplotlib.pyplot as plt

    fig_height = 4.8 if len(orders) == 1 else 8
    fig, axes = plt.subplots(len(orders), len(VARIANTS), figsize=(16, fig_height), squeeze=False)
    fig.suptitle(title, fontsize=14)

    total_panels = len(orders) * len(VARIANTS)
    with tqdm(total=total_panels, desc=title, unit="panel") as progress:
        for row_index, order_name in enumerate(orders):
            for col_index, variant in enumerate(VARIANTS):
                ax = axes[row_index][col_index]
                if panel_kind == "compression":
                    draw_compression_panel(ax, segment_rows, aggregate_rows, order_name, variant)
                else:
                    draw_rank_panel(ax, segment_rows, aggregate_rows, order_name, variant)
                progress.update(1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd-error", default="outputs/embedding_svd_error")
    parser.add_argument("--out", default="outputs/embedding_svd_error/plots")
    args = parser.parse_args()

    svd_error_dir = Path(args.svd_error)
    out_dir = Path(args.out)
    segment_path = svd_error_dir / "segment_svd_error.csv"
    aggregate_path = svd_error_dir / "aggregate_svd_error.csv"

    if not segment_path.exists():
        raise SystemExit(f"Missing SVD error segment file: {segment_path}")
    if not aggregate_path.exists():
        raise SystemExit(f"Missing SVD error aggregate file: {aggregate_path}")

    segment_rows = read_csv(segment_path)
    aggregate_rows = read_csv(aggregate_path)

    plot_grid(
        segment_rows,
        aggregate_rows,
        out_dir / "svd_error_targets_compression.png",
        "SVD Target Error: Compression vs Achieved Error",
        "compression",
        ORDERS,
    )
    plot_grid(
        segment_rows,
        aggregate_rows,
        out_dir / "svd_error_targets_k.png",
        "SVD Target Error: Selected Rank",
        "rank",
        ORDERS,
    )

    for order_name in ORDERS:
        order_dir = out_dir / order_name
        plot_grid(
            segment_rows,
            aggregate_rows,
            order_dir / "svd_error_targets_compression.png",
            f"SVD Target Error: Compression vs Achieved Error: {order_name}",
            "compression",
            [order_name],
        )
        plot_grid(
            segment_rows,
            aggregate_rows,
            order_dir / "svd_error_targets_k.png",
            f"SVD Target Error: Selected Rank: {order_name}",
            "rank",
            [order_name],
        )

    print(f"Wrote embedding SVD error-target plots to {out_dir}")


if __name__ == "__main__":
    main()
