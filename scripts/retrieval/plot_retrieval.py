#!/usr/bin/env python3
"""Plot retrieval recall against compression ratio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.plot_format import apply_three_decimal_ticks
from common.progress import tqdm
from common.tabular import number, read_csv


ORDERS = ["decode", "display"]
VARIANT_TITLES = {
    "embedding_frames": "Frame Embeddings",
    "embedding_delta_previous_i": "Delta From Previous I",
    "embedding_delta_adjacent": "Delta From Adjacent",
}
BASELINE_STYLES = {
    "pq": ("PQ", "#7F3C8D", "s"),
    "rabitq": ("RaBitQ", "#11A579", "D"),
}


def draw_order_panel(ax, rows: list[dict[str, str]], order_name: str) -> None:
    """Draw one recall-vs-compression panel for decode or display order."""
    svd_rows = [
        row
        for row in rows
        if row.get("method") == "svd"
        and row.get("order") == order_name
        and row.get("status") == "ok"
        and number(row.get("compression_ratio")) is not None
        and number(row.get("recall_at_k")) is not None
    ]

    variants = sorted({row["variant"] for row in svd_rows})
    for variant in variants:
        variant_rows = [row for row in svd_rows if row["variant"] == variant]
        variant_rows.sort(key=lambda row: number(row["compression_ratio"]) or 0)
        xs = [number(row["compression_ratio"]) for row in variant_rows]
        ys = [number(row["recall_at_k"]) for row in variant_rows]
        ax.plot(xs, ys, linewidth=1.0, alpha=0.75)
        ax.scatter(xs, ys, s=34, label=VARIANT_TITLES.get(variant, variant), zorder=3)

    has_baselines = False
    for method, (label, color, marker) in BASELINE_STYLES.items():
        baseline_rows = [
            row
            for row in rows
            if row.get("method") == method
            and row.get("status") == "ok"
            and number(row.get("compression_ratio")) is not None
            and number(row.get("recall_at_k")) is not None
        ]
        baseline_rows.sort(key=lambda row: number(row["compression_ratio"]) or 0)
        if not baseline_rows:
            continue

        xs = [number(row["compression_ratio"]) for row in baseline_rows]
        ys = [number(row["recall_at_k"]) for row in baseline_rows]
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.55)
        ax.scatter(xs, ys, s=42, color=color, marker=marker, label=label, zorder=4)
        has_baselines = True

    ax.set_title(order_name)
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("recall@k")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    apply_three_decimal_ticks(ax)
    ax.set_ylim(-0.02, 1.02)
    if variants or has_baselines:
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "no valid rows", transform=ax.transAxes, ha="center", va="center")


def main() -> None:
    """Render retrieval plots from retrieval_results.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="outputs/retrieval")
    parser.add_argument("--out", default="outputs/retrieval/plots")
    args = parser.parse_args()

    retrieval_dir = Path(args.retrieval)
    out_dir = Path(args.out)
    results_path = retrieval_dir / "retrieval_results.csv"

    if not results_path.exists():
        raise SystemExit(f"Missing retrieval results file: {results_path}")

    rows = read_csv(results_path)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ORDERS), figsize=(14, 4.8), squeeze=False)
    fig.suptitle("Retrieval Recall vs Compression", fontsize=14)

    with tqdm(total=len(ORDERS), desc="retrieval plot", unit="panel") as progress:
        for index, order_name in enumerate(ORDERS):
            draw_order_panel(axes[0][index], rows, order_name)
            progress.update(1)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "compression_vs_recall.png", dpi=160)
    plt.close(fig)

    print(f"Wrote retrieval plots to {out_dir}")


if __name__ == "__main__":
    main()
