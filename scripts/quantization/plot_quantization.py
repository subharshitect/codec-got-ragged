#!/usr/bin/env python3
"""Plot embedding quantization baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.plot_format import apply_three_decimal_ticks
from common.progress import tqdm


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


def draw_panel(ax, rows: list[dict[str, str]], y_column: str, y_label: str) -> None:
    for method, color in [("pq", "#4C78A8"), ("rabitq", "#F58518")]:
        method_rows = [
            row
            for row in rows
            if row.get("method") == method
            and row.get("status") == "ok"
            and number(row.get("compression_ratio")) is not None
            and number(row.get(y_column)) is not None
        ]
        method_rows.sort(key=lambda row: number(row["compression_ratio"]) or 0)
        if not method_rows:
            continue

        xs = [number(row["compression_ratio"]) for row in method_rows]
        ys = [number(row[y_column]) for row in method_rows]
        ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.65)
        ax.scatter(xs, ys, s=36, color=color, label=method, zorder=3)

        for x_value, y_value, row in zip(xs, ys, method_rows):
            ax.annotate(row["setting"], (x_value, y_value), fontsize=7, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("compression ratio")
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    apply_three_decimal_ticks(ax)
    ax.legend(loc="best", frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", default="outputs/quantization")
    parser.add_argument("--out", default="outputs/quantization/plots")
    args = parser.parse_args()

    quantization_dir = Path(args.quantization)
    out_dir = Path(args.out)
    results_path = quantization_dir / "quantization_results.csv"

    if not results_path.exists():
        raise SystemExit(f"Missing quantization results file: {results_path}")

    rows = read_csv(results_path)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Embedding Quantization Baselines", fontsize=14)

    panels = [
        ("relative_reconstruction_error", "relative reconstruction error"),
        ("mse", "MSE"),
        ("recall_at_k", "recall@k"),
    ]
    with tqdm(total=len(panels), desc="quantization plots", unit="panel") as progress:
        for ax, (column, label) in zip(axes, panels):
            draw_panel(ax, rows, column, label)
            progress.update(1)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "quantization_comparison.png", dpi=160)
    plt.close(fig)

    print(f"Wrote quantization plots to {out_dir}")


if __name__ == "__main__":
    main()
