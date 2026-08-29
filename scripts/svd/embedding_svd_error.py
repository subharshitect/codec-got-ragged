#!/usr/bin/env python3
"""Select the smallest SVD rank needed for target reconstruction errors."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from embedding_svd import (
    build_matrices,
    format_float,
    inter_i_segments,
    load_embedding_index,
    number,
    read_csv,
    write_csv,
)
from common.progress import tqdm


EPSILON_TOLERANCE = 1e-12

SEGMENT_FIELDS = [
    "order",
    "segment_id",
    "variant",
    "start_i_frame_id",
    "start_i_source_index",
    "end_i_frame_id",
    "end_i_source_index",
    "n_frames",
    "d_model",
    "matrix_rank",
    "matrix_frobenius_norm",
    "epsilon",
    "selected_k",
    "achieved_error",
    "original_params",
    "svd_params",
    "compression_ratio",
    "status",
]

AGGREGATE_FIELDS = [
    "order",
    "variant",
    "epsilon",
    "n_segments",
    "total_frames",
    "mean_selected_k",
    "median_selected_k",
    "mean_achieved_error",
    "median_achieved_error",
    "weighted_mean_achieved_error",
    "mean_compression_ratio",
    "median_compression_ratio",
]


def parse_epsilons(value: str) -> list[float]:
    epsilons = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        epsilon = float(part)
        if epsilon < 0:
            raise argparse.ArgumentTypeError("Epsilons must be non-negative.")
        epsilons.append(epsilon)
    if not epsilons:
        raise argparse.ArgumentTypeError("At least one epsilon is required.")
    return sorted(set(epsilons))


def relative_error_at_rank(
    matrix: np.ndarray,
    u_matrix: np.ndarray,
    singular_values: np.ndarray,
    vt_matrix: np.ndarray,
    rank: int,
    frobenius_norm: float,
) -> float:
    reconstructed = (u_matrix[:, :rank] * singular_values[:rank]) @ vt_matrix[:rank, :]
    # Equivalent SVD shortcut:
    # sqrt(sum(S[k:]^2) / sum(S^2))
    return float(np.linalg.norm(matrix - reconstructed, ord="fro") / frobenius_norm)


def selected_rank_for_epsilon(
    matrix: np.ndarray,
    u_matrix: np.ndarray,
    singular_values: np.ndarray,
    vt_matrix: np.ndarray,
    frobenius_norm: float,
    epsilon: float,
) -> tuple[int | None, float | None]:
    if frobenius_norm == 0:
        return None, None

    max_rank = len(singular_values)
    for rank in range(1, max_rank + 1):
        error = relative_error_at_rank(matrix, u_matrix, singular_values, vt_matrix, rank, frobenius_norm)
        if error <= epsilon + EPSILON_TOLERANCE:
            return rank, error

    return None, None


def rows_for_matrix(
    order_name: str,
    segment: dict[str, object],
    variant: str,
    matrix: np.ndarray,
    epsilons: list[float],
) -> list[dict[str, str]]:
    start_i = segment["start_i"]
    end_i = segment["end_i"]
    assert isinstance(start_i, dict)
    assert isinstance(end_i, dict)

    n_frames, d_model = matrix.shape
    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
    frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    matrix_rank = int(np.linalg.matrix_rank(matrix))
    original_params = n_frames * d_model

    output = []
    for epsilon in epsilons:
        selected_k, achieved_error = selected_rank_for_epsilon(
            matrix,
            u_matrix,
            singular_values,
            vt_matrix,
            frobenius_norm,
            epsilon,
        )
        svd_params = selected_k * (n_frames + d_model + 1) if selected_k is not None else None
        compression_ratio = original_params / svd_params if svd_params else None

        output.append(
            {
                "order": order_name,
                "segment_id": str(segment["segment_id"]),
                "variant": variant,
                "start_i_frame_id": start_i.get("frame_id", ""),
                "start_i_source_index": start_i.get("source_index", ""),
                "end_i_frame_id": end_i.get("frame_id", ""),
                "end_i_source_index": end_i.get("source_index", ""),
                "n_frames": str(n_frames),
                "d_model": str(d_model),
                "matrix_rank": str(matrix_rank),
                "matrix_frobenius_norm": format_float(frobenius_norm),
                "epsilon": format_float(epsilon),
                "selected_k": "" if selected_k is None else str(selected_k),
                "achieved_error": format_float(achieved_error),
                "original_params": str(original_params),
                "svd_params": "" if svd_params is None else str(svd_params),
                "compression_ratio": format_float(compression_ratio),
                "status": "zero_norm" if frobenius_norm == 0 else "met",
            }
        )

    return output


def aggregate_rows(segment_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = defaultdict(list)
    for row in segment_rows:
        if row.get("status") != "met":
            continue
        grouped[(row["order"], row["variant"], row["epsilon"])].append(row)

    output = []
    for (order_name, variant, epsilon), rows in sorted(grouped.items()):
        ranks = [number(row["selected_k"]) for row in rows]
        errors = [number(row["achieved_error"]) for row in rows]
        ratios = [number(row["compression_ratio"]) for row in rows]
        frames = [int(row["n_frames"]) for row in rows]

        total_frames = sum(frames)
        weighted_error = None
        if total_frames:
            weighted_error = sum(error * frame_count for error, frame_count in zip(errors, frames) if error is not None) / total_frames

        output.append(
            {
                "order": order_name,
                "variant": variant,
                "epsilon": epsilon,
                "n_segments": str(len(rows)),
                "total_frames": str(total_frames),
                "mean_selected_k": format_float(statistics.fmean(ranks)),
                "median_selected_k": format_float(statistics.median(ranks)),
                "mean_achieved_error": format_float(statistics.fmean(errors)),
                "median_achieved_error": format_float(statistics.median(errors)),
                "weighted_mean_achieved_error": format_float(weighted_error),
                "mean_compression_ratio": format_float(statistics.fmean(ratios)),
                "median_compression_ratio": format_float(statistics.median(ratios)),
            }
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="outputs/extracted/frames.csv")
    parser.add_argument("--embeddings", default="outputs/embeddings/frame_embeddings.npy")
    parser.add_argument("--index", default="outputs/embeddings/frame_embeddings.csv")
    parser.add_argument("--out", default="outputs/embedding_svd_error")
    parser.add_argument("--epsilons", type=parse_epsilons, default=parse_epsilons("0.01,0.05,0.10,0.20"))
    args = parser.parse_args()

    frames_path = Path(args.frames)
    embeddings_path = Path(args.embeddings)
    index_path = Path(args.index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frames_path.exists():
        raise SystemExit(f"Missing frames file: {frames_path}")
    if not embeddings_path.exists():
        raise SystemExit(f"Missing embeddings file: {embeddings_path}")
    if not index_path.exists():
        raise SystemExit(f"Missing embedding index file: {index_path}")

    frames = read_csv(frames_path)
    embedding_index = load_embedding_index(index_path)
    embeddings = np.load(embeddings_path)

    if embeddings.ndim != 2:
        raise SystemExit(f"Expected a 2D embedding array, got shape {embeddings.shape}")

    missing = [row["frame_id"] for row in frames if row["frame_id"] not in embedding_index]
    if missing:
        raise SystemExit(f"Missing embedding for frame: {missing[0]}")

    segment_rows = []
    segment_counts = {}

    orders = [
        ("decode", "decode_order_index"),
        ("display", "display_order_index"),
    ]
    for order_name, order_column in orders:
        segments = inter_i_segments(frames, order_name, order_column)
        segment_counts[order_name] = len(segments)
        for segment in tqdm(segments, desc=f"{order_name} SVD error targets", unit="segment"):
            matrix_rows: list[dict[str, str]] = []
            for variant, matrix in build_matrices(order_name, segment, embeddings, embedding_index, matrix_rows):
                segment_rows.extend(rows_for_matrix(order_name, segment, variant, matrix, args.epsilons))

    aggregate = aggregate_rows(segment_rows)

    write_csv(out_dir / "segment_svd_error.csv", segment_rows, SEGMENT_FIELDS)
    write_csv(out_dir / "aggregate_svd_error.csv", aggregate, AGGREGATE_FIELDS)

    metadata = {
        "epsilons": args.epsilons,
        "selection_rule": "For each epsilon, choose the smallest k where ||X - X_k||_F / ||X||_F <= epsilon + tolerance.",
        "epsilon_tolerance": EPSILON_TOLERANCE,
        "equivalent_error_shortcut": "sqrt(sum(S[k:]^2) / sum(S^2))",
        "compression_ratio": "N_i * d_model / (k * (N_i + d_model + 1)) using the selected k.",
        "achieved_error": "Relative reconstruction error at the selected k.",
        "segment_rule": "Segments are P/B frames between consecutive I-frames in the selected order.",
        "variants": {
            "embedding_frames": "X rows are e(current intermediate frame).",
            "embedding_delta_previous_i": "X rows are e(previous I-frame) - e(current intermediate frame).",
            "embedding_delta_adjacent": "X rows are e(previous frame in order) - e(current intermediate frame).",
        },
        "segment_counts": segment_counts,
    }
    (out_dir / "embedding_svd_error_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote embedding SVD error-target outputs to {out_dir}")


if __name__ == "__main__":
    main()
