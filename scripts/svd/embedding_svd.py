#!/usr/bin/env python3
"""Run SVD compression experiments on inter-I-frame embedding matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.progress import tqdm


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
    "k",
    "original_params",
    "svd_params",
    "compression_ratio",
    "relative_error",
]

AGGREGATE_FIELDS = [
    "order",
    "variant",
    "k",
    "n_segments",
    "total_frames",
    "mean_error",
    "median_error",
    "weighted_mean_error",
    "mean_compression_ratio",
    "median_compression_ratio",
]

MATRIX_ROW_FIELDS = [
    "order",
    "segment_id",
    "variant",
    "matrix_row_index",
    "frame_id",
    "source_index",
    "pict_type",
    "key_frame",
    "start_i_frame_id",
    "start_i_source_index",
    "end_i_frame_id",
    "end_i_source_index",
    "anchor_frame_id",
    "anchor_source_index",
]


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


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow({field: row.get(field, "") for field in fields})


def ordered_rows(rows: list[dict[str, str]], order_column: str) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            number(row.get(order_column)) is None,
            number(row.get(order_column)) if number(row.get(order_column)) is not None else math.inf,
            number(row.get("source_index")) or 0,
        ),
    )


def load_embedding_index(index_path: Path) -> dict[str, int]:
    return {row["frame_id"]: int(row["embedding_index"]) for row in read_csv(index_path)}


def parse_ranks(value: str) -> list[int]:
    ranks = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        rank = int(part)
        if rank <= 0:
            raise argparse.ArgumentTypeError("Ranks must be positive integers.")
        ranks.append(rank)
    if not ranks:
        raise argparse.ArgumentTypeError("At least one rank is required.")
    return sorted(set(ranks))


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def frame_embedding(row: dict[str, str], embeddings: np.ndarray, embedding_index: dict[str, int]) -> np.ndarray:
    return embeddings[embedding_index[row["frame_id"]]]


def inter_i_segments(rows: list[dict[str, str]], order_name: str, order_column: str) -> list[dict[str, object]]:
    ordered = ordered_rows(rows, order_column)
    i_positions = [index for index, row in enumerate(ordered) if row.get("pict_type") == "I"]
    segments = []

    for segment_number, (start_pos, end_pos) in enumerate(zip(i_positions, i_positions[1:])):
        start_i = ordered[start_pos]
        end_i = ordered[end_pos]
        intermediate_rows = [
            row
            for row in ordered[start_pos + 1 : end_pos]
            if row.get("pict_type") in {"P", "B"}
        ]
        if not intermediate_rows:
            continue

        segment_id = f"{order_name}_{segment_number:06d}"
        segments.append(
            {
                "segment_id": segment_id,
                "start_i": start_i,
                "end_i": end_i,
                "rows": intermediate_rows,
                "ordered_rows": ordered,
            }
        )

    return segments


def previous_frame_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output = {}
    previous = None
    for row in rows:
        if previous is not None:
            output[row["frame_id"]] = previous
        previous = row
    return output


def append_matrix_rows(
    matrix_rows: list[dict[str, str]],
    order_name: str,
    segment_id: str,
    variant: str,
    rows: list[dict[str, str]],
    start_i: dict[str, str],
    end_i: dict[str, str],
    anchors: list[dict[str, str] | None],
) -> None:
    for row_index, (row, anchor) in enumerate(zip(rows, anchors)):
        matrix_rows.append(
            {
                "order": order_name,
                "segment_id": segment_id,
                "variant": variant,
                "matrix_row_index": str(row_index),
                "frame_id": row.get("frame_id", ""),
                "source_index": row.get("source_index", ""),
                "pict_type": row.get("pict_type", ""),
                "key_frame": row.get("key_frame", ""),
                "start_i_frame_id": start_i.get("frame_id", ""),
                "start_i_source_index": start_i.get("source_index", ""),
                "end_i_frame_id": end_i.get("frame_id", ""),
                "end_i_source_index": end_i.get("source_index", ""),
                "anchor_frame_id": anchor.get("frame_id", "") if anchor else "",
                "anchor_source_index": anchor.get("source_index", "") if anchor else "",
            }
        )


def build_matrices(
    order_name: str,
    segment: dict[str, object],
    embeddings: np.ndarray,
    embedding_index: dict[str, int],
    matrix_rows: list[dict[str, str]],
) -> list[tuple[str, np.ndarray]]:
    segment_id = str(segment["segment_id"])
    start_i = segment["start_i"]
    end_i = segment["end_i"]
    rows = segment["rows"]
    ordered = segment["ordered_rows"]

    assert isinstance(start_i, dict)
    assert isinstance(end_i, dict)
    assert isinstance(rows, list)
    assert isinstance(ordered, list)

    previous_by_frame = previous_frame_map(ordered)
    start_embedding = frame_embedding(start_i, embeddings, embedding_index)

    variants = []

    frame_matrix = np.stack(
        [frame_embedding(row, embeddings, embedding_index) for row in rows],
        axis=0,
    ).astype(np.float32)
    append_matrix_rows(matrix_rows, order_name, segment_id, "embedding_frames", rows, start_i, end_i, [None] * len(rows))
    variants.append(("embedding_frames", frame_matrix))

    previous_i_matrix = np.stack(
        [start_embedding - frame_embedding(row, embeddings, embedding_index) for row in rows],
        axis=0,
    ).astype(np.float32)
    append_matrix_rows(matrix_rows, order_name, segment_id, "embedding_delta_previous_i", rows, start_i, end_i, [start_i] * len(rows))
    variants.append(("embedding_delta_previous_i", previous_i_matrix))

    adjacent_anchors = [previous_by_frame.get(row["frame_id"]) for row in rows]
    if all(anchor is not None for anchor in adjacent_anchors):
        adjacent_matrix = np.stack(
            [
                frame_embedding(anchor, embeddings, embedding_index) - frame_embedding(row, embeddings, embedding_index)
                for row, anchor in zip(rows, adjacent_anchors)
                if anchor is not None
            ],
            axis=0,
        ).astype(np.float32)
        append_matrix_rows(matrix_rows, order_name, segment_id, "embedding_delta_adjacent", rows, start_i, end_i, adjacent_anchors)
        variants.append(("embedding_delta_adjacent", adjacent_matrix))

    return variants


def svd_rows_for_matrix(
    order_name: str,
    segment: dict[str, object],
    variant: str,
    matrix: np.ndarray,
    ranks: list[int],
) -> list[dict[str, str]]:
    start_i = segment["start_i"]
    end_i = segment["end_i"]
    assert isinstance(start_i, dict)
    assert isinstance(end_i, dict)

    n_frames, d_model = matrix.shape
    max_rank = min(n_frames, d_model)
    valid_ranks = [rank for rank in ranks if rank <= max_rank]
    if not valid_ranks:
        return []

    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
    frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    matrix_rank = int(np.linalg.matrix_rank(matrix))
    original_params = n_frames * d_model

    output = []
    for rank in valid_ranks:
        svd_params = rank * (n_frames + d_model + 1)
        compression_ratio = original_params / svd_params if svd_params else None
        if frobenius_norm == 0:
            relative_error = None
        else:
            reconstructed = (u_matrix[:, :rank] * singular_values[:rank]) @ vt_matrix[:rank, :]
            relative_error = float(np.linalg.norm(matrix - reconstructed, ord="fro") / frobenius_norm)
            # Equivalent SVD shortcut:
            # sqrt(sum(S[k:]^2) / sum(S^2))

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
                "k": str(rank),
                "original_params": str(original_params),
                "svd_params": str(svd_params),
                "compression_ratio": format_float(compression_ratio),
                "relative_error": format_float(relative_error),
            }
        )

    return output


def aggregate_rows(segment_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = defaultdict(list)
    for row in segment_rows:
        if number(row.get("relative_error")) is None:
            continue
        grouped[(row["order"], row["variant"], row["k"])].append(row)

    output = []
    for (order_name, variant, rank), rows in sorted(grouped.items()):
        errors = [number(row["relative_error"]) for row in rows]
        ratios = [number(row["compression_ratio"]) for row in rows]
        frames = [int(row["n_frames"]) for row in rows]

        weighted_error = None
        total_frames = sum(frames)
        if total_frames:
            weighted_error = sum(error * frame_count for error, frame_count in zip(errors, frames) if error is not None) / total_frames

        output.append(
            {
                "order": order_name,
                "variant": variant,
                "k": rank,
                "n_segments": str(len(rows)),
                "total_frames": str(total_frames),
                "mean_error": format_float(statistics.fmean(errors)),
                "median_error": format_float(statistics.median(errors)),
                "weighted_mean_error": format_float(weighted_error),
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
    parser.add_argument("--out", default="outputs/embedding_svd")
    parser.add_argument("--ranks", type=parse_ranks, default=parse_ranks("1,5,10,20,50"))
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
    matrix_rows = []
    segment_counts = {}

    orders = [
        ("decode", "decode_order_index"),
        ("display", "display_order_index"),
    ]
    for order_name, order_column in orders:
        segments = inter_i_segments(frames, order_name, order_column)
        segment_counts[order_name] = len(segments)
        for segment in tqdm(segments, desc=f"{order_name} SVD segments", unit="segment"):
            for variant, matrix in build_matrices(order_name, segment, embeddings, embedding_index, matrix_rows):
                segment_rows.extend(svd_rows_for_matrix(order_name, segment, variant, matrix, args.ranks))

    aggregate = aggregate_rows(segment_rows)

    write_csv(out_dir / "segment_svd.csv", segment_rows, SEGMENT_FIELDS)
    write_csv(out_dir / "aggregate_svd.csv", aggregate, AGGREGATE_FIELDS)
    write_csv(out_dir / "segment_matrices.csv", matrix_rows, MATRIX_ROW_FIELDS)

    metadata = {
        "ranks": args.ranks,
        "rank_rule": "k <= min(N_i, d_model)",
        "compression_ratio": "N_i * d_model / (k * (N_i + d_model + 1))",
        "relative_error": "||X - X_k||_F / ||X||_F, computed by explicitly reconstructing X_k",
        "equivalent_error_shortcut": "sqrt(sum(S[k:]^2) / sum(S^2))",
        "segment_rule": "Segments are P/B frames between consecutive I-frames in the selected order.",
        "variants": {
            "embedding_frames": "X rows are e(current intermediate frame).",
            "embedding_delta_previous_i": "X rows are e(previous I-frame) - e(current intermediate frame).",
            "embedding_delta_adjacent": "X rows are e(previous frame in order) - e(current intermediate frame).",
        },
        "segment_counts": segment_counts,
    }
    (out_dir / "embedding_svd_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote embedding SVD outputs to {out_dir}")


if __name__ == "__main__":
    main()
