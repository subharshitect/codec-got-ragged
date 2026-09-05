"""Shared SVD matrix construction and reconstruction math."""

from __future__ import annotations

import numpy as np

from common.tabular import ordered_rows


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return cosine similarity with a zero-vector guard."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return None
    return float(np.dot(a, b) / denominator)


def frame_embedding(row: dict[str, str], embeddings: np.ndarray, embedding_index: dict[str, int]) -> np.ndarray:
    """Fetch the embedding vector for a frame row."""
    return embeddings[embedding_index[row["frame_id"]]]


def inter_i_segments(rows: list[dict[str, str]], order_name: str, order_column: str) -> list[dict[str, object]]:
    """Return P/B-frame segments between consecutive I-frames in the chosen order."""
    ordered = ordered_rows(rows, order_column)
    i_positions = [index for index, row in enumerate(ordered) if row.get("pict_type") == "I"]
    segments = []

    for segment_number, (start_pos, end_pos) in enumerate(zip(i_positions, i_positions[1:])): # Between consecutive I-frames
        start_i = ordered[start_pos]
        end_i = ordered[end_pos]
        intermediate_rows = [
            row
            for row in ordered[start_pos + 1 : end_pos]
            if row.get("pict_type") in {"P", "B"}
        ]
        if not intermediate_rows:
            continue

        segments.append(
            {
                "segment_id": f"{order_name}_{segment_number:06d}",
                "start_i": start_i,
                "end_i": end_i,
                "rows": intermediate_rows,
                "ordered_rows": ordered, # segment frames, ordered
            }
        )

    return segments


def previous_frame_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map each frame_id to its previous frame row in the same order."""
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
    """Record which frames became rows in an SVD segment matrix."""
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
    matrix_rows: list[dict[str, str]] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Build the three SVD variants; delta direction is e(anchor) - e(current)."""
    segment_id = str(segment["segment_id"])
    start_i = segment["start_i"]
    end_i = segment["end_i"]
    rows = segment["rows"]
    ordered = segment["ordered_rows"]

    assert isinstance(start_i, dict)
    assert isinstance(end_i, dict)
    assert isinstance(rows, list)
    assert isinstance(ordered, list)

    previous_by_frame = previous_frame_map(ordered) # maps current frame to the previous frame, "adjancent setting" 
    start_embedding = frame_embedding(start_i, embeddings, embedding_index) # Fetch the embedding vector for a "start_i" or prev I-frame for the segment

    variants = []

    frame_matrix = np.stack( # forms the matrix of embeddings for the frames in the segment
        [frame_embedding(row, embeddings, embedding_index) for row in rows],
        axis=0,
    ).astype(np.float32)
    if matrix_rows is not None:
        append_matrix_rows(
            matrix_rows,
            order_name,
            segment_id,
            "embedding_frames",
            rows,
            start_i,
            end_i,
            [None] * len(rows),
        )
    variants.append(("embedding_frames", frame_matrix))

    previous_i_matrix = np.stack( # forms the matrix of delta embeddings from the previous i-frame
        [start_embedding - frame_embedding(row, embeddings, embedding_index) for row in rows],
        axis=0,
    ).astype(np.float32)
    if matrix_rows is not None:
        append_matrix_rows(
            matrix_rows,
            order_name,
            segment_id,
            "embedding_delta_previous_i",
            rows,
            start_i,
            end_i,
            [start_i] * len(rows),
        )
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
        if matrix_rows is not None:
            append_matrix_rows(
                matrix_rows,
                order_name,
                segment_id,
                "embedding_delta_adjacent",
                rows,
                start_i,
                end_i,
                adjacent_anchors,
            )
        variants.append(("embedding_delta_adjacent", adjacent_matrix))

    return variants


def compression_params(n_frames: int, d_model: int, rank: int) -> tuple[int, int, float | None]:
    """Return original params, truncated-SVD params, and compression ratio."""
    original_params = n_frames * d_model
    svd_params = rank * (n_frames + d_model + 1)
    compression_ratio = original_params / svd_params if svd_params else None
    return original_params, svd_params, compression_ratio


def relative_error_at_rank(
    matrix: np.ndarray,
    u_matrix: np.ndarray,
    singular_values: np.ndarray,
    vt_matrix: np.ndarray,
    rank: int,
    frobenius_norm: float,
) -> float:
    """Compute ||X - X_k||_F / ||X||_F using explicit rank-k reconstruction."""
    reconstructed = (u_matrix[:, :rank] * singular_values[:rank]) @ vt_matrix[:rank, :]
    # Equivalent singular-value shortcut:
    # sqrt(sum(S[k:]^2) / sum(S^2))
    return float(np.linalg.norm(matrix - reconstructed, ord="fro") / frobenius_norm)
