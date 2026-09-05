#!/usr/bin/env python3
"""Compute embedding-vector deltas for extracted frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.embedding_data import load_embedding_inputs
from common.progress import tqdm
from common.svd_helpers import cosine_similarity
from common.tabular import add_ranks, ordered_rows, write_csv


FRAME_FIELDS = [
    "order",
    "order_index",
    "order_time_seconds",
    "frame_id",
    "source_index",
    "pict_type",
    "key_frame",
    "pts_time",
    "dts_time",
    "previous_i_frame_id",
    "previous_i_source_index",
    "previous_i_delta_vector_index",
    "previous_i_anchor_delta_cosine_similarity",
    "previous_i_anchor_delta_rank",
    "previous_adjacent_frame_id",
    "previous_adjacent_source_index",
    "previous_adjacent_delta_vector_index",
    "previous_adjacent_anchor_delta_cosine_similarity",
    "previous_adjacent_anchor_delta_rank",
]

VECTOR_FIELDS = [
    "delta_vector_index",
    "order",
    "setting",
    "frame_id",
    "source_index",
    "anchor_frame_id",
    "anchor_source_index",
    "pict_type",
    "key_frame",
    "anchor_delta_cosine_similarity",
]


def append_delta_vector(
    delta_vectors: list[np.ndarray],
    vector_rows: list[dict[str, str]],
    order_name: str,
    setting: str,
    row: dict[str, str],
    anchor: dict[str, str],
    current_embedding: np.ndarray,
    anchor_embedding: np.ndarray,
) -> tuple[int, float | None]:
    """Append delta = e(anchor) - e(current) and return its row index and cosine score."""
    delta = anchor_embedding - current_embedding
    similarity = cosine_similarity(anchor_embedding, delta)
    vector_index = len(delta_vectors)

    delta_vectors.append(delta.astype(np.float32))
    vector_rows.append(
        {
            "delta_vector_index": str(vector_index),
            "order": order_name,
            "setting": setting,
            "frame_id": row.get("frame_id", ""),
            "source_index": row.get("source_index", ""),
            "anchor_frame_id": anchor.get("frame_id", ""),
            "anchor_source_index": anchor.get("source_index", ""),
            "pict_type": row.get("pict_type", ""),
            "key_frame": row.get("key_frame", ""),
            "anchor_delta_cosine_similarity": "" if similarity is None else f"{similarity:.8f}",
        }
    )
    return vector_index, similarity


def compute_order(
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    embedding_index: dict[str, int],
    order_name: str,
    order_column: str,
    time_column: str,
    delta_vectors: list[np.ndarray],
    vector_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    output = []
    previous = None
    previous_i = None

    ordered = ordered_rows(rows, order_column)
    for row in tqdm(ordered, desc=f"{order_name} embedding vector deltas", unit="frame"):
        frame_id = row["frame_id"]
        current_embedding = embeddings[embedding_index[frame_id]]
        use_delta = row.get("pict_type") in {"P", "B"}

        output_row = {
            "order": order_name,
            "order_index": row.get(order_column, ""),
            "order_time_seconds": row.get(time_column, ""),
            "frame_id": frame_id,
            "source_index": row.get("source_index", ""),
            "pict_type": row.get("pict_type", ""),
            "key_frame": row.get("key_frame", ""),
            "pts_time": row.get("pts_time", ""),
            "dts_time": row.get("dts_time", ""),
            "previous_i_frame_id": "",
            "previous_i_source_index": "",
            "previous_i_delta_vector_index": "",
            "previous_i_anchor_delta_cosine_similarity": "",
            "previous_i_anchor_delta_rank": "",
            "previous_adjacent_frame_id": "",
            "previous_adjacent_source_index": "",
            "previous_adjacent_delta_vector_index": "",
            "previous_adjacent_anchor_delta_cosine_similarity": "",
            "previous_adjacent_anchor_delta_rank": "",
        }

        if use_delta and previous_i is not None:
            anchor_embedding = embeddings[embedding_index[previous_i["frame_id"]]]
            vector_index, similarity = append_delta_vector(
                delta_vectors,
                vector_rows,
                order_name,
                "previous_i",
                row,
                previous_i,
                current_embedding,
                anchor_embedding,
            )
            output_row["previous_i_frame_id"] = previous_i.get("frame_id", "")
            output_row["previous_i_source_index"] = previous_i.get("source_index", "")
            output_row["previous_i_delta_vector_index"] = str(vector_index)
            output_row["previous_i_anchor_delta_cosine_similarity"] = "" if similarity is None else f"{similarity:.8f}"

        if use_delta and previous is not None:
            anchor_embedding = embeddings[embedding_index[previous["frame_id"]]]
            vector_index, similarity = append_delta_vector(
                delta_vectors,
                vector_rows,
                order_name,
                "previous_adjacent",
                row,
                previous,
                current_embedding,
                anchor_embedding,
            )
            output_row["previous_adjacent_frame_id"] = previous.get("frame_id", "")
            output_row["previous_adjacent_source_index"] = previous.get("source_index", "")
            output_row["previous_adjacent_delta_vector_index"] = str(vector_index)
            output_row["previous_adjacent_anchor_delta_cosine_similarity"] = "" if similarity is None else f"{similarity:.8f}"

        output.append(output_row)

        if row.get("pict_type") == "I":
            previous_i = row
        previous = row

    add_ranks(output, "previous_i_anchor_delta_cosine_similarity", "previous_i_anchor_delta_rank")
    add_ranks(output, "previous_adjacent_anchor_delta_cosine_similarity", "previous_adjacent_anchor_delta_rank")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="outputs/extracted/frames.csv")
    parser.add_argument("--embeddings", default="outputs/embeddings/frame_embeddings.npy")
    parser.add_argument("--index", default="outputs/embeddings/frame_embeddings.csv")
    parser.add_argument("--out", default="outputs/embedding_delta")
    args = parser.parse_args()

    frames_path = Path(args.frames)
    embeddings_path = Path(args.embeddings)
    index_path = Path(args.index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, embeddings, embedding_index = load_embedding_inputs(frames_path, embeddings_path, index_path)

    delta_vectors: list[np.ndarray] = []
    vector_rows: list[dict[str, str]] = []
    decode_rows = compute_order(
        rows,
        embeddings,
        embedding_index,
        "decode",
        "decode_order_index",
        "decode_time_seconds",
        delta_vectors,
        vector_rows,
    )
    display_rows = compute_order(
        rows,
        embeddings,
        embedding_index,
        "display",
        "display_order_index",
        "display_time_seconds",
        delta_vectors,
        vector_rows,
    )

    write_csv(out_dir / "decode_order_embedding_delta_similarities.csv", decode_rows, FRAME_FIELDS)
    write_csv(out_dir / "display_order_embedding_delta_similarities.csv", display_rows, FRAME_FIELDS)
    write_csv(out_dir / "embedding_delta_vectors.csv", vector_rows, VECTOR_FIELDS)

    vector_array = np.stack(delta_vectors, axis=0) if delta_vectors else np.empty((0, 0), dtype=np.float32)
    np.save(out_dir / "embedding_delta_vectors.npy", vector_array)

    metadata = {
        "delta_formula": "delta = e(anchor_frame) - e(current_frame)",
        "similarity": "cosine similarity between e(anchor_frame) and delta",
        "settings": {
            "previous_i": "For each P/B frame, anchor is the previous I-frame in the same order.",
            "previous_adjacent": "For each P/B frame, anchor is the previous adjacent frame in the same order.",
        },
        "outputs": {
            "embedding_delta_vectors.npy": "Saved delta vectors, one row per valid comparison.",
            "embedding_delta_vectors.csv": "Index mapping for each saved delta vector.",
        },
        "notes": [
            "I-frames are kept as plot/row anchors but do not get delta values.",
            "Previous I-frame is an analysis anchor, not a claimed codec reference.",
            "This stage uses existing frame embeddings and does not rerun CLIP.",
        ],
    }
    (out_dir / "embedding_delta_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote embedding delta similarities to {out_dir}")


if __name__ == "__main__":
    main()
