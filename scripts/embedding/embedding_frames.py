#!/usr/bin/env python3
"""Compute direct frame-embedding cosine similarities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.progress import tqdm


FIELDS = [
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
    "previous_i_cosine_similarity",
    "previous_i_rank",
    "previous_adjacent_frame_id",
    "previous_adjacent_source_index",
    "previous_adjacent_cosine_similarity",
    "previous_adjacent_rank",
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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def ordered_rows(rows: list[dict[str, str]], order_column: str) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            number(row.get(order_column)) is None,
            number(row.get(order_column)) if number(row.get(order_column)) is not None else math.inf,
            number(row.get("source_index")) or 0,
        ),
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return None
    return float(np.dot(a, b) / denominator)


def add_ranks(rows: list[dict[str, str]], value_column: str, rank_column: str) -> None:
    ranked = [
        (index, number(row.get(value_column)))
        for index, row in enumerate(rows)
        if number(row.get(value_column)) is not None
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    for rank, (index, _) in enumerate(ranked, start=1):
        rows[index][rank_column] = str(rank)


def compute_order(
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    embedding_index: dict[str, int],
    order_name: str,
    order_column: str,
    time_column: str,
) -> list[dict[str, str]]:
    output = []
    previous = None
    previous_i = None

    ordered = ordered_rows(rows, order_column)
    for row in tqdm(ordered, desc=f"{order_name} frame embedding similarities", unit="frame"):
        frame_id = row["frame_id"]
        current_embedding = embeddings[embedding_index[frame_id]]
        use_similarity = row.get("pict_type") in {"P", "B"}

        previous_i_similarity = None
        if use_similarity and previous_i is not None:
            previous_i_similarity = cosine_similarity(
                embeddings[embedding_index[previous_i["frame_id"]]],
                current_embedding,
            )

        previous_adjacent_similarity = None
        if use_similarity and previous is not None:
            previous_adjacent_similarity = cosine_similarity(
                embeddings[embedding_index[previous["frame_id"]]],
                current_embedding,
            )

        output.append(
            {
                "order": order_name,
                "order_index": row.get(order_column, ""),
                "order_time_seconds": row.get(time_column, ""),
                "frame_id": frame_id,
                "source_index": row.get("source_index", ""),
                "pict_type": row.get("pict_type", ""),
                "key_frame": row.get("key_frame", ""),
                "pts_time": row.get("pts_time", ""),
                "dts_time": row.get("dts_time", ""),
                "previous_i_frame_id": previous_i.get("frame_id", "") if previous_i and use_similarity else "",
                "previous_i_source_index": previous_i.get("source_index", "") if previous_i and use_similarity else "",
                "previous_i_cosine_similarity": "" if previous_i_similarity is None else f"{previous_i_similarity:.8f}",
                "previous_i_rank": "",
                "previous_adjacent_frame_id": previous.get("frame_id", "") if previous and use_similarity else "",
                "previous_adjacent_source_index": previous.get("source_index", "") if previous and use_similarity else "",
                "previous_adjacent_cosine_similarity": "" if previous_adjacent_similarity is None else f"{previous_adjacent_similarity:.8f}",
                "previous_adjacent_rank": "",
            }
        )

        if row.get("pict_type") == "I":
            previous_i = row
        previous = row

    add_ranks(output, "previous_i_cosine_similarity", "previous_i_rank")
    add_ranks(output, "previous_adjacent_cosine_similarity", "previous_adjacent_rank")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="outputs/extracted/frames.csv")
    parser.add_argument("--embeddings", default="outputs/embeddings/frame_embeddings.npy")
    parser.add_argument("--index", default="outputs/embeddings/frame_embeddings.csv")
    parser.add_argument("--out", default="outputs/embedding_frames")
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

    rows = read_csv(frames_path)
    index_rows = read_csv(index_path)
    embedding_index = {row["frame_id"]: int(row["embedding_index"]) for row in index_rows}
    embeddings = np.load(embeddings_path)

    missing = [row["frame_id"] for row in rows if row["frame_id"] not in embedding_index]
    if missing:
        raise SystemExit(f"Missing embedding for frame: {missing[0]}")

    decode_rows = compute_order(rows, embeddings, embedding_index, "decode", "decode_order_index", "decode_time_seconds")
    display_rows = compute_order(rows, embeddings, embedding_index, "display", "display_order_index", "display_time_seconds")

    write_csv(out_dir / "decode_order_embedding_frame_similarities.csv", decode_rows)
    write_csv(out_dir / "display_order_embedding_frame_similarities.csv", display_rows)

    metadata = {
        "similarity": "cosine similarity between two frame embeddings",
        "settings": {
            "previous_i": "cosine_similarity(e(previous I-frame), e(current P/B frame))",
            "previous_adjacent": "cosine_similarity(e(previous adjacent frame), e(current P/B frame))",
        },
        "notes": [
            "I-frames are kept as plot/row anchors but do not get similarity values.",
            "Previous I-frame is an analysis anchor, not a claimed codec reference.",
            "This stage uses existing frame embeddings and does not rerun CLIP.",
        ],
    }
    (out_dir / "embedding_frame_similarity_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote frame embedding similarities to {out_dir}")


if __name__ == "__main__":
    main()
