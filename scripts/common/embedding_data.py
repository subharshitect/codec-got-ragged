"""Shared loader for extracted frame rows and frame embedding arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.tabular import load_embedding_index, read_csv


def load_embedding_inputs(
    frames_path: Path,
    embeddings_path: Path,
    index_path: Path,
) -> tuple[list[dict[str, str]], np.ndarray, dict[str, int]]:
    """Load frames.csv, frame_embeddings.npy, and frame_id -> embedding index mapping."""
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

    return frames, embeddings, embedding_index
