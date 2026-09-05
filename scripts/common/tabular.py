"""Small CSV, numeric, ordering, and ranking helpers for pipeline scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from common.progress import tqdm


def number(value: object) -> float | None:
    """Parse a CSV scalar into a float; blank-like values stay missing."""
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float | None) -> str:
    """Write float CSV values with the pipeline's current 8-decimal convention."""
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV stage artifact as row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    """Write row dictionaries to a CSV using an explicit stable field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow({field: row.get(field, "") for field in fields})


def ordered_rows(rows: list[dict[str, str]], order_column: str) -> list[dict[str, str]]:
    """Return frame rows sorted by a chosen order column.

    Input:
        rows: CSV-style frame rows.
        order_column: usually display_order_index or decode_order_index.

    Output:
        A new sorted list. Rows with missing order values are placed last, and
        source_index is used as a stable fallback/tiebreaker.

    Example:
        rows = [
            {"frame_id": "f2", "display_order_index": "2", "source_index": "0"},
            {"frame_id": "f0", "display_order_index": "0", "source_index": "1"},
            {"frame_id": "f1", "display_order_index": "1", "source_index": "2"},
        ]
        ordered_rows(rows, "display_order_index") returns rows in frame_id order:
        f0, f1, f2
    """
    return sorted(
        rows,
        key=lambda row: (
            number(row.get(order_column)) is None,
            number(row.get(order_column)) if number(row.get(order_column)) is not None else math.inf,
            number(row.get("source_index")) or 0,
        ),
    )


def add_ranks(rows: list[dict[str, str]], value_column: str, rank_column: str) -> None:
    """Add descending ranks in-place; rank 1 is the largest numeric value."""
    ranked = [
        (index, number(row.get(value_column)))
        for index, row in enumerate(rows)
        if number(row.get(value_column)) is not None
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    for rank, (index, _) in enumerate(ranked, start=1):
        rows[index][rank_column] = str(rank)


def load_embedding_index(index_path: Path) -> dict[str, int]:
    """Load frame_id -> embedding row index from frame_embeddings.csv."""
    return {row["frame_id"]: int(row["embedding_index"]) for row in read_csv(index_path)}
