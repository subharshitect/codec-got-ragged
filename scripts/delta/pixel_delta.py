#!/usr/bin/env python3
"""Compute adjacent-frame pixel deltas from decoded frame images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.progress import tqdm
from common.tabular import add_ranks, ordered_rows, read_csv, write_csv


DELTA_FIELDS = [
    "order",
    "order_index",
    "order_time_seconds",
    "frame_id",
    "source_index",
    "pict_type",
    "key_frame",
    "pts_time",
    "dts_time",
    "display_time_seconds",
    "decode_time_seconds",
    "frame_image",
    "previous_frame_id",
    "previous_source_index",
    "previous_frame_image",
    "pixel_delta_mean_abs",
    "delta_rank",
]

def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def pixel_delta(current_path: Path, previous_path: Path) -> float:
    current = read_image(current_path)
    previous = read_image(previous_path)
    if current.shape != previous.shape:
        raise ValueError(f"Frame size mismatch: {current_path} {current.shape} vs {previous_path} {previous.shape}")

    diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
    return float(diff.mean())

def compute_order_deltas(
    rows: list[dict[str, str]],
    extracted_dir: Path,
    order_name: str,
    order_column: str,
    time_column: str,
) -> list[dict[str, str]]:
    ordered = ordered_rows(rows, order_column)
    output = []

    previous = None
    for row in tqdm(ordered, desc=f"{order_name} pixel deltas", unit="frame"):
        delta = None
        if previous:
            delta = pixel_delta(extracted_dir / row["frame_image"], extracted_dir / previous["frame_image"])

        output.append(
            {
                "order": order_name,
                "order_index": row.get(order_column, ""),
                "order_time_seconds": row.get(time_column, ""),
                "frame_id": row.get("frame_id", ""),
                "source_index": row.get("source_index", ""),
                "pict_type": row.get("pict_type", ""),
                "key_frame": row.get("key_frame", ""),
                "pts_time": row.get("pts_time", ""),
                "dts_time": row.get("dts_time", ""),
                "display_time_seconds": row.get("display_time_seconds", ""),
                "decode_time_seconds": row.get("decode_time_seconds", ""),
                "frame_image": row.get("frame_image", ""),
                "previous_frame_id": previous.get("frame_id", "") if previous else "",
                "previous_source_index": previous.get("source_index", "") if previous else "",
                "previous_frame_image": previous.get("frame_image", "") if previous else "",
                "pixel_delta_mean_abs": "" if delta is None else f"{delta:.6f}",
                "delta_rank": "",
            }
        )
        previous = row

    add_ranks(output, "pixel_delta_mean_abs", "delta_rank")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/extracted/frames.csv")
    parser.add_argument("--out", default="outputs/deltas")
    args = parser.parse_args()

    input_path = Path(args.input)
    extracted_dir = input_path.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Missing extracted frames file: {input_path}")

    rows = read_csv(input_path)
    for row in tqdm(rows, desc="check frame images", unit="frame"):
        frame_image = row.get("frame_image")
        if not frame_image:
            raise SystemExit("frames.csv is missing frame_image paths; rerun make extract")
        if not (extracted_dir / frame_image).exists():
            raise SystemExit(f"Missing decoded frame image: {extracted_dir / frame_image}")

    decode_deltas = compute_order_deltas(
        rows,
        extracted_dir,
        "decode",
        "decode_order_index",
        "decode_time_seconds",
    )
    display_deltas = compute_order_deltas(
        rows,
        extracted_dir,
        "display",
        "display_order_index",
        "display_time_seconds",
    )

    write_csv(out_dir / "decode_order_deltas.csv", decode_deltas, DELTA_FIELDS)
    write_csv(out_dir / "display_order_deltas.csv", display_deltas, DELTA_FIELDS)

    metadata = {
        "delta_formula": "mean(abs(current_frame_pixels - previous_frame_pixels)) over decoded RGB frames.",
        "delta_types": {
            "decode_order_deltas.csv": "Adjacent-frame pixel deltas in decode/coding order.",
            "display_order_deltas.csv": "Adjacent-frame pixel deltas in display/presentation order.",
        },
        "notes": [
            "order_time_seconds is the x-axis used by plots.",
            "No keyframe-anchor dependency is guessed.",
            "Reference-frame deltas are intentionally not computed unless real reference frames are extracted later.",
        ],
    }
    (out_dir / "delta_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote delta outputs to {out_dir}")


if __name__ == "__main__":
    main()
