#!/usr/bin/env python3
"""Create CLIP embeddings for decoded frame images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.progress import tqdm


INDEX_FIELDS = [
    "embedding_index",
    "frame_id",
    "source_index",
    "display_order_index",
    "decode_order_index",
    "pict_type",
    "key_frame",
    "frame_image",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="outputs/extracted/frames.csv")
    parser.add_argument("--out", default="outputs/embeddings")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    frames_path = Path(args.frames)
    extracted_dir = frames_path.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not frames_path.exists():
        raise SystemExit(f"Missing frames file: {frames_path}")

    rows = read_csv(frames_path)
    image_paths = [extracted_dir / row["frame_image"] for row in rows]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing frame image: {missing[0]}")

    device = choose_device(args.device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model, use_safetensors=True).to(device)
    model.eval()

    batches = []
    with torch.no_grad():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="embed frames", unit="batch"):
            batch_paths = image_paths[start : start + args.batch_size]
            images = load_images(batch_paths)
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            batches.append(features.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)
    np.save(out_dir / "frame_embeddings.npy", embeddings)

    index_rows = []
    for index, row in enumerate(tqdm(rows, desc="build embedding index", unit="frame")):
        index_rows.append(
            {
                "embedding_index": str(index),
                "frame_id": row.get("frame_id", ""),
                "source_index": row.get("source_index", ""),
                "display_order_index": row.get("display_order_index", ""),
                "decode_order_index": row.get("decode_order_index", ""),
                "pict_type": row.get("pict_type", ""),
                "key_frame": row.get("key_frame", ""),
                "frame_image": row.get("frame_image", ""),
            }
        )
    write_csv(out_dir / "frame_embeddings.csv", index_rows)

    metadata = {
        "model": args.model,
        "device": device,
        "batch_size": args.batch_size,
        "frames": len(rows),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "normalized": True,
        "source_frames": str(frames_path),
    }
    (out_dir / "embedding_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote frame embeddings to {out_dir}")


if __name__ == "__main__":
    main()
