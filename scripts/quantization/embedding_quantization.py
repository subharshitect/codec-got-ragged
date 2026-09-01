#!/usr/bin/env python3
"""Run PQ and RaBitQ baselines on frame embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.progress import tqdm


FIELDS = [
    "method",
    "setting",
    "n_vectors",
    "d_model",
    "original_bytes",
    "code_size_bytes_per_vector",
    "code_bytes",
    "compression_ratio",
    "mse",
    "relative_reconstruction_error",
    "recall_at_k",
    "query_count",
    "k_neighbors",
    "index_file",
    "index_file_bytes",
    "status",
]


def parse_ints(value: str) -> list[int]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Values must be positive integers.")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return sorted(set(values))


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def format_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.8f}"


def recall_at_k(exact_indices: np.ndarray, approx_indices: np.ndarray, k: int) -> float:
    if len(exact_indices) == 0 or k == 0:
        return 0.0

    total = 0
    for exact_row, approx_row in zip(exact_indices, approx_indices):
        total += len(set(exact_row[:k]) & set(approx_row[:k]))
    return total / (len(exact_indices) * k)


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float | None]:
    mse = float(np.mean((original - reconstructed) ** 2))
    denominator = float(np.linalg.norm(original, ord="fro"))
    if denominator == 0:
        return mse, None
    relative_error = float(np.linalg.norm(original - reconstructed, ord="fro") / denominator)
    return mse, relative_error


def exact_neighbors(x: np.ndarray, query_count: int, k_neighbors: int) -> tuple[np.ndarray, int, int]:
    import faiss

    n_vectors, d_model = x.shape
    query_count = min(query_count, n_vectors)
    k_neighbors = min(k_neighbors, n_vectors)

    exact = faiss.IndexFlatL2(d_model)
    exact.add(x)
    _, exact_indices = exact.search(x[:query_count], k_neighbors)
    return exact_indices, query_count, k_neighbors


def maybe_write_index(index, path: Path, save_indexes: bool) -> tuple[str, str]:
    if not save_indexes:
        return "", ""

    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return str(path), str(path.stat().st_size)


def base_row(method: str, setting: str, x: np.ndarray, original_bytes: int, query_count: int, k_neighbors: int) -> dict[str, str]:
    n_vectors, d_model = x.shape
    return {
        "method": method,
        "setting": setting,
        "n_vectors": str(n_vectors),
        "d_model": str(d_model),
        "original_bytes": str(original_bytes),
        "query_count": str(query_count),
        "k_neighbors": str(k_neighbors),
    }


def run_pq(
    x: np.ndarray,
    m_values: list[int],
    nbits: int,
    exact_indices: np.ndarray,
    query_count: int,
    k_neighbors: int,
    out_dir: Path,
    save_indexes: bool,
) -> list[dict[str, str]]:
    import faiss

    n_vectors, d_model = x.shape
    original_bytes = int(x.nbytes)
    rows = []

    for m_value in tqdm(m_values, desc="PQ baselines", unit="M"):
        setting = f"M={m_value},nbits={nbits}"
        row = base_row("pq", setting, x, original_bytes, query_count, k_neighbors)

        if d_model % m_value != 0:
            row["status"] = "skipped_must_divide_d_model"
            rows.append(row)
            continue

        if n_vectors < 2**nbits:
            row["status"] = "skipped_not_enough_training_vectors"
            rows.append(row)
            continue

        pq = faiss.IndexPQ(d_model, m_value, nbits)
        pq.train(x)
        pq.add(x)

        codes = faiss.vector_to_array(pq.codes)
        reconstructed = pq.reconstruct_n(0, n_vectors)
        mse, relative_error = reconstruction_metrics(x, reconstructed)
        _, pq_indices = pq.search(x[:query_count], k_neighbors)
        recall = recall_at_k(exact_indices, pq_indices, k_neighbors)
        index_file, index_file_bytes = maybe_write_index(pq, out_dir / "indexes" / f"pq_M{m_value}_nbits{nbits}.index", save_indexes)

        row.update(
            {
                "code_size_bytes_per_vector": str(pq.code_size),
                "code_bytes": str(codes.nbytes),
                "compression_ratio": format_float(original_bytes / codes.nbytes if codes.nbytes else None),
                "mse": format_float(mse),
                "relative_reconstruction_error": format_float(relative_error),
                "recall_at_k": format_float(recall),
                "index_file": index_file,
                "index_file_bytes": index_file_bytes,
                "status": "ok",
            }
        )
        rows.append(row)

    return rows


def run_rabitq(
    x: np.ndarray,
    qb_values: list[int],
    exact_indices: np.ndarray,
    query_count: int,
    k_neighbors: int,
    out_dir: Path,
    save_indexes: bool,
) -> list[dict[str, str]]:
    import faiss

    n_vectors, d_model = x.shape
    original_bytes = int(x.nbytes)
    rows = []

    rabitq = faiss.IndexRaBitQ(d_model)
    rabitq.train(x)
    rabitq.add(x)

    codes = faiss.vector_to_array(rabitq.codes)
    reconstructed = rabitq.reconstruct_n(0, n_vectors)
    mse, relative_error = reconstruction_metrics(x, reconstructed)

    for qb in tqdm(qb_values, desc="RaBitQ baselines", unit="qb"):
        row = base_row("rabitq", f"qb={qb}", x, original_bytes, query_count, k_neighbors)

        params = faiss.RaBitQSearchParameters()
        params.qb = qb
        params.centered = rabitq.centered
        _, rabitq_indices = rabitq.search(x[:query_count], k_neighbors, params=params)
        recall = recall_at_k(exact_indices, rabitq_indices, k_neighbors)

        rabitq.qb = qb
        index_file, index_file_bytes = maybe_write_index(rabitq, out_dir / "indexes" / f"rabitq_qb{qb}.index", save_indexes)

        row.update(
            {
                "code_size_bytes_per_vector": str(rabitq.code_size),
                "code_bytes": str(codes.nbytes),
                "compression_ratio": format_float(original_bytes / codes.nbytes if codes.nbytes else None),
                "mse": format_float(mse),
                "relative_reconstruction_error": format_float(relative_error),
                "recall_at_k": format_float(recall),
                "index_file": index_file,
                "index_file_bytes": index_file_bytes,
                "status": "ok",
            }
        )
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="outputs/embeddings/frame_embeddings.npy")
    parser.add_argument("--out", default="outputs/quantization")
    parser.add_argument("--pq-m", type=parse_ints, default=parse_ints("16,32,64,128,256"))
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--rabitq-qb", type=parse_ints, default=parse_ints("1,2,3,4,5,6,7,8"))
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--save-indexes", type=parse_bool, default=False)
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        raise SystemExit(f"Missing embeddings file: {embeddings_path}")
    if args.pq_nbits <= 0:
        raise SystemExit("--pq-nbits must be positive")
    if args.query_count <= 0:
        raise SystemExit("--query-count must be positive")
    if args.k_neighbors <= 0:
        raise SystemExit("--k-neighbors must be positive")

    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing dependency: faiss") from exc

    x = np.ascontiguousarray(np.load(embeddings_path).astype("float32"))
    if x.ndim != 2:
        raise SystemExit(f"Expected a 2D embedding array, got shape {x.shape}")

    exact_indices, query_count, k_neighbors = exact_neighbors(x, args.query_count, args.k_neighbors)

    pq_rows = run_pq(x, args.pq_m, args.pq_nbits, exact_indices, query_count, k_neighbors, out_dir, args.save_indexes)
    rabitq_rows = run_rabitq(x, args.rabitq_qb, exact_indices, query_count, k_neighbors, out_dir, args.save_indexes)
    all_rows = pq_rows + rabitq_rows

    write_csv(out_dir / "pq_results.csv", pq_rows)
    write_csv(out_dir / "rabitq_results.csv", rabitq_rows)
    write_csv(out_dir / "quantization_results.csv", all_rows)

    metadata = {
        "input_embeddings": str(embeddings_path),
        "n_vectors": int(x.shape[0]),
        "d_model": int(x.shape[1]),
        "dtype": "float32",
        "original_bytes": int(x.nbytes),
        "pq_m": args.pq_m,
        "pq_nbits": args.pq_nbits,
        "rabitq_qb": args.rabitq_qb,
        "query_count": query_count,
        "k_neighbors": k_neighbors,
        "save_indexes": args.save_indexes,
        "compression_ratio": "original embedding bytes / quantized code bytes",
        "reconstruction_error": {
            "mse": "mean((x - reconstructed_x)^2)",
            "relative_reconstruction_error": "||x - reconstructed_x||_F / ||x||_F",
        },
        "recall_at_k": "overlap between exact IndexFlatL2 neighbors and approximate quantized-index neighbors",
    }
    (out_dir / "quantization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote quantization outputs to {out_dir}")


if __name__ == "__main__":
    main()
