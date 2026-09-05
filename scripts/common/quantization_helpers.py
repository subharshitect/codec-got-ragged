"""PQ and RaBitQ helpers for embedding retrieval baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.progress import tqdm
from common.tabular import format_float


def parse_ints(value: str) -> list[int]:
    """Parse comma-separated positive integers from Makefile/CLI values."""
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
    """Parse simple boolean flags passed as 0/1 or true/false strings."""
    return value.strip().lower() in {"1", "true", "yes", "y"}


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float | None]:
    """Compute MSE and relative Frobenius reconstruction error."""
    mse = float(np.mean((original - reconstructed) ** 2))
    denominator = float(np.linalg.norm(original, ord="fro"))
    if denominator == 0:
        return mse, None
    relative_error = float(np.linalg.norm(original - reconstructed, ord="fro") / denominator)
    return mse, relative_error


def maybe_write_index(index, path: Path, save_indexes: bool) -> tuple[str, str]:
    """Optionally persist a FAISS index and return its path/byte size."""
    if not save_indexes:
        return "", ""

    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return str(path), str(path.stat().st_size)


def base_stats(
    method: str,
    variant: str,
    x: np.ndarray,
    query_count: int,
    k_neighbors: int,
) -> dict[str, str]:
    """Create shared retrieval-result fields for quantized baselines."""
    n_vectors, d_model = x.shape
    return {
        "method": method,
        "order": "global",
        "variant": variant,
        "epsilon": "",
        "query_count": str(query_count),
        "k_neighbors": str(k_neighbors),
        "n_vectors": str(n_vectors),
        "d_model": str(d_model),
        "original_bytes": str(int(x.nbytes)),
    }


def run_pq_retrieval(
    x: np.ndarray,
    queries: np.ndarray,
    m_values: list[int],
    nbits: int,
    query_count: int,
    k_neighbors: int,
    search_k: int,
    out_dir: Path,
    save_indexes: bool,
) -> list[dict[str, object]]:
    """Train/search PQ indexes over the full embedding database."""
    import faiss

    n_vectors, d_model = x.shape
    outputs = []

    for m_value in tqdm(m_values, desc="PQ retrieval baselines", unit="M"):
        variant = f"M={m_value},nbits={nbits}"
        stats = base_stats("pq", variant, x, query_count, k_neighbors)

        if d_model % m_value != 0:
            stats["status"] = "skipped_must_divide_d_model"
            outputs.append({"stats": stats, "scores": None, "indices": None})
            continue

        if n_vectors < 2**nbits: # some PQ concept about centroids to be atleast 2^8
            stats["status"] = "skipped_not_enough_training_vectors"
            outputs.append({"stats": stats, "scores": None, "indices": None})
            continue

        pq = faiss.IndexPQ(d_model, m_value, nbits, faiss.METRIC_INNER_PRODUCT) # creates PQ inner product index
        pq.train(x) # 
        pq.add(x)

        scores, indices = pq.search(queries, search_k)
        codes = faiss.vector_to_array(pq.codes)
        reconstructed = pq.reconstruct_n(0, n_vectors)
        mse, relative_error = reconstruction_metrics(x, reconstructed)
        index_file, index_file_bytes = maybe_write_index(
            pq,
            out_dir / "indexes" / f"pq_M{m_value}_nbits{nbits}.index",
            save_indexes,
        )

        stats.update(
            {
                "compression_ratio": format_float(x.nbytes / codes.nbytes if codes.nbytes else None),
                "code_size_bytes_per_vector": str(pq.code_size),
                "code_bytes": str(codes.nbytes),
                "mse": format_float(mse),
                "relative_reconstruction_error": format_float(relative_error),
                "index_file": index_file,
                "index_file_bytes": index_file_bytes,
                "status": "ok",
            }
        )
        outputs.append({"stats": stats, "scores": scores, "indices": indices})

    return outputs


def run_rabitq_retrieval(
    x: np.ndarray,
    queries: np.ndarray,
    qb_values: list[int],
    query_count: int,
    k_neighbors: int,
    search_k: int,
    out_dir: Path,
    save_indexes: bool,
) -> list[dict[str, object]]:
    """Train/search one RaBitQ index over the full embedding database."""
    import faiss

    n_vectors, d_model = x.shape
    outputs = []

    rabitq = faiss.IndexRaBitQ(d_model)
    rabitq.train(x)
    rabitq.add(x)

    codes = faiss.vector_to_array(rabitq.codes)
    reconstructed = rabitq.reconstruct_n(0, n_vectors)
    mse, relative_error = reconstruction_metrics(x, reconstructed)

    for qb in tqdm(qb_values, desc="RaBitQ retrieval baselines", unit="qb"):
        variant = f"qb={qb}"
        stats = base_stats("rabitq", variant, x, query_count, k_neighbors)

        params = faiss.RaBitQSearchParameters()
        params.qb = qb
        params.centered = rabitq.centered
        distances, indices = rabitq.search(queries, search_k, params=params)

        rabitq.qb = qb
        index_file, index_file_bytes = maybe_write_index(
            rabitq,
            out_dir / "indexes" / f"rabitq_qb{qb}.index",
            save_indexes,
        )

        stats.update(
            {
                "compression_ratio": format_float(x.nbytes / codes.nbytes if codes.nbytes else None),
                "code_size_bytes_per_vector": str(rabitq.code_size),
                "code_bytes": str(codes.nbytes),
                "mse": format_float(mse),
                "relative_reconstruction_error": format_float(relative_error),
                "index_file": index_file,
                "index_file_bytes": index_file_bytes,
                "status": "ok",
            }
        )
        outputs.append({"stats": stats, "scores": 1.0 - (distances / 2.0), "indices": indices})

    return outputs
