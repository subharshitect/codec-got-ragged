#!/usr/bin/env python3
"""Evaluate retrieval quality for SVD, PQ, and RaBitQ compressed embeddings."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.embedding_data import load_embedding_inputs
from common.progress import tqdm
from common.quantization_helpers import parse_bool, parse_ints, run_pq_retrieval, run_rabitq_retrieval
from common.svd_helpers import build_matrices, inter_i_segments
from common.tabular import format_float, number, read_csv, write_csv


QUERY_FIELDS = [
    "query_index",
    "embedding_index",
    "frame_id",
    "source_index",
    "display_order_index",
    "decode_order_index",
    "pict_type",
    "key_frame",
    "frame_image",
]

RESULT_FIELDS = [
    "method",
    "order",
    "variant",
    "epsilon",
    "compression_ratio",
    "recall_at_k",
    "query_count",
    "k_neighbors",
    "n_vectors",
    "d_model",
    "n_segments",
    "total_reconstructed_frames",
    "mean_selected_k",
    "median_selected_k",
    "weighted_mean_achieved_error",
    "original_params",
    "raw_params",
    "svd_params",
    "compressed_params",
    "original_bytes",
    "code_size_bytes_per_vector",
    "code_bytes",
    "mse",
    "relative_reconstruction_error",
    "index_file",
    "index_file_bytes",
    "status",
]

NEIGHBOR_FIELDS = [
    "method",
    "order",
    "variant",
    "epsilon",
    "query_index",
    "query_embedding_index",
    "rank",
    "neighbor_embedding_index",
    "cosine_similarity",
]

ORDERS = [
    ("decode", "decode_order_index"),
    ("display", "display_order_index"),
]


def select_query_indices(n_vectors: int, query_count: int, seed: int) -> np.ndarray:
    """Choose a reproducible random query set from embedding row indices."""
    if query_count <= 0:
        raise SystemExit("--query-count must be positive")

    rng = np.random.default_rng(seed)
    count = min(query_count, n_vectors)
    return rng.choice(n_vectors, size=count, replace=False)


def query_rows(index_path: Path, query_indices: np.ndarray) -> list[dict[str, str]]:
    """Create a human-readable query-id table from the embedding index CSV."""
    rows = read_csv(index_path)
    rows_by_index = {int(row["embedding_index"]): row for row in rows}
    output = []

    for query_index, embedding_index in enumerate(query_indices):
        row = rows_by_index[int(embedding_index)]
        output.append(
            {
                "query_index": str(query_index),
                "embedding_index": str(int(embedding_index)),
                "frame_id": row.get("frame_id", ""),
                "source_index": row.get("source_index", ""),
                "display_order_index": row.get("display_order_index", ""),
                "decode_order_index": row.get("decode_order_index", ""),
                "pict_type": row.get("pict_type", ""),
                "key_frame": row.get("key_frame", ""),
                "frame_image": row.get("frame_image", ""),
            }
        )

    return output


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize rows so FAISS inner product is cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x.copy()
    np.divide(normalized, norms, out=normalized, where=norms > 0)
    return np.ascontiguousarray(normalized.astype("float32"))


def search_index(
    x: np.ndarray,
    queries: np.ndarray,
    query_indices: np.ndarray,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact cosine retrieval and remove each query's self-match."""
    import faiss

    if k_neighbors <= 0:
        raise SystemExit("--k-neighbors must be positive")
    if x.shape[0] <= 1:
        raise SystemExit("Need at least two vectors when excluding self matches")

    k = min(k_neighbors, x.shape[0] - 1)  # -1 because self-matches are excluded
    search_k = min(x.shape[0], k + 1)  # ask FAISS for one extra possible self-match to compensate for the self-match removal
    index = faiss.IndexFlatIP(x.shape[1])  # means: I'm going to store vectors containing d_model (s.shape[1]) numbers each, and when I search, compare them using dot-product/inner-product and search using brute force (flat) approach
    index.add(normalize_rows(x.astype("float32")))  # here is where i add the vectors to the index, and normalize them so that inner product is equivalent to cosine similarity
    raw_scores, raw_indices = index.search(normalize_rows(queries.astype("float32")), search_k) # i query here

    return filter_self_matches(raw_scores, raw_indices, query_indices, k) # removes the self-match and returns the top-k neighbors for each query


def filter_self_matches(
    raw_scores: np.ndarray,
    raw_indices: np.ndarray,
    query_indices: np.ndarray,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each query's own embedding index from returned neighbors."""
    scores = np.full((len(query_indices), k_neighbors), np.nan, dtype=np.float32)
    indices = np.full((len(query_indices), k_neighbors), -1, dtype=np.int64)
    for query_row_index, self_index in enumerate(query_indices):  # iterate query rows
        output_rank = 0
        for score, neighbor_index in zip(raw_scores[query_row_index], raw_indices[query_row_index]):
            # Exclude the query frame itself so recall is not inflated by trivial self-matches.
            if int(neighbor_index) == int(self_index):
                continue
            scores[query_row_index, output_rank] = score
            indices[query_row_index, output_rank] = neighbor_index
            output_rank += 1
            if output_rank == k_neighbors:
                break

    return scores, indices


def recall_at_k(reference_indices: np.ndarray, compressed_indices: np.ndarray, k_neighbors: int) -> float:
    """Measure mean top-k overlap against the uncompressed reference results."""
    if len(reference_indices) == 0 or k_neighbors == 0:
        return 0.0

    k = min(k_neighbors, reference_indices.shape[1], compressed_indices.shape[1])
    total = 0

    # iters over each queries' top-k neighbors 
    for reference_row, compressed_row in zip(reference_indices, compressed_indices):
        reference_set = {int(index) for index in reference_row[:k] if int(index) >= 0}
        compressed_set = {int(index) for index in compressed_row[:k] if int(index) >= 0}
        total += len(reference_set & compressed_set)  # intersection
    return total / (len(reference_indices) * k)  # len(reference_indices) = number of queries


def neighbor_rows(
    method: str,
    order_name: str,
    variant: str,
    epsilon: str,
    query_indices: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
) -> list[dict[str, str]]:
    """Flatten FAISS neighbor arrays into auditable CSV rows."""
    rows = []
    for query_index, query_embedding_index in enumerate(query_indices):
        for rank, (score, neighbor_index) in enumerate(zip(scores[query_index], indices[query_index]), start=1):
            rows.append(
                {
                    "method": method,
                    "order": order_name,
                    "variant": variant,
                    "epsilon": epsilon,
                    "query_index": str(query_index),
                    "query_embedding_index": str(int(query_embedding_index)),
                    "rank": str(rank),
                    "neighbor_embedding_index": "" if int(neighbor_index) < 0 else str(int(neighbor_index)),
                    "cosine_similarity": format_float(float(score)) if np.isfinite(score) else "",
                }
            )
    return rows


def svd_reconstruct(u_matrix: np.ndarray, singular_values: np.ndarray, vt_matrix: np.ndarray, rank: int) -> np.ndarray:
    """Rebuild a rank-k approximation from cached SVD factors."""
    return ((u_matrix[:, :rank] * singular_values[:rank]) @ vt_matrix[:rank, :]).astype(np.float32)


def build_svd_matrix_entries(
    frames: list[dict[str, str]],
    embeddings: np.ndarray,
    embedding_index: dict[str, int],
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Precompute segment matrices and SVD factors used by each epsilon group.

    Output example:
        {
            ("display", "display_000003", "embedding_delta_adjacent"): {
                "segment": {
                    "segment_id": "display_000003",
                    "start_i": row_for_previous_i_frame,
                    "end_i": row_for_next_i_frame,
                    "rows": [row_for_B_or_P_frame, ...],
                    "ordered_rows": all_frames_in_this_order,
                },
                "matrix_shape": (N_i, d_model),
                "u_matrix": np.ndarray,
                "singular_values": np.ndarray,
                "vt_matrix": np.ndarray,
            }
        }

    Note:
        The original matrix is not kept here. Rank-k reconstruction only needs
        U, S, Vt, plus matrix_shape for rank bounds.
    """
    entries = {}

    for order_name, order_column in ORDERS:
        segments = inter_i_segments(frames, order_name, order_column)
        for segment in tqdm(segments, desc=f"{order_name} retrieval matrices", unit="segment"):
            variant_matrices: list[tuple[str, np.ndarray]] = build_matrices(order_name, segment, embeddings, embedding_index)
            for variant, matrix in variant_matrices:
                u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
                key = (order_name, str(segment["segment_id"]), variant)
                entries[key] = {
                    "segment": segment,
                    "matrix_shape": matrix.shape,
                    "u_matrix": u_matrix,
                    "singular_values": singular_values,
                    "vt_matrix": vt_matrix,
                }

    return entries


def svd_error_groups(svd_error_path: Path) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    """Group successful selected-rank rows by order, variant, and epsilon.

    Input CSV row example:
        {
            "order": "display",
            "variant": "embedding_delta_adjacent",
            "segment_id": "display_000003",
            "start_i_frame_id": "f000420",
            "end_i_frame_id": "f000511",
            "n_frames": "90",
            "d_model": "512",
            "epsilon": "0.05000000",
            "selected_k": "42",
            "achieved_error": "0.04910000",
            "svd_params": "25326",
            "compression_ratio": "1.81910000",
            "status": "met",
        }

    Output example:
        {
            ("display", "embedding_delta_adjacent", "0.05000000"): [
                {
                    "segment_id": "display_000003",
                    "n_frames": "90",
                    "selected_k": "42",
                    "achieved_error": "0.04910000",
                    "svd_params": "25326",
                    "status": "met",
                },
                ...
            ]
        }

    The output list keeps the full CSV rows; only a few important fields are
    shown above. See outputs/embedding_svd_error/segment_svd_error.csv.
    """
    if not svd_error_path.exists():
        raise SystemExit(f"Missing SVD error file: {svd_error_path}")

    groups = defaultdict(list)
    for row in read_csv(svd_error_path):
        if row.get("status") != "met" or not row.get("selected_k"):
            continue
        groups[(row["order"], row["variant"], row["epsilon"])].append(row)

    return dict(groups)


def reconstruct_database_for_group(
    x: np.ndarray,
    embedding_index: dict[str, int],
    matrix_entries: dict[tuple[str, str, str], dict[str, object]],
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, dict[str, object]]:
    """Create one full reconstructed embedding database for an SVD setting."""
    n_vectors, d_model = x.shape
    original_params = n_vectors * d_model
    reconstructed = x.copy() # this keeps the i-frames untouched and manipulates the B/P frames based on the variant
    replaced_indices = set()
    selected_ks = []
    achieved_errors = []
    weighted_error_sum = 0.0
    weighted_error_frames = 0
    total_svd_params = 0

    for row in rows: # iterating over the segments in this group, and reconstructing the B/P frames based on the variant
        key = (row["order"], row["segment_id"], row["variant"])
        entry = matrix_entries.get(key)
        if entry is None:
            continue

        rank = int(row["selected_k"])
        matrix_shape = entry["matrix_shape"]
        u_matrix = entry["u_matrix"]
        singular_values = entry["singular_values"]
        vt_matrix = entry["vt_matrix"]
        segment = entry["segment"]
        assert isinstance(matrix_shape, tuple)
        assert isinstance(u_matrix, np.ndarray)
        assert isinstance(singular_values, np.ndarray)
        assert isinstance(vt_matrix, np.ndarray)
        assert isinstance(segment, dict)

        rank = min(rank, min(matrix_shape))
        reconstructed_matrix = svd_reconstruct(u_matrix, singular_values, vt_matrix, rank)
        segment_rows = segment["rows"]
        start_i = segment["start_i"]
        assert isinstance(segment_rows, list)
        assert isinstance(start_i, dict)

        """
        In the "embedding_delta_adjacent" variant, each row is reconstructed
        sequentially using the previous reconstructed row as an anchor. In other
        variants, each row is reconstructed independently.

            e.g.                I0, B1, P2
            where,              I0, deltaB1 = I0 - B1, deltaP2 = B1 - P2
            reconstruction:     I0, recon(B1) = I0 - deltaB1,
                                recon(P2) = recon(B1) - deltaP2 ... keep going sequentially
        """
        if row["variant"] == "embedding_delta_adjacent":
            # Adjacent deltas must decode sequentially; original B/P anchors are not reused.
            anchor = x[embedding_index[start_i["frame_id"]]]
            for matrix_row_index, frame_row in enumerate(segment_rows):
                frame_id = frame_row["frame_id"]
                output_index = embedding_index[frame_id]
                reconstructed_delta = reconstructed_matrix[matrix_row_index]
                reconstructed_frame = anchor - reconstructed_delta
                reconstructed[output_index] = reconstructed_frame
                anchor = reconstructed_frame
                replaced_indices.add(output_index)
        else:
            for matrix_row_index, frame_row in enumerate(segment_rows):
                frame_id = frame_row["frame_id"]
                output_index = embedding_index[frame_id]

                if row["variant"] == "embedding_frames":
                    reconstructed[output_index] = reconstructed_matrix[matrix_row_index]
                elif row["variant"] == "embedding_delta_previous_i":
                    anchor = x[embedding_index[start_i["frame_id"]]]
                    reconstructed_delta = reconstructed_matrix[matrix_row_index]
                    reconstructed[output_index] = anchor - reconstructed_delta
                else:
                    continue

                replaced_indices.add(output_index)

        selected_ks.append(rank)  # per segment-variant-order selected k values for mean/median
        achieved_error = number(row.get("achieved_error"))  # segment reconstruction error from SVD-error stage
        frame_count = int(row["n_frames"])  # number of P/B rows in this segment matrix
        if achieved_error is not None:
            achieved_errors.append(achieved_error)  # keep unweighted segment errors
            weighted_error_sum += achieved_error * frame_count  # weight larger segments more
            weighted_error_frames += frame_count  # denominator for frame-weighted error
        total_svd_params += int(row["svd_params"])  # compressed params for this segment's rank-k SVD

    raw_params = (n_vectors - len(replaced_indices)) * d_model  # untouched frames stored as raw embeddings like I-frames
    compressed_params = raw_params + total_svd_params  # full compressed database cost
    compression_ratio = original_params / compressed_params if compressed_params else None  # higher means smaller storage
    weighted_error = weighted_error_sum / weighted_error_frames if weighted_error_frames else None  # per-frame segment error

    stats = {
        "n_segments": len(rows),
        "total_reconstructed_frames": len(replaced_indices),
        "mean_selected_k": statistics.fmean(selected_ks) if selected_ks else None,
        "median_selected_k": statistics.median(selected_ks) if selected_ks else None,
        "weighted_mean_achieved_error": weighted_error,
        "original_params": original_params,
        "raw_params": raw_params,
        "svd_params": total_svd_params,
        "compressed_params": compressed_params,
        "compression_ratio": compression_ratio,
    }
    return reconstructed, stats


def evaluate_svd_groups(
    x: np.ndarray,
    queries: np.ndarray,
    query_indices: np.ndarray,
    reference_indices: np.ndarray,
    frames: list[dict[str, str]],
    embedding_index: dict[str, int],
    svd_error_path: Path,
    query_count: int,
    k_neighbors: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Evaluate retrieval recall for every order/variant/epsilon database."""

    # matrix entries are precomputed SVD factors for each segment 
    # keyed by (order, segment_id, variant)
    # where: order[decode, display], segment_id[display_000003], variant[embedding_frames, embedding_delta_previous_i, embedding_delta_adjacent]
    matrix_entries: dict[tuple[str, str, str], dict[str, object]] = build_svd_matrix_entries(frames, x, embedding_index)

    # group the SVD-error rows by (order, variant, epsilon) for evaluation
    # for: display, embedding_frames, epsilon=0.05
    # we get: (segment_0 -> n_frames=90, selected_k=4), (segment_1 -> n_frames=148, selected_k=6) ... look inside
    groups = svd_error_groups(svd_error_path)
    result_rows = []
    compressed_neighbor_rows = []

    for (order_name, variant, epsilon), rows in tqdm(sorted(groups.items()), desc="SVD retrieval", unit="database"):
        # receives the reconstructed database for "this order/variant/epsilon group" (and stats)
        reconstructed, stats = reconstruct_database_for_group(x, embedding_index, matrix_entries, rows) # for one group "row/rows" are composed of the segments forming the whole video-embeddings
        # reconstructed: [num_frames, d_model]; full "video" embedding database (per group)

        compressed_scores, compressed_indices = search_index(reconstructed, queries, query_indices, k_neighbors) # received scores for the group
        # compressed_scores:[query_count, k], compressed_indices:[query_count, k]

        recall = recall_at_k(reference_indices, compressed_indices, k_neighbors)
        compressed_neighbor_rows.extend(
            neighbor_rows("svd", order_name, variant, epsilon, query_indices, compressed_scores, compressed_indices)
        )

        result_rows.append(
            {
                "method": "svd",
                "order": order_name,
                "variant": variant,
                "epsilon": epsilon,
                "compression_ratio": format_float(stats["compression_ratio"]),
                "recall_at_k": format_float(recall),
                "query_count": str(query_count),
                "k_neighbors": str(min(k_neighbors, x.shape[0] - 1)),
                "n_vectors": str(x.shape[0]),
                "d_model": str(x.shape[1]),
                "n_segments": str(stats["n_segments"]),
                "total_reconstructed_frames": str(stats["total_reconstructed_frames"]),
                "mean_selected_k": format_float(stats["mean_selected_k"]),
                "median_selected_k": format_float(stats["median_selected_k"]),
                "weighted_mean_achieved_error": format_float(stats["weighted_mean_achieved_error"]),
                "original_params": str(stats["original_params"]),
                "raw_params": str(stats["raw_params"]),
                "svd_params": str(stats["svd_params"]),
                "compressed_params": str(stats["compressed_params"]),
                "status": "ok",
            }
        )

    return result_rows, compressed_neighbor_rows


def evaluate_quantization_baselines(
    x: np.ndarray,
    queries: np.ndarray,
    query_indices: np.ndarray,
    reference_indices: np.ndarray,
    pq_m: list[int],
    pq_nbits: int,
    rabitq_qb: list[int],
    query_count: int,
    k_neighbors: int,
    out_dir: Path,
    save_indexes: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Evaluate PQ/RaBitQ on the full embedding database with the shared queries."""
    x_search = normalize_rows(x.astype("float32"))
    queries_search = normalize_rows(queries.astype("float32"))
    search_k = min(x.shape[0], k_neighbors + 1)
    result_rows = []
    compressed_neighbor_rows = []

    runs = []
    runs.extend(
        run_pq_retrieval(x_search, queries_search, pq_m, pq_nbits, query_count, k_neighbors, search_k, out_dir, save_indexes)
    )
    runs.extend(
        run_rabitq_retrieval(x_search, queries_search, rabitq_qb, query_count, k_neighbors, search_k, out_dir, save_indexes)
    )

    for run in runs:
        stats = run["stats"]
        assert isinstance(stats, dict)

        row = {field: str(stats.get(field, "")) for field in RESULT_FIELDS}
        row["recall_at_k"] = ""
        if stats.get("status") == "ok":
            scores = run["scores"]
            indices = run["indices"]
            assert isinstance(scores, np.ndarray)
            assert isinstance(indices, np.ndarray)
            scores, indices = filter_self_matches(scores, indices, query_indices, k_neighbors)
            recall = recall_at_k(reference_indices, indices, k_neighbors)
            row["recall_at_k"] = format_float(recall)
            compressed_neighbor_rows.extend(
                neighbor_rows(
                    str(stats["method"]),
                    str(stats["order"]),
                    str(stats["variant"]),
                    "",
                    query_indices,
                    scores,
                    indices,
                )
            )

        result_rows.append(row)

    return result_rows, compressed_neighbor_rows


def main() -> None:
    """Run the retrieval experiment from existing embedding and SVD artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="outputs/extracted/frames.csv")
    parser.add_argument("--embeddings", default="outputs/embeddings/frame_embeddings.npy")
    parser.add_argument("--index", default="outputs/embeddings/frame_embeddings.csv")
    parser.add_argument("--svd-error", default="outputs/embedding_svd_error/segment_svd_error.csv")
    parser.add_argument("--out", default="outputs/retrieval")
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pq-m", type=parse_ints, default=parse_ints("16,32,64,128,256"))
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--rabitq-qb", type=parse_ints, default=parse_ints("1,2,3,4,5,6,7,8"))
    parser.add_argument("--save-indexes", type=parse_bool, default=False)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing dependency: faiss") from exc

    frames_path = Path(args.frames)
    embeddings_path = Path(args.embeddings)
    index_path = Path(args.index)
    svd_error_path = Path(args.svd_error)

    frames, embeddings, embedding_index = load_embedding_inputs(frames_path, embeddings_path, index_path)
    x = np.ascontiguousarray(embeddings.astype("float32"))
    if x.shape[0] == 0:
        raise SystemExit("Embedding array is empty")
    if args.pq_nbits <= 0:
        raise SystemExit("--pq-nbits must be positive")

    query_indices = select_query_indices(x.shape[0], args.query_count, args.seed)
    queries = x[query_indices]
    query_count = len(query_indices)
    k_neighbors = min(args.k_neighbors, x.shape[0] - 1) # Uncompressed reference neighbors, sorted most-similar-first with self-match excluded
    reference_scores, reference_indices = search_index(x, queries, query_indices, k_neighbors) # here, gather for the "uncompressed" or the reference, which can be further used for calculating recall for the compressed methods

    write_csv(out_dir / "query_ids.csv", query_rows(index_path, query_indices), QUERY_FIELDS)
    write_csv(
        out_dir / "reference_neighbors.csv",
        neighbor_rows("reference", "", "", "", query_indices, reference_scores, reference_indices),
        NEIGHBOR_FIELDS,
    )

    # SVD eval
    # results: summary per group i.e. len(groups)
    # compressed_neighbor_rows: queries * top_k * groups
    results, compressed_neighbor_rows = evaluate_svd_groups(
        x,
        queries,
        query_indices,
        reference_indices, # ucompressed winners for recall
        frames,
        embedding_index,
        svd_error_path,
        query_count,
        k_neighbors,
    )

    # Quant - PQ, RabitQ eval
    # quantization_results: summary per group 
    # quantization_neighbor_rows: queries * top_k * groups
    quantization_results, quantization_neighbor_rows = evaluate_quantization_baselines(
        x,
        queries,
        query_indices,
        reference_indices,
        args.pq_m,
        args.pq_nbits,
        args.rabitq_qb,
        query_count,
        k_neighbors,
        out_dir,
        args.save_indexes,
    )
    results.extend(quantization_results)
    compressed_neighbor_rows.extend(quantization_neighbor_rows)

    write_csv(out_dir / "retrieval_results.csv", results, RESULT_FIELDS)
    write_csv(out_dir / "compressed_neighbors.csv", compressed_neighbor_rows, NEIGHBOR_FIELDS)

    metadata = {
        "frames": str(frames_path),
        "embeddings": str(embeddings_path),
        "embedding_index": str(index_path),
        "svd_error": str(svd_error_path),
        "n_vectors": int(x.shape[0]),
        "d_model": int(x.shape[1]),
        "query_seed": args.seed,
        "requested_query_count": args.query_count,
        "query_count": query_count,
        "k_neighbors": k_neighbors,
        "index": "faiss.IndexFlatIP over L2-normalized vectors, equivalent to cosine similarity",
        "recall_at_k": "mean overlap between original top-k and reconstructed top-k, divided by k",
        "self_match": "Excluded by embedding index before writing neighbors and computing recall.",
        "methods": {
            "svd": "Implemented from segment_svd_error.csv using one reconstructed database per order, variant, and epsilon.",
            "pq": "Implemented on the full normalized embedding database using the same query ids and reference neighbors.",
            "rabitq": "Implemented on the full normalized embedding database using the same query ids and reference neighbors.",
        },
        "pq_m": args.pq_m,
        "pq_nbits": args.pq_nbits,
        "rabitq_qb": args.rabitq_qb,
        "save_indexes": args.save_indexes,
        "svd_reconstruction": {
            "embedding_frames": "Reconstructed matrix rows are written directly into the database.",
            "embedding_delta_previous_i": "Frame embeddings are recovered as e(previous I-frame) - reconstructed_delta.",
            "embedding_delta_adjacent": "Frame embeddings are recovered sequentially: each row uses the previously reconstructed frame as its anchor.",
        },
        "compression_ratio": "Full database original params / compressed params. Untouched frames are counted as raw d_model vectors; reconstructed segment frames are counted by SVD params.",
    }
    (out_dir / "retrieval_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote retrieval outputs to {out_dir}")


if __name__ == "__main__":
    # from pdb import set_trace; set_trace()
    main()
