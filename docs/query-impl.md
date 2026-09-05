# Query / Retrieval Implementation Strategy

You want to implement a retrieval-quality experiment on top of the existing compression outputs.

After SVD/PQ/RaBitQ compress the embedding database, we ask:

```text
If I retrieve nearest frames from the compressed database, how close are the results to retrieval from the original embeddings?
```

## Strategy

1. Add a new stage, likely `make retrieval` or `make query`.

   It should depend on:

   ```text
   outputs/extracted/frames.csv
   outputs/embeddings/frame_embeddings.npy
   outputs/embeddings/frame_embeddings.csv
   outputs/embedding_svd_error/segment_svd_error.csv
   ```

   PQ/RaBitQ are evaluated inside this stage from the same embeddings, query ids,
   and reference neighbors.

2. Create a script like:

   ```text
   scripts/retrieval/embedding_retrieval.py
   ```

3. Load original embeddings:

   ```text
   X = frame_embeddings.npy
   ```

4. Select query IDs deterministically:

   ```python
   rng = np.random.default_rng(seed)
   query_indices = rng.choice(n_vectors, size=min(query_count, n_vectors), replace=False)
   queries = X[query_indices]
   ```

   This fixes the current mismatch where quantization uses `x[:query_count]`.

5. Build the uncompressed reference FAISS index:

   ```text
   IndexFlatIP over L2-normalized X
   ```

   Search with `queries`, and save:

   ```text
   reference_topk_indices
   reference_topk_scores
   ```

   This is cosine-similarity search. Exclude each query frame's own embedding index from its neighbor list.

6. For SVD, reconstruct full embedding databases per:

   ```text
   order: decode/display
   variant: embedding_frames / embedding_delta_previous_i / embedding_delta_adjacent
   epsilon
   ```

   Using `segment_svd_error.csv`, get each segment's `selected_k`.

   Reconstruction differs by variant:

   ```text
   embedding_frames
   reconstructed rows are directly e(current)

   embedding_delta_previous_i
   e(current) = e(previous I) - reconstructed_delta

   embedding_delta_adjacent
   e(current) = e(previous reconstructed frame) - reconstructed_delta
   ```

   For adjacent, this needs care: use the actual anchor embedding convention from `build_matrices`.
   The faithful retrieval implementation reconstructs adjacent deltas sequentially, starting from the previous I-frame.

7. Put reconstructed rows back into full database shape:

   ```text
   X_reconstructed = copy(X)
   X_reconstructed[segment_frame_indices] = reconstructed_embeddings
   ```

   I-frames and frames outside inter-I segments stay original.

8. Build a FAISS index over each `X_reconstructed`, search the same `queries`, and compute recall:

   ```text
   recall@k = average overlap(reference_topk, compressed_topk) / k
   ```

   Save both reference and compressed neighbor lists so recall can be audited.

9. For PQ/RaBitQ, use the original full embedding database directly:

   ```text
   X = [video_len, d_model]
   ```

   They do not use:

   ```text
   embedding_frames segment matrices
   embedding_delta_previous_i segment matrices
   embedding_delta_adjacent segment matrices
   ```

   They reuse the same query indices and reference top-k as SVD retrieval.

10. Write outputs:

    ```text
    outputs/retrieval/retrieval_results.csv
    outputs/retrieval/query_ids.csv
    outputs/retrieval/reference_neighbors.csv
    outputs/retrieval/compressed_neighbors.csv
    outputs/retrieval/retrieval_metadata.json
    outputs/retrieval/plots/compression_vs_recall.png
    ```

    CSV rows should include:

    ```text
    method, order, variant, epsilon, compression_ratio, recall_at_k,
    query_count, k_neighbors, n_segments, status
    ```

## Summary

Use original embeddings as the gold retrieval database. For SVD, reconstruct full embedding databases from compressed inter-I segments. For PQ/RaBitQ, search quantized indexes over the full embedding database. Query all methods with the same fixed queries and plot compression ratio vs recall@k.
