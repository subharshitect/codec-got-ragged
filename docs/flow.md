# Pipeline Flow

Simple stage map for the current Makefile.

## `make extract`

Input:

```text
VIDEO
FPS
```

Main outputs:

```text
outputs/extracted/frames.csv
outputs/extracted/frames.json
outputs/extracted/packets.csv
outputs/extracted/motion_vectors.csv
outputs/extracted/summary.json
outputs/extracted/frame_images/frame_*.png
outputs/extracted/plots/frame_types.png
```

Produces:

```text
codec/frame metadata
decoded frame images
frame-type/keyframe plot
summary counts and packet-size stats
```

Note:

```text
FPS=0 uses original encoded video information.
FPS>0 re-encodes the video at that FPS, then extracts from the re-encoded video.
```

## `make pixel-delta`

Input:

```text
outputs/extracted/frames.csv
outputs/extracted/frame_images/frame_*.png
```

Main outputs:

```text
outputs/deltas/pixel/decode_order_deltas.csv
outputs/deltas/pixel/display_order_deltas.csv
outputs/deltas/pixel/delta_metadata.json
outputs/deltas/pixel/plots/decode_order_deltas.png
outputs/deltas/pixel/plots/display_order_deltas.png
```

Produces:

```text
adjacent-frame pixel deltas
delta ranks
decode-order and display-order plots
```

Formula:

```text
mean(abs(current_frame_pixels - previous_frame_pixels))
```

## `make embedding`

Input:

```text
outputs/extracted/frames.csv
outputs/extracted/frame_images/frame_*.png
```

Main outputs:

```text
outputs/embeddings/frame_embeddings.npy
outputs/embeddings/frame_embeddings.csv
outputs/embeddings/embedding_metadata.json
```

Produces:

```text
CLIP image embedding per decoded frame
frame_id -> embedding_index mapping
```

Shape:

```text
[num_frames, d_model]
```

## `make embedding-frames`

Input:

```text
outputs/extracted/frames.csv
outputs/embeddings/frame_embeddings.npy
outputs/embeddings/frame_embeddings.csv
```

Main outputs:

```text
outputs/embedding_frames/decode_order_embedding_frame_similarities.csv
outputs/embedding_frames/display_order_embedding_frame_similarities.csv
outputs/embedding_frames/embedding_frame_similarity_metadata.json
outputs/embedding_frames/plots/decode_order_embedding_frame_similarities.png
outputs/embedding_frames/plots/display_order_embedding_frame_similarities.png
```

Produces:

```text
cosine similarity between frame embeddings
```

Settings:

```text
cos_sim(e(previous I-frame), e(current P/B frame))
cos_sim(e(previous adjacent frame), e(current P/B frame))
```

## `make embedding-delta`

Input:

```text
outputs/extracted/frames.csv
outputs/embeddings/frame_embeddings.npy
outputs/embeddings/frame_embeddings.csv
```

Main outputs:

```text
outputs/embedding_delta/decode_order_embedding_delta_similarities.csv
outputs/embedding_delta/display_order_embedding_delta_similarities.csv
outputs/embedding_delta/embedding_delta_vectors.npy
outputs/embedding_delta/embedding_delta_vectors.csv
outputs/embedding_delta/embedding_delta_metadata.json
outputs/embedding_delta/plots/decode_order_embedding_delta_similarities.png
outputs/embedding_delta/plots/display_order_embedding_delta_similarities.png
```

Produces:

```text
embedding delta vectors
cosine similarity between anchor embedding and delta vector
```

Convention:

```text
delta = e(anchor frame) - e(current frame)
```

Settings:

```text
previous_i:        e(previous I-frame) - e(current P/B frame)
previous_adjacent: e(previous adjacent frame) - e(current P/B frame)
```

## `make embedding-svd`

Input:

```text
outputs/extracted/frames.csv
outputs/embeddings/frame_embeddings.npy
outputs/embeddings/frame_embeddings.csv
SVD_RANKS
```

Main outputs:

```text
outputs/embedding_svd/segment_svd.csv
outputs/embedding_svd/aggregate_svd.csv
outputs/embedding_svd/segment_matrices.csv
outputs/embedding_svd/embedding_svd_metadata.json
outputs/embedding_svd/plots/svd_error_vs_compression.png
outputs/embedding_svd/plots/decode/svd_error_vs_compression.png
outputs/embedding_svd/plots/display/svd_error_vs_compression.png
```

Produces:

```text
fixed-rank SVD reconstruction error and compression ratio
per segment and aggregate
```

Variants:

```text
embedding_frames:            rows are e(current intermediate frame)
embedding_delta_previous_i:  rows are e(previous I-frame) - e(current intermediate frame)
embedding_delta_adjacent:    rows are e(previous frame) - e(current intermediate frame)
```

Formula:

```text
relative_error = ||X - X_k||_F / ||X||_F
compression_ratio = (N_i * d_model) / (k * (N_i + d_model + 1))
```

## `make embedding-svd-error`

Input:

```text
outputs/extracted/frames.csv
outputs/embeddings/frame_embeddings.npy
outputs/embeddings/frame_embeddings.csv
SVD_ERROR_EPSILONS
```

Main outputs:

```text
outputs/embedding_svd_error/segment_svd_error.csv
outputs/embedding_svd_error/aggregate_svd_error.csv
outputs/embedding_svd_error/embedding_svd_error_metadata.json
outputs/embedding_svd_error/plots/svd_error_targets_compression.png
outputs/embedding_svd_error/plots/svd_error_targets_k.png
outputs/embedding_svd_error/plots/decode/svd_error_targets_compression.png
outputs/embedding_svd_error/plots/decode/svd_error_targets_k.png
outputs/embedding_svd_error/plots/display/svd_error_targets_compression.png
outputs/embedding_svd_error/plots/display/svd_error_targets_k.png
```

Produces:

```text
target-error SVD results
smallest selected_k per segment and epsilon
achieved error and compression ratio
```

Selection rule:

```text
binary search for the smallest k where ||X - X_k||_F / ||X||_F <= epsilon
```

Rows:

```text
segment_svd_error.csv: one row per order + variant + segment + epsilon
aggregate_svd_error.csv: one row per order + variant + epsilon
```

## `make quantization`

Input:

```text
outputs/embeddings/frame_embeddings.npy
QUANT_PQ_M
QUANT_PQ_NBITS
QUANT_RABITQ_QB
QUANT_QUERY_COUNT
QUANT_K
QUANT_SAVE_INDEXES
```

`QUANT_QUERY_COUNT` defaults to `QUERY_COUNT`.
`QUANT_K` defaults to `QUERY_K`.

Main outputs:

```text
outputs/quantization/pq_results.csv
outputs/quantization/rabitq_results.csv
outputs/quantization/quantization_results.csv
outputs/quantization/quantization_metadata.json
outputs/quantization/plots/quantization_comparison.png
```

Produces:

```text
PQ and RaBitQ compression baselines
MSE
relative reconstruction error
recall@k
compression ratio
```

Compression ratio:

```text
original embedding bytes / quantized code bytes
```

## `make retrieval`

Input:

```text
outputs/extracted/frames.csv
outputs/embeddings/frame_embeddings.npy
outputs/embeddings/frame_embeddings.csv
outputs/embedding_svd_error/segment_svd_error.csv
QUERY_COUNT
QUERY_K
QUERY_SEED
QUANT_PQ_M
QUANT_PQ_NBITS
QUANT_RABITQ_QB
QUANT_SAVE_INDEXES
```

Main outputs:

```text
outputs/retrieval/retrieval_results.csv
outputs/retrieval/query_ids.csv
outputs/retrieval/reference_neighbors.csv
outputs/retrieval/compressed_neighbors.csv
outputs/retrieval/retrieval_metadata.json
outputs/retrieval/plots/compression_vs_recall.png
```

Produces:

```text
fixed query set
cosine FAISS reference over original embeddings
cosine FAISS retrieval over SVD-reconstructed embeddings
PQ/RaBitQ retrieval over the full embedding database
recall@k against the original neighbor list
```

Notes:

```text
self-matches are excluded before recall
embedding_delta_adjacent is decoded sequentially from the previous reconstructed frame
PQ/RaBitQ use outputs/embeddings/frame_embeddings.npy, not SVD segment matrices
retrieval_results.csv has one summary row per SVD group, PQ config, or RaBitQ config
compressed_neighbors.csv stores method/config * query_count * top_k neighbor rows
```

For SVD:

```text
group = order + variant + epsilon
with 2 orders, 3 variants, and 10 epsilons, there are up to 60 SVD groups
```

## `make all`

Runs:

```text
extract
embedding
embedding-frames
embedding-delta
embedding-svd
embedding-svd-error
```

Note:

```text
quantization is separate.
Run make all quantization to include it.
```
