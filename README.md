# codec-got-ragged

## Makefile

Targets:

- `make extract`: extract codec metadata, decoded frame images, summary, and frame-type plot.
- `make pixel-delta`: compute pixel deltas and pixel-delta plots.
- `make embedding`: run extract, then create CLIP frame embeddings.
- `make embedding-frames`: use existing frame embeddings, then compute/plot direct frame-embedding cosine similarities.
- `make embedding-delta`: use existing frame embeddings, then compute/plot embedding-vector delta cosine similarities.
- `make embedding-svd`: use existing frame embeddings, then run inter-I-frame SVD compression experiments.
- `make embedding-svd-error`: use existing frame embeddings, then find the smallest SVD rank for each target error.
- `make all`: run extract, embedding, embedding-frames, embedding-delta, embedding-svd, and embedding-svd-error.
- `make clean`: clean generated `outputs/` files.

Knobs:

- `VIDEO=path/to/video.mp4`
- `FPS=0`: analyze original encoded video.
- `FPS>0`: re-encode at this FPS before extraction.
- `EMBED_MODEL=openai/clip-vit-base-patch32`
- `EMBED_BATCH=32`
- `DEVICE=auto`
- `SVD_RANKS=1,5,10,20,50`
- `SVD_ERROR_EPSILONS=0.01,0.05,0.10,0.20`

## Stages

### Extract

Prepares the video for analysis and records frame-level data.

- `FPS=0`: use the original encoded video.
- `FPS>0`: re-encode the video at that FPS, then analyze that new encoded video.
- Extract codec metadata: codec, profile, level, frame types, keyframes, `PTS`/`DTS`, packet sizes.
- Decode each frame into an image file under `outputs/extracted/frame_images/`.
- Write an extraction summary to `outputs/extracted/summary.json`.
- Write a frame-type plot to `outputs/extracted/plots/frame_types.png`.

So extraction gives:

```text
codec metadata + decoded frame images
```

### Pixel Delta

Compares actual decoded frame images.

For both display order and decode order, it computes:

```text
mean(abs(current_frame_pixels - previous_frame_pixels))
```

So this is pixel/image difference, not codec-feature distance.

### Embedding

Creates one CLIP embedding per decoded frame and saves it under `outputs/embeddings/`.

Outputs:

- `outputs/embeddings/frame_embeddings.npy`: array shaped `total_frames x embedding_dim`.
- `outputs/embeddings/frame_embeddings.csv`: frame id/type/order mapping for each embedding row.
- `outputs/embeddings/embedding_metadata.json`: model/device/batch metadata.

Current CLIP embedding is one pooled/global vector per frame, not patch embeddings.

This stage does not compare frames.

### Embedding Frames

Uses existing frame embeddings from `make embedding`.

It computes direct frame-embedding cosine similarity.

Setting 1, previous I-frame anchor:

```text
I0, B1, P1, I2, B2, P2
=> I0, cosine_sim(e(I0), e(B1)), cosine_sim(e(I0), e(P1)), I2, cosine_sim(e(I2), e(B2)), cosine_sim(e(I2), e(P2))
```

Setting 2, previous adjacent-frame anchor:

```text
I0, B1, P1, I2, B2, P2
=> I0, cosine_sim(e(I0), e(B1)), cosine_sim(e(B1), e(P1)), I2, cosine_sim(e(I2), e(B2)), cosine_sim(e(B2), e(P2))
```

Outputs:

- `outputs/embedding_frames/decode_order_embedding_frame_similarities.csv`
- `outputs/embedding_frames/display_order_embedding_frame_similarities.csv`
- `outputs/embedding_frames/plots/`

### Embedding Delta

Uses existing frame embeddings from `make embedding`.

For each `P` or `B` frame, it first computes an embedding-vector delta:

```text
delta = e(anchor frame) - e(current frame)
```

Then it plots:

```text
cosine_sim(e(anchor frame), delta)
```

Setting 1, previous I-frame anchor:

```text
I0, B1, P1, I2, B2, P2
=> I0, e(I0)-e(B1), e(I0)-e(P1), I2, e(I2)-e(B2), e(I2)-e(P2)
```

Setting 2, previous adjacent-frame anchor:

```text
I0, B1, P1, I2, B2, P2
=> I0, e(I0)-e(B1), e(B1)-e(P1), I2, e(I2)-e(B2), e(B2)-e(P2)
```

The previous I-frame is an analysis anchor, not a claimed codec reference.

Run `make embedding` before this stage when embeddings are missing.

This stage does not rerun `make embedding`.

Outputs:

- `outputs/embedding_delta/decode_order_embedding_delta_similarities.csv`
- `outputs/embedding_delta/display_order_embedding_delta_similarities.csv`
- `outputs/embedding_delta/embedding_delta_vectors.npy`
- `outputs/embedding_delta/embedding_delta_vectors.csv`
- `outputs/embedding_delta/plots/`

### Embedding SVD

Uses existing frame embeddings from `make embedding`.

For every pair of consecutive I-frames, it creates one segment from the intermediate `B/P` frames:

```text
I0, B1, P2, B3, I4
=> segment = B1, P2, B3
```

For each segment, it builds three matrices:

```text
embedding_frames
X = [e(B1), e(P2), e(B3), ...]

embedding_delta_previous_i
X = [e(I0)-e(B1), e(I0)-e(P2), e(I0)-e(B3), ...]

embedding_delta_adjacent
X = [e(I0)-e(B1), e(B1)-e(P2), e(P2)-e(B3), ...]
```

For each matrix `X`, valid ranks follow:

```text
k <= min(N_i, d_model)
```

Compression and error:

```text
original_params = N_i * d_model
svd_params = k * (N_i + d_model + 1)
compression_ratio = original_params / svd_params
relative_error = ||X - X_k||_F / ||X||_F
```

Outputs:

- `outputs/embedding_svd/segment_svd.csv`
- `outputs/embedding_svd/aggregate_svd.csv`
- `outputs/embedding_svd/segment_matrices.csv`
- `outputs/embedding_svd/embedding_svd_metadata.json`
- `outputs/embedding_svd/plots/svd_error_vs_compression.png`: per-segment points plus median aggregate trend.
- `outputs/embedding_svd/plots/decode/svd_error_vs_compression.png`
- `outputs/embedding_svd/plots/display/svd_error_vs_compression.png`

### Embedding SVD Error

Uses the same inter-I-frame matrices as `make embedding-svd`, but asks the inverse question:

```text
for a target error epsilon, what is the smallest rank k needed?
```

For each segment matrix and epsilon:

```text
relative_error(k) = ||X - X_k||_F / ||X||_F
selected_k = first k where relative_error(k) <= epsilon + tolerance
```

The tolerance is `1e-12`, only to avoid floating-point boundary misses.

Equivalent SVD shortcut:

```text
sqrt(sum(S[k:]^2) / sum(S^2))
```

Outputs:

- `outputs/embedding_svd_error/segment_svd_error.csv`
- `outputs/embedding_svd_error/aggregate_svd_error.csv`
- `outputs/embedding_svd_error/embedding_svd_error_metadata.json`
- `outputs/embedding_svd_error/plots/svd_error_targets_compression.png`
- `outputs/embedding_svd_error/plots/svd_error_targets_k.png`
- `outputs/embedding_svd_error/plots/decode/`
- `outputs/embedding_svd_error/plots/display/`
