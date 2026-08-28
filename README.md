# codec-got-ragged

## Makefile

Targets:

- `make extract`: extract codec metadata, decoded frame images, summary, and frame-type plot.
- `make pixel-delta`: compute pixel deltas and pixel-delta plots.
- `make embedding`: run extract, then create CLIP frame embeddings.
- `make embedding-frames`: use existing frame embeddings, then compute/plot direct frame-embedding cosine similarities.
- `make embedding-delta`: use existing frame embeddings, then compute/plot embedding-vector delta cosine similarities.
- `make all`: run extract, pixel-delta, embedding, embedding-frames, and embedding-delta.
- `make clean`: clean generated `outputs/` files.

Knobs:

- `VIDEO=path/to/video.mp4`
- `FPS=0`: analyze original encoded video.
- `FPS>0`: re-encode at this FPS before extraction.
- `EMBED_MODEL=openai/clip-vit-base-patch32`
- `EMBED_BATCH=32`
- `DEVICE=auto`

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
