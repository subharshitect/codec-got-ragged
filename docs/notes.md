# Codec Bitstream Notes

## Layout

- extraction code: `scripts/extract/extract.py`
- extraction plot code: `scripts/extract/plot_frames.py`
- pixel delta code: `scripts/delta/pixel_delta.py`
- pixel plot code: `scripts/delta/pixel_plot.py`
- extraction summary: `outputs/extracted/summary.json`
- extraction frame plot: `outputs/extracted/plots/frame_types.png`
- pixel delta outputs: `outputs/deltas/pixel/`
- pixel plots: `outputs/deltas/pixel/plots/`
- embedding code: `scripts/embedding/embed_frames.py`
- embedding frame similarity code: `scripts/embedding/embedding_frames.py`
- embedding delta code: `scripts/embedding/embedding_delta.py`
- embedding plot code: `scripts/embedding/embedding_plot.py`
- frame embeddings: `outputs/embeddings/`
- embedding frame similarity outputs: `outputs/embedding_frames/`
- embedding frame similarity plots: `outputs/embedding_frames/plots/`
- embedding delta similarity outputs: `outputs/embedding_delta/`
- embedding delta similarity plots: `outputs/embedding_delta/plots/`

## Frame Terms

- `I-frame`: intra-coded frame. Mostly self-contained image data.
- `P-frame`: predicted frame. Uses earlier reference frame(s).
- `B-frame`: bidirectionally predicted frame. Can use earlier and/or later reference frame(s).
- `key_frame`: random-access/seek anchor flag. Usually an I-frame, but a decoder can start afresh from this frame.

Keep both:

- `pict_type`: tells how the frame is coded: `I`, `P`, `B`.
- `key_frame`: tells whether the frame can act as a safe access/seek anchor.

## Ordering

- Display order: the order frames are shown during playback.
- Decode order: the order frames are fed to / processed by the decoder.
- `PTS`: presentation timestamp. This comes from ffprobe/FFmpeg and is used to build display order.
- `DTS`: decode timestamp. This comes from ffprobe/FFmpeg and is used to build decode order.
- `display_order_index`: assigned by us after sorting frames by `PTS`. `#we_invented`
- `decode_order_index`: assigned by us after sorting frames by matched packet `DTS`, then packet position if needed. `#we_invented`

With B-frames, decode order and display order can differ because future reference frames may need to be decoded before earlier B-frames are shown.

Example:

```text
Display order: I0 B1 B2 P3
Decode order:  I0 P3 B1 B2
```

## References

- `reference_frames`: actual frames used by codec prediction, if we can extract them.
- Do not guess that a frame is derived from the nearest previous keyframe.
- Do not use `gop_anchor_keyframe`; that was only a plotting heuristic and is not faithful enough.

If `reference_frames` are unavailable, leave them unavailable.

## Motion Vector Flags

FFmpeg `codecview` motion-vector flags:

- `pf`: forward-predicted motion vectors of P-frames.
- `bf`: forward-predicted motion vectors of B-frames.
- `bb`: backward-predicted motion vectors of B-frames.

Meaning:

- `bf`: B-frame prediction using an earlier reference frame.
- `bb`: B-frame prediction using a later reference frame.
- A B-frame can have both, which is why it is bidirectional.

## Extracted Features

Possible codec/bitstream fields:

- codec name: video codec, for example `h264`, `hevc`, `av1`.
- profile: codec feature set, for example H.264 `High`.
- level: codec constraint level, roughly related to decoder limits.
- `source_index`: row number in ffprobe frame output. `#we_invented`
- `display_order_index`: row number after sorting by `PTS`. `#we_invented`
- `decode_order_index`: row number after sorting by `DTS`, then packet position if needed. `#we_invented`
- `pict_type`: how the frame is coded: `I`, `P`, or `B`.
- `key_frame`: whether the frame is a safe random-access/seek anchor.
- `PTS`: presentation timestamp, meaning display time/order.
- `DTS`: decode timestamp, meaning decoder input order.
- packet/frame size: encoded size for that packet/frame.
- raw motion vectors: codec prediction movement data, if available.
- QP values: quantization/compression-strength values, if available.
- block/macroblock information: per-block coding details, if available.
- reference frames: actual frames used by codec prediction, if available.

## Extraction Summary

`outputs/extracted/summary.json` contains:

- total frames
- total I/P/B frames
- total keyframes
- average distance between I-frames in frames
- average distance between I-frames in seconds
- packet size stats by frame type
- duration, FPS, codec, profile, level

`outputs/extracted/plots/frame_types.png` shows the frame-type/keyframe strip from extraction.

## FPS Knob

- `FPS=0`: default; keep all extracted frames and source timing from `PTS` and `DTS`.
- `FPS>0`: re-encode the input video at the requested FPS, then extract from that encoded analysis video.
- Example: `FPS=1` creates a new 1 fps encoded video and extracts codec facts from it.
- Raw `PTS` and `DTS` come from the video being analyzed: original video for `FPS=0`, re-encoded video for `FPS>0`.
- The re-encoded video is written to `outputs/encoded/`.

## `frames.csv` Columns

- `frame_id`: stable frame id assigned by us. `#we_invented`
- `source_index`: row number in ffprobe frame output. `#we_invented`
- `display_order_index`: row number after sorting by `PTS`. `#we_invented`
- `decode_order_index`: row number after sorting by `DTS`, then packet position if needed. `#we_invented`
- `key_frame`: ffprobe flag; `1` means random-access/seek frame.
- `pict_type`: ffprobe frame type: `I`, `P`, or `B`.
- `pts_time`: presentation/display timestamp.
- `dts_time`: decode/coding timestamp.
- `display_time_seconds`: display time used by analysis, copied from extracted display timestamp. `#we_invented`
- `decode_time_seconds`: decode time used by analysis, copied from extracted decode timestamp. `#we_invented`
- `frame_image`: decoded PNG frame image path used for pixel delta. `#we_invented`
- `best_effort_timestamp_time`: ffmpeg fallback display timestamp.
- `pkt_duration_time`: packet/frame duration.
- `pkt_pos`: encoded packet byte position in the file.
- `pkt_size`: encoded packet size in bytes.
- `coded_picture_number`: codec-reported coded picture number, if available.
- `display_picture_number`: codec-reported display picture number, if available.
- `mv_count`: number of motion vectors extracted for this frame. `#we_invented`
- `mv_mean_magnitude`: mean motion-vector length for this frame. `#we_invented`
- `mv_max_magnitude`: max motion-vector length for this frame. `#we_invented`
- `mv_source_neg_count`: count of motion vectors with negative `source`. `#we_invented`
- `mv_source_pos_count`: count of motion vectors with positive `source`. `#we_invented`
- `mv_source_unknown_count`: count where motion-vector source is missing or zero. `#we_invented`
- `mv_zero_count`: count of zero-length motion vectors. `#we_invented`
- `mv_zero_ratio`: `mv_zero_count / mv_count`. `#we_invented`

## Delta Ideas

Use decoded frame pixels for delta.

Formula:

```text
mean(abs(current_frame_pixels - previous_frame_pixels))
```

`decode_adjacent_delta`: compare each frame with the previous frame in decode/coding order. `#we_invented`

```text
DTS order:
previous decoded frame -> current decoded frame
```

`display_adjacent_delta`: compare each frame with the previous frame in display/presentation order. `#we_invented`

```text
PTS order:
previous displayed frame -> current displayed frame
```

`reference_delta`: compare a frame with actual `reference_frames`, only if those references are extracted from the codec/tool.

Ranks are computed after deltas, using the delta values within that ordering. `#we_invented`

Plots use `order_time_seconds` on the x-axis. `#we_invented`

## Embedding Stage

`make embedding` creates one CLIP embedding per decoded frame.

- default model: `openai/clip-vit-base-patch32`
- saved embeddings: `outputs/embeddings/frame_embeddings.npy`
- embedding index: `outputs/embeddings/frame_embeddings.csv`
- metadata: `outputs/embeddings/embedding_metadata.json`

The embedding index keeps codec labels from extraction:

- `pict_type`
- `key_frame`
- `display_order_index`
- `decode_order_index`

## Embedding Frames

`make embedding-frames` uses existing frame embeddings and computes direct frame-embedding cosine similarities.

Setting 1, previous I-frame anchor: `#we_invented`

```text
I0, B1, P1, I2, B2, P2
=> I0, cosine_sim(e(I0), e(B1)), cosine_sim(e(I0), e(P1)), I2, cosine_sim(e(I2), e(B2)), cosine_sim(e(I2), e(P2))
```

Setting 2, previous adjacent-frame anchor: `#we_invented`

```text
I0, B1, P1, I2, B2, P2
=> I0, cosine_sim(e(I0), e(B1)), cosine_sim(e(B1), e(P1)), I2, cosine_sim(e(I2), e(B2)), cosine_sim(e(B2), e(P2))
```

- `previous_i_cosine_similarity`: `cosine_sim(e(previous I-frame), e(current frame))`. `#we_invented`
- `previous_adjacent_cosine_similarity`: `cosine_sim(e(previous adjacent frame), e(current frame))`. `#we_invented`

## Embedding Delta

`make embedding-delta` uses existing frame embeddings and computes embedding-vector deltas.

Run `make embedding` first if `outputs/embeddings/frame_embeddings.npy` is missing.

Embedding-vector delta: `#we_invented`

```text
delta = e(anchor frame) - e(current frame)
score = cosine_sim(e(anchor frame), delta)
```

Setting 1, previous I-frame anchor: `#we_invented`

```text
I0, B1, P1, I2, B2, P2
=> I0, e(I0)-e(B1), e(I0)-e(P1), I2, e(I2)-e(B2), e(I2)-e(P2)
```

Setting 2, previous adjacent-frame anchor: `#we_invented`

```text
I0, B1, P1, I2, B2, P2
=> I0, e(I0)-e(B1), e(B1)-e(P1), I2, e(I2)-e(B2), e(B2)-e(P2)
```

- `previous_i_anchor_delta_cosine_similarity`: `cosine_sim(e(previous I-frame), e(previous I-frame)-e(current frame))`. `#we_invented`
- `previous_adjacent_anchor_delta_cosine_similarity`: `cosine_sim(e(previous adjacent frame), e(previous adjacent frame)-e(current frame))`. `#we_invented`
- `embedding_delta_vectors.npy`: saved delta vectors. `#we_invented`
- `embedding_delta_vectors.csv`: index for saved delta vectors. `#we_invented`

The previous I-frame is an analysis anchor, not a claimed codec reference.

Plots:

- subplot 1: cosine similarity between previous I-frame embedding and its delta vector
- subplot 2: cosine similarity between previous adjacent-frame embedding and its delta vector
- subplot 3: frame-type/keyframe strip
