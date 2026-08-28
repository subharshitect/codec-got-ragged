PYTHON ?= python3
# VIDEO ?= /home/shubh/workspace/centroids-are-all-you-need/TimeLensBench/data/TimeLensBench/videos/activitynet/v_0fvL6IHKYF0.mp4
VIDEO ?= /home/shubh/workspace/codec-got-ragged/data/traffic.mp4
FPS ?= 0
EMBED_MODEL ?= openai/clip-vit-base-patch32
EMBED_BATCH ?= 32
DEVICE ?= auto

EXTRACTED_DIR := outputs/extracted
ENCODED_DIR := outputs/encoded
EXTRACT_PLOTS_DIR := outputs/extracted/plots
PIXEL_DELTAS_DIR := outputs/deltas/pixel
PIXEL_PLOTS_DIR := outputs/deltas/pixel/plots
EMBEDDINGS_DIR := outputs/embeddings
EMBEDDING_FRAMES_DIR := outputs/embedding_frames
EMBEDDING_FRAMES_PLOTS_DIR := outputs/embedding_frames/plots
EMBEDDING_DELTA_DIR := outputs/embedding_delta
EMBEDDING_DELTA_PLOTS_DIR := outputs/embedding_delta/plots
OUTPUTS_DIR := outputs

.PHONY: all extract delta pixel-delta embedding embedding-frames embedding-delta clean check-video check-embedding

all: extract pixel-delta embedding embedding-frames embedding-delta

# /home/shubh/workspace/centroids-are-all-you-need/TimeLensBench/data/TimeLensBench/videos/activitynet/v_0fvL6IHKYF0.mp4
extract: check-video
	@mkdir -p $(EXTRACTED_DIR) $(ENCODED_DIR) $(EXTRACT_PLOTS_DIR)
	$(PYTHON) scripts/extract/extract.py --video "$(VIDEO)" --fps "$(FPS)" --out "$(EXTRACTED_DIR)" --encoded-out "$(ENCODED_DIR)"
	$(PYTHON) scripts/extract/plot_frames.py --extracted "$(EXTRACTED_DIR)" --out "$(EXTRACT_PLOTS_DIR)"

pixel-delta:
	@mkdir -p $(PIXEL_DELTAS_DIR) $(PIXEL_PLOTS_DIR)
	$(PYTHON) scripts/delta/pixel_delta.py --input "$(EXTRACTED_DIR)/frames.csv" --out "$(PIXEL_DELTAS_DIR)"
	$(PYTHON) scripts/delta/pixel_plot.py --deltas "$(PIXEL_DELTAS_DIR)" --out "$(PIXEL_PLOTS_DIR)"

delta: pixel-delta

embedding: extract
	@mkdir -p $(EMBEDDINGS_DIR)
	$(PYTHON) scripts/embedding/embed_frames.py --frames "$(EXTRACTED_DIR)/frames.csv" --out "$(EMBEDDINGS_DIR)" --model "$(EMBED_MODEL)" --batch-size "$(EMBED_BATCH)" --device "$(DEVICE)"

# cosine_sim[e(I-frame), e(B/P-frame)] (no delta here)
embedding-frames: check-embedding
	@mkdir -p $(EMBEDDING_FRAMES_DIR) $(EMBEDDING_FRAMES_PLOTS_DIR)
	$(PYTHON) scripts/embedding/embedding_frames.py --frames "$(EXTRACTED_DIR)/frames.csv" --embeddings "$(EMBEDDINGS_DIR)/frame_embeddings.npy" --index "$(EMBEDDINGS_DIR)/frame_embeddings.csv" --out "$(EMBEDDING_FRAMES_DIR)"
	$(PYTHON) scripts/embedding/embedding_plot.py \
		--scores "$(EMBEDDING_FRAMES_DIR)" \
		--out "$(EMBEDDING_FRAMES_PLOTS_DIR)" \
		--decode-file "decode_order_embedding_frame_similarities.csv" \
		--display-file "display_order_embedding_frame_similarities.csv" \
		--decode-plot "decode_order_embedding_frame_similarities.png" \
		--display-plot "display_order_embedding_frame_similarities.png" \
		--first-column "previous_i_cosine_similarity" \
		--second-column "previous_adjacent_cosine_similarity" \
		--first-label "cos sim: e(prev I), e(current)" \
		--second-label "cos sim: e(prev adj), e(current)" \
		--title-prefix "Frame Embedding Similarities"

# delta1 = e(I1) - e(I0), delta2 = e(B/P1) - e(B/P0), then cosine_sim[delta1, delta2]
embedding-delta: check-embedding
	@mkdir -p $(EMBEDDING_DELTA_DIR) $(EMBEDDING_DELTA_PLOTS_DIR)
	$(PYTHON) scripts/embedding/embedding_delta.py --frames "$(EXTRACTED_DIR)/frames.csv" --embeddings "$(EMBEDDINGS_DIR)/frame_embeddings.npy" --index "$(EMBEDDINGS_DIR)/frame_embeddings.csv" --out "$(EMBEDDING_DELTA_DIR)"
	$(PYTHON) scripts/embedding/embedding_plot.py \
		--scores "$(EMBEDDING_DELTA_DIR)" \
		--out "$(EMBEDDING_DELTA_PLOTS_DIR)" \
		--decode-file "decode_order_embedding_delta_similarities.csv" \
		--display-file "display_order_embedding_delta_similarities.csv" \
		--decode-plot "decode_order_embedding_delta_similarities.png" \
		--display-plot "display_order_embedding_delta_similarities.png" \
		--first-column "previous_i_anchor_delta_cosine_similarity" \
		--second-column "previous_adjacent_anchor_delta_cosine_similarity" \
		--first-label "cos sim: e(prev I), delta" \
		--second-label "cos sim: e(prev adj), delta" \
		--title-prefix "Embedding Delta Similarities"

clean:
	@mkdir -p $(OUTPUTS_DIR)
	@find $(OUTPUTS_DIR) -type f -delete
	@find $(OUTPUTS_DIR) -depth -mindepth 1 -type d -empty -delete
	@mkdir -p $(EXTRACTED_DIR) $(ENCODED_DIR) $(EXTRACT_PLOTS_DIR) $(PIXEL_DELTAS_DIR) $(PIXEL_PLOTS_DIR) $(EMBEDDINGS_DIR) $(EMBEDDING_FRAMES_DIR) $(EMBEDDING_FRAMES_PLOTS_DIR) $(EMBEDDING_DELTA_DIR) $(EMBEDDING_DELTA_PLOTS_DIR)

check-video:
	@test -n "$(VIDEO)" || (echo "Usage: make extract VIDEO=path/to/video.mp4"; exit 1)

check-embedding:
	@test -f "$(EXTRACTED_DIR)/frames.csv" || (echo "Missing $(EXTRACTED_DIR)/frames.csv. Run: make embedding"; exit 1)
	@test -f "$(EMBEDDINGS_DIR)/frame_embeddings.npy" || (echo "Missing $(EMBEDDINGS_DIR)/frame_embeddings.npy. Run: make embedding"; exit 1)
	@test -f "$(EMBEDDINGS_DIR)/frame_embeddings.csv" || (echo "Missing $(EMBEDDINGS_DIR)/frame_embeddings.csv. Run: make embedding"; exit 1)
