PYTHON ?= python3

# VIDEO ?= /home/shubh/workspace/centroids-are-all-you-need/TimeLensBench/data/TimeLensBench/videos/activitynet/v_0fvL6IHKYF0.mp4
# VIDEO ?= /home/shubh/workspace/codec-got-ragged/data/orange_juice_10mins.mp4
VIDEO ?= /home/shubh/workspace/codec-got-ragged/data/traffic_10m.mp4
FPS ?= 0

EMBED_MODEL ?= openai/clip-vit-base-patch32
EMBED_BATCH ?= 2048
DEVICE ?= auto

# SVD_RANKS ?= 1,5,10,20,50
SVD_RANKS ?= 100,50,25,5
SVD_ERROR_EPSILONS ?= 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10

# Shared retrieval/query params
# Number of frame embeddings sampled as queries.
# Used by: make retrieval; also defaults QUANT_QUERY_COUNT below.
# How many queries do we sample from the video embeddings.
QUERY_COUNT ?= 1000
# Top-k neighbors used for recall.
# Used by: make retrieval; also defaults QUANT_K below.
QUERY_K ?= 10
# Random seed for deterministic query sampling.
# Used by: make retrieval.
QUERY_SEED ?= 42

# Quantization params
# PQ sub-vector counts to sweep; each value must divide d_model.
QUANT_PQ_M ?= 16,32,64,128,256
# Number of bits used per PQ sub-vector code.
QUANT_PQ_NBITS ?= 8

# RaBitQ query-time bit/refinement settings to sweep.
QUANT_RABITQ_QB ?= 1,2,3,4,5,6,7,8
# Query count used by standalone make quantization; defaults to QUERY_COUNT.
QUANT_QUERY_COUNT ?= $(QUERY_COUNT)
# Top-k neighbors used by standalone make quantization; defaults to QUERY_K.
QUANT_K ?= $(QUERY_K)
# Set to 1 to save FAISS PQ/RaBitQ index files.
# Used by: make quantization and make retrieval.
QUANT_SAVE_INDEXES ?= 0
