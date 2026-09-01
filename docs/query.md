# Query / Retrieval Experiment

Goal:
measure how compression affects retrieval quality.

## 1. Original Database

Video frames are converted into embeddings:

```text
X = [video_len, d_model]
```

This is the uncompressed embedding database.

## 2. Queries

Select fixed query frame ids using:

```text
fixed seed + fixed query_count
```

Then:

```text
queries = X[query_ids]
```

Shape:

```text
[query_count, d_model]
```

## 3. Uncompressed Baseline

Build a FAISS index over the original embeddings:

```text
index(X)
```

Query it with:

```text
queries
```

FAISS returns:

```text
topk_indices = [query_count, top_k]
topk_scores  = [query_count, top_k]
```

This is the reference retrieval result.

## 4. Our SVD Compression

Codec extraction gives frame metadata:

```text
I/P/B labels, keyframes, display order, decode order
```

Embeddings give:

```text
X = [video_len, d_model]
```

Together, we form inter-I-frame segment matrices.

Variants:

```text
embedding_frames
X_i = [e(B1), e(P2), e(B3), ...]

embedding_delta_previous_i
X_i = [e(I0)-e(B1), e(I0)-e(P2), e(I0)-e(B3), ...]

embedding_delta_adjacent
X_i = [e(I0)-e(B1), e(B1)-e(P2), e(P2)-e(B3), ...]
```

For each segment and target error epsilon, use the selected rank:

```text
selected_k
```

from:

```text
outputs/embedding_svd_error/segment_svd_error.csv
```

Then reconstruct:

```text
X_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
```

Place reconstructed segment rows back into their original frame positions.

This produces:

```text
X_reconstructed = [video_len, d_model]
```

## 5. Compressed Retrieval

Build a FAISS index over:

```text
X_reconstructed
```

Query it using the same:

```text
queries
```

Get:

```text
compressed_topk_indices = [query_count, top_k]
compressed_topk_scores  = [query_count, top_k]
```

## 6. Recall

Compare uncompressed and compressed top-k neighbors:

```text
recall@k = overlap(uncompressed_topk, compressed_topk) / k
```

Average across all query vectors.

Example:

```text
top_k = 10
overlap = 7
recall@10 = 7 / 10 = 0.7
```

## 7. Baselines

Run the same retrieval evaluation for:

```text
PQ
RaBitQ
```

Use the same:

```text
query_ids
top_k
uncompressed_topk reference
```

## 8. Main Plot

```text
x-axis: compression ratio
y-axis: recall@k
```

Compare:

```text
SVD embedding_frames
SVD embedding_delta_previous_i
SVD embedding_delta_adjacent
PQ
RaBitQ
```
