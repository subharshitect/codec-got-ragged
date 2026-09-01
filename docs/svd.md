# Size of the rank
k <= min(N_i, d_model)

# Compression ratio
original params = N * d_model
svd params at rank k = k * (N + d_model + 1)
compression_ratio = original params / svd params

# Error
Relative reconstruction error: ||X - X_k||_F / ||X||_F
- ||X - X_k||_F = total reconstruction error
- ||X||_F = total magnitude/energy of the original matrix (cause the matrices sizes (N) are variable)

where;
Formula: ||X - X_k||_F = sqrt( sum over all i,j of (X[i,j] - X_k[i,j])^2 )

- X = original matrix
- X_k = rank-k approximation (reconstructed matrix)
- X[i,j] - X_k[i,j] = reconstruction error at one individual cell (i,j)

Example:
    X - X_k =
    [ 1  -2 ]
    [ 0   3 ]

    Then:

    ||X - X_k||_F
    = sqrt(1^2 + (-2)^2 + 0^2 + 3^2) = sqrt(1 + 4 + 0 + 9) = sqrt(14) ≈ 3.74

# Experiment plan

1. We have: I0, B1, P2, B3, I4, B5, P6, I7

2. Create segments: 
        segment_0 = B1, P2, B3   between I0 and I4
        segment_1 = B5, P6       between I4 and I7

3. Then create matrices for each segment:
    a. embedding_frames: rows are raw frame embeddings: 
            we have: e(I0), e(B1), e(P2), e(B3), e(I4), e(B5), e(P6), e(I7)
                    where; segments will be formed like in 2.

    b. embedding_deltas: rows are deltas
            embedding_delta_previous_i: rows are delta: e(previous I)     -   e(current intermediate frame like B1/P2)
            embedding_delta_adjacent:   rows are delta: e(previous frame) -   e(current intermediate frame like B1/P2)
        what do we store here:
            we have: e(I0), e(B1), e(P2), e(B3), e(I4), e(B5), e(P6), e(I7)
            delta:   
                setting 1 (from I-frames):  e(I0), [e(I0) - e(B1)], [e(I0) - e(P2)], [e(I0) - e(B3)], e(I4), [e(I4) - e(B5)], ... 
                setting 2 (from adjascent): e(I0), [e(I0) - e(B1)], [e(B1) - e(P2)], [e(P2) - e(B3)], e(I4), [e(I4) - e(B5)], ...

    hence, three matrix will be made for the SVD experiment.

4. For each matrix X, run SVD:
    
    X = U S Vt

    For each rank k, reconstruct:
        X_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]

    Then compute:
        relative_error = ||X - X_k||_F / ||X||_F
        compression_ratio = original_params / svd_params

Sure, there will be different N sized matrices, we plan to report both per segment and the aggregate. 
e.g.
    segment_id, start_i_frame, end_i_frame, n_frames, variant, k, compression_ratio, relative_error (variant: embedding_frame, embedding-delta-iframe, embedding-delta-adjacent)
        where k per segment: k <= min(N_i, d_model)

                        embedding_frames
                        X = [e(B1), e(P2), e(B3), ...]

                        embedding_delta_previous_i
                        X = [e(I0)-e(B1), e(I0)-e(P2), e(I0)-e(B3), ...]

                        embedding_delta_adjacent
                        X = [e(I0)-e(B1), e(B1)-e(P2), e(P2)-e(B3), ...]

5. Plot:

    x-axis: compression ratio
    y-axis: relative reconstruction error
    points/lines: k values

# Target-error experiment

For each target error epsilon:

    try k = 1, 2, ..., min(N_i, d_model)
    compute relative_error(k) = ||X - X_k||_F / ||X||_F
    pick the first k where relative_error(k) <= epsilon + tolerance

Tolerance is 1e-12, only to avoid floating-point boundary misses.

Equivalent SVD shortcut: sqrt(sum(S[k:]^2) / sum(S^2)).

Here, first k means the smallest rank that satisfies the target error.

Plots:

    svd_error_targets_compression.png
        x-axis: compression ratio
        y-axis: achieved reconstruction error

    svd_error_targets_k.png
        x-axis: target error epsilon
        y-axis: selected k

# Experiment forward

For each inter-I segment, we build three matrices:

    embedding_frames
    embedding_delta_previous_i
    embedding_delta_adjacent


Then for each matrix, for each valid rank `k`:

    compute SVD
    compute relative reconstruction error
    compute compression ratio
    plot one point

Repeat this for all segments, separately for display order and decode order.

# Give error range and let it determine the k values/ranks

--- 
target_error = epsilon

rank_for_error = smallest k such that:    
                                        ||X - X_k||_F / ||X||_F <= epsilon

---
Given epsilons: [0.01, 0.05, 0.10, 0.20]

For each segment matrix X:
  compute SVD once: X = U S Vt

 # Baseline:
 - FAISS PQ and RaBitQ are implemented as a separate global embedding-quantization stage.
 - They require embeddings in numpy format: outputs/embeddings/frame_embeddings.npy.
 - Their outputs are under outputs/quantization/.

 # [NOTE]: binary search!
  For each epsilon:
    try k = 1, 2, ..., min(N_i, d_model)
    compute relative_error(k) #two ways although same: sqrt(sum(S[k:]^2) / sum(S^2)) && ||X - X_k||_F / ||X||_F
    pick smallest k where relative_error(k) <= epsilon

    using selected k:
      achieved_error = ||X - X_k||_F / ||X||_F
      compression_ratio = original_params / svd_params
      PLOT!

## SVD Notes:

Example: I0, B1, P2, B3, I4, B5, P6, I7

So, we have (say, adjacent setting):

X = [    e(I0)−e(B1)   ]
    [    e(B1)−e(P2)   ]
    [    e(P2)−e(B3)​   ]

U, S, Vt = np.linalg.svd(X, full_matrices=False)

where, S is simply the diagonal of sigma, returned as a 1D array: S = [s1, s2, s3, ...] 

example: S = [10, 5, 2, 1] (singular values)
    means roughly:
    component 1 -> strength 10
    component 2 -> strength 5
    component 3 -> strength 2
    component 4 -> strength 1

SVD orders them from largest to smallest: s1 > s2 > s3 > s4 ... 

`This is why low-rank approximation works. If the first few values are huge and the rest tiny, `
`most of the matrix can be represented using only those first few components.`
like: S = [100, 40, 3, 0.5, 0.1]
is highly promising for low-rank compression because almost all the energy is concentrated in the first two components.

Now connect S to rank: Suppose: S = [10, 5, 2, 1]
If you choose: k = 2; it means: Keep the first two SVD components. [Mk x kk x kN]
keep [10,5]; discard [2,1]
