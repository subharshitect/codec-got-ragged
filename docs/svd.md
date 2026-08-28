# Size of the rank
k <= min(N_i, d_model)

# Compression ratio
original params = N * d_model
svd params at rank k = k * (N + d_model + 1)
compression_ratio = original params / svd params

# Error
Relative reconstruction error: ||X - X_k||_F / ||X||_F
- ||X - X_k||_F = total reconstruction error
- ||X||_F = total magnitude/energy of the original matrix

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
            embedding_delta_previous_i: rows are e(previous I)     -   delta1_with_I-frame
            embedding_delta_adjacent:   rows are e(previous frame) -   delta1_with_adjacent
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

