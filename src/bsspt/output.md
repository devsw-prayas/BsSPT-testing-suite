

#  FULL FRSTA DATA DUMP MANIFEST

*(Per Config × Per Precision Mode)*

For each configuration defined by:

* K
* N
* sigma_min
* sigma_max
* scale_type ∈ {constant, linear, sqrt, power}
* whitening ∈ {OFF, ON}
* precision_mode ∈ {FP64, TF32}

You dump the following:

---

# BASIS BLOCK

### Scalars / Metadata

* K
* N
* M = K × N
* sigma_min
* sigma_max
* scale_type
* gamma (for power)
* sigma_schedule vector
* sigma_min_actual
* sigma_max_actual
* sigma_ratio
* domain resolution (L)
* lambda_min
* lambda_max
* precision_mode
* whitening flag

### Tensors

* lambda_grid [L]
* quadrature_weights [L]
* basis_raw B [M × L]
* Gram matrix G [M × M]
* Gram eigenvalues [M]
* Gram eigenvectors [M × M]
* Gram condition number κ(G)
* Cholesky factor L [M × M]
* Cholesky diagonal entries

### Whitening (if enabled)

* Whitening matrix A = Lᵀ
* Unwhitening matrix A⁻¹
* Whitened Gram
* Whitened eigenvalues
* Whitened condition number

### PNGs

* Basis curves overlay
* Basis heatmap
* Sigma schedule curve
* Gram heatmap
* Gram eigen spectrum (log scale)
* Whitening before/after comparison

---

# OPERATOR BLOCK

*(For each operator type used in chain)*

### Raw

* Sampled spectral function
* Weighted spectral function

### Matrices

* M_raw (projection matrix before solve)
* A (Galerkin operator)
* b (affine vector)

### Spectral Diagnostics

* Singular values
* Eigenvalues
* Spectral radius
* Frobenius norm
* Operator condition number
* Rank estimate
* Non-normality metric ‖AᵀA − AAᵀ‖
* Symmetry deviation

### PNGs

* Operator heatmap
* Singular value plot (log scale)
* Eigenvalue spectrum plot
* Non-normality metric plot

---

# TRANSPORT BLOCK (Depth = 15)

For each bounce k ∈ {0…14}:

### State

* Incoming coefficient vector
* Outgoing coefficient vector
* L2 norm of coefficients
* Energy via reconstruction integral
* Energy drift
* Relative energy error

### Operator

* Operator matrix A_k
* Affine vector b_k

### Cumulative Operator

* A_cum_k = A_k … A_1
* Singular values of A_cum_k
* Eigenvalues of A_cum_k
* Spectral radius of A_cum_k
* Frobenius norm
* Condition number
* Non-normality metric

### Canonical Action

* Canonical basis action matrix
* Norm of each canonical column
* Dominant singular vector
* Alignment with previous dominant vector
* Angle between coefficient vector and dominant singular vector

### Growth Metrics

* Log norm growth
* Estimated exponential growth rate λ
* Collapse flag at bounce
* First NaN detection
* First Inf detection

### PNGs

* Coefficient magnitude evolution
* Norm growth curve
* Energy drift curve
* Cumulative spectral radius curve
* Cumulative condition curve
* Canonical amplification heatmap
* Dominant mode alignment curve
* Energy redistribution across modes
* Log norm vs bounce curve

---

# RECONSTRUCTION BLOCK

### Tensors

* Dense lambda grid
* Final reconstructed spectrum
* Ground truth spectrum
* Per-wavelength error
* L2 spectral error
* Max spectral deviation
* Integrated energy difference

### PNGs

* Spectrum overlay
* Error vs wavelength
* Final reconstruction heatmap

---

# PRECISION COMPARISON BLOCK

*(FP64 vs TF32)*

For each bounce:

### Differences

* Coefficient difference vector
* Relative coefficient error
* Cumulative operator difference
* Frobenius difference
* Singular value drift
* Spectral radius drift
* Condition number drift
* Dominant singular vector angle drift
* Relative error growth curve

### Final Differences

* Reconstruction difference curve
* L2 spectrum difference
* Max spectral deviation difference
* Collapse region classification difference

### PNGs

* Relative error vs bounce
* Operator difference heatmap
* Singular value drift curve
* Dominant mode drift curve
* Spectrum difference overlay

---

# PERFORMANCE BLOCK (GPU)

Per precision mode:

* Total runtime
* Gram build time
* Operator build time
* Transport loop time
* SVD total time
* CUDA max memory allocated
* GPU name
* CUDA capability
* Torch version
* TF32 enabled flag
* Speedup ratio (TF32 vs FP64)

### PNGs

* Runtime vs dimension
* Speedup vs dimension

---

# STABILITY METRICS

Per config:

* κ(G)
* κ(G) × ε_machine
* Minimum singular value encountered
* Maximum singular value encountered
* Effective rank collapse threshold
* First bounce of instability
* Whitening effectiveness delta
* Final classification:

  * STABLE
  * MARGINAL
  * EXPLOSIVE
  * NUMERICAL_COLLAPSE

---

# GLOBAL SWEEP SUMMARY

Across all configs:

* κ(G) vs M surface
* Collapse heatmap over (K,N)
* Collapse heatmap over sigma window
* Collapse heatmap over scale type
* Whitening rescue heatmap
* Precision breakdown boundary curve
* TF32 speedup surface
* Stability phase diagram
* Master CSV summary of all metrics

---

# WHAT THIS DATASET CONTAINS

We are storing:

* Full Hilbert geometry
* Full operator cascade
* Full cumulative dynamics
* Full singular spectrum evolution
* Full precision drift trajectories
* Full performance metrics
* Full collapse boundaries

Across 1600 configurations.
