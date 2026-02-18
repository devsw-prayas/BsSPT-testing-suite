#  MASTER IMPLEMENTATION PHASES

---

#  PHASE 0 — Precision & Environment Control

**Goal:** Guarantee FP64 and TF32 behave exactly as intended.

### Implement:

* Precision mode switch
* TF32 enable/disable toggle
* Device logging
* CUDA + Torch version logging
* Deterministic flags
* Seed control

### Validate:

* FP64 tensors truly float64
* TF32 matmul speed differs from FP64
* No silent dtype mixing

If this fails → everything else invalid.

---

# PHASE 1 — Geometry Block (Per Config)

**Goal:** Build and dump basis geometry only.

### Implement:

* SpectralDomain
* GHGSFMultiLobeBasisFlexible
* Gram matrix
* Eigenvalues
* Cholesky
* Whitening (optional)

### Dump:

* geometry/ folder
* All geometry tensors
* 512 DPI plots

### Validate:

* κ(G) reasonable
* No NaNs
* Whitening reduces conditioning (when enabled)
* Folder structure correct

No transport yet.

---

#  PHASE 2 — Operator Construction Block

**Goal:** Build operator chains (without spectra yet).

### Implement:

* 3 operator chains
* Compose operators
* Dump operator matrices
* Compute singular values
* Compute non-normality metric

### Dump:

* precision/fp64/operators/
* precision/tf32/operators/

### Validate:

* Operators non-trivial
* Singular spectrum stable
* TF32 vs FP64 difference small initially

Still no transport loop.

---

#  PHASE 3 — Deterministic Transport (Single Spectrum)

**Goal:** Validate transport loop using 1 test spectrum.

### Implement:

* Projection to basis
* 15-bounce loop
* Cumulative operator tracking
* Reconstruction

### Dump:

* single_run block
* Norm curves
* Energy drift
* Reconstruction error

### Validate:

* FP64 stable
* TF32 small drift
* No explosion for small config

Transport skeleton proven.

---

#  PHASE 4 — All 16 Spectra Integration

**Goal:** Integrate IRL + Hostile spectra layer.

### Implement:

* Loop over 16 spectra
* For each:

  * Project
  * Run 3 operator chains
  * Single deterministic run

### Dump:

* spectra/spectrum_X/chain_Y/single_run

### Validate:

* Hostile spectra activate higher modes
* IRL spectra smoother
* Some configs show instability under hostile only

Now geometry + transport + spectra integrated.

---

#  PHASE 5 — Monte Carlo Layer (Mean Only)

**Goal:** Add 64-run MC averaging.

### Implement:

For each (spectrum, chain):

* Run 64 times
* Compute:

  * Mean coefficients
  * Variance
  * Mean reconstruction
  * Variance reconstruction

### Dump:

* mc_mean block
* Error statistics

### Validate:

* Mean converges
* Variance non-zero
* MC mean close to single deterministic baseline (if expected)

Now stochastic layer active.

---

#  PHASE 6 — Precision Comparison Layer

**Goal:** Compare FP64 vs TF32 results.

### Implement:

For each (spectrum, chain):

* Compute:

  * Coefficient drift
  * Relative error curve
  * Reconstruction difference
  * Collapse classification difference

### Dump:

* precision_comparison/

### Validate:

* Small configs → tiny drift
* Large configs → measurable drift
* No false positives

Now hardware precision impact measurable.

---

#  PHASE 7 — Stability & Collapse Classification

**Goal:** Automate failure detection.

### Implement:

* NaN detection
* Inf detection
* κ threshold
* Relative error threshold
* Collapse probability (in MC)
* First failure bounce

### Dump:

* stability_metrics.json per config

### Validate:

* Artificial unstable config triggers correctly
* Stable config not misclassified

Now system self-evaluates.

---

#  PHASE 8 — Performance Logging

**Goal:** Capture runtime + GPU stats.

### Implement:

* CUDA events timing
* Total runtime
* Transport loop time
* SVD time
* Max CUDA memory
* TF32 speedup ratio

### Dump:

* performance.json

### Validate:

* TF32 faster than FP64
* No memory leak
* Runtime scales with M

Now performance axis active.

---

#  PHASE 9 — Mini Sweep (10–20 Configs)

**Goal:** Sanity test full pipeline.

### Mix:

* Small M
* Mid M (~48)
* Max M (64)
* Constant scale
* Sqrt scale
* Whitening ON/OFF

### Validate:

* No crashes
* Folder structure correct
* Disk growth predictable
* Precision drift visible
* MC stable

Only after this passes → full sweep.

---

#  PHASE 10 — Full 800-Config Sweep

**Goal:** Launch entire grid.

Rules:

* No code changes mid-run
* No parameter tweaks
* Log progress
* Monitor disk + GPU

---

#  PHASE 11 — Post-Processing

After sweep completes:

* Build master CSV
* Collapse heatmap
* κ(G) surface
* Precision breakdown boundary
* Whitening effectiveness map
* Runtime surface

This is analysis phase.

---

#  Summary of Phases

0. Precision Control
1. Geometry
2. Operators
3. Deterministic Transport
4. 16 Spectra
5. Monte Carlo Mean
6. Precision Comparison
7. Collapse Classification
8. Performance Logging
9. Mini Sweep
10. Full Sweep
11. Analysis

