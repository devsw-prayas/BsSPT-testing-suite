

#  FINAL DATASET STRUCTURE

Root:

```
FRSTA_v1_0/
│
├── metadata/
├── global_summary/
└── configs/
```

---

#  metadata/

Global, single-copy info:

```
metadata/
│
├── experiment_manifest.json
├── hardware_info.json
├── torch_cuda_info.json
├── sweep_definition.json
└── git_commit.txt
```

Contains:

* K range
* N range
* sigma windows
* scale types
* whitening toggle
* number of spectra
* operator chains definition
* MC count (64)
* transport depth (15)
* DPI (512)
* device name
* CUDA version
* TF32 flags

This guarantees reproducibility.

---

#  configs/

Each configuration gets ONE folder.

Naming convention:

```
K{K}_N{N}_S{sigmaMin}-{sigmaMax}_{scale}_W{0|1}
```

Example:

```
K8_N6_S5.0-10.0_sqrt_W1
```

Inside each config:

```
config_X/
│
├── geometry/
├── precision/
└── spectra/
```

---

#  geometry/

Stored ONCE per config (not per precision).

```
geometry/
│
├── lambda_grid.pt
├── weights.pt
├── basis_raw.pt
├── sigma_schedule.pt
├── gram.pt
├── gram_eigenvalues.pt
├── chol.pt
├── geometry_metrics.json
└── plots/
```

`plots/` contains:

* basis_overlay.png
* basis_heatmap.png
* sigma_schedule.png
* gram_heatmap.png
* gram_eigen_log.png

No precision duplication here.

---

#  precision/

Two subfolders:

```
precision/
│
├── fp64/
└── tf32/
```

Inside each:

```
fp64/
│
├── operators/
├── cumulative/
├── performance.json
└── stability_metrics.json
```

Operators and cumulative matrices are precision-dependent.

---

#  spectra/

Here’s where scale explodes, so structure must be clean.

```
spectra/
│
├── spectrum_00_IRL/
├── spectrum_01_IRL/
...
├── spectrum_07_IRL/
├── spectrum_08_HOSTILE/
...
├── spectrum_15_HOSTILE/
```

Inside each spectrum:

```
spectrum_X/
│
├── chain_0/
├── chain_1/
└── chain_2/
```

---

#  chain_X/

Inside each chain:

```
chain_X/
│
├── fp64/
└── tf32/
```

Inside precision folder:

```
fp64/
│
├── single_run/
├── mc_mean/
└── precision_comparison/   (only in tf32 folder)
```

---

#  single_run/

```
single_run/
│
├── final_coeffs.pt
├── reconstruction.pt
├── reconstruction_error.json
├── norm_curve.pt
├── energy_curve.pt
└── plots/
```

Plots:

* reconstruction.png
* error_curve.png
* norm_growth.png
* energy_drift.png

---

#  mc_mean/

```
mc_mean/
│
├── mean_coeffs.pt
├── var_coeffs.pt
├── mean_reconstruction.pt
├── var_reconstruction.pt
├── error_stats.json
└── plots/
```

Plots:

* mean_reconstruction.png
* variance_curve.png
* mc_error_distribution.png

---

#  precision_comparison/

Inside TF32 chain folder only:

```
precision_comparison/
│
├── coeff_diff.pt
├── rel_error_curve.pt
├── reconstruction_diff.pt
├── drift_metrics.json
└── plots/
```

Plots:

* rel_error_vs_bounce.png
* reconstruction_overlay.png
* singular_value_drift.png

---

#  global_summary/

Generated after sweep finishes.

```
global_summary/
│
├── master_metrics.csv
├── collapse_heatmap.png
├── precision_boundary_surface.png
├── whitening_effect_heatmap.png
├── runtime_surface.png
└── phase_diagram.png
```

---

#  Why This Structure Is Optimal

* Geometry stored once per config
* Operators stored per precision only
* Spectra nested cleanly
* Chains separated cleanly
* MC data separate from deterministic run
* Precision comparison isolated
* PNG confined to `plots/`
* No redundant duplication
* Easily script-parsable
* Easy to tar/compress later

---

#  Final Storage Expectation @ 512 DPI

* Geometry block: ~24 GB
* Spectrum + chain block: ~100–110 GB
* Metadata: negligible

Total:

>  ~120–140 GB

Fits comfortably on 1TB NVMe.
Leaves room for future sweeps.

