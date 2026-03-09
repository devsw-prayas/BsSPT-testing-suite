import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PARQUET_PATH = "datasets/stability_dataset.parquet"

print("Loading dataset...")
df = pd.read_parquet(PARQUET_PATH)

# remove SPD failures
df = df[df["spd_fail_flag"] == 0]

# raw basis only
df = df[df["whitened"] == 0]

# compute basis size
df["basis_size"] = df["K"] * df["order"]

# --------------------------------------------------------
# Stability score (same logic you used)
# --------------------------------------------------------

df["stability_score"] = (
    -np.log10(df["condition_number"]) +
    0.5 * df["spectral_entropy"] -
    0.25 * df["std_eigen"]
)

# --------------------------------------------------------
# Aggregate per (family, basis_size)
# --------------------------------------------------------

agg = (
    df.groupby(["family_id", "basis_size"])
    .agg({
        "stability_score": "mean"
    })
    .reset_index()
)

# --------------------------------------------------------
# Plot
# --------------------------------------------------------

plt.figure(figsize=(10,6))

families = sorted(agg["family_id"].unique())

for fam in families:
    sub = agg[agg["family_id"] == fam]
    plt.plot(
        sub["basis_size"],
        sub["stability_score"],
        marker="o",
        label=f"Family {fam}"
    )

plt.xlabel("Basis Size (K × order)")
plt.ylabel("Stability Score")
plt.title("GHGSF Stability vs Basis Size")
plt.legend()
plt.grid(True)

plt.show()