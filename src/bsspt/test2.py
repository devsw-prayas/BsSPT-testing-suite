import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PARQUET_PATH = "datasets/stability_dataset.parquet"

df = pd.read_parquet(PARQUET_PATH)

# remove failures
df = df[df["spd_fail_flag"] == 0]
df = df[df["whitened"] == 0]

# stability score
df["stability_score"] = (
    -np.log10(df["condition_number"]) +
    0.5 * df["spectral_entropy"] -
    0.25 * df["std_eigen"]
)

agg = (
    df.groupby(["K","order"])
    .agg({"stability_score":"mean"})
    .reset_index()
)

plt.figure(figsize=(10,6))

for k in sorted(agg["K"].unique()):
    sub = agg[agg["K"]==k]
    plt.plot(sub["order"], sub["stability_score"], marker="o", label=f"K={k}")

plt.xlabel("Polynomial Order")
plt.ylabel("Stability Score")
plt.title("GHGSF Stability vs Hermite Order")
plt.legend()
plt.grid(True)

plt.show()