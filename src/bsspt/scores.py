import numpy as np
import pandas as pd

from plotting.Plot import SurfaceEngine


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DATA_PATH = "datasets/stability_dataset.parquet"


print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)

print(f"Total rows: {len(df):,}")


# -------------------------------------------------
# FILTER BAD CONFIGS
# -------------------------------------------------

df = df[df["spd_fail_flag"] == 0]

print(f"Rows after SPD filter: {len(df):,}")


# -------------------------------------------------
# BUILD STABILITY SCORE
# -------------------------------------------------

safe_lambda = df["lambda_min"].clip(lower=1e-20)

df["log_lambda_min"] = np.log10(safe_lambda)

df["stability_score"] = -df["log10_condition"]

# -------------------------------------------------
# AGGREGATE OVER HIDDEN AXES
# -------------------------------------------------

agg = (
    df.groupby(["family_id", "K", "order"])
    .agg({
        "stability_score": "mean"
    })
    .reset_index()
)

print("Aggregation complete")


# -------------------------------------------------
# VISUALIZE PER FAMILY
# -------------------------------------------------

families = sorted(agg["family_id"].unique())


for family in families:

    print(f"\nRendering family {family}")

    family_df = agg[agg["family_id"] == family]

    pivot = family_df.pivot(
        index="order",
        columns="K",
        values="stability_score"
    )

    pivot = pivot.sort_index().sort_index(axis=1)

    K_vals = pivot.columns.values
    order_vals = pivot.index.values

    X, Y = np.meshgrid(K_vals, order_vals)
    Z = pivot.values


    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------

    engine = SurfaceEngine(figsize=(10, 8))

    engine.setTitle(f"GHGSF Stability Landscape | Family {family}")

    engine.addSurface(X, Y, Z)

    engine.setLabels(
        "Lobes (K)",
        "Polynomial Order",
        "Stability Score"
    )

    engine.setView(elev=35, azim=50)

    engine.show()
    engine.close()


# -------------------------------------------------
# PRINT BEST CONFIGS
# -------------------------------------------------

best = df.sort_values("stability_score", ascending=False).head(20)

print("\nTop 20 configurations:")
print(
    best[[
        "family_id",
        "K",
        "order",
        "stability_score",
        "log10_condition",
        "lambda_min",
        "spectral_entropy"
    ]]
)