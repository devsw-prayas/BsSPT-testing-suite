import numpy as np
import pandas as pd

# Import your plotting engine
from plotting.Plot import SurfaceEngine


# -----------------------------
# CONFIG
# -----------------------------
PARQUET_PATH = "datasets/stability_dataset.parquet"   # change if needed
FIX_SCALING_ID = None                  # set to int if you want a specific scaling
FIX_WHITENED = None                    # set to 0 or 1 if desired
FIX_PRECISION_ID = None                # optional precision filtering


# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading parquet...")
df = pd.read_parquet(PARQUET_PATH)

print(f"Total rows: {len(df):,}")


# -----------------------------
# REMOVE SPD FAILS
# -----------------------------
df = df[df["spd_fail_flag"] == 0]

print(f"Rows after SPD removal: {len(df):,}")


# -----------------------------
# OPTIONAL GLOBAL FILTERS
# -----------------------------
if FIX_SCALING_ID is not None:
    df = df[df["scaling_id"] == FIX_SCALING_ID]

if FIX_WHITENED is not None:
    df = df[df["whitened"] == FIX_WHITENED]

if FIX_PRECISION_ID is not None:
    df = df[df["precision_id"] == FIX_PRECISION_ID]


# -----------------------------
# PROCESS PER FAMILY
# -----------------------------
families = sorted(df["family_id"].unique())

print(f"Found families: {families}")

for family in families:

    print(f"\nProcessing family {family}")

    family_df = df[df["family_id"] == family]

    if len(family_df) == 0:
        print("No valid rows after filtering. Skipping.")
        continue

    # ---------------------------------
    # Aggregate per (K, order)
    # ---------------------------------
    agg = (
        family_df
        .groupby(["K", "order"])
        .agg({
            "log10_condition": "mean"
        })
        .reset_index()
    )

    if len(agg) == 0:
        print("Aggregation empty. Skipping.")
        continue

    # ---------------------------------
    # Pivot to grid
    # ---------------------------------
    pivot = agg.pivot(
        index="order",
        columns="K",
        values="log10_condition"
    )

    # Sort axes for consistent geometry
    pivot = pivot.sort_index().sort_index(axis=1)

    # Drop incomplete rows/columns (optional safety)
    pivot = pivot.dropna()

    if pivot.shape[0] == 0 or pivot.shape[1] == 0:
        print("Pivot grid empty after dropna. Skipping.")
        continue

    # ---------------------------------
    # Build meshgrid
    # ---------------------------------
    K_vals = pivot.columns.values
    order_vals = pivot.index.values

    X, Y = np.meshgrid(K_vals, order_vals)
    Z = pivot.values

    # ---------------------------------
    # Plot 3D Surface
    # ---------------------------------
    engine = SurfaceEngine(figsize=(10, 8))

    engine.addSurface(X, Y, Z)

    engine.setLabels(
        "Lobe (K)",
        "Order",
        "log10(condition)"
    )

    engine.setTitle(
        f"Family {family} | Condition Surface (SPD-safe)"
    )

    engine.setView(elev=35, azim=55)

    engine.show()

    engine.close()