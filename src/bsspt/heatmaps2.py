import numpy as np
import pandas as pd

from plotting.Plot import MultiPanelEngine


# -----------------------------
# CONFIG
# -----------------------------
PARQUET_PATH = "datasets/stability_dataset.parquet"


print("Loading parquet...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Total rows: {len(df):,}")

# Remove SPD failures
df = df[df["spd_fail_flag"] == 0]

# RAW ONLY
df = df[df["whitened"] == 0]

print(f"Rows after SPD removal + raw filter: {len(df):,}")


# -----------------------------
# GLOBAL RANGES
# -----------------------------
global_ranges = {
    "log10_condition": (df["log10_condition"].min(), df["log10_condition"].max()),
    "lambda_min": (np.log10(df["lambda_min"].clip(lower=1e-20)).min(),
                   np.log10(df["lambda_min"].clip(lower=1e-20)).max()),
    "spectral_entropy": (df["spectral_entropy"].min(), df["spectral_entropy"].max()),
    "std_eigen": (df["std_eigen"].min(), df["std_eigen"].max()),
    "lambda_max": (df["lambda_max"].min(), df["lambda_max"].max()),
    "mean_eigen": (df["mean_eigen"].min(), df["mean_eigen"].max()),
    "trace_G": (df["trace_G"].min(), df["trace_G"].max()),
    "dominance_gap": (df["dominance_gap"].min(), df["dominance_gap"].max()),
}


families = sorted(df["family_id"].unique())


for family in families:

    print(f"\nProcessing family {family}")

    family_df = df[df["family_id"] == family]

    if len(family_df) == 0:
        continue

    # -----------------------------
    # Aggregate per (K, order)
    # -----------------------------
    agg = (
        family_df
        .groupby(["K", "order"])
        .agg({
            "log10_condition": "mean",
            "lambda_min": "mean",
            "spectral_entropy": "mean",
            "std_eigen": "mean",
            "lambda_max": "mean",
            "mean_eigen": "mean",
            "trace_G": "mean",
            "dominance_gap": "mean",
        })
        .reset_index()
    )

    if len(agg) == 0:
        continue


    # -----------------------------
    # Build Heatmap Matrix
    # -----------------------------
    def build_surface(metric_name, log_transform=False):

        pivot = agg.pivot(
            index="order",
            columns="K",
            values=metric_name
        )

        pivot = pivot.sort_index().sort_index(axis=1)

        if log_transform:
            pivot = np.log10(pivot.clip(lower=1e-20))

        Z = pivot.values
        K_vals = pivot.columns.values
        order_vals = pivot.index.values

        return Z, K_vals, order_vals


    Zc, K_vals, order_vals = build_surface("log10_condition")
    Zl, _, _ = build_surface("lambda_min", log_transform=True)
    Ze, _, _ = build_surface("spectral_entropy")
    Zs, _, _ = build_surface("std_eigen")
    Zm, _, _ = build_surface("lambda_max")
    Zme, _, _ = build_surface("mean_eigen")
    Zt, _, _ = build_surface("trace_G")
    Zd, _, _ = build_surface("dominance_gap")


    # -----------------------------
    # Multi Panel (2x4)
    # -----------------------------
    engine = MultiPanelEngine(
        nrows=2,
        ncols=4,
        figsize=(18, 10)
    )

    engine.setMainTitle(f"Family {family} | Spectral Geometry Dashboard")


    panels = [
        ("log10(condition)", Zc, *global_ranges["log10_condition"]),
        ("log10(lambda_min)", Zl, *global_ranges["lambda_min"]),
        ("spectral_entropy", Ze, *global_ranges["spectral_entropy"]),
        ("std_eigen", Zs, *global_ranges["std_eigen"]),
        ("lambda_max", Zm, *global_ranges["lambda_max"]),
        ("mean_eigen", Zme, *global_ranges["mean_eigen"]),
        ("trace_G", Zt, *global_ranges["trace_G"]),
        ("dominance_gap", Zd, *global_ranges["dominance_gap"]),
    ]


    for i, (title, Z, vmin, vmax) in enumerate(panels):

        panel = engine.getPanel(i)

        im = panel.m_axes.imshow(
            Z,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto"
        )

        panel.m_axes.set_xlabel("Lobe (K)")
        panel.m_axes.set_ylabel("Order")
        panel.m_axes.set_title(title)

        panel.m_axes.set_xticks(range(len(K_vals)))
        panel.m_axes.set_xticklabels(K_vals)

        panel.m_axes.set_yticks(range(len(order_vals)))
        panel.m_axes.set_yticklabels(order_vals)

        engine.m_figure.colorbar(
            im,
            ax=panel.m_axes,
            shrink=0.75,
            aspect=10
        )


    engine.show()
    engine.close()