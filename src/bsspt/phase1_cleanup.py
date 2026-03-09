import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PARQUET = "datasets/stability_dataset.parquet"
OUTPUT_PARQUET = "datasets/phase1_cleaned.parquet"
OUTPUT_CSV = "phase1_cleaned.csv"

LOGCOND_MIN = 4.0
LOGCOND_MAX = 8.0

VALID_FAMILIES = [0, 4]  # uniform, sawblade


# ============================================================
# Load Dataset
# ============================================================

print("Loading parquet dataset...")
df = pd.read_parquet(INPUT_PARQUET)

print("Total rows:", len(df))


# ============================================================
# Drop invalid numerical rows
# ============================================================

print("Removing SPD failures...")
df = df[df["spd_fail_flag"] == 0]


# ============================================================
# Stability filter
# ============================================================

print("Filtering by condition number stability...")
df = df[
    (df["log10_condition"] >= LOGCOND_MIN) &
    (df["log10_condition"] <= LOGCOND_MAX)
]


# ============================================================
# Family filter
# ============================================================

print("Filtering topology families...")
df = df[df["family_id"].isin(VALID_FAMILIES)]


# ============================================================
# Drop error rows
# ============================================================

if "error_msg" in df.columns:
    df = df[df["error_msg"].isna()]


# ============================================================
# Reset index
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# Save cleaned dataset
# ============================================================

print("Saving cleaned dataset...")

df.to_parquet(OUTPUT_PARQUET)
df.to_csv(OUTPUT_CSV, index=False)

print("Cleaned rows:", len(df))
print("Done.")