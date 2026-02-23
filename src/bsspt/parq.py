import pandas as pd
import glob

# Path to your shards
csv_files = sorted(glob.glob("phase1_output/phase1_batch_*.csv"))

print(f"Found {len(csv_files)} batch files.")

# Merge all shards
df = pd.concat(
    (pd.read_csv(f) for f in csv_files),
    ignore_index=True
)

print("Loaded shape:", df.shape)

# Reduce small integer columns safely
int8_cols = [
    "family_id",
    "scaling_id",
    "precision_id",
    "whitened",
    "tf32_safe_flag",
    "fp32_safe_flag",
    "fp64_safe_flag",
    "spd_fail_flag"
]

for col in int8_cols:
    if col in df.columns:
        df[col] = df[col].astype("int8")

print("Writing Parquet...")

df.to_parquet(
    "stability_dataset.parquet",
    engine="pyarrow",
    compression="zstd", 
    index=False
)

print("Conversion complete.")