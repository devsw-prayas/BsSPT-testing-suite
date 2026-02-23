PHASE1_VERSION = "phase1_v2.1"

CONFIG_COLUMNS = [
    "family_id",
    "K",
    "order",
    "scaling_id",
    "precision_id",
    "whitened",
    "wide_min",
    "wide_max",
    "narrow_min",
    "narrow_max",
]

METRIC_COLUMNS = [
    "basis_size",
    "wide_bandwidth",
    "narrow_bandwidth",
    "dominance_gap",
    "domain_ratio",
    "bandwidth_ratio",
    "lambda_min",
    "lambda_2",
    "lambda_max",
    "condition_number",
    "log10_condition",
    "trace_G",
    "mean_eigen",
    "std_eigen",
    "spectral_entropy",
    "eigen_gap_ratio",
    "tf32_safe_flag",
    "fp32_safe_flag",
    "fp64_safe_flag",
    "spd_fail_flag",
]

# Stored separately from METRIC_COLUMNS — not part of the numeric tensor.
# Written as its own column in the CSV only.
ERROR_COLUMN = "error_msg"

# Maps scaling_id integer (from config tensor) to ScaleType string
SCALING_ID_MAP = {
    0: "constant",
    1: "linear",
    2: "sqrt",
    3: "power",
}