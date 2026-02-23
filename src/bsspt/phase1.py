import os
import torch
import pandas as pd
import traceback

from engine.spectraldomain import SpectralDomain
from engine.ghgsfexp import GHGSFMultiLobeBasisDualDomain
from spectral_topology import generate_topology
from torchconfig import TorchConfig
from build_configs import build_phase1_configs
from schema import CONFIG_COLUMNS, METRIC_COLUMNS, SCALING_ID_MAP, PHASE1_VERSION

# ============================================================
# GLOBAL SETTINGS
# ============================================================

DISK_BATCH_SIZE = 1024
SUB_BATCH_SIZE  = 64

# D65 domain
LAMBDA_MIN     = 380.0
LAMBDA_MAX     = 830.0
LAMBDA_SAMPLES = 4096

OUTPUT_DIR = "phase1_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# METRIC COMPUTATION
# Returns:
#   (row_index, metrics_tensor, error_string)
# ============================================================

def compute_metrics(args):

    row_index, config_vals = args

    try:
        (
            family_id, K, order, scaling_id, precision_id, whitened,
            wide_min, wide_max, narrow_min, narrow_max
        ) = config_vals

        K          = int(K)
        order      = int(order)
        family_id  = int(family_id)
        scaling_id = int(scaling_id)
        whitened   = int(whitened)

        precision_mode = "performance" if precision_id == 0 else "reference"
        torch_info = TorchConfig.set_mode(precision_mode, verbose=False)

        device = torch_info["device"]
        dtype  = torch_info["dtype"]

        scale_type = SCALING_ID_MAP[scaling_id]

        domain = SpectralDomain(
            LAMBDA_MIN, LAMBDA_MAX, LAMBDA_SAMPLES,
            device=device, dtype=dtype
        )

        centers  = generate_topology(family_id, K)
        num_wide = K // 2

        basis = GHGSFMultiLobeBasisDualDomain(
            domain=domain,
            centers=centers,
            num_wide=num_wide,
            wide_sigma_min=float(wide_min),
            wide_sigma_max=float(wide_max),
            wide_scale_type=scale_type,
            narrow_sigma_min=float(narrow_min),
            narrow_sigma_max=float(narrow_max),
            narrow_scale_type=scale_type,
            order=order
        )

        # ---- Gram selection ----
        if whitened:
            L   = basis.m_chol
            LiG = torch.linalg.solve_triangular(L, basis.m_gram, upper=False)
            G   = torch.linalg.solve_triangular(L, LiG.T, upper=False).T
        else:
            G = basis.m_gram

        eigenvals = torch.linalg.eigvalsh(G)

        # SPD guard
        if eigenvals[0] <= 0.0:
            raise ValueError(
                f"Non-positive min eigenvalue {eigenvals[0].item():.3e} — Gram not SPD."
            )

        lam_min = eigenvals[0]
        lam_2   = eigenvals[1] if eigenvals.shape[0] > 1 else lam_min
        lam_max = eigenvals[-1]

        cond     = lam_max / lam_min
        log_cond = torch.log10(cond)
        trace_G  = torch.trace(G)
        mean_eig = torch.mean(eigenvals)
        std_eig  = torch.std(eigenvals)

        prob             = eigenvals / torch.sum(eigenvals)
        spectral_entropy = -torch.sum(prob * torch.log(prob + 1e-12))
        eigen_gap_ratio  = lam_2 / lam_min

        wide_bandwidth   = wide_max - wide_min
        narrow_bandwidth = narrow_max - narrow_min
        dominance_gap    = wide_bandwidth - narrow_bandwidth
        domain_ratio     = wide_max / (narrow_max + 1e-8)
        bandwidth_ratio  = wide_bandwidth / (narrow_bandwidth + 1e-8)

        metrics = torch.tensor([
            float(K * order),
            float(wide_bandwidth),
            float(narrow_bandwidth),
            float(dominance_gap),
            float(domain_ratio),
            float(bandwidth_ratio),
            lam_min.item(),
            lam_2.item(),
            lam_max.item(),
            cond.item(),
            log_cond.item(),
            trace_G.item(),
            mean_eig.item(),
            std_eig.item(),
            spectral_entropy.item(),
            eigen_gap_ratio.item(),
            1.0 if cond.item() < 1e4  else 0.0,
            1.0 if cond.item() < 1e6  else 0.0,
            1.0 if cond.item() < 1e12 else 0.0,
            0.0
        ], dtype=torch.float64)

        return row_index, metrics, ""

    except Exception:
        error_str = traceback.format_exc()

        failed = torch.zeros(len(METRIC_COLUMNS), dtype=torch.float64)
        failed[19] = 1.0  # spd_fail_flag

        return row_index, failed, error_str


# ============================================================
# SUB BATCH PROCESSING
# ============================================================

def process_sub_batch(config_tensor: torch.Tensor):

    B = config_tensor.shape[0]
    metrics_list = []
    error_list   = []

    for i in range(B):
        row = config_tensor[i].tolist()
        row_idx, metrics, error_str = compute_metrics((i, row))
        metrics_list.append(metrics)
        error_list.append(error_str)

    return torch.stack(metrics_list), error_list


# ============================================================
# MAIN SWEEP
# ============================================================

def run_phase1():

    torch.set_grad_enabled(False)

    configs       = build_phase1_configs()
    total_configs = configs.shape[0]
    num_batches   = (total_configs + DISK_BATCH_SIZE - 1) // DISK_BATCH_SIZE

    print(f"Phase 1 sweep")
    print(f"  Total configs  : {total_configs:,}")
    print(f"  Disk batches   : {num_batches}")
    print(f"  Lambda samples : {LAMBDA_SAMPLES}")
    print(f"  Output dir     : {OUTPUT_DIR}")

    for batch_id in range(num_batches):

        pt_path  = os.path.join(OUTPUT_DIR, f"phase1_batch_{batch_id}.pt")
        csv_path = pt_path.replace(".pt", ".csv")

        if os.path.exists(pt_path):
            print(f"  Batch {batch_id:4d} — exists, skipping.")
            continue

        try:
            print(f"  Batch {batch_id:4d} — starting...")

            start      = batch_id * DISK_BATCH_SIZE
            end        = min(start + DISK_BATCH_SIZE, total_configs)

            disk_batch = configs[start:end].clone()

            all_metrics = []
            all_errors  = []

            for sub_start in range(0, disk_batch.shape[0], SUB_BATCH_SIZE):
                sub_end   = min(sub_start + SUB_BATCH_SIZE, disk_batch.shape[0])
                sub_batch = disk_batch[sub_start:sub_end]

                metrics, errors = process_sub_batch(sub_batch)
                all_metrics.append(metrics)
                all_errors.extend(errors)

                pct = 100.0 * sub_end / disk_batch.shape[0]
                print(f"    {sub_end}/{disk_batch.shape[0]} ({pct:.0f}%)", end="\r")

            print()

            all_metrics = torch.cat(all_metrics, dim=0)

            torch.save({
                "phase1_version": PHASE1_VERSION,
                "configs":        disk_batch,
                "metrics":        all_metrics,
                "config_columns": CONFIG_COLUMNS,
                "metric_columns": METRIC_COLUMNS,
            }, pt_path)

            df_cfg = pd.DataFrame(disk_batch.numpy(), columns=CONFIG_COLUMNS)
            df_met = pd.DataFrame(all_metrics.numpy(), columns=METRIC_COLUMNS)
            df_err = pd.DataFrame({"error_msg": all_errors})

            int_cols = ["family_id", "K", "order", "scaling_id", "precision_id", "whitened"]
            for col in int_cols:
                df_cfg[col] = df_cfg[col].astype(int)

            pd.concat([df_cfg, df_met, df_err], axis=1).to_csv(csv_path, index=False)

            spd_fails = int(all_metrics[:, -1].sum().item())
            real_errors = sum(1 for e in all_errors if e)

            print(
                f"  Batch {batch_id:4d} — done. "
                f"SPD failures: {spd_fails}/{disk_batch.shape[0]}  "
                f"({real_errors} with traceback)"
            )

        except Exception:
            print(f"  Batch {batch_id:4d} — CRASHED.")
            traceback.print_exc()


if __name__ == "__main__":
    run_phase1()
