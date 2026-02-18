# precision_comparison.py

import os
import json
import torch

from plotting.Plot import PlotEngine


class PrecisionComparator:

    def __init__(self, root_path):
        self.root = root_path
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "plots"), exist_ok=True)

    def _save_tensor(self, tensor, name):
        torch.save(
            tensor.detach().cpu(),
            os.path.join(self.root, name + ".pt")
        )

    def compare_single_run(
        self,
        fp64_path,
        tf32_path
    ):
        # --------------------------------------------
        # Load data
        # --------------------------------------------

        coeff_fp64 = torch.load(
            os.path.join(fp64_path, "coeff_history.pt")
        )

        coeff_tf32 = torch.load(
            os.path.join(tf32_path, "coeff_history.pt")
        )

        spectral_radius_fp64 = torch.load(
            os.path.join(fp64_path, "cumulative_spectral_radius.pt")
        )

        spectral_radius_tf32 = torch.load(
            os.path.join(tf32_path, "cumulative_spectral_radius.pt")
        )

        final_recon_fp64 = torch.load(
            os.path.join(fp64_path, "final_reconstruction.pt")
        )

        final_recon_tf32 = torch.load(
            os.path.join(tf32_path, "final_reconstruction.pt")
        )

        # --------------------------------------------
        # Coefficient Drift
        # --------------------------------------------

        diff = coeff_fp64 - coeff_tf32.to(coeff_fp64.dtype)
        rel_error = torch.linalg.norm(diff, dim=1) / (
            torch.linalg.norm(coeff_fp64, dim=1) + 1e-16
        )

        self._save_tensor(diff, "coeff_diff")
        self._save_tensor(rel_error, "rel_error_curve")

        # --------------------------------------------
        # Spectral Radius Drift
        # --------------------------------------------

        radius_drift = spectral_radius_fp64 - spectral_radius_tf32.to(
            spectral_radius_fp64.dtype
        )

        self._save_tensor(radius_drift, "spectral_radius_drift")

        # --------------------------------------------
        # Reconstruction Difference
        # --------------------------------------------

        recon_diff = final_recon_fp64 - final_recon_tf32.to(
            final_recon_fp64.dtype
        )

        l2_recon_error = torch.linalg.norm(recon_diff).item()

        self._save_tensor(recon_diff, "reconstruction_diff")

        metrics = {
            "final_l2_reconstruction_error": l2_recon_error,
            "max_relative_coeff_error": rel_error.max().item()
        }

        with open(
            os.path.join(self.root, "drift_metrics.json"),
            "w"
        ) as f:
            json.dump(metrics, f, indent=4)

        # --------------------------------------------
        # Plot Relative Error Curve
        # --------------------------------------------

        plot = PlotEngine(figsize=(6, 4))

        x = list(range(len(rel_error)))
        plot.addLine(x, rel_error.cpu().numpy())

        plot.setTitle("Relative Coefficient Error")
        plot.setLabels("Bounce", "Relative Error")

        plot.saveFigure(
            os.path.join(self.root, "plots", "rel_error_vs_bounce.png"),
            dpi=512
        )

        plot.close()
