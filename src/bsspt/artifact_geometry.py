# artifact_geometry.py

import torch
import os
import json
from bsspt.plotting.Plot import PlotEngine


class GeometryEmitter:

    def __init__(self, root):
        self.root = root
        self.geometry_path = os.path.join(root, "geometry")
        os.makedirs(self.geometry_path, exist_ok=True)

    def _save_tensor(self, tensor, name):
        torch.save(tensor.detach().cpu(),
                   os.path.join(self.geometry_path, name + ".pt"))

    def dump_geometry(self, basis, whitening):

        # Save core tensors
        self._save_tensor(basis.m_basisRaw, "basis_raw")
        self._save_tensor(basis.m_gram, "gram")
        self._save_tensor(basis.m_chol, "chol")
        self._save_tensor(basis.m_sigma_schedule, "sigma_schedule")

        # Eigenvalues
        eigvals = torch.linalg.eigvalsh(basis.m_gram)
        self._save_tensor(eigvals, "gram_eigenvalues")

        cond = (eigvals.max() / eigvals.min()).item()

        # Metrics
        metrics = {
            "dimension": basis.m_M,
            "sigma_min": basis.m_sigma_schedule.min().item(),
            "sigma_max": basis.m_sigma_schedule.max().item(),
            "condition_number": cond,
            "whitening": whitening
        }

        with open(os.path.join(self.geometry_path, "geometry_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # 512 DPI plots
        self._plot_basis(basis)
        self._plot_gram(basis)

    def _plot_basis(self, basis):
        plot = PlotEngine(figsize=(10, 6))

        x = basis.m_domain.m_lambda.cpu().numpy()

        for i in range(basis.m_M):
            y = basis.m_basisRaw[i].cpu().numpy()
            plot.addLine(x, y, label=f"mode {i}")

        plot.setTitle("GHGSF Basis Functions")
        plot.setLabels("Wavelength (nm)", "Amplitude")
        plot.addLegend(location="upper right")

        plot.saveFigure(
            os.path.join(self.geometry_path, "basis_overlay.png"),
            dpi=512
        )

        plot.close()

    def _plot_gram(self, basis):
        plot = PlotEngine(figsize=(8, 6))

        gram = basis.m_gram.detach().cpu().numpy()

        im = plot.m_axes.imshow(
            gram,
            cmap="viridis",
            aspect="auto"
        )

        plot.setTitle("Gram Matrix Heatmap")
        plot.setLabels("Mode Index", "Mode Index")

        plot.m_figure.colorbar(im, ax=plot.m_axes)

        plot.saveFigure(
            os.path.join(self.geometry_path, "gram_heatmap.png"),
            dpi=512
        )

        plot.close()
