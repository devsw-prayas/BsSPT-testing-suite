# transport_emitter.py

import os
import json
import torch

from engine.spectraloperator import SpectralOperator
from engine.spectralstate import SpectralState
from plotting.Plot import PlotEngine


class TransportEmitter:

    def __init__(self, root_path, transport_depth=15):
        self.root = root_path
        self.depth = transport_depth

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "plots"), exist_ok=True)

    def _save_tensor(self, tensor, name):
        torch.save(
            tensor.detach().cpu(),
            os.path.join(self.root, name + ".pt")
        )

    def run_single_deterministic(
        self,
        basis,
        domain,
        operator_chain: SpectralOperator,
        initial_coeffs: torch.Tensor
    ):

        state = SpectralState(basis, initial_coeffs)
        cumulative = SpectralOperator.identity(basis)

        coeff_history = []
        norm_curve = []
        energy_curve = []
        spectral_radius_curve = []
        singular_value_curve = []

        first_nan = -1
        first_inf = -1

        for bounce in range(self.depth):

            # ---- Save state before applying operator ----
            coeff_history.append(state.m_coeffs.clone())

            norm_curve.append(state.norm().item())

            recon = basis.reconstruct(state.m_coeffs)
            energy = domain.integrate(recon).item()
            energy_curve.append(energy)

            # ---- Apply operator ----
            operator_chain.apply(state)

            # ---- Update cumulative ----
            cumulative = operator_chain.compose(cumulative)

            A_cum = cumulative.m_A

            # ---- Spectral diagnostics ----
            S = torch.linalg.svdvals(A_cum)
            singular_value_curve.append(S)

            eigvals = torch.linalg.eigvals(A_cum)
            spectral_radius = eigvals.abs().max().item()
            spectral_radius_curve.append(spectral_radius)

            # ---- Instability detection ----
            if torch.isnan(state.m_coeffs).any() and first_nan == -1:
                first_nan = bounce

            if torch.isinf(state.m_coeffs).any() and first_inf == -1:
                first_inf = bounce

        # --------------------------------------------------------
        # Save tensors
        # --------------------------------------------------------

        self._save_tensor(torch.stack(coeff_history), "coeff_history")
        self._save_tensor(torch.tensor(norm_curve), "norm_curve")
        self._save_tensor(torch.tensor(energy_curve), "energy_curve")
        self._save_tensor(torch.stack(singular_value_curve), "cumulative_singular_values")
        self._save_tensor(torch.tensor(spectral_radius_curve), "cumulative_spectral_radius")
        self._save_tensor(state.m_coeffs, "final_coeffs")

        final_recon = basis.reconstruct(state.m_coeffs)
        self._save_tensor(final_recon, "final_reconstruction")

        # --------------------------------------------------------
        # Stability Metrics
        # --------------------------------------------------------

        metrics = {
            "first_nan_bounce": first_nan,
            "first_inf_bounce": first_inf,
            "max_norm": max(norm_curve),
            "min_norm": min(norm_curve),
            "final_spectral_radius": spectral_radius_curve[-1],
            "max_singular_value": torch.stack(singular_value_curve).max().item(),
            "min_singular_value": torch.stack(singular_value_curve).min().item()
        }

        with open(os.path.join(self.root, "stability_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # --------------------------------------------------------
        # Plots
        # --------------------------------------------------------

        self._plot_curve(norm_curve, "Norm Growth", "Bounce", "||α||", "norm_growth.png")
        self._plot_curve(energy_curve, "Energy Drift", "Bounce", "Energy", "energy_drift.png")
        self._plot_curve(spectral_radius_curve, "Cumulative Spectral Radius",
                         "Bounce", "ρ(A_cum)", "spectral_radius.png")

    def _plot_curve(self, y, title, xlabel, ylabel, filename):

        plot = PlotEngine(figsize=(6, 4))

        x = list(range(len(y)))
        plot.addLine(x, y)

        plot.setTitle(title)
        plot.setLabels(xlabel, ylabel)

        plot.saveFigure(
            os.path.join(self.root, "plots", filename),
            dpi=512
        )

        plot.close()
