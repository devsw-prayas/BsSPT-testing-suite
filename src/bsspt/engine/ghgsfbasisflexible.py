import torch
from torch import Tensor
from typing import List, Literal, Optional

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis


ScaleType = Literal["constant", "linear", "sqrt", "power"]


class GHGSFMultiLobeBasisFlexible:
    """
    Gaussian-Hermite Multi-Lobe Basis
    with configurable sigma growth per Hermite order.

    Scaling modes:
        - constant
        - linear
        - sqrt
        - power (requires gamma)
    """

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma_min: float,
        sigma_max: Optional[float],
        order: int,
        scale_type: ScaleType = "sqrt",
        gamma: float = 0.5
    ):
        self.m_domain   = domain
        self.m_centers  = torch.tensor(
            centers, device=domain.m_device, dtype=domain.m_dtype
        )

        self.m_sigma_min  = sigma_min
        self.m_sigma_max  = sigma_max if sigma_max is not None else sigma_min
        self.m_order      = order
        self.m_scale_type = scale_type
        self.m_gamma      = gamma

        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        self.m_basisRaw      = None
        self.m_gram          = None
        self.m_chol          = None
        self.m_sigma_schedule = None

        self._buildBasis()
        self._buildGram()
        self._buildCholesky()

    # ---------------------------------------------------------
    # Sigma Schedule  [N]
    # ---------------------------------------------------------

    def _build_sigma_schedule(self, device, dtype) -> Tensor:

        N = self.m_N

        if N <= 1 or self.m_scale_type == "constant":
            return torch.full((N,), self.m_sigma_min, device=device, dtype=dtype)

        t     = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
        delta = self.m_sigma_max - self.m_sigma_min

        if self.m_scale_type == "linear":
            return self.m_sigma_min + delta * t
        elif self.m_scale_type == "sqrt":
            return self.m_sigma_min + delta * torch.sqrt(t)
        elif self.m_scale_type == "power":
            return self.m_sigma_min + delta * torch.pow(t, self.m_gamma)
        else:
            raise ValueError(f"Unknown scale_type: {self.m_scale_type}")

    # ---------------------------------------------------------
    # Basis Construction — fully batched
    # Previously: K*N separate hermiteBasis calls in nested loops
    # ---------------------------------------------------------

    def _buildBasis(self):

        lbda    = self.m_domain.m_lambda   # [L]
        centers = self.m_centers           # [K]
        device  = lbda.device
        dtype   = lbda.dtype
        K       = self.m_K
        N       = self.m_N
        L       = lbda.shape[0]

        sigma_sched = self._build_sigma_schedule(device, dtype)  # [N]
        self.m_sigma_schedule = sigma_sched

        # Normalization constants [N]
        n_idx      = torch.arange(N, device=device, dtype=dtype)
        factorials = torch.exp(torch.lgamma(n_idx + 1))
        sqrt_pi    = torch.tensor(torch.pi, device=device, dtype=dtype).sqrt()
        norms      = torch.sqrt((2.0 ** n_idx) * factorials * sqrt_pi)  # [N]

        # x[k, n, l] = (lambda[l] - centers[k]) / sigma_sched[n]
        lbda_exp    = lbda.unsqueeze(0).unsqueeze(0)       # [1, 1, L]
        centers_exp = centers.unsqueeze(1).unsqueeze(2)    # [K, 1, 1]
        sigma_exp   = sigma_sched.unsqueeze(0).unsqueeze(2) # [1, N, 1]

        x_full = (lbda_exp - centers_exp) / sigma_exp      # [K, N, L]
        x_flat = x_full.reshape(K * N, L)                  # [K*N, L]

        H_full = hermiteBasis(N, x_flat)                   # [K*N, N, L]

        row_idx = torch.arange(K * N, device=device)
        ord_idx = row_idx % N
        H_diag  = H_full[row_idx, ord_idx, :]              # [K*N, L]

        gaussian    = torch.exp(-0.5 * x_flat ** 2)        # [K*N, L]
        norms_tiled = norms.repeat(K)                      # [K*N]

        self.m_basisRaw = (H_diag * gaussian) / norms_tiled.unsqueeze(1)  # [M, L]

    # ---------------------------------------------------------
    # Gram / Cholesky
    # ---------------------------------------------------------

    def _buildGram(self):
        B = self.m_basisRaw
        w = self.m_domain.m_weights
        self.m_gram = (B * w) @ B.T

    def _buildCholesky(self):
        self.m_chol = torch.linalg.cholesky(self.m_gram)

    # ---------------------------------------------------------
    # Projection
    # ---------------------------------------------------------

    def project(self, spectrum: Tensor) -> Tensor:

        B = self.m_basisRaw
        w = self.m_domain.m_weights

        if spectrum.device != B.device:
            spectrum = spectrum.to(B.device)
        if spectrum.dtype != B.dtype:
            spectrum = spectrum.to(B.dtype)

        b     = ((B * w) @ spectrum).unsqueeze(1)
        y     = torch.linalg.solve_triangular(self.m_chol,   b, upper=False)
        alpha = torch.linalg.solve_triangular(self.m_chol.T, y, upper=True)

        return alpha.squeeze(1)

    # ---------------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------------

    def reconstruct(self, coeffs: Tensor) -> Tensor:

        B = self.m_basisRaw

        if coeffs.device != B.device:
            coeffs = coeffs.to(B.device)
        if coeffs.dtype != B.dtype:
            coeffs = coeffs.to(B.dtype)

        return coeffs @ B

    def get_sigma_schedule(self) -> Tensor:
        return self.m_sigma_schedule.clone()

    def get_sigma_stats(self):
        sigma = self.m_sigma_schedule
        return {
            "sigma_min_actual": sigma.min().item(),
            "sigma_max_actual": sigma.max().item(),
            "sigma_mean":       sigma.mean().item(),
            "sigma_std":        sigma.std().item(),
            "sigma_ratio":      (sigma.max() / sigma.min()).item()
        }
