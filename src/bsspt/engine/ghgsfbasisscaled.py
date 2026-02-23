import torch
from torch import Tensor
from typing import List

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis


class GHGSFMultiLobeBasisScaled:
    """
    Gaussian-Hermite Multi-Lobe Basis
    with sqrt growth of sigma per Hermite order.

        sigma_n = sigma_min + beta * sqrt(n)
        beta    = (sigma_max - sigma_min) / sqrt(N - 1)
    """

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma_min: float,
        sigma_max: float,
        order: int
    ):
        self.m_domain   = domain
        self.m_centers  = torch.tensor(
            centers, device=domain.m_device, dtype=domain.m_dtype
        )

        self.m_sigma_min = sigma_min
        self.m_sigma_max = sigma_max
        self.m_order     = order

        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        self.m_basisRaw = None
        self.m_gram     = None
        self.m_chol     = None

        self._buildBasis()
        self._buildGram()
        self._buildCholesky()

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

        # Sigma schedule: sigma_n = sigma_min + beta * sqrt(n)
        n_idx = torch.arange(N, device=device, dtype=dtype)
        if N > 1:
            beta = (self.m_sigma_max - self.m_sigma_min) / (
                torch.tensor(float(N - 1), device=device, dtype=dtype).sqrt()
            )
        else:
            beta = torch.tensor(0.0, device=device, dtype=dtype)

        sigma_sched = self.m_sigma_min + beta * torch.sqrt(n_idx)  # [N]

        # Normalization constants [N]
        factorials  = torch.exp(torch.lgamma(n_idx + 1))
        sqrt_pi     = torch.tensor(torch.pi, device=device, dtype=dtype).sqrt()
        norms       = torch.sqrt((2.0 ** n_idx) * factorials * sqrt_pi)  # [N]

        # x[k, n, l] = (lambda[l] - centers[k]) / sigma_sched[n]
        lbda_exp    = lbda.unsqueeze(0).unsqueeze(0)        # [1, 1, L]
        centers_exp = centers.unsqueeze(1).unsqueeze(2)     # [K, 1, 1]
        sigma_exp   = sigma_sched.unsqueeze(0).unsqueeze(2) # [1, N, 1]

        x_full = (lbda_exp - centers_exp) / sigma_exp       # [K, N, L]
        x_flat = x_full.reshape(K * N, L)                   # [K*N, L]

        H_full = hermiteBasis(N, x_flat)                    # [K*N, N, L]

        row_idx = torch.arange(K * N, device=device)
        ord_idx = row_idx % N
        H_diag  = H_full[row_idx, ord_idx, :]               # [K*N, L]

        gaussian    = torch.exp(-0.5 * x_flat ** 2)         # [K*N, L]
        norms_tiled = norms.repeat(K)                       # [K*N]

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
