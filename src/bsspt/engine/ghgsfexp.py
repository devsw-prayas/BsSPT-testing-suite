import torch
from torch import Tensor
from typing import List, Literal, Optional

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis


ScaleType = Literal["constant", "linear", "sqrt", "power"]

# Exported so phase1.py can map scaling_id -> ScaleType without duplicating the table
SCALE_TYPE_MAP = {
    0: "constant",
    1: "linear",
    2: "sqrt",
    3: "power",
}


class GHGSFMultiLobeBasisDualDomain:
    """
    Gaussian-Hermite Multi-Lobe Basis
    with two independent sigma domains:

        - Wide lobes  (first num_wide centers)
        - Narrow lobes (remaining centers)

    Used strictly for Phase 1 Gram conditioning experiments.
    """

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],

        num_wide: int,

        wide_sigma_min: float,
        wide_sigma_max: Optional[float],
        wide_scale_type: ScaleType = "sqrt",
        wide_gamma: float = 0.5,

        narrow_sigma_min: float = 4.0,
        narrow_sigma_max: Optional[float] = None,
        narrow_scale_type: ScaleType = "sqrt",
        narrow_gamma: float = 0.5,

        order: int = 6
    ):
        self.m_domain  = domain
        self.m_centers = torch.tensor(
            centers, device=domain.m_device, dtype=domain.m_dtype
        )

        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        if num_wide > self.m_K:
            raise ValueError("num_wide cannot exceed number of centers.")

        self.m_num_wide   = num_wide
        self.m_num_narrow = self.m_K - num_wide

        self.m_wide_sigma_min  = wide_sigma_min
        self.m_wide_sigma_max  = wide_sigma_max if wide_sigma_max is not None else wide_sigma_min
        self.m_wide_scale_type = wide_scale_type
        self.m_wide_gamma      = wide_gamma

        self.m_narrow_sigma_min  = narrow_sigma_min
        self.m_narrow_sigma_max  = narrow_sigma_max if narrow_sigma_max is not None else narrow_sigma_min
        self.m_narrow_scale_type = narrow_scale_type
        self.m_narrow_gamma      = narrow_gamma

        self.m_basisRaw = None
        self.m_gram     = None
        self.m_chol     = None

        self._buildBasis()
        self._buildGram()
        self._buildCholesky()

    # ---------------------------------------------------------
    # Sigma schedule for one group  →  [N]
    # ---------------------------------------------------------

    def _sigma_schedule(
        self,
        sigma_min: float,
        sigma_max: float,
        scale_type: ScaleType,
        gamma: float,
        device,
        dtype
    ) -> Tensor:

        N = self.m_N

        if N <= 1 or scale_type == "constant":
            return torch.full((N,), sigma_min, device=device, dtype=dtype)

        t     = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
        delta = sigma_max - sigma_min

        if scale_type == "linear":
            return sigma_min + delta * t
        elif scale_type == "sqrt":
            return sigma_min + delta * torch.sqrt(t)
        elif scale_type == "power":
            return sigma_min + delta * torch.pow(t, gamma)
        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")

    # ---------------------------------------------------------
    # Basis Construction — fully batched across all K*N functions
    # Previously: K*N separate hermiteBasis calls in nested loops
    # Now: single hermiteBasis call on [K*N, L], diagonal gather
    # ---------------------------------------------------------

    def _buildBasis(self):

        lbda    = self.m_domain.m_lambda   # [L]
        device  = lbda.device
        dtype   = lbda.dtype
        K       = self.m_K
        N       = self.m_N
        L       = lbda.shape[0]

        # Sigma schedules per group, assembled into [K, N] matrix
        wide_sigmas   = self._sigma_schedule(
            self.m_wide_sigma_min, self.m_wide_sigma_max,
            self.m_wide_scale_type, self.m_wide_gamma, device, dtype
        )   # [N]
        narrow_sigmas = self._sigma_schedule(
            self.m_narrow_sigma_min, self.m_narrow_sigma_max,
            self.m_narrow_scale_type, self.m_narrow_gamma, device, dtype
        )   # [N]

        sigma_matrix = torch.empty(K, N, device=device, dtype=dtype)
        sigma_matrix[:self.m_num_wide, :]  = wide_sigmas.unsqueeze(0)
        sigma_matrix[self.m_num_wide:, :]  = narrow_sigmas.unsqueeze(0)
        # sigma_matrix[k, n] = sigma for center k at Hermite order n

        # x[k, n, l] = (lambda[l] - centers[k]) / sigma_matrix[k, n]
        lbda_exp    = lbda.unsqueeze(0).unsqueeze(0)        # [1, 1, L]
        centers_exp = self.m_centers.unsqueeze(1).unsqueeze(2)  # [K, 1, 1]
        sigma_exp   = sigma_matrix.unsqueeze(2)             # [K, N, 1]

        x_full = (lbda_exp - centers_exp) / sigma_exp       # [K, N, L]
        x_flat = x_full.reshape(K * N, L)                   # [K*N, L]

        # Single batched Hermite evaluation
        H_full = hermiteBasis(N, x_flat)                    # [K*N, N, L]

        # Gather the diagonal: row r = k*N+n needs H[r, n, :]
        row_idx = torch.arange(K * N, device=device)
        ord_idx = row_idx % N
        H_diag  = H_full[row_idx, ord_idx, :]               # [K*N, L]

        # Normalization constants
        n_idx      = torch.arange(N, device=device, dtype=dtype)
        factorials = torch.exp(torch.lgamma(n_idx + 1))
        sqrt_pi    = torch.tensor(torch.pi, device=device, dtype=dtype).sqrt()
        norms      = torch.sqrt((2.0 ** n_idx) * factorials * sqrt_pi)  # [N]
        norms_tiled = norms.repeat(K)                       # [K*N]

        gaussian = torch.exp(-0.5 * x_flat ** 2)            # [K*N, L]

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
