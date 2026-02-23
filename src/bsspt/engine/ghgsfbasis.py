import torch
from torch import Tensor
from typing import List

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis


class GHGSFMultiLobeBasis:
    """
    Gaussian-Hermite Global Spectral Function (Multi-Lobe)

    Discretized basis over spectral domain.

    Owns:
        - Raw basis matrix B      [M, L]
        - Gram matrix G           [M, M]
        - Cholesky factor L       [M, M]  (G = L L^T)

    Does NOT:
        - Store inverse
        - Perform whitening
        - Perform operator logic
    """

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma: float,
        order: int
    ):
        self.m_domain = domain
        self.m_centers = torch.tensor(
            centers,
            device=domain.m_device,
            dtype=domain.m_dtype
        )

        self.m_sigma = sigma
        self.m_order = order

        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        self.m_basisRaw = None
        self.m_gram = None
        self.m_chol = None

        self._buildBasis()
        self._buildGram()
        self._buildCholesky()

    # ---------------------------------------------------------
    # Basis Construction — fully batched across all K centers
    # Previously: K*N separate hermiteBasis calls in nested loops
    # Now: single batched call on [K*N, L], diagonal gather
    # ---------------------------------------------------------

    def _buildBasis(self):

        lbda    = self.m_domain.m_lambda    # [L]
        sigma   = self.m_sigma
        centers = self.m_centers            # [K]
        device  = lbda.device
        dtype   = lbda.dtype
        K       = self.m_K
        N       = self.m_N
        L       = lbda.shape[0]

        # Normalization constants for orders 0..N-1
        n_idx      = torch.arange(N, device=device, dtype=dtype)
        factorials = torch.exp(torch.lgamma(n_idx + 1))
        sqrt_pi    = torch.tensor(torch.pi, device=device, dtype=dtype).sqrt()
        norms      = torch.sqrt((2.0 ** n_idx) * factorials * sqrt_pi)  # [N]

        # x[k, l] = (lambda[l] - centers[k]) / sigma  =>  [K, L]
        x = (lbda.unsqueeze(0) - centers.unsqueeze(1)) / sigma   # [K, L]

        # Expand x to [K, N, L] by repeating for each order, then flatten
        # x_full[k, n, l] = x[k, l]  (same x, different Hermite order applied)
        x_rep  = x.unsqueeze(1).expand(K, N, L)                  # [K, N, L]
        x_flat = x_rep.reshape(K * N, L)                          # [K*N, L]

        # Single batched Hermite call
        H_full = hermiteBasis(N, x_flat)                          # [K*N, N, L]

        # For row r = k*N + n we need H[r, n, :] (order-n polynomial)
        row_idx = torch.arange(K * N, device=device)
        ord_idx = row_idx % N
        H_diag  = H_full[row_idx, ord_idx, :]                    # [K*N, L]

        # Gaussian envelopes using the original x (not x_flat which repeats)
        # x_flat[k*N+n, :] == x[k, :] for all n, so we can tile x directly
        x_tiled  = x.unsqueeze(1).expand(K, N, L).reshape(K * N, L)
        gaussian = torch.exp(-0.5 * x_tiled ** 2)                # [K*N, L]

        # Tile norms [N] -> [K*N]
        norms_tiled = norms.repeat(K)                             # [K*N]

        self.m_basisRaw = (H_diag * gaussian) / norms_tiled.unsqueeze(1)  # [M, L]

    # ---------------------------------------------------------
    # Gram Matrix
    # ---------------------------------------------------------

    def _buildGram(self):
        B = self.m_basisRaw
        w = self.m_domain.m_weights
        self.m_gram = (B * w) @ B.T

    # ---------------------------------------------------------
    # Cholesky
    # ---------------------------------------------------------

    def _buildCholesky(self):
        self.m_chol = torch.linalg.cholesky(self.m_gram)

    # ---------------------------------------------------------
    # Projection  — solve G alpha = b  via Cholesky
    # ---------------------------------------------------------

    def project(self, spectrum: Tensor) -> Tensor:

        B = self.m_basisRaw
        w = self.m_domain.m_weights

        if spectrum.device != B.device:
            spectrum = spectrum.to(B.device)
        if spectrum.dtype != B.dtype:
            spectrum = spectrum.to(B.dtype)

        b = ((B * w) @ spectrum).unsqueeze(1)   # [M, 1]

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
