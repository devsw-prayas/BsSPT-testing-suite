import torch
from torch import Tensor
from typing import Callable, Union

from engine.spectraloperator import SpectralOperator
from engine.ghgsfbasis import GHGSFMultiLobeBasis

AnyBasis = Union[
    "GHGSFMultiLobeBasis",
    "GHGSFMultiLobeBasisFlexible",
    "GHGSFMultiLobeBasisScaled",
    "GHGSFMultiLobeBasisDualDomain",
]


class EmissionOperator:
    """
    Projects an emitter spectrum E(λ) into the basis subspace.

    Solves:  G b = raw   where   raw_i = ∫ E(λ) b_i(λ) dλ

    Returns a SpectralOperator with A=0, b=projected_coefficients.
    Applied once at path initialization: α_0 = b.

    Previously crashed: solve_triangular requires 2D input [M, K],
    but raw was 1D [M]. Fixed with unsqueeze(1) / squeeze(1).
    """

    @staticmethod
    def create(
        basis: AnyBasis,
        emissionFn: Callable[[Tensor], Tensor]
    ) -> SpectralOperator:

        B    = basis.m_basisRaw
        w    = basis.m_domain.m_weights
        L    = basis.m_chol
        lbda = basis.m_domain.m_lambda

        spectrum = emissionFn(lbda)               # [L]
        raw      = (B * w) @ spectrum             # [M]  — inner products

        # solve_triangular requires (..., n, k) — needs at least 2D
        raw_2d = raw.unsqueeze(1)                 # [M, 1]

        y = torch.linalg.solve_triangular(L,   raw_2d, upper=False)  # [M, 1]
        b = torch.linalg.solve_triangular(L.T, y,      upper=True)   # [M, 1]
        b = b.squeeze(1)                          # [M]

        A = torch.zeros(
            (basis.m_M, basis.m_M),
            device=b.device,
            dtype=b.dtype
        )

        return SpectralOperator(basis, A, b)
