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


class AbsorptionOperator:
    """
    Beer-Lambert absorption along a path segment:

        T(λ) = exp(-σ_a(λ) · distance)

    Galerkin operator matrix:
        M_ij = ∫ T(λ) b_i(λ) b_j(λ) dλ

    Update rule:
        α_{k+1} = G⁻¹ M α_k
    """

    @staticmethod
    def create(
        basis: AnyBasis,
        sigmaA: Callable[[Tensor], Tensor],
        distance: float
    ) -> SpectralOperator:

        B    = basis.m_basisRaw
        w    = basis.m_domain.m_weights
        L    = basis.m_chol
        lbda = basis.m_domain.m_lambda

        T = torch.exp(-sigmaA(lbda) * distance)   # [L]

        M_raw = (B * (w * T)) @ B.T               # [M, M]

        # Solve G A = M_raw  →  A = G⁻¹ M_raw  via Cholesky
        Y = torch.linalg.solve_triangular(L,   M_raw, upper=False)
        A = torch.linalg.solve_triangular(L.T, Y,     upper=True)

        b = torch.zeros(basis.m_M, device=A.device, dtype=A.dtype)

        return SpectralOperator(basis, A, b)
