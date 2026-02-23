import torch
from torch import Tensor
from typing import Union

from engine.spectraloperator import SpectralOperator
from engine.ghgsfbasis import GHGSFMultiLobeBasis

AnyBasis = Union[
    "GHGSFMultiLobeBasis",
    "GHGSFMultiLobeBasisFlexible",
    "GHGSFMultiLobeBasisScaled",
    "GHGSFMultiLobeBasisDualDomain",
]


class WhitenOperator:
    """
    Transforms raw coefficients α into whitened coefficients α̃:

        α̃ = Lᵀ α

    where G = L Lᵀ  (Cholesky factorization).

    This makes the inner product Euclidean:
        ‖α̃‖² = α̃ᵀ α̃ = αᵀ Lᵀᵀ Lᵀ α = αᵀ L Lᵀ α = αᵀ G α = ‖α‖²_G

    Math verified: A = Lᵀ is correct.
    Previously imported from engine.ghgsfbasis — fixed to accept all basis types.
    """

    @staticmethod
    def create(basis: AnyBasis) -> SpectralOperator:

        L = basis.m_chol   # lower triangular, G = L Lᵀ
        A = L.T            # apply Lᵀ to raw coefficients

        b = torch.zeros(basis.m_M, device=L.device, dtype=L.dtype)

        return SpectralOperator(basis, A, b)


class UnwhitenOperator:
    """
    Inverse of WhitenOperator — recovers raw α from whitened α̃:

        α = L⁻ᵀ α̃

    Computed via triangular solve: solve Lᵀ α = α̃.
    Previously formed the full explicit inverse L⁻ᵀ which is unnecessary.
    """

    @staticmethod
    def create(basis: AnyBasis) -> SpectralOperator:

        L      = basis.m_chol
        M      = basis.m_M
        device = L.device
        dtype  = L.dtype

        I = torch.eye(M, device=device, dtype=dtype)

        # Solve Lᵀ X = I  →  X = L⁻ᵀ
        A = torch.linalg.solve_triangular(L.T, I, upper=True)

        b = torch.zeros(M, device=device, dtype=dtype)

        return SpectralOperator(basis, A, b)
