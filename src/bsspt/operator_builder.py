import torch

from engine.spectraloperator import SpectralOperator


def build_multiplication_operator(basis, spectral_function_lambda):

    B = basis.m_basisRaw            # [M, L]
    G = basis.m_gram                # [M, M]
    w = basis.m_domain.m_weights    # [L]

    # Ensure shape consistency
    f = spectral_function_lambda.view(1, -1)   # [1, L]
    w = w.view(1, -1)                          # [1, L]

    weighted_B = B * w                         # [M, L]
    weighted_fB = weighted_B * f               # [M, L]

    M = weighted_fB @ B.T                      # [M, M]

    A = torch.linalg.solve(G, M)

    # --- ZERO affine term ---
    b = torch.zeros(
        basis.m_M,
        device=A.device,
        dtype=A.dtype
    )

    return SpectralOperator(basis, A, b)



def build_absorption_operator(basis):

    lbda = basis.m_domain.m_lambda

    assert isinstance(lbda, torch.Tensor), "Lambda must be torch tensor"

    device = lbda.device
    dtype = lbda.dtype

    center = torch.tensor(550.0, device=device, dtype=dtype)
    width  = torch.tensor(40.0, device=device, dtype=dtype)

    absorption = torch.exp(-((lbda - center) ** 2) / (2.0 * width ** 2))

    transmission = torch.exp(-0.8 * absorption)

    return build_multiplication_operator(basis, transmission)


def build_emission_operator(basis):

    lbda = basis.m_domain.m_lambda

    device = lbda.device
    dtype  = lbda.dtype

    center = torch.tensor(600.0, device=device, dtype=dtype)
    width  = torch.tensor(25.0, device=device, dtype=dtype)

    emission = 1.0 + 0.5 * torch.exp(-((lbda - center) ** 2) / (2.0 * width ** 2))

    return build_multiplication_operator(basis, emission)


def build_dispersion_operator(basis):

    lbda = basis.m_domain.m_lambda

    modulation = 1.0 + 0.3 * torch.sin(0.02 * lbda)

    return build_multiplication_operator(basis, modulation)

def build_chain_0(basis):

    A_abs  = build_absorption_operator(basis)
    A_disp = build_dispersion_operator(basis)
    A_emit = build_emission_operator(basis)

    return A_emit.compose(A_disp).compose(A_abs)


def build_chain_1(basis):

    A_emit = build_emission_operator(basis)
    A_abs  = build_absorption_operator(basis)

    return A_abs.compose(A_emit)


def build_chain_2(basis):

    return build_dispersion_operator(basis)

from engine.whitening import WhitenOperator, UnwhitenOperator

def apply_whitening_if_enabled(chain, basis, whitening_enabled):

    if not whitening_enabled:
        return chain

    W = WhitenOperator.create(basis)
    W_inv = UnwhitenOperator.create(basis)

    return W.compose(chain).compose(W_inv)
