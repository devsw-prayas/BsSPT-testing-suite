import os
import torch

from precision import configure_precision, configure_determinism
from operator_builder import (
    build_chain_0,
    build_chain_1,
    build_chain_2,
    apply_whitening_if_enabled
)
from operator_analysis import analyze_operator

from engine.spectraldomain import SpectralDomain
from engine.ghgsfbasisflexible import GHGSFMultiLobeBasisFlexible


# ============================================================
# CONFIG
# ============================================================

K = 6
N = 6
sigma_min = 5.0
sigma_max = 10.0
scale_type = "sqrt"
whitening_enabled = True

transport_depth = 15

# ============================================================
# MAIN
# ============================================================

def run_phase2_test():

    configure_determinism(1337)

    root = "FRSTA_v1_0/PHASE2_TEST"
    os.makedirs(root, exist_ok=True)

    for precision_mode in ["fp64", "tf32"]:

        print("\n==============================")
        print("Precision:", precision_mode)
        print("==============================")

        dtype = configure_precision(precision_mode)
        device = torch.device("cuda")

        # ----------------------------------------------------
        # Build domain
        # ----------------------------------------------------

        domain = SpectralDomain(
            lambdaMin=400.0,
            lambdaMax=700.0,
            numSamples=1024,
            device=device,
            dtype=dtype
        )

        centers = torch.linspace(450, 650, K).tolist()

        basis = GHGSFMultiLobeBasisFlexible(
            domain=domain,
            centers=centers,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            order=N,
            scale_type=scale_type,
            gamma=0.5
        )

        # ----------------------------------------------------
        # Phase 2 operator block
        # ----------------------------------------------------

        precision_path = os.path.join(
            root,
            f"{precision_mode}",
            "operators"
        )

        os.makedirs(precision_path, exist_ok=True)

        chains = [
            build_chain_0(basis),
            build_chain_1(basis),
            build_chain_2(basis)
        ]

        for i, chain in enumerate(chains):

            chain = apply_whitening_if_enabled(
                chain,
                basis,
                whitening_enabled
            )

            chain_path = os.path.join(
                precision_path,
                f"chain_{i}"
            )

            analyze_operator(chain, chain_path)

            # Console diagnostics
            A = chain.m_A.detach()

            eigvals = torch.linalg.eigvals(A)
            spectral_radius = eigvals.abs().max().item()

            U, S, Vh = torch.linalg.svd(A)
            cond = (S.max() / S.min()).item()

            ATA = A.T @ A
            AAT = A @ A.T
            non_normality = torch.norm(ATA - AAT, p="fro").item()

            print(f"\nChain {i}:")
            print("  Spectral Radius :", spectral_radius)
            print("  Condition Number:", cond)
            print("  Non-Normality   :", non_normality)

    print("\nPHASE 2 TEST COMPLETE.\n")


# ============================================================

if __name__ == "__main__":
    run_phase2_test()
