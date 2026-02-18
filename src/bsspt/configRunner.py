# config_runner.py

import os
import torch

from engine.spectraldomain import SpectralDomain
from engine.ghgsfbasisflexible import GHGSFMultiLobeBasisFlexible
from operator_analysis import analyze_operator
from precisionComparison import PrecisionComparator

from operator_builder import (
    build_chain_0,
    build_chain_1,
    build_chain_2,
    apply_whitening_if_enabled
)

from artifact_geometry import GeometryEmitter
from transportEmitter import TransportEmitter
from precision import configure_precision, configure_determinism


# ============================================================
# Root
# ============================================================

FRSTA_ROOT = "./FRSTA_v1_0/configs"

TRANSPORT_DEPTH = 15


# ============================================================
# Config Name Builder
# ============================================================

def build_config_name(K, N, sigma_min, sigma_max, scale_type, whitening):
    return f"K{K}_N{N}_S{sigma_min}-{sigma_max}_{scale_type}_W{int(whitening)}"


# ============================================================
# Run Single Config
# ============================================================

def run_config(
    K,
    N,
    sigma_min,
    sigma_max,
    scale_type,
    whitening_enabled
):

    configure_determinism()

    config_name = build_config_name(
        K, N, sigma_min, sigma_max, scale_type, whitening_enabled
    )

    config_root = os.path.join(FRSTA_ROOT, config_name)
    geometry_root = os.path.join(config_root, "geometry")

    os.makedirs(config_root, exist_ok=True)

    # =========================================================
    # 1. Build Domain + Basis (FP64 for geometry)
    # =========================================================

    dtype_geom = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain = SpectralDomain(
        lambdaMin=380.0,
        lambdaMax=780.0,
        numSamples=2048,
        device=device,
        dtype=dtype_geom
    )

    centers = torch.linspace(420.0, 720.0, K).tolist()

    basis = GHGSFMultiLobeBasisFlexible(
        domain=domain,
        centers=centers,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        order=N,
        scale_type=scale_type
    )

    # Emit geometry once
    geometry_emitter = GeometryEmitter(config_root)
    geometry_emitter.dump_geometry(basis, whitening_enabled)

    # =========================================================
    # 2. Loop Over Precision Modes
    # =========================================================

    for precision_mode in ["fp64", "tf32"]:

        dtype = configure_precision(precision_mode)

        precision_root = os.path.join(
            config_root,
            "precision",
            precision_mode
        )

        os.makedirs(precision_root, exist_ok=True)

        # -----------------------------------------------------
        # Rebuild domain + basis in this precision
        # -----------------------------------------------------

        domain_p = SpectralDomain(
            380.0,
            780.0,
            2048,
            device=device,
            dtype=dtype
        )

        basis_p = GHGSFMultiLobeBasisFlexible(
            domain=domain_p,
            centers=centers,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            order=N,
            scale_type=scale_type
        )

        # -----------------------------------------------------
        # Build operator chains (ordered list)
        # -----------------------------------------------------

        chains = [
            ("chain_0", build_chain_0(basis_p)),
            ("chain_1", build_chain_1(basis_p)),
            ("chain_2", build_chain_2(basis_p)),
        ]

        # -----------------------------------------------------
        # Apply whitening if enabled
        # -----------------------------------------------------

        chains = [
            (
                name,
                apply_whitening_if_enabled(op, basis_p, whitening_enabled)
            )
            for name, op in chains
        ]

        # -----------------------------------------------------
        # Operator Diagnostics
        # -----------------------------------------------------

        operator_root = os.path.join(precision_root, "operators")
        os.makedirs(operator_root, exist_ok=True)

        for name, op in chains:
            chain_path = os.path.join(operator_root, name)
            os.makedirs(chain_path, exist_ok=True)

            analyze_operator(op, chain_path)

        print(f"[{precision_mode}] operator diagnostics complete.")

        # =====================================================
        # Phase 3: Deterministic Transport
        # =====================================================

        # Build IRL spectrum (placeholder)
        lbda = domain_p.m_lambda
        spectrum = torch.exp(
            -((lbda - 540.0) ** 2) / (2.0 * 30.0 ** 2)
        )

        initial_coeffs = basis_p.project(spectrum)

        for name, op in chains:
            single_run_root = os.path.join(
                precision_root,
                "spectra",
                "spectrum_00_IRL",
                name,
                "single_run"
            )

            os.makedirs(single_run_root, exist_ok=True)

            emitter = TransportEmitter(
                root_path=single_run_root,
                transport_depth=TRANSPORT_DEPTH
            )

            emitter.run_single_deterministic(
                basis_p,
                domain_p,
                op,
                initial_coeffs
            )

            print(f"[{precision_mode}] {name} transport complete.")

            # =========================================================
            # Precision Comparison Block
            # =========================================================


            comparison_root = os.path.join(
                config_root,
                "precision",
                "precision_comparison",
                "spectrum_00_IRL"
            )

            chains = ["chain_0", "chain_1", "chain_2"]

            for chain_name in chains:
                fp64_path = os.path.join(
                    config_root,
                    "precision",
                    "fp64",
                    "spectra",
                    "spectrum_00_IRL",
                    chain_name,
                    "single_run"
                )

                tf32_path = os.path.join(
                    config_root,
                    "precision",
                    "tf32",
                    "spectra",
                    "spectrum_00_IRL",
                    chain_name,
                    "single_run"
                )

                chain_comparison_root = os.path.join(
                    comparison_root,
                    chain_name
                )

                comparator = PrecisionComparator(chain_comparison_root)

                comparator.compare_single_run(
                    fp64_path,
                    tf32_path
                )

            print("Precision comparison complete.")

    print(f"Config {config_name} finished.\n")

if __name__ == "__main__":

    run_config(
        K=6,              # 6 lobes
        N=5,              # 5 Hermite orders
        sigma_min=6.0,
        sigma_max=10.0,
        scale_type="sqrt",
        whitening_enabled=True
    )
