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
from plotting.Plot import PlotEngine
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

DEPTH = 15

# ============================================================

def run_phase3_test():

    configure_determinism(1337)

    root = "FRSTA_v1_0/PHASE3_TEST"
    os.makedirs(root, exist_ok=True)

    for precision_mode in ["fp64", "tf32"]:

        print("\n==============================")
        print("Precision:", precision_mode)
        print("==============================")

        dtype = configure_precision(precision_mode)
        device = torch.device("cuda")

        # ----------------------------------------------------
        # Build geometry
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

        chains = [
            build_chain_0(basis),
            build_chain_1(basis),
            build_chain_2(basis)
        ]

        for chain_idx, chain in enumerate(chains):

            print(f"\nChain {chain_idx}")

            chain = apply_whitening_if_enabled(
                chain,
                basis,
                whitening_enabled
            )

            chain_root = os.path.join(
                root,
                precision_mode,
                f"chain_{chain_idx}"
            )
            os.makedirs(chain_root, exist_ok=True)

            A = chain.m_A.detach()

            A_cum = torch.eye(
                basis.m_M,
                device=A.device,
                dtype=A.dtype
            )

            spectral_radii = []
            condition_numbers = []
            fro_norms = []
            non_normalities = []

            for k in range(DEPTH):

                A_cum = A @ A_cum

                # SVD
                U, S, Vh = torch.linalg.svd(A_cum)

                cond = (S.max() / S.min()).item()
                fro_norm = torch.norm(A_cum, p="fro").item()

                eigvals = torch.linalg.eigvals(A_cum)
                spectral_radius = eigvals.abs().max().item()

                ATA = A_cum.T @ A_cum
                AAT = A_cum @ A_cum.T
                non_normality = torch.norm(ATA - AAT, p="fro").item()

                spectral_radii.append(spectral_radius)
                condition_numbers.append(cond)
                fro_norms.append(fro_norm)
                non_normalities.append(non_normality)

                print(
                    f"  Bounce {k+1}: "
                    f"ρ={spectral_radius:.6f} "
                    f"cond={cond:.6f}"
                )

            # ------------------------------------------------
            # Save curves
            # ------------------------------------------------

            torch.save(
                torch.tensor(spectral_radii),
                os.path.join(chain_root, "spectral_radius_curve.pt")
            )

            torch.save(
                torch.tensor(condition_numbers),
                os.path.join(chain_root, "condition_curve.pt")
            )

            torch.save(
                torch.tensor(fro_norms),
                os.path.join(chain_root, "fro_norm_curve.pt")
            )

            torch.save(
                torch.tensor(non_normalities),
                os.path.join(chain_root, "non_normality_curve.pt")
            )

            # ------------------------------------------------
            # Plot spectral radius growth
            # ------------------------------------------------

            plot = PlotEngine(figsize=(6,4))
            plot.addLine(range(1, DEPTH+1), spectral_radii)
            plot.setTitle("Spectral Radius Growth")
            plot.setLabels("Bounce", "Spectral Radius")
            plot.saveFigure(
                os.path.join(chain_root, "spectral_radius_growth.png"),
                dpi=512
            )
            plot.close()

    print("\nPHASE 3 COMPLETE.\n")


if __name__ == "__main__":
    run_phase3_test()
