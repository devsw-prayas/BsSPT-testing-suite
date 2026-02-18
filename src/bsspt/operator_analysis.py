import os
import json

import numpy
import torch
from plotting.Plot import PlotEngine

def analyze_operator(operator, output_path):

    os.makedirs(output_path, exist_ok=True)

    A = operator.m_A.detach()

    torch.save(A.cpu(), os.path.join(output_path, "matrix.pt"))

    # SVD
    U, S, Vh = torch.linalg.svd(A)
    torch.save(S.cpu(), os.path.join(output_path, "singular_values.pt"))

    # Eigenvalues
    eigvals = torch.linalg.eigvals(A)
    spectral_radius = eigvals.abs().max().item()

    # Frobenius norm
    fro_norm = torch.norm(A, p="fro").item()

    # Condition number
    cond = (S.max() / S.min()).item()

    # Non-normality
    ATA = A.T @ A
    AAT = A @ A.T
    non_normality = torch.norm(ATA - AAT, p="fro").item()

    metrics = {
        "spectral_radius": spectral_radius,
        "frobenius_norm": fro_norm,
        "condition_number": cond,
        "non_normality": non_normality
    }

    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Singular value plot
    plot = PlotEngine(figsize=(6, 4))
    plot.addLine(
        numpy.array(list(range(len(S)))),
        S.cpu().numpy()
    )
    plot.setTitle("Singular Values")
    plot.setLabels("Index", "Singular Value")
    plot.saveFigure(
        os.path.join(output_path, "singular_values.png"),
        dpi=512
    )
    plot.close()
