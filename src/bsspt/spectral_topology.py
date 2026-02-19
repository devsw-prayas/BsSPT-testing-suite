# ============================================================
# Spectral Topology Family Generators
# ============================================================

import torch
from typing import List

L_MIN_DEFAULT = 380.0
L_MAX_DEFAULT = 780.0


# ============================================================
# 1. Uniform
# Even spacing across full domain
# ============================================================

def topology_uniform(
    K: int,
    lambda_min: float = L_MIN_DEFAULT,
    lambda_max: float = L_MAX_DEFAULT
) -> List[float]:

    centers = torch.linspace(lambda_min, lambda_max, K)
    return centers.tolist()


# ============================================================
# 2. Bell Curve (Dense Middle)
# Warped using cosine compression toward center
# ============================================================

def topology_bell(
    K: int,
    lambda_min: float = L_MIN_DEFAULT,
    lambda_max: float = L_MAX_DEFAULT
) -> List[float]:

    t = torch.linspace(0.0, 1.0, K)
    warped = 0.5 * (1.0 - torch.cos(torch.pi * t))
    centers = lambda_min + warped * (lambda_max - lambda_min)
    return centers.tolist()


# ============================================================
# 3. Tristimulus (RGB Anchored)
# 3 primary regions around 450 / 550 / 650 nm
# ============================================================

def topology_tristimulus(
    K: int,
    lambda_min: float = L_MIN_DEFAULT,
    lambda_max: float = L_MAX_DEFAULT
) -> List[float]:

    anchors = torch.tensor([450.0, 550.0, 650.0])

    if K <= 3:
        return anchors[:K].tolist()

    per_band = K // 3
    remainder = K % 3

    centers = []

    for i in range(3):
        count = per_band + (1 if i < remainder else 0)

        if count == 1:
            centers.append(anchors[i].item())
        else:
            spread = 40.0
            local = torch.linspace(-spread/2, spread/2, count)
            centers.extend((anchors[i] + local).tolist())

    return centers[:K]


# ============================================================
# 4. Valley (Dense Edges, Sparse Middle)
# Inverse bell mapping
# ============================================================

def topology_valley(
    K: int,
    lambda_min: float = L_MIN_DEFAULT,
    lambda_max: float = L_MAX_DEFAULT
) -> List[float]:

    t = torch.linspace(0.0, 1.0, K)
    warped = torch.sin(torch.pi * t / 2.0) ** 2
    centers = lambda_min + warped * (lambda_max - lambda_min)
    return centers.tolist()


# ============================================================
# 5. Saw Blade (Alternating Compression)
# Uniform base + alternating perturbation
# ============================================================

def topology_sawblade(
    K: int,
    lambda_min: float = L_MIN_DEFAULT,
    lambda_max: float = L_MAX_DEFAULT
) -> List[float]:

    base = torch.linspace(lambda_min, lambda_max, K)
    perturb = (lambda_max - lambda_min) / (4.0 * K)

    for i in range(K):
        if i % 2 == 0:
            base[i] -= perturb
        else:
            base[i] += perturb

    base = torch.clamp(base, lambda_min, lambda_max)

    return base.tolist()


# ============================================================
# Dispatcher
# ============================================================

def generate_topology(
    topology_id: int,
    K: int,
    lambda_min: float = L_MIN_DEFAULT,
    lambda_max: float = L_MAX_DEFAULT
) -> List[float]:

    if topology_id == 0:
        return topology_uniform(K, lambda_min, lambda_max)

    elif topology_id == 1:
        return topology_bell(K, lambda_min, lambda_max)

    elif topology_id == 2:
        return topology_tristimulus(K, lambda_min, lambda_max)

    elif topology_id == 3:
        return topology_valley(K, lambda_min, lambda_max)

    elif topology_id == 4:
        return topology_sawblade(K, lambda_min, lambda_max)

    else:
        raise ValueError("Invalid topology_id (must be 0–4)")
