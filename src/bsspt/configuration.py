# ============================================================
# Phase 1 Experiment Configuration
# ============================================================

import numpy as np
import itertools
from dataclasses import dataclass
from typing import Iterator, Tuple


# ============================================================
# Global Spectral Domain Settings
# ============================================================

LAMBDA_MIN = 380.0
LAMBDA_MAX = 780.0
NUM_SAMPLES = 4096


# ============================================================
# Sweep Ranges
# ============================================================

LOBE_MIN = 4
LOBE_MAX = 12

ORDER_MIN = 4
ORDER_MAX = 12

SIGMA_MIN = 6.0
SIGMA_MAX = 12.0
SIGMA_STEP = 0.5

SCALING_LAWS = [
    "constant",
    "linear",
    "sqrt",
    "power"
]

GAMMA_VALUES = [0.5]  # only used for power law

PRECISION_MODES = [
    "performance",  # TF32
    "reference"     # FP64
]

WHITENING_MODES = [
    False,
    True
]

NUM_TOPOLOGIES = 5


# ============================================================
# Derived Grids
# ============================================================

LOBE_RANGE = range(LOBE_MIN, LOBE_MAX + 1)
ORDER_RANGE = range(ORDER_MIN, ORDER_MAX + 1)

SIGMA_VALUES = np.arange(
    SIGMA_MIN,
    SIGMA_MAX + SIGMA_STEP,
    SIGMA_STEP
)


# ============================================================
# Configuration Dataclass
# ============================================================

@dataclass(frozen=True)
class Phase1Config:

    topology_id: int
    lobe_count: int
    order: int
    sigma_min: float
    sigma_max: float
    scaling_law: str
    gamma: float
    precision_mode: str
    whitened: bool


# ============================================================
# Configuration Iterator
# ============================================================

def generate_phase1_configs() -> Iterator[Phase1Config]:

    for topology_id in range(NUM_TOPOLOGIES):

        for lobe_count in LOBE_RANGE:

            for order in ORDER_RANGE:

                for sigma_min, sigma_max in itertools.combinations(SIGMA_VALUES, 2):

                    for scaling_law in SCALING_LAWS:

                        gamma_list = (
                            GAMMA_VALUES
                            if scaling_law == "power"
                            else [0.0]
                        )

                        for gamma in gamma_list:

                            for precision_mode in PRECISION_MODES:

                                for whitened in WHITENING_MODES:

                                    yield Phase1Config(
                                        topology_id=topology_id,
                                        lobe_count=lobe_count,
                                        order=order,
                                        sigma_min=float(sigma_min),
                                        sigma_max=float(sigma_max),
                                        scaling_law=scaling_law,
                                        gamma=float(gamma),
                                        precision_mode=precision_mode,
                                        whitened=whitened
                                    )


# ============================================================
# Experiment Size Estimation
# ============================================================

def estimate_total_configs() -> int:

    num_lobes = len(LOBE_RANGE)
    num_orders = len(ORDER_RANGE)
    num_sigma_pairs = len(list(itertools.combinations(SIGMA_VALUES, 2)))
    num_scaling = len(SCALING_LAWS)
    num_precisions = len(PRECISION_MODES)
    num_whitening = len(WHITENING_MODES)

    return (
        NUM_TOPOLOGIES *
        num_lobes *
        num_orders *
        num_sigma_pairs *
        num_scaling *
        num_precisions *
        num_whitening
    )


# ============================================================
# Quick Sanity Print
# ============================================================

if __name__ == "__main__":

    total = estimate_total_configs()
    print(f"Total Phase 1 configurations: {total}")
