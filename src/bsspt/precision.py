import torch
import json
import os
import random
import numpy as np


def configure_precision(mode: str):
    """
    mode: "fp64" or "tf32"
    """

    if mode == "fp64":
        dtype = torch.float64
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    elif mode == "tf32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    else:
        raise ValueError("Unknown precision mode")

    return dtype

def configure_determinism(seed: int = 1337):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

def collect_environment_info():

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": torch.cuda.get_device_capability(0),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }

    return info

def dump_environment_info(root_folder: str):

    os.makedirs(root_folder, exist_ok=True)

    info = collect_environment_info()

    with open(os.path.join(root_folder, "environment.json"), "w") as f:
        json.dump(info, f, indent=4)
