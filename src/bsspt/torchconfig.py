import torch


class TorchConfig:

    @staticmethod
    def resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def set_mode(
        mode: str,
        device: torch.device = None,
        verbose: bool = False
    ) -> dict:
        """
        Configure torch for either:
            - 'performance'  (TF32 tensor core mode, float32)
            - 'reference'    (strict mode, float64)

        Parameters
        ----------
        mode : str
        device : torch.device, optional
            Defaults to CUDA if available, else CPU.
            Previously hardcoded to cuda — crashed on CPU-only machines.
        verbose : bool
            Suppress prints during sweep (default False).
            Previously always printed, flooding stdout across millions of rows.
        """

        if mode not in ["performance", "reference"]:
            raise ValueError("Mode must be 'performance' or 'reference'.")

        if device is None:
            device = TorchConfig.resolve_device()

        # Previously called torch.no_grad() which returns a context manager
        # and does nothing when not used as one. Fixed to actually disable grads.
        torch.set_grad_enabled(False)

        if mode == "performance":
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            dtype = torch.float32
            if verbose:
                print(f"Torch set to PERFORMANCE mode (TF32) on {device}.")

        else:
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            dtype = torch.float64
            if verbose:
                print(f"Torch set to REFERENCE mode (FP64) on {device}.")

        return {
            "device": device,
            "dtype": dtype
        }
