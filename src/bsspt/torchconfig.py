import torch

class TorchConfig:

    @staticmethod
    def set_mode(mode: str, device: torch.device = torch.device("cuda")):
        """
        Configure torch for either:
            - 'performance'  (TF32 tensor core mode)
            - 'reference'    (FP64 strict mode)
        """

        if mode not in ["performance", "reference"]:
            raise ValueError("Mode must be 'performance' or 'reference'.")

        if mode == "performance":

            # Enable TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            torch.set_float32_matmul_precision("high")

            dtype = torch.float32

            print("Torch set to PERFORMANCE mode (TF32).")

        else:

            # Disable TF32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            torch.set_float32_matmul_precision("high")

            dtype = torch.float64

            print("Torch set to REFERENCE mode (FP64).")

        torch.no_grad()
        return {
            "device": device,
            "dtype": dtype
        }
