import torch

data = torch.load("phase1_output/phase1_batch_0.pt")
for k, v in data.items():
    print(k, type(v))
    if isinstance(v, torch.Tensor):
        print(v.shape, v.dtype)
        print(v)
