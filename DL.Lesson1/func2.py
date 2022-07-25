import torch


def function02(tensor: torch.Tensor):
    return torch.rand(tensor.shape[1], dtype=torch.float32, requires_grad=True)