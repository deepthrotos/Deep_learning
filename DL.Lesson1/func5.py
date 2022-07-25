import torch
from torch import nn


def count_parameters(layer: nn.Module):
    n = 0
    for param in layer.parameters():
        if param.requires_grad:
            n += param.numel()
    return n
