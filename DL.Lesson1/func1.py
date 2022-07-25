import torch


def function01(tensor: torch.Tensor, count_over: str):
    if count_over == 'columns':
        return torch.mean(tensor, dim = 0)
    if count_over == 'rows':
        return torch.mean(tensor, dim=1)
