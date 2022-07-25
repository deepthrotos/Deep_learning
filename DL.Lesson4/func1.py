import torch


def get_normalize(features: torch.Tensor):
    mu = torch.mean(features, dim=[0, 2, 3])
    sigma = torch.std(features, dim=[0, 2, 3])
    return mu, sigma

