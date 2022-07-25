import torch
from torch import nn


def create_linear():
    return nn.Linear(10, 100, bias=False)