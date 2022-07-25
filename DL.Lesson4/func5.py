import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, iterations: int = 2):
    model.eval()
    iter = []
    with torch.no_grad():
        for i in range(iterations):
            predicts = []
            for x, y in loader:
                out = model(x)
                predicts.append(out)
            iter.append(torch.concat(predicts))
    return torch.concat(iter).reshape(iterations, len(loader.dataset), out.size(1)).mean(dim=0).argmax(dim=1)