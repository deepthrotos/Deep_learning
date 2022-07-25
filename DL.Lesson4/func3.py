import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            y_pred = torch.argmax(output, dim=1)
            predictions.append(y_pred)
    result = torch.cat(predictions)
    return result

