import torch.nn as nn
from torch.utils.data import DataLoader
import torch


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    list_loss = []
    with torch.no_grad():
        model.eval()
        for x, y in data_loader:
            output = model(x)

            loss = loss_fn(output, y)

            list_loss.append(loss.item())

        return sum(list_loss) / len(list_loss)