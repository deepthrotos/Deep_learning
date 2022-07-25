import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()
    list_loss = []
    for x, y in data_loader:
        optimizer.zero_grad()
        output = model(x)

        loss = loss_fn(output, y)

        loss.backward()

        print(f'{loss.item():.5f}')
        list_loss.append(loss.item())
        optimizer.step()
    return sum(list_loss) / len(list_loss)




