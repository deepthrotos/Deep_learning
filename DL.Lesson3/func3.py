from time import perf_counter

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import torch
import torch.nn as nn
import torchvision.transforms as t
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

mnist_train = MNIST(
    "../datasets/mnist",
    train=True,
    download=True,
    transform=t.ToTensor()
)

mnist_valid = MNIST(
    "../datasets/mnist",
    train=False,
    download=True,
    transform=t.ToTensor()
)

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
valid_loader = DataLoader(mnist_valid, batch_size=64, shuffle=False)

second_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(4 * 4 * 64, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = torch.optim.Adam(second_model.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss()


def train(model: nn.Module) -> float:
    model.train()

    train_loss = 0

    for x, y in tqdm(train_loader, desc='Train'):
        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

    train_loss /= len(train_loader)

    return train_loss


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc='Evaluation'):
        output = model(x)

        loss = loss_fn(output, y)

        total_loss += loss.item()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    total_loss /= len(loader)
    accuracy = correct / total

    return total_loss, accuracy

def plot_stats(
    train_loss: list[float],
    valid_loss: list[float],
    valid_accuracy: list[float],
    title: str
):
    plt.figure(figsize=(16, 8))

    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + ' accuracy')

    plt.plot(valid_accuracy)
    plt.grid()

    plt.show()

def create_conv_model():
    num_epochs = 100

    train_loss_history, valid_loss_history = [], []
    valid_accuracy_history = []
    start = perf_counter()


    for epoch in range(num_epochs):
        train_loss = train(second_model)

        valid_loss, valid_accuracy = evaluate(second_model, valid_loader)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        valid_accuracy_history.append(valid_accuracy)

        clear_output()

        plot_stats(train_loss_history, valid_loss_history, valid_accuracy_history, 'CONV model')

    print(f'Total training and evaluation time {perf_counter() - start:.5f}')

    return second_model

model = create_conv_model()

torch.save(model.state_dict(), 'cock.pth')
