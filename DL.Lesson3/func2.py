from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torchvision.transforms as t
from tqdm import tqdm

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

first_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = torch.optim.Adam(first_model.parameters(), lr=1e-3)

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


def create_mlp_model():
    num_epochs = 16

    train_loss_history, valid_loss_history = [], []
    valid_accuracy_history = []

    for epoch in range(num_epochs):
        train_loss = train(first_model)

        valid_loss, valid_accuracy = evaluate(first_model, valid_loader)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        valid_accuracy_history.append(valid_accuracy)

    return first_model


model = create_mlp_model()

torch.save(model.state_dict(), 'model.pth')
