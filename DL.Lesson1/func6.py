import torch
from torch import nn

def function04(x: torch.Tensor, y: torch.Tensor):
    y = y.view(y.shape[0], 1)
    layer = nn.Linear(x.shape[1], y.shape[1])

    step_size = 1e-2
    step_number = 1
    while True:
        y_pred = layer(x)
        #y_pred = layer(x).ravel()
        mse = torch.mean((y_pred-y)**2)

        print(f"MSE Ha шаге {step_number} {mse.item(): .5f}")
        if (mse.item() < 0.3) or (step_number > 2000):
            return layer

        mse.backward()
        with torch.no_grad():
            for param in layer.parameters():
                param -= step_size * param.grad

        layer.zero_grad()

        step_number = step_number + 1