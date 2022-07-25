import torch


def function03(x: torch.Tensor, y: torch.Tensor):
    n_steps = 50
    step_size = 1e-2
    w_list = torch.empty(n_steps, x.size(dim=1))
    w = torch.rand(x.size(dim=1), dtype=torch.float32)
    for i in range(n_steps):
        w.requires_grad = True
        w_list[i] = w.detach().clone()
        y_pred = x @ w
        mse = torch.mean((y_pred - y) ** 2)
        mse.backward()
        with torch.no_grad():
            w = w - w.grad * step_size
    return w
