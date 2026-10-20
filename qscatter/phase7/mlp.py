"""Small PyTorch MLP for (E, V_L, V_R) -> (Re t, Im t), with validation-only model selection."""
from __future__ import annotations

import itertools
import time

import numpy as np
import torch


def _net(width: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(3, width), torch.nn.SiLU(),
        torch.nn.Linear(width, width), torch.nn.SiLU(),
        torch.nn.Linear(width, width), torch.nn.SiLU(),
        torch.nn.Linear(width, 2),
    ).double()


def fit_mlp(X, t, Xv, tv, lo, hi, width: int, lr: float, epochs: int, seed: int, patience: int = 500):
    """Full-batch Adam with cosine decay; keeps the parameters with the lowest validation loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    scale = lambda A: torch.tensor((np.asarray(A) - lo) / (hi - lo))  # noqa: E731
    Xt, Xvt = scale(X), scale(Xv)
    Y = torch.tensor(np.column_stack([t.real, t.imag]))
    Yv = torch.tensor(np.column_stack([tv.real, tv.imag]))
    net = _net(width)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 1e-3)
    best, best_state, since = np.inf, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(Xt) - Y) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            v = float(torch.mean((net(Xvt) - Yv) ** 2))
        if v < best:
            best, best_state, since = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            since += 1
            if since > patience:
                break
    net.load_state_dict(best_state)

    def predict(Xq):
        with torch.no_grad():
            Z = net(scale(Xq)).numpy()
        return Z[:, 0] + 1j * Z[:, 1]

    return predict, best


def select_and_fit(X, t, Xv, tv, lo, hi, grid: dict, seed: int):
    """Choose (width, lr) by validation loss only; returns predictor, choice and fit time."""
    t0 = time.perf_counter()
    results = []
    for width, lr in itertools.product(grid["width"], grid["lr"]):
        pred, v = fit_mlp(X, t, Xv, tv, lo, hi, width, lr, grid["epochs"], seed)
        results.append((v, width, lr, pred))
    v, width, lr, pred = min(results, key=lambda r: r[0])
    return pred, {"width": width, "lr": lr, "val_mse": v}, time.perf_counter() - t0
