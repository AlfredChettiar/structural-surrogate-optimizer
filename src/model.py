"""PyTorch surrogate networks: Residual MLP and Physics-Guided (multi-fidelity) Residual MLP."""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch import nn

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.LayerNorm(width),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResidualMLP(nn.Module):
    """Input projection -> N residual blocks -> linear multi-output head."""

    def __init__(self, n_in: int, n_out: int, width: int = 96, n_blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, width), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.inp(x)))


def train_network(
    net: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    epochs: int = 800,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    patience: int = 120,
    min_epochs: int = 150,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict:
    """Train on already-standardised arrays with AdamW, cosine LR and early stopping
    (patience-based, but never before ``min_epochs`` so the cosine schedule can do its work).

    Returns a history dict; the network is left holding the best-validation weights.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(round(val_fraction * len(X))))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt = torch.as_tensor(X[tr_idx], dtype=torch.float32)
    Yt = torch.as_tensor(Y[tr_idx], dtype=torch.float32)
    Xv = torch.as_tensor(X[val_idx], dtype=torch.float32)
    Yv = torch.as_tensor(Y[val_idx], dtype=torch.float32)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)
    loss_fn = nn.MSELoss()
    best, best_state, wait = float("inf"), copy.deepcopy(net.state_dict()), 0
    history = {"train": [], "val": []}
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        running = 0.0
        for start in range(0, len(Xt), batch_size):
            b = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xv), Yv).item()
        history["train"].append(running / len(Xt))
        history["val"].append(val)
        if val < best - 1e-7:
            best, best_state, wait = val, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience and len(history["val"]) >= min(min_epochs, epochs):
                break
    net.load_state_dict(best_state)
    net.eval()
    history["best_val"] = best
    history["epochs_run"] = len(history["val"])
    return history


@torch.no_grad()
def predict_network(net: nn.Module, X: np.ndarray) -> np.ndarray:
    net.eval()
    return net(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(float)
