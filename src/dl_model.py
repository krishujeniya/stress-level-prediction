"""
Deep learning model: 3-layer MLP using PyTorch.

Architecture: Linear(8→64) → BN → ReLU → Drop →
              Linear(64→32) → BN → ReLU → Drop →
              Linear(32→5)

Training: Adam + CosineAnnealing + EarlyStopping (patience=25)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class StressMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden1: int = 64,
        hidden2: int = 32,
        output_dim: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_tensor(arr, dtype=torch.float32) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        return torch.tensor(arr, dtype=dtype)
    return torch.tensor(np.array(arr), dtype=dtype)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    patience: int = 25,
) -> StressMLP:
    """Train MLP with early stopping, return best model."""
    device = torch.device("cpu")

    X_tr = _to_tensor(X_train)
    y_tr = _to_tensor(y_train, dtype=torch.long)
    X_v = _to_tensor(X_val).to(device)
    y_v = _to_tensor(y_val, dtype=torch.long).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    model = StressMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def mlp_predict(model: StressMLP, X: np.ndarray):
    """Return (predictions, probabilities) arrays."""
    model.eval()
    with torch.no_grad():
        logits = model(_to_tensor(X))
        proba = torch.softmax(logits, dim=1).numpy()
        preds = proba.argmax(axis=1)
    return preds, proba


def mlp_predict_proba(model: StressMLP, X) -> np.ndarray:
    """Return probability matrix."""
    model.eval()
    with torch.no_grad():
        X_t = _to_tensor(np.array(X, dtype=np.float32))
        logits = model(X_t)
        proba = torch.softmax(logits, dim=1).numpy()
    return proba
