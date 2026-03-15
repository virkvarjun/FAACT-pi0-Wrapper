"""FACT training loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fact.fact.models import FactMLP, FactTemporal

logger = logging.getLogger(__name__)


def train_fact(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "mlp",
    output_dir: str | Path = "checkpoints/fact",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    """Train FACT model on chunk-level data.

    Returns:
        Dict with best_val_loss, metrics, checkpoint_path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    input_dim = X.shape[1]
    if model_type == "mlp":
        model = FactMLP(input_dim=input_dim)
    elif model_type == "temporal":
        # Temporal expects (B, seq_len, dim); use seq_len=1 for per-chunk input
        model = FactTemporal(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if model_type == "temporal":
                xb = xb.unsqueeze(1)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d loss=%.4f", epoch + 1, avg_loss)

    if best_state:
        model.load_state_dict(best_state)
    ckpt_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    config = {
        "model_type": model_type,
        "input_dim": input_dim,
        "epochs": epochs,
    }
    with open(output_dir / "config.json", "w") as f:
        import json
        json.dump(config, f, indent=2)

    return {
        "best_val_loss": best_loss,
        "checkpoint_path": str(ckpt_path),
        "config": config,
    }
