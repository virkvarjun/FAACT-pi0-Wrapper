"""FACT model variants: MLP and temporal encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class FactMLP(nn.Module):
    """Simple MLP baseline for failure prediction.

    Input: single feature vector per chunk.
    Output: logit for P(failure within K).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        dims = [input_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [B]."""
        return self.mlp(x).squeeze(-1)

    def predict_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Standard interface: batch with 'x' key."""
        return self.forward(batch["x"])

    def predict_risk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return probability via sigmoid."""
        return torch.sigmoid(self.predict_logits(batch))


class FactTemporal(nn.Module):
    """Temporal transformer encoder for failure prediction.

    Input: sequence of feature vectors (recent chunks).
    Output: logit for P(failure within K).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, input_dim) -> logits (B,)."""
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)

    def predict_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """batch['x']: (B, seq_len, input_dim)."""
        return self.forward(batch["x"])

    def predict_risk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.predict_logits(batch))
