"""Spike-risk model trained from scratch: a small 1-D CNN over the pre-meal sensor window plus profile features."""

from __future__ import annotations

import torch
from torch import nn


class SensorCNN(nn.Module):
    """Two 1-D convolutions read the 2-hour glucose / heart-rate / missing-mask window; the summary of the whole
    window and of its last steps (the trend right before eating) is joined with the profile features and mapped
    to one spike logit. Kept deliberately small (about 5k parameters) because only ~1,000 training meals exist and
    no pretrained model for CGM traces is available to start from."""

    def __init__(self, in_channels: int = 3, n_tabular: int = 8, channels: int = 16, hidden: int = 32,
                 dropout: float = 0.2, last_steps: int = 6) -> None:  # fmt: skip
        """Randomly initialised (PyTorch default Kaiming-uniform for conv/linear); no pretrained weights."""
        super().__init__()
        self.last_steps = last_steps
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, 2 * channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * channels + n_tabular, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sequence: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """sequence: (B, 3, 24), tabular: (B, n_tabular) -> spike logits (B,). Apply sigmoid only at inference."""
        h = self.conv(sequence)  # (B, 2C, 24)
        summary = torch.cat([h.mean(dim=-1), h[..., -self.last_steps :].mean(dim=-1)], dim=1)  # whole + recent
        return self.head(torch.cat([summary, tabular], dim=1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Number of parameters (trainable or not)."""
    return sum(p.numel() for p in model.parameters())
