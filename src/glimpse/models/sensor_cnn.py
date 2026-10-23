"""SensorCNN: the model trained from scratch. It reads the 2 hours before a meal and outputs a spike risk.

Where it sits: plain-code checks run first (missing readings, gaps); this model scores meals that pass them;
thresholds and costs (outside the model) turn its risk into low risk / caution / can't judge."""

from __future__ import annotations

import torch
from torch import nn


class SensorCNN(nn.Module):
    """Two 1-D convolutions slide along the 24 five-minute slots (glucose, heart rate, missing-glucose flag) and
    learn short shapes such as a rise or a dip (kernel 5 = 25 minutes). Their output is summarised over the whole
    2 hours and over the last 30 minutes, joined with the 8 profile inputs, and a small layer gives one number."""

    def __init__(self, in_channels: int = 3, n_profile: int = 8, channels: int = 16, hidden: int = 32,
                 dropout: float = 0.2, last_slots: int = 6) -> None:  # fmt: skip
        """Build the layers. All weights start random (PyTorch default initialisation): nothing is pretrained."""
        super().__init__()
        self.last_slots = last_slots  # 6 slots x 5 min = the last 30 minutes before the meal
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, 2 * channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * (2 * channels) + n_profile, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),  # randomly silences units during training to reduce memorising
            nn.Linear(hidden, 1),
        )

    def forward(self, sequence: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        """sequence (batch, 3, 24) + profile (batch, 8) -> one raw score (logit) per meal, shape (batch,).
        The probability is sigmoid(logit); it is applied outside so the loss can use the stable logit form."""
        f = self.conv(sequence)
        summary = torch.cat([f.mean(dim=2), f[:, :, -self.last_slots :].mean(dim=2)], dim=1)
        return self.head(torch.cat([summary, profile], dim=1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
