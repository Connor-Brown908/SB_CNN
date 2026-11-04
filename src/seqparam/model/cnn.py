"""The Stage-1 sequence CNN.

One-hot DNA (4, L) -> stacked 1D convolutions (motif detectors) -> position-
tolerant adaptive max-pool -> MLP -> head. With `heteroscedastic=True` the head
emits both a mean and a log-variance per sequence, so each prediction carries its
own uncertainty (consumed later by the mechanistic simulator). One model predicts
one parameter (promoter OR rbs).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SeqCNN(nn.Module):
    def __init__(self, *, channels=(64, 64), kernel_sizes=(12, 8), pool_size=16,
                 mlp_hidden=128, dropout=0.3, heteroscedastic=True, in_channels=4):
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must match in length")
        self.heteroscedastic = heteroscedastic

        convs = []
        c_in = in_channels
        for c_out, k in zip(channels, kernel_sizes):
            convs += [nn.Conv1d(c_in, c_out, kernel_size=k, padding="same"),
                      nn.ReLU()]
            c_in = c_out
        convs.append(nn.AdaptiveMaxPool1d(pool_size))
        self.conv = nn.Sequential(*convs)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * pool_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(mlp_hidden, 1)
        self.logvar_head = nn.Linear(mlp_hidden, 1) if heteroscedastic else None

    def forward(self, x):
        """x: (B, 4, L) -> (mean (B,), log_var (B,) | None)."""
        h = self.mlp(self.conv(x))
        mean = self.mean_head(h).squeeze(-1)
        if self.logvar_head is None:
            return mean, None
        log_var = self.logvar_head(h).squeeze(-1)
        return mean, log_var

    @torch.no_grad()
    def predict(self, x):
        """Eval-mode mean and standard deviation (std=None if homoscedastic)."""
        self.eval()
        mean, log_var = self(x)
        std = None if log_var is None else torch.exp(0.5 * log_var)
        return mean, std
