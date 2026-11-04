"""Training losses.

Heteroscedastic Gaussian negative log-likelihood lets the model report its own
uncertainty: predicting a larger variance softens the penalty on hard examples,
so log-variance is learned jointly with the mean. Falls back to MSE when the
model is homoscedastic (log_var is None).
"""
from __future__ import annotations

import torch


def gaussian_nll(mean, log_var, y, reduction: str = "mean"):
    """0.5 * (exp(-s) * (y-mean)^2 + s), s = log variance. Per-element, then reduced."""
    log_var = torch.clamp(log_var, min=-10.0, max=10.0)  # numerical safety
    loss = 0.5 * (torch.exp(-log_var) * (y - mean) ** 2 + log_var)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def mse_loss(mean, y, reduction: str = "mean"):
    loss = (y - mean) ** 2
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def compute_loss(mean, log_var, y):
    """Pick NLL when the model is heteroscedastic, else MSE."""
    if log_var is None:
        return mse_loss(mean, y)
    return gaussian_nll(mean, log_var, y)
