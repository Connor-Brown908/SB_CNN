"""Regression + calibration metrics (numpy, framework-free).

Point metrics (RMSE, MAE, R^2, Pearson, Spearman) say whether the mean is right;
calibration metrics (NLL, 95% coverage, sharpness) say whether the predicted
uncertainty is trustworthy -- essential because the downstream simulator consumes
the std, not just the mean.
"""
from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank (ties shared), avoiding a scipy dependency."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearson(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y, p):
    return pearson(_rankdata(np.asarray(y, float)), _rankdata(np.asarray(p, float)))


def gaussian_nll_np(y, mean, std):
    y, mean, std = (np.asarray(a, float) for a in (y, mean, std))
    var = np.clip(std ** 2, 1e-12, None)
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)))


def coverage(y, mean, std, z: float = 1.96):
    """Fraction of targets inside mean +/- z*std (should be ~0.95 at z=1.96)."""
    y, mean, std = (np.asarray(a, float) for a in (y, mean, std))
    lo, hi = mean - z * std, mean + z * std
    return float(np.mean((y >= lo) & (y <= hi)))


def regression_metrics(y_true, y_pred, y_std=None) -> dict:
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "n": int(len(y_true)),
    }
    if y_std is not None:
        out["nll"] = gaussian_nll_np(y_true, y_pred, y_std)
        out["coverage95"] = coverage(y_true, y_pred, y_std)
        out["sharpness"] = float(np.mean(np.asarray(y_std, float)))
    return out
