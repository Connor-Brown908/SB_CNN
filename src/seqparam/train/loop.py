"""The training / evaluation loop.

`train` wires the dataset bridge, model, loss, monitor, early stopping and
checkpointing into one run for a single element type (promoter or rbs). Early
stopping watches the validation loss (normalized space); reported metrics are
computed in NATIVE units (predictions inverted per source) so RMSE/R^2 are
interpretable and comparable to the source papers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data.dataset import build_dataloaders
from ..data.encode import one_hot_batch
from ..model.params import ModelParams, build_model, load_checkpoint, save_checkpoint
from .losses import compute_loss
from .metrics import regression_metrics
from .monitor import EarlyStopping, make_monitor


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one_epoch(model, loader, opt, grad_clip, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        mean, log_var = model(x)
        loss = compute_loss(mean, log_var, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += loss.detach().item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def _val_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mean, log_var = model(x)
        total += float(compute_loss(mean, log_var, y)) * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_native(model, rows: list[dict], norm, seq_len: int, device,
                    batch_size: int = 512) -> dict:
    """Predict on `rows`, invert to native units per source, score."""
    model.eval()
    if not rows:
        return {}
    seqs = [r["sequence"] for r in rows]
    sources = [r["source"] for r in rows]
    y_true = np.array([r["value"] for r in rows], dtype=float)

    means_z, stds_z = [], []
    for i in range(0, len(seqs), batch_size):
        xb = torch.from_numpy(one_hot_batch(seqs[i:i + batch_size], seq_len)).to(device)
        mean, std = model.predict(xb)
        means_z.append(mean.cpu().numpy())
        stds_z.append(np.zeros(len(mean)) if std is None else std.cpu().numpy())
    mean_z = np.concatenate(means_z)
    std_z = np.concatenate(stds_z)

    # invert mean to native units; propagate std by local linearization of expm1
    y_pred = norm.inverse(mean_z, sources)
    scale = np.array([norm._ms(s)[1] for s in sources])      # per-source log-std
    y_std = (1.0 + y_pred) * scale * std_z if np.any(std_z) else None
    return regression_metrics(y_true, y_pred, y_std)


def train(cfg: Config, engine, element_type: str, out_dir: str | Path,
          monitor_backend: str | None = None) -> dict:
    device = _device()
    data = build_dataloaders(engine, element_type, cfg)
    loaders, norm = data["loaders"], data["norm"]
    seq_len = cfg.data.seq_len_for(element_type)

    params = ModelParams.from_config(
        cfg, element_type=element_type, unit=data["unit"], seq_len=seq_len,
        norm=norm, sources=data["sources"])
    model = build_model(params).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr,
                           weight_decay=cfg.optim.weight_decay)

    mon_cfg = cfg.monitor.model_copy(update={"backend": monitor_backend}) \
        if monitor_backend else cfg.monitor
    monitor = make_monitor(mon_cfg, run_name=f"{element_type}",
                           config={"element_type": element_type, **cfg.model.model_dump()})
    stopper = EarlyStopping(patience=cfg.optim.early_stop_patience, mode="min")

    out_dir = Path(out_dir)
    history = []
    best_val = float("inf")
    for epoch in range(cfg.optim.epochs):
        tr = _train_one_epoch(model, loaders["train"], opt, cfg.optim.grad_clip, device)
        va = _val_loss(model, loaders["val"], device)
        val_metrics = evaluate_native(model, data["rows"]["val"], norm, seq_len, device)
        rec = {"train_loss": tr, "val_loss": va,
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        monitor.log(epoch, rec)
        history.append(rec)

        improved, stop = stopper.step(va)
        if improved:
            best_val = va
            save_checkpoint(out_dir, model, params)
        if stop:
            break

    monitor.finish()

    # reload best weights and score the held-out test set in native units
    best_model, _ = load_checkpoint(out_dir)
    best_model.to(device)
    test_metrics = evaluate_native(best_model, data["rows"]["test"], norm, seq_len, device)
    return {"element_type": element_type, "out_dir": str(out_dir),
            "best_val_loss": best_val, "epochs_ran": len(history),
            "counts": data["counts"], "test_metrics": test_metrics,
            "history": history}


def evaluate(cfg: Config, engine, element_type: str, ckpt_dir: str | Path,
             split: str = "test") -> dict:
    """Load a trained checkpoint and score one split in native units."""
    from ..data.store import read_sequences
    device = _device()
    model, params = load_checkpoint(ckpt_dir)
    model.to(device)
    norm = params.norm_stats()
    rows = read_sequences(engine, element_type=element_type, split=split)
    return evaluate_native(model, rows, norm, params.seq_len, device)
