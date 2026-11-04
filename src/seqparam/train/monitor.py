"""Training monitors + early stopping.

Monitor backends:
  - "wandb"    Weights & Biases (offline by default -> no account/network needed)
  - "csv"      append metric rows to <log_dir>/<run>.csv
  - "disabled" no-op (used in tests)

All share one interface: log(step, metrics), finish(). W&B import failure falls
back to CSV so a run never dies for lack of the optional dependency.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path


class _CSVMonitor:
    def __init__(self, log_dir: str, run_name: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(log_dir) / f"{run_name}.csv"
        self._fh = open(self.path, "w", newline="")
        self._writer = None

    def log(self, step: int, metrics: dict):
        row = {"step": step, **metrics}
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def finish(self):
        self._fh.close()


class _WandbMonitor:
    def __init__(self, cfg, run_name, config):
        import wandb
        self._wandb = wandb
        self.run = wandb.init(project=cfg.project, name=run_name,
                              mode=cfg.wandb_mode, dir=cfg.log_dir, config=config,
                              reinit=True)

    def log(self, step, metrics):
        self._wandb.log(metrics, step=step)

    def finish(self):
        self.run.finish()


class _NullMonitor:
    def log(self, step, metrics):
        pass

    def finish(self):
        pass


def make_monitor(cfg, run_name: str | None = None, config: dict | None = None):
    run_name = run_name or cfg.run_name or f"run-{int(time.time())}"
    if cfg.backend == "disabled":
        return _NullMonitor()
    if cfg.backend == "wandb":
        try:
            return _WandbMonitor(cfg, run_name, config or {})
        except Exception:
            return _CSVMonitor(cfg.log_dir, run_name)   # graceful fallback
    return _CSVMonitor(cfg.log_dir, run_name)


class EarlyStopping:
    """Stop when the monitored value hasn't improved for `patience` checks."""

    def __init__(self, patience: int = 8, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad = 0

    def step(self, value: float) -> tuple[bool, bool]:
        """Return (improved, should_stop)."""
        if self.best is None:
            self.best = value
            return True, False
        improved = (value < self.best - self.min_delta) if self.mode == "min" \
            else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
            return True, False
        self.num_bad += 1
        return False, self.num_bad >= self.patience
