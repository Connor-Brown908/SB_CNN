"""Per-source target normalization (log -> z-score), fit on the train split only.

The promoter sources (La Fleur, Urtecho, Hossain) all report transcription in
"arbitrary units" but from different experiments, so their scales differ. A
single global z-score would fold those batch effects into the target. Instead we
standardize each source independently: for source s,

    z = (log1p(value) - mean_s) / std_s

with mean_s, std_s estimated over that source's TRAIN rows. Inversion maps model
outputs back to native units for evaluation. Stats are JSON-serializable so they
travel inside the model's param file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class NormStats:
    # source -> {"mean": float, "std": float, "n": int}, in log1p space
    per_source: dict[str, dict] = field(default_factory=dict)
    global_mean: float = 0.0
    global_std: float = 1.0

    # ---- fitting -------------------------------------------------------------
    @classmethod
    def fit(cls, rows: list[dict]) -> "NormStats":
        """Estimate per-source log-space mean/std from rows with 'value','source'."""
        by_src: dict[str, list[float]] = {}
        for r in rows:
            by_src.setdefault(r["source"], []).append(float(r["value"]))
        per_source = {}
        for src, vals in by_src.items():
            logs = np.log1p(np.asarray(vals, dtype=float))
            std = float(logs.std())
            per_source[src] = {"mean": float(logs.mean()),
                               "std": std if std > 0 else 1.0,
                               "n": int(len(vals))}
        all_logs = np.log1p(np.asarray([float(r["value"]) for r in rows], dtype=float)) \
            if rows else np.array([0.0])
        gstd = float(all_logs.std())
        return cls(per_source=per_source,
                   global_mean=float(all_logs.mean()),
                   global_std=gstd if gstd > 0 else 1.0)

    def _ms(self, source: str) -> tuple[float, float]:
        s = self.per_source.get(source)
        if s is None:                      # source unseen in train -> global fallback
            return self.global_mean, self.global_std
        return s["mean"], s["std"]

    # ---- transforms ----------------------------------------------------------
    def transform(self, values, sources) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        logs = np.log1p(values)
        means = np.array([self._ms(s)[0] for s in sources])
        stds = np.array([self._ms(s)[1] for s in sources])
        return ((logs - means) / stds).astype(np.float32)

    def inverse(self, z, sources) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        means = np.array([self._ms(s)[0] for s in sources])
        stds = np.array([self._ms(s)[1] for s in sources])
        return np.expm1(z * stds + means)

    # ---- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {"per_source": self.per_source,
                "global_mean": self.global_mean, "global_std": self.global_std}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(per_source=d.get("per_source", {}),
                   global_mean=d.get("global_mean", 0.0),
                   global_std=d.get("global_std", 1.0))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        return cls.from_dict(json.loads(Path(path).read_text()))
