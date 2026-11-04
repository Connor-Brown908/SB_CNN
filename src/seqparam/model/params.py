"""Model param file: everything needed to rebuild a trained model and interpret
its outputs, independent of the training code.

A saved model is a directory:
    <dir>/params.json   architecture + metadata + per-source NormStats
    <dir>/weights.pt    torch state_dict

`ModelParams` itself is torch-free so configs/metadata can be inspected without
importing torch; `build_model`, `save_checkpoint`, `load_checkpoint` import it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..data.normalize import NormStats

PARAMS_FILE = "params.json"
WEIGHTS_FILE = "weights.pt"


@dataclass
class ModelParams:
    element_type: str                 # "promoter" | "rbs"
    unit: str                         # native target unit, e.g. "PNI_1_100"
    seq_len: int
    channels: tuple[int, ...] = (64, 64)
    kernel_sizes: tuple[int, ...] = (12, 8)
    pool_size: int = 16
    mlp_hidden: int = 128
    dropout: float = 0.3
    heteroscedastic: bool = True
    sources: list[str] = field(default_factory=list)
    norm: dict = field(default_factory=dict)      # NormStats.to_dict()
    version: str = "0.1.0"
    created_at: str = ""

    @classmethod
    def from_config(cls, cfg, *, element_type: str, unit: str, seq_len: int,
                    norm: NormStats, sources: list[str]) -> "ModelParams":
        m = cfg.model
        return cls(
            element_type=element_type, unit=unit, seq_len=seq_len,
            channels=tuple(m.channels), kernel_sizes=tuple(m.kernel_sizes),
            pool_size=m.pool_size, mlp_hidden=m.mlp_hidden, dropout=m.dropout,
            heteroscedastic=m.heteroscedastic, sources=list(sources),
            norm=norm.to_dict(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def norm_stats(self) -> NormStats:
        return NormStats.from_dict(self.norm)

    # ---- json ----------------------------------------------------------------
    def to_json(self) -> str:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["kernel_sizes"] = list(self.kernel_sizes)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelParams":
        d = json.loads(s)
        d["channels"] = tuple(d["channels"])
        d["kernel_sizes"] = tuple(d["kernel_sizes"])
        return cls(**d)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelParams":
        return cls.from_json(Path(path).read_text())


def build_model(params: ModelParams):
    """Instantiate an (untrained) SeqCNN matching `params`."""
    from .cnn import SeqCNN
    return SeqCNN(channels=params.channels, kernel_sizes=params.kernel_sizes,
                  pool_size=params.pool_size, mlp_hidden=params.mlp_hidden,
                  dropout=params.dropout, heteroscedastic=params.heteroscedastic)


def save_checkpoint(directory: str | Path, model, params: ModelParams) -> None:
    import torch
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    params.save_json(d / PARAMS_FILE)
    torch.save(model.state_dict(), d / WEIGHTS_FILE)


def load_checkpoint(directory: str | Path):
    """Return (model, params) with weights loaded, in eval mode."""
    import torch
    d = Path(directory)
    params = ModelParams.load_json(d / PARAMS_FILE)
    model = build_model(params)
    model.load_state_dict(torch.load(d / WEIGHTS_FILE, map_location="cpu"))
    model.eval()
    return model, params
