"""Torch dataset bridge: Postgres split -> (one-hot tensor, normalized target).

`build_dataloaders` reads the train/val/test splits for one element type, fits
per-source normalization on TRAIN only, and returns ready DataLoaders plus the
fitted `NormStats` (which get saved inside the model's param file).

Targets are normalized; keep `source` so predictions can be inverted to native
units at eval time. Sequences are one-hot encoded lazily per item to keep memory
flat over hundreds of thousands of rows.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .encode import one_hot
from .normalize import NormStats
from .store import read_sequences


class SeqParamDataset(Dataset):
    def __init__(self, rows: list[dict], norm: NormStats, seq_len: int):
        self.sequences = [r["sequence"] for r in rows]
        self.sources = [r["source"] for r in rows]
        self.seq_len = seq_len
        self.ids = [r["id"] for r in rows]
        self.y = norm.transform([r["value"] for r in rows], self.sources)  # (N,) float32

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
        x = torch.from_numpy(one_hot(self.sequences[i], self.seq_len))  # (4, L)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


def build_dataloaders(engine, element_type: str, cfg) -> dict:
    """Return {'loaders': {split: DataLoader}, 'norm', 'sources', 'counts'}."""
    seq_len = cfg.data.seq_len_for(element_type)
    rows = {sp: read_sequences(engine, element_type=element_type, split=sp)
            for sp in ("train", "val", "test")}
    if not rows["train"]:
        raise ValueError(f"no train rows for element_type={element_type!r}; "
                         "did you run ingestion + splits?")

    norm = NormStats.fit(rows["train"])
    datasets = {sp: SeqParamDataset(rows[sp], norm, seq_len) for sp in rows}

    loaders = {}
    for sp, ds in datasets.items():
        loaders[sp] = DataLoader(
            ds, batch_size=cfg.optim.batch_size, shuffle=(sp == "train"),
            num_workers=cfg.optim.num_workers, drop_last=False)

    sources = sorted({s for r in rows["train"] for s in [r["source"]]})
    counts = {sp: len(rows[sp]) for sp in rows}
    return {"loaders": loaders, "norm": norm, "sources": sources, "counts": counts}
