"""Dataset bridge tests (needs torch + the ephemeral Postgres `engine`)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seqparam.config import load_config  # noqa: E402
from seqparam.data.dataset import build_dataloaders  # noqa: E402
from seqparam.data.records import RawRecord  # noqa: E402
from seqparam.data.store import (read_sequences, update_cluster_and_split,  # noqa: E402
                                 write_dataset)


def _seq(i: int, n: int = 96) -> str:
    s = ""
    j = i
    while len(s) < n:
        s += "ACGT"[j & 3]
        j = (j >> 2) or (j + 7)
    return s[:n]


def _populate(engine, source, values, element_type="rbs"):
    recs = [RawRecord(source, element_type, f"{source}{i}", _seq(i + hash(source) % 97),
                      float(v), "PNI_1_100", None) for i, v in enumerate(values)]
    write_dataset(engine, name=f"{source}_ds", source=source,
                  element_type=element_type, unit="PNI_1_100", records=recs,
                  clean_report={})


def _assign_splits(engine, element_type):
    rows = read_sequences(engine, element_type=element_type)
    updates = []
    for k, r in enumerate(rows):
        sp = "train" if k % 10 < 7 else ("val" if k % 10 < 8 else "test")
        updates.append({"id": r["id"], "cluster_id": k, "split": sp})
    update_cluster_and_split(engine, updates)


def test_build_dataloaders_shapes_and_counts(engine):
    cfg = load_config(None)
    _populate(engine, "cambray2018", list(range(10, 50)))      # 40 rows
    _assign_splits(engine, "rbs")

    out = build_dataloaders(engine, "rbs", cfg)
    assert set(out["loaders"]) == {"train", "val", "test"}
    assert out["counts"]["train"] > 0 and out["counts"]["val"] > 0

    xb, yb = next(iter(out["loaders"]["train"]))
    assert xb.shape[1:] == (4, cfg.data.rbs_seq_len)   # (B, 4, 96)
    assert yb.ndim == 1 and xb.dtype == torch.float32
    # one-hot: each column sums to 1
    assert torch.allclose(xb.sum(dim=1), torch.ones_like(xb.sum(dim=1)))


def test_norm_is_fit_on_train_only_and_per_source(engine):
    cfg = load_config(None)
    _populate(engine, "srcA", list(range(1, 41)))
    _populate(engine, "srcB", list(range(100, 140)))
    _assign_splits(engine, "rbs")

    out = build_dataloaders(engine, "rbs", cfg)
    norm = out["norm"]
    assert set(norm.per_source) == {"srcA", "srcB"}
    # the two sources get different means (different scales) -> per-source, not global
    assert norm.per_source["srcA"]["mean"] != norm.per_source["srcB"]["mean"]


def test_raises_without_train_rows(engine):
    cfg = load_config(None)
    with pytest.raises(ValueError):
        build_dataloaders(engine, "rbs", cfg)
