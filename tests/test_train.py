"""End-to-end training on tiny data (needs torch + ephemeral Postgres)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seqparam.config import load_config  # noqa: E402
from seqparam.data.records import RawRecord  # noqa: E402
from seqparam.data.store import (read_sequences, update_cluster_and_split,  # noqa: E402
                                 write_dataset)
from seqparam.model.params import PARAMS_FILE, WEIGHTS_FILE  # noqa: E402
from seqparam.train.loop import evaluate, train  # noqa: E402


def _seq_with_signal(i: int, n: int = 96) -> tuple[str, float]:
    """Sequence whose A-count drives the target -> a learnable mapping."""
    rng = np.random.default_rng(i)
    bases = rng.choice(list("ACGT"), size=n, p=[0.4, 0.2, 0.2, 0.2])
    s = "".join(bases)
    value = 1.0 + (s.count("A") / n) * 90.0      # within PNI-like 1..100
    return s, float(value)


def _populate_and_split(engine, source="cambray2018", n=80):
    recs = []
    for i in range(n):
        s, v = _seq_with_signal(i)
        recs.append(RawRecord(source, "rbs", f"{source}{i}", s, v, "PNI_1_100", None))
    write_dataset(engine, name=f"{source}_ds", source=source, element_type="rbs",
                  unit="PNI_1_100", records=recs, clean_report={})
    rows = read_sequences(engine, element_type="rbs")
    updates = [{"id": r["id"], "cluster_id": k,
                "split": "train" if k % 10 < 7 else ("val" if k % 10 < 8 else "test")}
               for k, r in enumerate(rows)]
    update_cluster_and_split(engine, updates)


def _tiny_cfg():
    cfg = load_config(None)
    cfg = cfg.model_copy(update={
        "optim": cfg.optim.model_copy(update={
            "epochs": 5, "batch_size": 16, "early_stop_patience": 3, "lr": 1e-2}),
        "model": cfg.model.model_copy(update={
            "channels": (8, 8), "kernel_sizes": (5, 3), "pool_size": 4, "mlp_hidden": 16}),
    })
    return cfg


def test_train_produces_checkpoint_and_metrics(engine, tmp_path):
    _populate_and_split(engine)
    cfg = _tiny_cfg()
    out = tmp_path / "rbs"
    summary = train(cfg, engine, "rbs", out, monitor_backend="disabled")

    # checkpoint written
    assert (out / PARAMS_FILE).exists() and (out / WEIGHTS_FILE).exists()
    # ran, stopped within the epoch budget, produced finite native-unit metrics
    assert 1 <= summary["epochs_ran"] <= cfg.optim.epochs
    tm = summary["test_metrics"]
    assert np.isfinite(tm["rmse"]) and np.isfinite(tm["r2"])
    assert "coverage95" in tm            # heteroscedastic -> uncertainty metrics present


def test_training_reduces_train_loss(engine, tmp_path):
    _populate_and_split(engine)
    cfg = _tiny_cfg().model_copy(update={
        "optim": load_config(None).optim.model_copy(update={
            "epochs": 15, "batch_size": 16, "early_stop_patience": 20, "lr": 1e-2})})
    summary = train(cfg, engine, "rbs", tmp_path / "rbs", monitor_backend="disabled")
    hist = summary["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]   # it actually learns


def test_evaluate_matches_training_test_metrics(engine, tmp_path):
    _populate_and_split(engine)
    cfg = _tiny_cfg()
    out = tmp_path / "rbs"
    summary = train(cfg, engine, "rbs", out, monitor_backend="disabled")
    reval = evaluate(cfg, engine, "rbs", out, split="test")
    assert abs(reval["rmse"] - summary["test_metrics"]["rmse"]) < 1e-6
