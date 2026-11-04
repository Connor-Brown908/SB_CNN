"""End-to-end Phase 1 on tiny fixtures: load -> clean -> store -> split."""
from seqparam.config import load_config
from seqparam.data.ingest import run_phase1
from seqparam.data.store import element_counts, read_sequences


def test_run_phase1_end_to_end(engine, cambray_zip, lafleur_xlsx):
    cfg = load_config(None)
    summary = run_phase1(cfg, engine, cambray_path=cambray_zip,
                         lafleur_path=lafleur_xlsx)

    # Cambray: 5 raw rows -> NaN skipped at load; dup + >100 dropped in clean -> 2 kept
    assert summary["clean"]["cambray"]["n_out"] == 2
    rbs = read_sequences(engine, element_type="rbs")
    assert len(rbs) == 2 and all(1.0 <= r["value"] <= 100.0 for r in rbs)

    # Promoter: 3 sources, one Urtecho NaN row skipped -> 4 kept total
    promoter = read_sequences(engine, element_type="promoter")
    assert len(promoter) == 4
    assert {r["source"] for r in promoter} == {"lafleur2022", "urtecho2019", "hossain2020"}

    # every row got a split label
    assert all(r["split"] in {"train", "val", "test"} for r in rbs + promoter)


def test_run_phase1_is_idempotent(engine, cambray_zip, lafleur_xlsx):
    cfg = load_config(None)
    run_phase1(cfg, engine, cambray_path=cambray_zip, lafleur_path=lafleur_xlsx)
    run_phase1(cfg, engine, cambray_path=cambray_zip, lafleur_path=lafleur_xlsx)
    # re-ingesting replaces datasets by name rather than duplicating rows
    counts = element_counts(engine)
    assert sum(counts["rbs"].values()) == 2
    assert sum(counts["promoter"].values()) == 4


def test_splits_do_not_leak_clusters(engine, cambray_zip, lafleur_xlsx):
    cfg = load_config(None)
    run_phase1(cfg, engine, cambray_path=cambray_zip, lafleur_path=lafleur_xlsx)
    for et in ("rbs", "promoter"):
        rows = read_sequences(engine, element_type=et)
        by_cluster = {}
        for r in rows:
            by_cluster.setdefault(r["cluster_id"], set()).add(r["split"])
        assert all(len(v) == 1 for v in by_cluster.values())
