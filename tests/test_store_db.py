"""Storage-layer tests against an ephemeral Postgres (the `engine` fixture)."""
from seqparam.data.records import RawRecord
from seqparam.data.store import (element_counts, read_sequences,
                                 update_cluster_and_split, write_dataset)


def _recs(n, et="rbs", source="src"):
    # unique sequences per record so none are dropped as duplicates
    return [RawRecord(source, et, f"id{i}", ("ACGT" * 24)[:-1] + "ACGT"[i % 4],
                      float(i + 1), "PNI_1_100", 1.0) for i in range(n)]


def test_write_and_read_roundtrip(engine):
    ds_id = write_dataset(engine, name="d1", source="src", element_type="rbs",
                          unit="PNI_1_100", records=_recs(3), clean_report={"n_out": 3})
    assert isinstance(ds_id, int)
    rows = read_sequences(engine, element_type="rbs")
    assert len(rows) == 3
    assert {r["value"] for r in rows} == {1.0, 2.0, 3.0}
    assert all(r["split"] is None for r in rows)   # not yet split


def test_write_is_idempotent_by_name(engine):
    write_dataset(engine, name="d1", source="src", element_type="rbs",
                  unit="PNI_1_100", records=_recs(3), clean_report={})
    write_dataset(engine, name="d1", source="src", element_type="rbs",
                  unit="PNI_1_100", records=_recs(2), clean_report={})  # replace
    assert len(read_sequences(engine, element_type="rbs")) == 2


def test_update_cluster_and_split_and_counts(engine):
    write_dataset(engine, name="d1", source="src", element_type="rbs",
                  unit="PNI_1_100", records=_recs(4), clean_report={})
    rows = read_sequences(engine, element_type="rbs")
    updates = [{"id": r["id"], "cluster_id": i, "split": ["train", "val", "test", "train"][i]}
               for i, r in enumerate(rows)]
    update_cluster_and_split(engine, updates)

    counts = element_counts(engine)["rbs"]
    assert counts == {"train": 2, "val": 1, "test": 1}
    assert len(read_sequences(engine, element_type="rbs", split="train")) == 2


def test_source_filter(engine):
    write_dataset(engine, name="a", source="srcA", element_type="promoter",
                  unit="TX_au", records=_recs(2, et="promoter", source="srcA"), clean_report={})
    write_dataset(engine, name="b", source="srcB", element_type="promoter",
                  unit="TX_au", records=_recs(3, et="promoter", source="srcB"), clean_report={})
    assert len(read_sequences(engine, source="srcA")) == 2
    assert len(read_sequences(engine, source="srcB")) == 3
