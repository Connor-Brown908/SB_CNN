from seqparam.config import ElementClean
from seqparam.data.clean import clean
from seqparam.data.records import RawRecord


def _rec(seq, val, sd=None, source="s", unit="u", et="rbs"):
    return RawRecord(source, et, "id", seq, val, unit, sd)


def test_each_defect_is_dropped_and_counted():
    cfg = ElementClean(min_len=4, max_len=8, value_min=1.0, value_max=100.0,
                       label_z_clip=0.0)
    recs = [
        _rec("ACGT", 50.0),          # keep
        _rec("acgtAC", 60.0),        # keep (lower-cased)
        _rec("ACGN", 10.0),          # alphabet
        _rec("AC", 10.0),            # too short
        _rec("ACGTACGTAC", 10.0),    # too long
        _rec("ACGT", float("nan")),  # value invalid
        _rec("ACGT", 0.5),           # below value_min
        _rec("ACGT", 250.0),         # above value_max
        _rec("ACGT", 50.0),          # duplicate of row 0
    ]
    kept, rep = clean(recs, cfg, source="s")
    seqs = {k.sequence for k in kept}
    assert seqs == {"ACGT", "ACGTAC"}
    d = rep.dropped
    assert d["alphabet"] == 1 and d["length"] == 2
    assert d["value_invalid"] == 1 and d["value_range"] == 2
    assert d["duplicate"] == 1
    assert rep.n_in == 9 and rep.n_out == 2


def test_uppercasing_is_applied_to_kept():
    cfg = ElementClean(min_len=1, max_len=99, label_z_clip=0.0)
    kept, _ = clean([_rec("acgt", 1.0)], cfg)
    assert kept[0].sequence == "ACGT"


def test_rel_sd_filter():
    cfg = ElementClean(min_len=1, max_len=99, label_z_clip=0.0, max_rel_sd=0.1)
    kept, rep = clean([_rec("ACGT", 100.0, sd=5.0),     # rel sd 0.05 keep
                       _rec("CGTA", 100.0, sd=50.0)],   # rel sd 0.5 drop
                      cfg)
    assert len(kept) == 1 and kept[0].sequence == "ACGT"
    assert rep.dropped["noisy"] == 1


def _uniq_seq(i: int) -> str:
    s = ""
    for _ in range(6):
        s += "ACGT"[i & 3]
        i >>= 2
    return s


def test_label_outlier_clipping():
    cfg = ElementClean(min_len=1, max_len=99, label_z_clip=3.0)
    # many tight, unique inliers so one extreme value is a clear >3-SD log-space outlier
    recs = [_rec(_uniq_seq(i), 10.0 + (i % 5)) for i in range(15)]
    recs.append(_rec("ACGTACGTAC", 1_000_000.0))   # the outlier
    kept, rep = clean(recs, cfg)
    assert rep.dropped["label_outlier"] == 1
    assert all(k.value < 1000 for k in kept)
