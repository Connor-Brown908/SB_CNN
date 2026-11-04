"""Per-source normalization (torch-free)."""
import math

import numpy as np

from seqparam.data.normalize import NormStats


def _rows(source, values):
    return [{"source": source, "value": v} for v in values]


def test_per_source_standardizes_each_source():
    rows = _rows("A", [1, 2, 3, 4, 5]) + _rows("B", [100, 200, 300, 400, 500])
    norm = NormStats.fit(rows)
    assert set(norm.per_source) == {"A", "B"}

    # transforming each source's own train values gives ~zero mean, unit std
    for src, vals in (("A", [1, 2, 3, 4, 5]), ("B", [100, 200, 300, 400, 500])):
        z = norm.transform(vals, [src] * len(vals))
        assert abs(float(z.mean())) < 1e-5
        assert abs(float(z.std()) - 1.0) < 1e-5


def test_sources_are_independent():
    # same z-score should map back to very different native values per source
    rows = _rows("A", [1, 2, 3, 4, 5]) + _rows("B", [100, 200, 300, 400, 500])
    norm = NormStats.fit(rows)
    inv = norm.inverse([0.0, 0.0], ["A", "B"])   # z=0 -> each source's (log) mean
    assert inv[0] < 10 and inv[1] > 50            # clearly different scales


def test_inverse_roundtrip():
    rows = _rows("A", [5, 10, 20, 40, 80])
    norm = NormStats.fit(rows)
    vals = np.array([7.0, 33.0, 60.0])
    z = norm.transform(vals, ["A"] * 3)
    back = norm.inverse(z, ["A"] * 3)
    assert np.allclose(back, vals, rtol=1e-5)


def test_unseen_source_falls_back_to_global():
    rows = _rows("A", [1, 2, 3, 4, 5])
    norm = NormStats.fit(rows)
    z = norm.transform([3.0], ["UNSEEN"])         # must not raise
    assert math.isfinite(float(z[0]))


def test_save_load_roundtrip(tmp_path):
    rows = _rows("A", [1, 2, 3]) + _rows("B", [10, 20, 30])
    norm = NormStats.fit(rows)
    p = tmp_path / "norm.json"
    norm.save(p)
    loaded = NormStats.load(p)
    assert loaded.per_source == norm.per_source
    z1 = norm.transform([2.0], ["A"])
    z2 = loaded.transform([2.0], ["A"])
    assert np.allclose(z1, z2)
