import numpy as np

from seqparam.data.split import assign_splits, cluster


def test_identical_and_near_identical_cluster_together():
    base = "ACGT" * 24          # 96 bp
    near = base[:-1] + "A"      # differs by 1 bp -> shares most k-mers
    far = "TGCA" * 24
    cids = cluster([base, near, far], k=12, threshold=0.4)
    assert cids[0] == cids[1]   # base and near-dup in the same cluster
    assert cids[2] != cids[0]   # far sequence separate


def test_distinct_random_sequences_are_singletons():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 96)) for _ in range(40)]
    cids = cluster(seqs, k=12, threshold=0.5)
    assert len(np.unique(cids)) == len(seqs)  # all singletons


def test_no_cluster_spans_two_splits():
    # 10 clusters of 5 near-identical sequences each
    seqs, truth = [], []
    for c in range(10):
        base = f"{c:02d}" + "ACGT" * 23 + "AA"      # 96 bp, unique per cluster
        for j in range(5):
            s = list(base)
            s[90 + (j % 4)] = "ACGT"[j % 4]          # tiny within-cluster edits
            seqs.append("".join(s)[:96])
            truth.append(c)
    cids = cluster(seqs, k=10, threshold=0.3)
    splits = assign_splits(cids, val_fraction=0.2, test_fraction=0.2, seed=1)

    # every sequence sharing a cluster id must share its split (no leakage)
    by_cluster = {}
    for cid, sp in zip(cids, splits):
        by_cluster.setdefault(cid, set()).add(sp)
    assert all(len(v) == 1 for v in by_cluster.values())


def test_assign_splits_cover_all_three_and_are_deterministic():
    cids = np.arange(100)  # 100 singleton clusters
    a = assign_splits(cids, 0.2, 0.2, seed=7)
    b = assign_splits(cids, 0.2, 0.2, seed=7)
    assert np.array_equal(a, b)                 # deterministic under a seed
    assert set(a) == {"train", "val", "test"}
    # fractions roughly honored
    frac_test = (a == "test").mean()
    assert 0.1 <= frac_test <= 0.3
