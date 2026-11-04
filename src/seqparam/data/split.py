"""Similarity-based train/val/test splitting.

Random splits leak: near-identical sequences landing in both train and test make
a model look better than it is. Here, sequences are clustered by k-mer Jaccard
similarity and whole clusters are assigned to one split, so related sequences
never straddle the boundary.

Scales to hundreds of thousands of sequences: candidate pairs are generated from
a k-mer inverted index (only sequences that share a k-mer are ever compared), and
over-common k-mer buckets are skipped. For a well-separated random library this
yields mostly singleton clusters (→ effectively a random split, which is correct
when there are no near-duplicates); for structured/mutant libraries it groups the
related variants.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _kmer_ints(seq: str, k: int) -> set[int]:
    """2-bit-encode every k-mer of seq into an int set (non-ACGT k-mers skipped)."""
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    out: set[int] = set()
    val = 0
    valid = 0
    mask = (1 << (2 * k)) - 1
    for ch in seq:
        c = code.get(ch)
        if c is None:
            val = 0
            valid = 0
            continue
        val = ((val << 2) | c) & mask
        valid += 1
        if valid >= k:
            out.add(val)
    return out


def cluster(sequences: list[str], k: int, threshold: float,
            bucket_cap: int = 200) -> np.ndarray:
    """Return an int cluster id per sequence (union-find over similar pairs)."""
    n = len(sequences)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    kmers = [_kmer_ints(s, k) for s in sequences]

    index: dict[int, list[int]] = defaultdict(list)
    for i, ks in enumerate(kmers):
        for km in ks:
            index[km].append(i)

    # candidate pairs: co-occur in a (not-too-common) bucket -> count shared k-mers
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for idxs in index.values():
        if len(idxs) < 2 or len(idxs) > bucket_cap:
            continue
        for a in range(len(idxs)):
            ia = idxs[a]
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                shared[(ia, ib) if ia < ib else (ib, ia)] += 1

    for (i, j), sh in shared.items():
        union_sz = len(kmers[i]) + len(kmers[j]) - sh
        if union_sz > 0 and sh / union_sz >= threshold:
            union(i, j)

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    # relabel roots to contiguous 0..C-1
    _, cid = np.unique(roots, return_inverse=True)
    return cid.astype(np.int64)


def assign_splits(cluster_ids: np.ndarray, val_fraction: float,
                  test_fraction: float, seed: int = 1234) -> np.ndarray:
    """Assign whole clusters to 'train'/'val'/'test', targeting the given fractions."""
    n = len(cluster_ids)
    rng = np.random.default_rng(seed)

    clusters = np.unique(cluster_ids)
    rng.shuffle(clusters)
    sizes = {int(c): int((cluster_ids == c).sum()) for c in clusters}

    want_test = test_fraction * n
    want_val = val_fraction * n
    split_of_cluster: dict[int, str] = {}
    acc_test = acc_val = 0
    for c in clusters:
        c = int(c)
        if acc_test < want_test:
            split_of_cluster[c] = "test"
            acc_test += sizes[c]
        elif acc_val < want_val:
            split_of_cluster[c] = "val"
            acc_val += sizes[c]
        else:
            split_of_cluster[c] = "train"

    return np.array([split_of_cluster[int(c)] for c in cluster_ids], dtype=object)
