"""Sequence <-> tensor encoding.

One-hot encodes DNA over the fixed alphabet ACGT into a (4, L) float array,
the channel-first layout a 1D CNN expects. Sequences are padded with zero
columns (all-zero = "no base") or truncated to a fixed length.
"""
from __future__ import annotations

import numpy as np

from .. import ALPHABET

# base -> row index; anything else -> -1 (encoded as an all-zero column)
_LUT = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(ALPHABET):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i


def is_valid_sequence(seq: str) -> bool:
    """True iff every character is in the (upper-case) alphabet."""
    return len(seq) > 0 and all(c in ALPHABET for c in seq)


def one_hot(seq: str, length: int) -> np.ndarray:
    """Encode one sequence to a (4, length) float32 array (pad/truncate)."""
    idx = _LUT[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    idx = idx[:length]
    out = np.zeros((len(ALPHABET), length), dtype=np.float32)
    pos = np.arange(idx.shape[0])
    valid = idx >= 0
    out[idx[valid], pos[valid]] = 1.0
    return out


def one_hot_batch(seqs: list[str], length: int) -> np.ndarray:
    """Encode a list of sequences to (N, 4, length) float32."""
    arr = np.zeros((len(seqs), len(ALPHABET), length), dtype=np.float32)
    for i, s in enumerate(seqs):
        arr[i] = one_hot(s, length)
    return arr


def gc_content(seq: str) -> float:
    """Fraction of G/C bases (0 for empty)."""
    if not seq:
        return 0.0
    gc = sum(c in "GCgc" for c in seq)
    return gc / len(seq)
