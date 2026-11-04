"""Quality control for canonical records.

Rules, in order (each drop is counted in the report):
  1. schema        - sequence is a non-empty string, value present
  2. alphabet      - only A/C/G/T after upper-casing
  3. length        - within [min_len, max_len]
  4. value finite  - value finite; within [value_min, value_max] if set
  5. rel. sd       - drop noisy measurements if sd/|value| > max_rel_sd (when set)
  6. dedup         - drop exact duplicate sequences (keep first seen)
  7. label outlier - drop values beyond label_z_clip SDs in log-space (when > 0)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .encode import is_valid_sequence
from .records import RawRecord


@dataclass
class CleanReport:
    source: str = ""
    n_in: int = 0
    n_out: int = 0
    dropped: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {"source": self.source, "n_in": self.n_in, "n_out": self.n_out,
                "dropped": dict(self.dropped)}


def clean(records: list[RawRecord], clean_cfg, *, source: str = "") -> tuple[list[RawRecord], CleanReport]:
    rep = CleanReport(source=source, n_in=len(records))
    kept: list[RawRecord] = []
    seen: set[str] = set()

    for r in records:
        seq = r.sequence
        if not isinstance(seq, str) or not seq or r.value is None:
            rep.dropped["schema"] += 1
            continue
        seq = seq.upper()
        if not is_valid_sequence(seq):
            rep.dropped["alphabet"] += 1
            continue
        if not (clean_cfg.min_len <= len(seq) <= clean_cfg.max_len):
            rep.dropped["length"] += 1
            continue

        v = r.value
        if not np.isfinite(v):
            rep.dropped["value_invalid"] += 1
            continue
        if clean_cfg.value_min is not None and v < clean_cfg.value_min:
            rep.dropped["value_range"] += 1
            continue
        if clean_cfg.value_max is not None and v > clean_cfg.value_max:
            rep.dropped["value_range"] += 1
            continue
        if (clean_cfg.max_rel_sd is not None and r.value_sd is not None
                and abs(v) > 0 and (r.value_sd / abs(v)) > clean_cfg.max_rel_sd):
            rep.dropped["noisy"] += 1
            continue
        if seq in seen:
            rep.dropped["duplicate"] += 1
            continue

        seen.add(seq)
        kept.append(RawRecord(r.source, r.element_type, r.seq_id, seq, float(v),
                              r.unit, r.value_sd, dict(r.meta)))

    # log-space label outlier clipping
    if kept and clean_cfg.label_z_clip and clean_cfg.label_z_clip > 0:
        x = np.log1p(np.array([k.value for k in kept], dtype=float))
        sd = x.std()
        if sd > 0:
            z = np.abs((x - x.mean()) / sd)
            mask = z <= clean_cfg.label_z_clip
            n_drop = int((~mask).sum())
            if n_drop:
                rep.dropped["label_outlier"] += n_drop
                kept = [k for k, m in zip(kept, mask) if m]

    rep.n_out = len(kept)
    return kept, rep
