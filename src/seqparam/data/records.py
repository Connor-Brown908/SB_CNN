"""The canonical record every loader emits.

One row = one measured regulatory sequence. Loaders translate each source's
idiosyncratic columns into this shape; everything downstream (cleaning, storage,
splitting, datasets) speaks only `RawRecord`. Values are kept in the source's
native units (tagged by `unit`); harmonization/normalization is a modeling
decision made later, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawRecord:
    source: str              # e.g. "cambray2018", "lafleur2022", "urtecho2019"
    element_type: str        # "promoter" | "rbs"
    seq_id: str              # stable id within the source
    sequence: str            # DNA, will be upper-cased in cleaning
    value: float             # measured target, native units
    unit: str                # e.g. "PNI_1_100", "TX_au", "log_TX_ratio"
    value_sd: float | None = None   # per-measurement spread, if the source reports it
    meta: dict = field(default_factory=dict)  # extra provenance (series, group, ...)

    def as_row(self) -> dict:
        """Flat dict for bulk DB insert (meta kept as a JSON-able dict)."""
        return {
            "source": self.source,
            "element_type": self.element_type,
            "seq_id": self.seq_id,
            "sequence": self.sequence,
            "length": len(self.sequence),
            "value": float(self.value),
            "value_sd": None if self.value_sd is None else float(self.value_sd),
            "unit": self.unit,
            "meta": dict(self.meta),
        }
