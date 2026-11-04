"""Persistence: write cleaned records to Postgres, read them back, update splits."""
from __future__ import annotations

from sqlalchemy import func, select, update

from .db import Dataset, Sequence, make_session_factory
from .records import RawRecord


def write_dataset(engine, *, name: str, source: str, element_type: str, unit: str,
                  records: list[RawRecord], clean_report: dict,
                  chunk: int = 5000, replace: bool = True) -> int:
    """Insert a dataset row and its sequences. Returns the dataset id.

    If `replace` and a dataset with the same name exists, it is deleted first
    (cascade), so re-ingesting is idempotent.
    """
    Session = make_session_factory(engine)
    with Session() as s:
        if replace:
            existing = s.scalar(select(Dataset).where(Dataset.name == name))
            if existing is not None:
                s.delete(existing)
                s.commit()

        ds = Dataset(name=name, source=source, element_type=element_type,
                     unit=unit, n_records=len(records), clean_report=clean_report)
        s.add(ds)
        s.flush()                      # assign ds.id
        ds_id = ds.id

        rows = []
        for r in records:
            row = r.as_row()
            row["dataset_id"] = ds_id
            rows.append(row)
        for i in range(0, len(rows), chunk):
            s.bulk_insert_mappings(Sequence, rows[i:i + chunk])
        s.commit()
    return ds_id


def read_sequences(engine, *, element_type: str | None = None,
                   split: str | None = None, source: str | None = None) -> list[dict]:
    """Read sequences as plain dicts (id, sequence, value, value_sd, split, ...)."""
    Session = make_session_factory(engine)
    cols = (Sequence.id, Sequence.source, Sequence.element_type, Sequence.seq_id,
            Sequence.sequence, Sequence.value, Sequence.value_sd, Sequence.unit,
            Sequence.cluster_id, Sequence.split)
    stmt = select(*cols)
    if element_type is not None:
        stmt = stmt.where(Sequence.element_type == element_type)
    if split is not None:
        stmt = stmt.where(Sequence.split == split)
    if source is not None:
        stmt = stmt.where(Sequence.source == source)
    with Session() as s:
        return [dict(row._mapping) for row in s.execute(stmt)]


def update_cluster_and_split(engine, updates: list[dict]) -> None:
    """Bulk-apply {'id', 'cluster_id', 'split'} updates to sequences."""
    Session = make_session_factory(engine)
    with Session() as s:
        s.bulk_update_mappings(Sequence.__mapper__, updates)
        s.commit()


def element_counts(engine) -> dict:
    """Per element_type and split, how many rows (for QC reporting)."""
    Session = make_session_factory(engine)
    stmt = select(Sequence.element_type, Sequence.split, func.count()).group_by(
        Sequence.element_type, Sequence.split)
    with Session() as s:
        out: dict = {}
        for et, split, n in s.execute(stmt):
            out.setdefault(et, {})[split or "unassigned"] = int(n)
        return out
