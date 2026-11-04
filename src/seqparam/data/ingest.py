"""Ingestion orchestration: load -> clean -> store -> split.

`run_phase1` is the end-to-end entry: it initializes the schema, ingests Cambray
(RBS) and La Fleur (promoter), then computes a similarity-based split per element
type. Everything is idempotent (re-ingesting replaces a source's dataset by name).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import Config
from .clean import clean
from .db import init_db
from .loaders import load_cambray, load_lafleur
from .records import RawRecord
from .split import assign_splits, cluster
from .store import (element_counts, read_sequences, update_cluster_and_split,
                    write_dataset)


def _ingest_source(cfg: Config, engine, *, name: str, source: str,
                   element_type: str, unit: str, records: list[RawRecord]) -> dict:
    kept, rep = clean(records, cfg.data.clean_for(element_type), source=source)
    write_dataset(engine, name=name, source=source, element_type=element_type,
                  unit=unit, records=kept, clean_report=rep.as_dict(),
                  chunk=cfg.db.insert_chunk)
    return rep.as_dict()


def ingest_cambray(cfg: Config, engine, zip_or_csv: str | Path,
                   limit: int | None = None) -> dict:
    recs = load_cambray(zip_or_csv, limit=limit)
    return _ingest_source(cfg, engine, name="cambray2018_rbs", source="cambray2018",
                          element_type="rbs", unit="PNI_1_100", records=recs)


def ingest_lafleur(cfg: Config, engine, xlsx: str | Path,
                   include_fig3a: bool = False) -> list[dict]:
    recs = load_lafleur(xlsx, include_fig3a=include_fig3a)
    by_source: dict[str, list[RawRecord]] = defaultdict(list)
    for r in recs:
        by_source[r.source].append(r)
    reports = []
    for source, rs in by_source.items():
        unit = rs[0].unit
        reports.append(_ingest_source(
            cfg, engine, name=f"{source}_promoter", source=source,
            element_type="promoter", unit=unit, records=rs))
    return reports


def make_splits(cfg: Config, engine, element_type: str) -> dict:
    """Cluster all sequences of an element type and assign train/val/test."""
    rows = read_sequences(engine, element_type=element_type)
    if not rows:
        return {"element_type": element_type, "n": 0}
    seqs = [r["sequence"] for r in rows]
    cids = cluster(seqs, k=cfg.data.kmer_for(element_type),
                   threshold=cfg.data.similarity_threshold,
                   bucket_cap=cfg.data.bucket_cap)
    splits = assign_splits(cids, cfg.data.val_fraction, cfg.data.test_fraction,
                           seed=cfg.data.split_seed)
    updates = [{"id": r["id"], "cluster_id": int(c), "split": str(sp)}
               for r, c, sp in zip(rows, cids, splits)]
    update_cluster_and_split(engine, updates)
    uniq, counts = np.unique(splits, return_counts=True)
    return {"element_type": element_type, "n": len(rows),
            "n_clusters": int(len(np.unique(cids))),
            "splits": dict(zip(uniq.tolist(), counts.tolist()))}


def run_phase1(cfg: Config, engine, *, cambray_path: str | Path,
               lafleur_path: str | Path, cambray_limit: int | None = None,
               include_fig3a: bool = False) -> dict:
    init_db(engine)
    summary: dict = {"clean": {}, "splits": {}}
    summary["clean"]["cambray"] = ingest_cambray(cfg, engine, cambray_path, limit=cambray_limit)
    summary["clean"]["lafleur"] = ingest_lafleur(cfg, engine, lafleur_path,
                                                 include_fig3a=include_fig3a)
    summary["splits"]["rbs"] = make_splits(cfg, engine, "rbs")
    summary["splits"]["promoter"] = make_splits(cfg, engine, "promoter")
    summary["counts"] = element_counts(engine)
    return summary
