"""Command-line interface for the data pipeline (Phase 1).

    seqparam init-db
    seqparam ingest --cambray path/to/MOESM58.zip --lafleur path/to/MOESM5.xlsx
    seqparam splits --element promoter
    seqparam stats

The DB DSN comes from the config or the SEQPARAM_DB_URL env var.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .data.db import drop_db, init_db, make_engine
from .data.ingest import ingest_cambray, ingest_lafleur, make_splits, run_phase1
from .data.store import element_counts


def _engine(args):
    cfg = load_config(args.config)
    return cfg, make_engine(cfg.db.url, echo=cfg.db.echo)


def cmd_init_db(args):
    cfg, engine = _engine(args)
    if args.drop:
        drop_db(engine)
    init_db(engine)
    print("schema ready at", cfg.db.url)


def cmd_ingest(args):
    cfg, engine = _engine(args)
    if args.all:
        summary = run_phase1(cfg, engine, cambray_path=args.cambray,
                             lafleur_path=args.lafleur,
                             cambray_limit=args.cambray_limit,
                             include_fig3a=args.include_fig3a)
    else:
        init_db(engine)
        summary = {"clean": {}}
        if args.cambray:
            summary["clean"]["cambray"] = ingest_cambray(
                cfg, engine, args.cambray, limit=args.cambray_limit)
        if args.lafleur:
            summary["clean"]["lafleur"] = ingest_lafleur(
                cfg, engine, args.lafleur, include_fig3a=args.include_fig3a)
        if not args.skip_splits:
            summary["splits"] = {et: make_splits(cfg, engine, et)
                                 for et in ("rbs", "promoter")}
    print(json.dumps(summary, indent=2))


def cmd_splits(args):
    cfg, engine = _engine(args)
    print(json.dumps(make_splits(cfg, engine, args.element), indent=2))


def cmd_stats(args):
    cfg, engine = _engine(args)
    print(json.dumps(element_counts(engine), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="seqparam", description="Stage-1 data pipeline")
    p.add_argument("--config", default=None, help="path to YAML config")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init-db", help="create tables")
    pi.add_argument("--drop", action="store_true", help="drop existing tables first")
    pi.set_defaults(func=cmd_init_db)

    pg = sub.add_parser("ingest", help="load, clean, store (and split)")
    pg.add_argument("--cambray", help="Cambray MOESM58 .zip or extracted .csv")
    pg.add_argument("--lafleur", help="La Fleur MOESM5 .xlsx")
    pg.add_argument("--all", action="store_true", help="full run_phase1 (both + splits)")
    pg.add_argument("--cambray-limit", type=int, default=None, help="cap rows (debug)")
    pg.add_argument("--include-fig3a", action="store_true",
                    help="also load the log-ratio promoter sheet")
    pg.add_argument("--skip-splits", action="store_true")
    pg.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("splits", help="recompute similarity split for one element")
    ps.add_argument("--element", choices=("promoter", "rbs"), required=True)
    ps.set_defaults(func=cmd_splits)

    pst = sub.add_parser("stats", help="row counts per element/split")
    pst.set_defaults(func=cmd_stats)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
