"""Shared fixtures: ephemeral Postgres + tiny in-memory stand-ins for the
real supplementary files, so no test depends on the 300 MB downloads.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import zipfile

import pandas as pd
import pytest
from sqlalchemy import URL

# make `src/` importable without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pytest_postgresql import factories  # noqa: E402

from seqparam.data.db import init_db, make_engine  # noqa: E402

# Point pytest-postgresql at the local pg_ctl (Debian/Ubuntu path fallback).
_PG_CTL = shutil.which("pg_ctl") or "/usr/lib/postgresql/16/bin/pg_ctl"
postgresql_proc = factories.postgresql_proc(executable=_PG_CTL)
postgresql = factories.postgresql("postgresql_proc")


@pytest.fixture
def engine(postgresql):
    """A SQLAlchemy engine bound to a fresh ephemeral Postgres, schema created."""
    info = postgresql.info
    url = URL.create(
        "postgresql+psycopg2",
        username=info.user,
        password=info.password or "",
        host=info.host,
        port=info.port,
        database=info.dbname,
    )
    eng = make_engine(str(url))
    init_db(eng)
    yield eng
    eng.dispose()


def _seq(prefix: str, n: int) -> str:
    """Deterministic 96-ish bp sequence for fixtures (pads/truncates to n)."""
    base = (prefix + "ACGT" * 30)
    return base[:n].ljust(n, "A")


@pytest.fixture
def cambray_zip(tmp_path) -> pathlib.Path:
    """A tiny Cambray-shaped CSV zipped under the real inner filename."""
    rows = [
        {"gs.sequence": _seq("AAAA", 96), "clean.lin.prot.mean": 40.0,
         "clean.lin.prot.var": 4.0, "group": 1},
        {"gs.sequence": _seq("CCCC", 96), "clean.lin.prot.mean": 70.0,
         "clean.lin.prot.var": 9.0, "group": 2},
        {"gs.sequence": _seq("GGGG", 96), "clean.lin.prot.mean": float("nan"),
         "clean.lin.prot.var": 1.0, "group": 1},   # dropped: NaN value
        {"gs.sequence": _seq("AAAA", 96), "clean.lin.prot.mean": 55.0,
         "clean.lin.prot.var": 2.0, "group": 3},   # dropped: duplicate of row 0
        {"gs.sequence": _seq("TTTT", 96), "clean.lin.prot.mean": 250.0,
         "clean.lin.prot.var": 1.0, "group": 1},   # dropped: value > 100
    ]
    csv = tmp_path / "Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    zpath = tmp_path / "cambray.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(csv, arcname=csv.name)
    return zpath


@pytest.fixture
def lafleur_xlsx(tmp_path) -> pathlib.Path:
    """A tiny La Fleur-shaped workbook with the three promoter sheets."""
    path = tmp_path / "lafleur.xlsx"
    fig1d = pd.DataFrame([
        {"UP": "ACGTACGTACGTACGT", "h35": "TTGACA",
         "spacs": "CTATTTCCTATTTCTCT", "h10": "TATAAT",
         "Txave [au]": 2.5, "Txstddev [au]": 0.3},
        {"UP": "GGGGCCCCGGGGCCCC", "h35": "TTGACC",
         "spacs": "CTATTTCCTATTTCTCA", "h10": "TACAAT",
         "Txave [au]": 1.1, "Txstddev [au]": 0.1},
    ])
    urtecho = pd.DataFrame([
        {"Promoter Sequence": _seq("ACGT", 50), "Observed TX [au]": 5.0},
        {"Promoter Sequence": _seq("TGCA", 50), "Observed TX [au]": float("nan")},  # dropped
    ])
    hossain = pd.DataFrame([
        {"Upstream DNA": "ACGTACGTAC", "Promoter Sequence": _seq("GG", 30),
         "Downstream DNA": "TTTTGGGG", "Observed TX [au]": 3.3},
    ])
    with pd.ExcelWriter(path) as xl:
        fig1d.to_excel(xl, sheet_name="La Fleur et al (Fig 1d, S1)", index=False)
        urtecho.to_excel(xl, sheet_name="Urtecho et al (Fig 3c, S7b)", index=False)
        hossain.to_excel(xl, sheet_name="Hossain et al (Fig 3d, S7d)", index=False)
    return path
