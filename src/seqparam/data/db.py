"""SQLAlchemy 2.0 ORM models and engine/session helpers.

Two tables:
  datasets  - one row per ingestion run (provenance: source, counts, QC report)
  sequences - one row per cleaned measurement, FK to its dataset

`meta` and `clean_report` use JSON (JSONB on Postgres).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Index, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    source: Mapped[str] = mapped_column(String(64))
    element_type: Mapped[str] = mapped_column(String(16))
    unit: Mapped[str] = mapped_column(String(32))
    n_records: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    clean_report: Mapped[dict] = mapped_column(JSON, default=dict)

    sequences: Mapped[list["Sequence"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan")


class Sequence(Base):
    __tablename__ = "sequences"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(64))
    element_type: Mapped[str] = mapped_column(String(16))
    seq_id: Mapped[str] = mapped_column(String(128))
    sequence: Mapped[str] = mapped_column(Text)
    length: Mapped[int] = mapped_column(Integer)
    value: Mapped[float] = mapped_column(Float)
    value_sd: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str] = mapped_column(String(32))
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    split: Mapped[str | None] = mapped_column(String(8), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    dataset: Mapped[Dataset] = relationship(back_populates="sequences")

    __table_args__ = (
        Index("ix_seq_element_split", "element_type", "split"),
        Index("ix_seq_dataset", "dataset_id"),
    )


def make_engine(url: str, echo: bool = False):
    return create_engine(url, echo=echo, future=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def drop_db(engine) -> None:
    Base.metadata.drop_all(engine)
