"""Typed configuration: a pydantic schema plus a YAML loader.

The whole pipeline is driven by one config object so a run is fully described
(and reproducible) by a single file. `load_config` reads YAML, applies the
SEQPARAM_DB_URL environment override, and validates.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ElementClean(BaseModel):
    """Cleaning / QC thresholds for one element type."""
    min_len: int = Field(8, ge=1)
    max_len: int = Field(400, ge=1)
    value_min: float | None = None          # drop labels below this (native units)
    value_max: float | None = None          # drop labels above this
    label_z_clip: float = Field(6.0, ge=0)  # drop label outliers (log-space SDs); 0 disables
    max_rel_sd: float | None = None          # drop measurements with sd/|value| above this


class DataConfig(BaseModel):
    # model input widths (pad/truncate target); promoters vary, Cambray RBS = 96 bp
    promoter_seq_len: int = Field(150, ge=8)
    rbs_seq_len: int = Field(96, ge=8)

    promoter_clean: ElementClean = ElementClean(min_len=20, max_len=400,
                                                value_min=0.0, label_z_clip=6.0)
    rbs_clean: ElementClean = ElementClean(min_len=96, max_len=96,
                                           value_min=1.0, value_max=100.0, label_z_clip=0.0)

    # similarity-based splitting (per element type)
    promoter_kmer: int = Field(8, ge=2)
    rbs_kmer: int = Field(12, ge=2)
    similarity_threshold: float = Field(0.5, gt=0, le=1,
                                        description="Jaccard >= threshold links two sequences")
    bucket_cap: int = Field(200, ge=2,
                            description="skip k-mer buckets larger than this when pairing")
    val_fraction: float = Field(0.15, gt=0, lt=1)
    test_fraction: float = Field(0.15, gt=0, lt=1)
    split_seed: int = 1234

    def clean_for(self, element_type: str) -> ElementClean:
        return self.rbs_clean if element_type == "rbs" else self.promoter_clean

    def kmer_for(self, element_type: str) -> int:
        return self.rbs_kmer if element_type == "rbs" else self.promoter_kmer

    def seq_len_for(self, element_type: str) -> int:
        return self.rbs_seq_len if element_type == "rbs" else self.promoter_seq_len


class ModelConfig(BaseModel):
    channels: tuple[int, ...] = (64, 64)
    kernel_sizes: tuple[int, ...] = (12, 8)
    pool_size: int = 16
    mlp_hidden: int = 128
    dropout: float = Field(0.3, ge=0.0, lt=1.0)
    heteroscedastic: bool = True


class OptimConfig(BaseModel):
    epochs: int = Field(40, ge=1)
    batch_size: int = Field(256, ge=1)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    grad_clip: float = Field(5.0, ge=0)
    early_stop_patience: int = Field(8, ge=1)
    num_workers: int = 0
    seed: int = 0


class MonitorConfig(BaseModel):
    backend: Literal["wandb", "csv"] = "wandb"
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    project: str = "seqparam"
    run_name: str | None = None
    log_dir: str = "runs"


class DBConfig(BaseModel):
    url: str = "postgresql+psycopg2://seqparam:seqparam@localhost:5432/seqparam"
    echo: bool = False
    insert_chunk: int = Field(5000, ge=1)


class PathsConfig(BaseModel):
    artifacts: str = "artifacts"


class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    monitor: MonitorConfig = MonitorConfig()
    db: DBConfig = DBConfig()
    paths: PathsConfig = PathsConfig()

    def ensure_dirs(self) -> None:
        for p in (self.paths.artifacts, self.monitor.log_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None, **overrides) -> Config:
    """Load config from YAML (optional), apply SEQPARAM_DB_URL env override, validate."""
    raw: dict = {}
    if path is not None:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    cfg = Config(**raw)

    env_url = os.environ.get("SEQPARAM_DB_URL")
    if env_url:
        cfg = cfg.model_copy(update={"db": cfg.db.model_copy(update={"url": env_url})})
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg
