# SynBioModel

Stage-1 CNNs that predict *E. coli* circuit parameters from DNA sequence, with
calibrated uncertainty, for a downstream mechanistic (ODE/SSA) circuit simulator:

- **promoter strength / transcription rate** (`alpha1`) — La Fleur et al. 2022 (bundles La Fleur, Urtecho, Hossain)
- **RBS / translation rate** (`alpha2`) — Cambray et al. 2018 (244k synthetic sequences, PNI readout)

Two models, one shared pipeline. Each model emits a mean **and** a variance per
sequence, so the simulator can propagate parameter uncertainty.

## Pipeline

```
raw supplementary files
  └─ loaders ──► clean/QC ──► Postgres ──► similarity split
                                              └─ dataset bridge (per-source z-score, fit on train)
                                                    └─ SeqCNN (mean + log-var head)
                                                          └─ train (NLL, early stop, W&B) ──► checkpoint (params.json + weights.pt)
```

## Layout

```
configs/default.yaml      pydantic-validated run config
src/seqparam/
  config.py               typed config + loader
  data/
    records.py            canonical RawRecord
    encode.py             sequence <-> one-hot
    clean.py              QC / cleaning
    normalize.py          per-source log z-score (fit on train only)
    db.py  store.py       Postgres models + persistence
    split.py              k-mer-Jaccard similarity clustering + split
    dataset.py            torch Dataset + build_dataloaders
    ingest.py             load -> clean -> store -> split
    loaders/              cambray.py, lafleur.py
  model/
    cnn.py                SeqCNN (mean + log-variance head)
    params.py             ModelParams + checkpoint save/load
  train/
    losses.py             Gaussian NLL / MSE
    metrics.py            RMSE, MAE, R^2, Pearson, Spearman, NLL, coverage, sharpness
    monitor.py            W&B / CSV monitors + early stopping
    loop.py               train / evaluate
  cli.py                  `seqparam` CLI
tests/                    pytest (ephemeral Postgres via pytest-postgresql)
docker-compose.yml        local Postgres     Dockerfile / Makefile
```

## Getting the data

Not committed (large, publisher terms). Download and point the CLI at them:

- Cambray Supplementary Data 15 (`...MOESM58_ESM.zip`) — 244k sequences, PNI
- La Fleur Supplementary Data 5 (`...MOESM5_ESM.xlsx`) — promoter transcription rates

## Quickstart

```bash
pip install -e ".[dev]"
docker compose up -d db
export SEQPARAM_DB_URL=postgresql+psycopg2://seqparam:seqparam@localhost:5432/seqparam

# 1. ingest + clean + store + split
seqparam --config configs/default.yaml ingest --all \
    --cambray path/to/MOESM58_ESM.zip \
    --lafleur path/to/MOESM5_ESM.xlsx
seqparam --config configs/default.yaml stats

# 2. train each model (checkpoints land in artifacts/<element>/)
seqparam --config configs/default.yaml train --element rbs
seqparam --config configs/default.yaml train --element promoter

# 3. score a checkpoint on the held-out test split (native units)
seqparam --config configs/default.yaml evaluate --element rbs --ckpt artifacts/rbs
```

Weights & Biases logs **offline** by default (no account/network). Set
`monitor.wandb_mode: online` in the config, or `--monitor csv`, to change that.

## Tests

```bash
pytest     # spins up an ephemeral Postgres; model tests skip if torch is absent
```

## Notes

- GPU is used automatically if available; otherwise CPU.
- Metrics are reported in **native units** (predictions inverted per source), so
  RMSE / R^2 are directly comparable to the source papers.
