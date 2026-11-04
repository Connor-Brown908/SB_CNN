.PHONY: install test db-up db-down ingest train-rbs train-promoter

install:
	pip install -e ".[dev]"

test:
	pytest

db-up:
	docker compose up -d db

db-down:
	docker compose down

# Usage: make ingest CAMBRAY=path/to/MOESM58.zip LAFLEUR=path/to/MOESM5.xlsx
ingest:
	seqparam --config configs/default.yaml ingest --all --cambray "$(CAMBRAY)" --lafleur "$(LAFLEUR)"

train-rbs:
	seqparam --config configs/default.yaml train --element rbs

train-promoter:
	seqparam --config configs/default.yaml train --element promoter
