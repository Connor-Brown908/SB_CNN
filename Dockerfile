# CPU training/inference image. For GPU, swap the base for an nvidia/cuda image
# and install a CUDA torch wheel.
FROM python:3.11-slim

WORKDIR /app

# psycopg2-binary ships wheels, so no libpq build deps are needed.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[dev]"

# DSN is injected at runtime (compose/CI set SEQPARAM_DB_URL).
ENV WANDB_MODE=offline
ENTRYPOINT ["seqparam"]
CMD ["--help"]
