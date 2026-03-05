FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root
RUN pip install "psycopg[binary]>=3.1,<4.0" "celery>=5.6,<6.0"

# Copy project sources and install the package itself.
COPY configs ./configs
COPY medical_rag_reranker ./medical_rag_reranker
COPY .dvc ./.dvc
COPY .dvcignore ./.dvcignore
# Do not call Poetry twice with system-site installs:
# project deps (notably via dvc) can change `dulwich` and break Poetry runtime.
# Install the local package with pip using already-installed deps.
RUN pip install --no-deps .

RUN mkdir -p /app/data /app/artifacts /app/runs /app/reports /app/.cache/huggingface

CMD ["python", "-m", "medical_rag_reranker.commands"]
