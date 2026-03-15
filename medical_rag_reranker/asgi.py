from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from medical_rag_reranker.jobs.http_api import create_jobs_api
from medical_rag_reranker.utils.hydra_cfg import load_cfg


@lru_cache(maxsize=1)
def build_app() -> FastAPI:
    """Build the production ASGI app once per process."""
    cfg = load_cfg(config_name="config")
    return create_jobs_api(cfg)


app = build_app()
