from __future__ import annotations

from medical_rag_reranker.jobs.bootstrap import build_job_service
from medical_rag_reranker.jobs.celery_app import build_celery_app_from_cfg
from medical_rag_reranker.utils.hydra_cfg import load_cfg

_cfg = load_cfg(config_name="config")
celery_app, _broker_cfg = build_celery_app_from_cfg(_cfg)


@celery_app.task(name=_broker_cfg.task_name)
def process_inference_job(job_id: str) -> None:
    """Celery consumer task that executes a single inference job."""
    cfg = load_cfg(config_name="config")
    service = build_job_service(cfg)
    service.process(job_id)
