from __future__ import annotations

from dataclasses import dataclass

from celery import Celery
from omegaconf import DictConfig


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "null"):
        return None
    return text


@dataclass(frozen=True)
class CeleryBrokerConfig:
    broker_url: str
    queue: str
    task_name: str
    routing_key: str


def resolve_broker_config(cfg: DictConfig) -> CeleryBrokerConfig:
    broker_url = _as_optional_str(getattr(cfg.jobs.broker, "url", None))
    if broker_url is None:
        raise ValueError(
            "jobs.dispatch=broker requires jobs.broker.url "
            "(or JOBS_BROKER_URL env var)."
        )

    queue = _as_optional_str(getattr(cfg.jobs.broker, "queue", None))
    if queue is None:
        raise ValueError("jobs.broker.queue must be set for broker mode.")

    task_name = _as_optional_str(getattr(cfg.jobs.broker, "task_name", None))
    if task_name is None:
        raise ValueError("jobs.broker.task_name must be set for broker mode.")

    routing_key = _as_optional_str(getattr(cfg.jobs.broker, "routing_key", None))
    if routing_key is None:
        routing_key = queue

    return CeleryBrokerConfig(
        broker_url=broker_url,
        queue=queue,
        task_name=task_name,
        routing_key=routing_key,
    )


def build_celery_app(name: str, broker_cfg: CeleryBrokerConfig) -> Celery:
    app = Celery(name, broker=broker_cfg.broker_url)
    app.conf.update(
        task_default_queue=broker_cfg.queue,
        task_default_exchange=broker_cfg.queue,
        task_default_exchange_type="direct",
        task_default_routing_key=broker_cfg.routing_key,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
    )
    return app


def build_celery_app_from_cfg(cfg: DictConfig) -> tuple[Celery, CeleryBrokerConfig]:
    broker_cfg = resolve_broker_config(cfg)
    app = build_celery_app("medical_rag_reranker", broker_cfg)
    return app, broker_cfg
