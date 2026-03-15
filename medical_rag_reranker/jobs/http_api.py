from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from pydantic import BaseModel
import uvicorn

from medical_rag_reranker.jobs.bootstrap import JobRuntime, build_job_runtime


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _read_cfg_host_port(cfg: DictConfig) -> tuple[str, int]:
    api_cfg = getattr(cfg.jobs, "api", None)
    host = str(getattr(api_cfg, "host", "0.0.0.0")).strip() or "0.0.0.0"
    port = int(getattr(api_cfg, "port", 8080))
    return host, port


@dataclass(frozen=True)
class JobsApiContext:
    runtime: JobRuntime
    auto_dispatch: bool


class SubmitJobPayload(BaseModel):
    question: str
    metadata: dict[str, Any] | None = None


def create_jobs_api(cfg: DictConfig) -> FastAPI:
    runtime = build_job_runtime(cfg)
    auto_dispatch = _as_bool(getattr(cfg.jobs, "auto_dispatch", True), default=True)
    context = JobsApiContext(runtime=runtime, auto_dispatch=auto_dispatch)

    app = FastAPI(
        title="Medical RAG Jobs API",
        version="1.0.0",
        debug=False,
    )
    app.state.context = context

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs")
    def submit_job(payload: SubmitJobPayload):
        question = str(payload.question).strip()
        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "`question` must be a non-empty string."},
            )

        service = context.runtime.service
        dispatcher = context.runtime.dispatcher
        job = service.create_job(question=question, metadata=payload.metadata)

        dispatched = False
        if context.auto_dispatch:
            try:
                dispatcher.dispatch(job.job_id)
                dispatched = True
            except Exception as exc:
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": f"Failed to dispatch job: {exc}",
                        "job_id": job.job_id,
                    },
                )

        current = service.get_job(job.job_id) or job
        response = current.to_dict()
        response["result"] = service.get_result(job.job_id)
        response["dispatched"] = dispatched

        status_code = 202 if dispatched else 201
        return JSONResponse(status_code=status_code, content=response)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = context.runtime.service.get_job(job_id)
        if job is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Job not found", "job_id": job_id},
            )
        return job.to_dict()

    @app.get("/jobs/{job_id}/result")
    def get_result(job_id: str):
        service = context.runtime.service
        result = service.get_result(job_id)
        if result is not None:
            return result

        job = service.get_job(job_id)
        if job is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Job not found", "job_id": job_id},
            )

        return JSONResponse(
            status_code=409,
            content={
                "error": "Result is not ready yet.",
                "job_id": job_id,
                "status": job.status.value,
            },
        )

    return app


def serve_jobs_api(
    cfg: DictConfig,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    cfg_host, cfg_port = _read_cfg_host_port(cfg)
    bind_host = str(host).strip() if host is not None else cfg_host
    bind_port = int(port) if port is not None else cfg_port

    app = create_jobs_api(cfg)
    print(f"Jobs API listening on http://{bind_host}:{bind_port}")
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="info")
