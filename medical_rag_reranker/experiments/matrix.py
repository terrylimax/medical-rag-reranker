from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, ListConfig, OmegaConf

from medical_rag_reranker.artifacts.sync import pull_artifacts, push_artifacts


PRIMARY_RETRIEVAL_METRICS = (
    "Hit@1",
    "Hit@3",
    "Hit@5",
    "MRR@10",
    "latency_p50_ms",
    "latency_p95_ms",
)
SECONDARY_RETRIEVAL_METRICS = (
    "NDCG@10",
    "gold_retention@10",
    "graph_expansion_size",
    "graph_expansion_latency_ms",
)
E2E_COLUMNS = (
    "run_id",
    "retrieval_job_id",
    "generation_job_id",
    "method",
    "retrieval_backend",
    "retrieval_params",
    "retrieval_run_path",
    "qdrant_collection",
    "reranker_enabled",
    "generation_top_k",
    "generation_retrieve_top_k",
    "generation_run_name",
    "generation_remote_concurrency",
    "judge_mode",
    "generator_model",
    "judge_model",
    "hit@1",
    "hit@3",
    "hit@5",
    "mrr@10",
    "ndcg@10",
    "gold_retention@10",
    "graph_expansion_size",
    "graph_expansion_latency_ms",
    "retrieval_latency_p50_ms",
    "retrieval_latency_p95_ms",
    "answer_pass_rate",
    "avg_faithfulness",
    "avg_relevance",
    "avg_completeness",
    "avg_safety",
    "citation_support_rate",
    "unsupported_citation_rate",
    "generation_latency_p50_ms",
    "e2e_latency_p50_ms",
    "e2e_latency_p95_ms",
    "best_by_hit1",
    "best_by_judge_pass_rate",
    "best_overall",
    "best_overall_score",
)
SECRET_FRAGMENTS = ("api_key", "secret", "password", "token", "credential")
RETRIEVAL_PARAM_KEYS = (
    "retrieval.alpha",
    "retrieval.cand_k",
    "retrieval.rrf_k",
    "retrieval.seed_k",
    "retrieval.expand_k",
    "retrieval.max_hops",
    "retrieval.hop_decay",
    "retrieval.base_weight",
    "retrieval.graph_weight",
    "retrieval.num_queries",
    "retrieval.include_original",
)


@dataclass(frozen=True)
class MatrixJob:
    job_id: str
    method: str
    stage: str
    overrides: tuple[str, ...]
    output_dir: Path
    status_path: Path


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _format_ks_override(value: object) -> str:
    if isinstance(value, (list, tuple, ListConfig)):
        values = [str(int(v)) for v in value]
    else:
        values = [part.strip() for part in str(value).split(",") if part.strip()]
    if not values:
        raise ValueError("run.experiment_matrix.metrics.eval_ks is empty")
    return f"run.eval_retrieval.ks=[{','.join(values)}]"


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return text


def _as_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_list(value: object) -> list[Any]:
    value = _as_plain(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_env(keys: Iterable[str]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in keys:
        value = os.getenv(str(key))
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in SECRET_FRAGMENTS):
            safe[str(key)] = bool(value)
        else:
            safe[str(key)] = value or ""
    return safe


def _job_hash(parts: Iterable[object]) -> str:
    payload = json.dumps(list(parts), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _override_dict(
    status_or_overrides: dict[str, Any] | Iterable[str],
) -> dict[str, str]:
    if isinstance(status_or_overrides, dict):
        overrides = status_or_overrides.get("overrides", [])
    else:
        overrides = status_or_overrides

    values: dict[str, str] = {}
    for override in overrides:
        text = str(override)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key] = value
    return values


def _retrieval_params(
    status_or_overrides: dict[str, Any] | Iterable[str],
) -> dict[str, str]:
    values = _override_dict(status_or_overrides)
    return {
        key.removeprefix("retrieval."): values[key]
        for key in RETRIEVAL_PARAM_KEYS
        if key in values
    }


def _retrieval_overrides(status: dict[str, Any]) -> list[str]:
    return [
        str(override)
        for override in status.get("overrides", [])
        if not str(override).startswith("generation.")
        and not str(override).startswith("run.eval_generation.")
    ]


def _retrieval_key(status_or_overrides: dict[str, Any] | Iterable[str]) -> str:
    values = _override_dict(status_or_overrides)
    return values.get("run.eval_retrieval.out_run", "")


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _positive_float(value: Any, *, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed <= 0:
        return float(default)
    return float(parsed)


def _generation_remote_concurrency(
    status: dict[str, Any],
    generation: dict[str, Any] | None = None,
) -> float:
    if "generation_remote_concurrency" in status:
        return _positive_float(status.get("generation_remote_concurrency"))

    override_value = _override_value(status, "generation.remote_concurrency")
    if override_value:
        return _positive_float(override_value)

    if generation is not None and "generation_remote_concurrency" in generation:
        return _positive_float(generation.get("generation_remote_concurrency"))

    return 1.0


def _normalize_generation_latency(
    value: float | None,
    *,
    concurrency: float,
) -> float | None:
    if value is None:
        return None
    return float(value) / _positive_float(concurrency)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(E2E_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in E2E_COLUMNS})


def _command_args(command: str, overrides: list[str]) -> list[str]:
    args = [sys.executable, "-m", "medical_rag_reranker.commands", command]
    if overrides:
        args.extend(["--overrides", json.dumps(overrides, ensure_ascii=False)])
    return args


def _run_command(
    command: str,
    overrides: list[str],
    *,
    cwd: Path,
    dry_run: bool,
) -> dict[str, Any]:
    args = _command_args(command, overrides)
    if dry_run:
        return {"command": args, "dry_run": True}

    subprocess.run(args, cwd=str(cwd), check=True)
    return {"command": args, "dry_run": False}


def _request_json(
    url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 180.0,
    label: str = "request",
) -> dict[str, Any]:
    headers = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Preflight `{label}` failed: GET {url} returned {exc.code}. {body}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Preflight `{label}` timed out after {float(timeout_seconds):.0f}s: "
            f"GET {url}. If this is RunPod Serverless, the worker may be cold "
            "starting. Increase EXPERIMENT_PREFLIGHT_TIMEOUT_SECONDS or warm the "
            "endpoint with curl."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Preflight `{label}` failed: GET {url}. {exc}") from exc


def _checked_request_json(
    checks: dict[str, Any],
    output_dir: Path,
    *,
    name: str,
    url: str,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        return _request_json(
            url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            label=name,
        )
    except Exception as exc:
        checks["checks"][name] = {"ok": False, "url": url, "error": str(exc)}
        _write_json(output_dir / "preflight.json", checks)
        raise


def _preflight(
    cfg: DictConfig,
    *,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    run_cfg = cfg.run.experiment_matrix
    remote_cfg = run_cfg.remote
    checks: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dry_run": bool(dry_run),
        "env": _safe_env(
            [
                "ARTIFACT_REMOTE_URI",
                "ARTIFACT_DVC_REMOTE",
                "AWS_REGION",
                "AWS_DEFAULT_REGION",
                "QDRANT_URL",
                "QDRANT_API_KEY",
                "QDRANT_COLLECTION",
                "VLLM_BASE_URL",
                "VLLM_API_KEY",
                "GENERATION_LLM_MODEL_NAME",
                "LLM_JUDGE_BASE_URL",
                "LLM_JUDGE_API_KEY",
                "LLM_JUDGE_MODEL",
            ]
        ),
        "checks": {},
    }

    pull_result = pull_artifacts(
        remote_uri=_as_optional_str(remote_cfg.artifact_remote_uri),
        local_root=".",
        remote_name=str(remote_cfg.artifact_dvc_remote),
        dry_run=dry_run,
    )
    checks["checks"]["artifact_pull"] = {
        "ok": True,
        "dvc_remote": pull_result.get("dvc_remote"),
        "dry_run": bool(dry_run),
    }

    if not dry_run:
        qdrant_url = (_as_optional_str(remote_cfg.qdrant_url) or "").rstrip("/")
        if qdrant_url:
            _checked_request_json(
                checks,
                output_dir,
                name="qdrant",
                url=f"{qdrant_url}/collections",
                api_key=_as_optional_str(remote_cfg.qdrant_api_key),
                timeout_seconds=float(remote_cfg.timeout_seconds),
            )
            checks["checks"]["qdrant"] = {"ok": True, "url": qdrant_url}

        generator_url = (_as_optional_str(remote_cfg.vllm_base_url) or "").rstrip("/")
        if generator_url:
            payload = _checked_request_json(
                checks,
                output_dir,
                name="vllm_generator",
                url=f"{generator_url}/models",
                api_key=_as_optional_str(remote_cfg.vllm_api_key),
                timeout_seconds=float(remote_cfg.timeout_seconds),
            )
            checks["checks"]["vllm_generator"] = {
                "ok": True,
                "models": [row.get("id") for row in payload.get("data", [])],
            }

        judge_url = (_as_optional_str(remote_cfg.llm_judge_base_url) or "").rstrip("/")
        if judge_url:
            payload = _checked_request_json(
                checks,
                output_dir,
                name="llm_judge",
                url=f"{judge_url}/models",
                api_key=_as_optional_str(remote_cfg.llm_judge_api_key),
                timeout_seconds=float(remote_cfg.timeout_seconds),
            )
            checks["checks"]["llm_judge"] = {
                "ok": True,
                "models": [row.get("id") for row in payload.get("data", [])],
            }
    else:
        checks["checks"]["qdrant"] = {"ok": True, "dry_run": True}
        checks["checks"]["vllm_generator"] = {"ok": True, "dry_run": True}
        checks["checks"]["llm_judge"] = {"ok": True, "dry_run": True}

    _write_json(output_dir / "preflight.json", checks)
    return checks


def _profile_cfg(cfg: DictConfig, profile: str) -> dict[str, Any]:
    profiles = OmegaConf.to_container(cfg.run.experiment_matrix.profiles, resolve=True)
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"Unknown experiment_matrix profile: {profile}")
    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise ValueError(f"Bad experiment profile: {profile}")
    return selected


def _method_cfg(cfg: DictConfig, method: str) -> dict[str, Any]:
    methods = OmegaConf.to_container(cfg.run.experiment_matrix.methods, resolve=True)
    if not isinstance(methods, dict) or method not in methods:
        raise ValueError(f"Unknown experiment method: {method}")
    selected = methods[method]
    if not isinstance(selected, dict):
        raise ValueError(f"Bad experiment method: {method}")
    return selected


def _expand_retrieval_overrides(
    cfg: DictConfig,
    *,
    method: str,
    output_dir: Path,
    profile: dict[str, Any],
) -> list[list[str]]:
    method_cfg = _method_cfg(cfg, method)
    base = [str(item) for item in _as_list(method_cfg.get("overrides"))]
    retrieval_name = str(method_cfg.get("retrieval", method))
    retrieval_kind = _retrieval_kind(retrieval_name, method)
    run_name = str(method_cfg.get("run_name", method))

    sweep_rows: list[list[str]] = [[]]
    sweeps = profile.get("retrieval_sweeps", {})
    if retrieval_kind == "hybrid":
        sweep_rows = []
        for alpha in sweeps.get("hybrid_alpha", [None]):
            for cand_k in sweeps.get("hybrid_cand_k", [None]):
                for rrf_k in sweeps.get("hybrid_rrf_k", [None]):
                    row = []
                    if alpha is not None:
                        row.append(f"retrieval.alpha={alpha}")
                        row.append(f"run.retrieval_index.alpha={alpha}")
                    if cand_k is not None:
                        row.append(f"retrieval.cand_k={cand_k}")
                        row.append(f"run.retrieval_index.cand_k={cand_k}")
                    if rrf_k is not None:
                        row.append(f"retrieval.rrf_k={rrf_k}")
                        row.append(f"run.retrieval_index.rrf_k={rrf_k}")
                    sweep_rows.append(row)
    elif retrieval_kind == "graph":
        sweep_rows = []
        for seed_k in sweeps.get("graph_seed_k", [None]):
            for expand_k in sweeps.get("graph_expand_k", [None]):
                for max_hops in sweeps.get("graph_max_hops", [None]):
                    for hop_decay in sweeps.get("graph_hop_decay", [None]):
                        for weights in sweeps.get(
                            "graph_base_graph_weights", [[None, None]]
                        ):
                            row = []
                            if seed_k is not None:
                                row.append(f"retrieval.seed_k={seed_k}")
                            if expand_k is not None:
                                row.append(f"retrieval.expand_k={expand_k}")
                            if max_hops is not None:
                                row.append(f"retrieval.max_hops={max_hops}")
                            if hop_decay is not None:
                                row.append(f"retrieval.hop_decay={hop_decay}")
                            if weights[0] is not None:
                                row.append(f"retrieval.base_weight={weights[0]}")
                                row.append(f"retrieval.graph_weight={weights[1]}")
                            sweep_rows.append(row)
    elif retrieval_kind == "rag_fusion":
        sweep_rows = []
        for num_queries in sweeps.get("rag_fusion_num_queries", [None]):
            for rrf_k in sweeps.get("rag_fusion_rrf_k", [None]):
                for include_original in sweeps.get(
                    "rag_fusion_include_original", [None]
                ):
                    row = []
                    if num_queries is not None:
                        row.append(f"retrieval.num_queries={num_queries}")
                    if rrf_k is not None:
                        row.append(f"retrieval.rrf_k={rrf_k}")
                    if include_original is not None:
                        row.append(f"retrieval.include_original={include_original}")
                    sweep_rows.append(row)

    expanded: list[list[str]] = []
    for row in sweep_rows:
        variant_id = _job_hash(["retrieval-sweep", method, retrieval_name, base, row])
        variant_run_name = f"{run_name}_{variant_id}"
        run_path = output_dir / "runs" / method / f"{variant_id}.trec"
        index_path = _index_path_for(
            output_dir=output_dir,
            method=method,
            retrieval_name=retrieval_name,
            retrieval_kind=retrieval_kind,
            base_retriever=str(method_cfg.get("base_retriever", "")),
            variant_id=variant_id,
        )

        common = [
            f"retrieval={retrieval_name}",
            f"retrieval_run.out={run_path}",
            f"retrieval_run.run_name={variant_run_name}",
            f"retrieval_run.index={index_path}",
            f"run.retrieval_index.out={index_path}",
            f"run.eval_retrieval.out_run={run_path}",
            f"run.eval_retrieval.run_name={variant_run_name}",
            f"run.eval_retrieval.retriever={method}",
            f"run.eval_retrieval.run_tag={cfg.run.experiment_matrix.run_id}",
            _format_ks_override(cfg.run.experiment_matrix.metrics.eval_ks),
            f"generation.index={index_path}",
        ]
        expanded.append(base + common + row)

    return expanded


def _index_path_for(
    *,
    output_dir: Path,
    method: str,
    retrieval_name: str,
    retrieval_kind: str,
    base_retriever: str = "",
    variant_id: str | None = None,
) -> Path:
    index_dir = output_dir / "indices" / method
    if variant_id:
        index_dir = index_dir / variant_id
    if retrieval_kind == "bm25":
        return index_dir / "bm25_index.json.gz"
    if retrieval_kind in {"dense", "bi_encoder"}:
        return index_dir / "dense_index.pkl"
    if retrieval_kind == "hybrid":
        return index_dir / "hybrid_index.json"
    if retrieval_kind == "graph":
        return index_dir / "graph_index.json"
    if retrieval_kind == "rag_fusion":
        if "bm25" in retrieval_name or base_retriever == "bm25":
            return index_dir / "bm25_index.json.gz"
        if "hybrid" in retrieval_name or base_retriever == "hybrid":
            return index_dir / "hybrid_index.json"
        return index_dir / "dense_index.pkl"
    return index_dir / "index.json"


def _retrieval_kind(retrieval_name: str, method: str) -> str:
    name = str(retrieval_name)
    method_name = str(method)
    if name == "bm25":
        return "bm25"
    if name in {"dense", "qdrant"}:
        return "dense"
    if name in {"medcpt", "bi_encoder"} or "medcpt" in method_name:
        if method_name.startswith("hybrid") or name.startswith("hybrid"):
            return "hybrid"
        if method_name.startswith("graph") or name.startswith("graph"):
            return "graph"
        return "bi_encoder"
    if name.startswith("hybrid") or method_name.startswith("hybrid"):
        return "hybrid"
    if name.startswith("graph") or method_name.startswith("graph"):
        return "graph"
    if name.startswith("rag_fusion") or method_name.startswith("rag_fusion"):
        return "rag_fusion"
    return name


def _expand_generation_overrides(
    cfg: DictConfig,
    *,
    method: str,
    output_dir: Path,
    profile: dict[str, Any],
    retrieval_overrides: list[str],
) -> list[list[str]]:
    sweeps = profile.get("generation_sweeps", {})
    top_ks = sweeps.get("top_k", [int(cfg.generation.top_k)])
    retrieve_top_ks = sweeps.get("retrieve_top_k", [int(cfg.generation.retrieve_top_k)])
    reranker_values = sweeps.get("use_reranker", [False])
    judge_modes = sweeps.get("judge_mode", ["heuristic"])
    examples_limit = int(
        sweeps.get("examples_limit", cfg.run.eval_generation.examples_limit)
    )
    rows: list[list[str]] = []
    retrieval_job_id = _job_hash(["retrieval", method, retrieval_overrides])

    for top_k in top_ks:
        for retrieve_top_k in retrieve_top_ks:
            for use_reranker in reranker_values:
                for judge_mode in judge_modes:
                    suffix = (
                        f"{method}_{retrieval_job_id}_top{top_k}_ret{retrieve_top_k}_"
                        f"rerank{str(use_reranker).lower()}_{judge_mode}"
                    )
                    rows.append(
                        retrieval_overrides
                        + [
                            "generation.backend=openai_compatible",
                            f"generation.top_k={top_k}",
                            f"generation.retrieve_top_k={retrieve_top_k}",
                            f"generation.use_reranker={str(use_reranker).lower()}",
                            f"run.eval_generation.examples_limit={examples_limit}",
                            f"run.eval_generation.judge_mode={judge_mode}",
                            f"run.eval_generation.run_name={suffix}",
                            f"run.eval_generation.run_tag={cfg.run.experiment_matrix.run_id}",
                            f"run.eval_generation.output_jsonl={output_dir / 'generation' / (suffix + '.jsonl')}",
                            f"run.eval_generation.summary_json={output_dir / 'generation' / (suffix + '.summary.json')}",
                            f"run.eval_generation.output_report={output_dir / 'generation' / (suffix + '.md')}",
                            f"run.eval_generation.generation_results_jsonl_path={output_dir / 'generation' / (suffix + '.raw.jsonl')}",
                            f"run.eval_generation.generation_report_path={output_dir / 'generation' / (suffix + '.raw.md')}",
                        ]
                    )
    return rows


def _build_jobs(
    cfg: DictConfig,
    *,
    profile: str,
    output_dir: Path,
    stages: set[str],
) -> list[MatrixJob]:
    selected_profile = _profile_cfg(cfg, profile)
    methods = [str(item) for item in selected_profile.get("methods", [])]
    jobs: list[MatrixJob] = []

    for method in methods:
        retrieval_rows = _expand_retrieval_overrides(
            cfg, method=method, output_dir=output_dir, profile=selected_profile
        )
        for retrieval_overrides in retrieval_rows:
            retrieval_hash = _job_hash(["retrieval", method, retrieval_overrides])
            retrieval_job_dir = output_dir / "jobs" / retrieval_hash
            if "index" in stages:
                jobs.append(
                    MatrixJob(
                        job_id=retrieval_hash,
                        method=method,
                        stage="index",
                        overrides=tuple(retrieval_overrides),
                        output_dir=retrieval_job_dir,
                        status_path=retrieval_job_dir / "index.status.json",
                    )
                )
            if "retrieval" in stages:
                jobs.append(
                    MatrixJob(
                        job_id=retrieval_hash,
                        method=method,
                        stage="retrieval",
                        overrides=tuple(retrieval_overrides),
                        output_dir=retrieval_job_dir,
                        status_path=retrieval_job_dir / "retrieval.status.json",
                    )
                )
            if "generation" in stages:
                for generation_overrides in _expand_generation_overrides(
                    cfg,
                    method=method,
                    output_dir=output_dir,
                    profile=selected_profile,
                    retrieval_overrides=retrieval_overrides,
                ):
                    generation_hash = _job_hash(
                        ["generation", method, generation_overrides]
                    )
                    generation_job_dir = output_dir / "jobs" / generation_hash
                    jobs.append(
                        MatrixJob(
                            job_id=generation_hash,
                            method=method,
                            stage="generation",
                            overrides=tuple(generation_overrides),
                            output_dir=generation_job_dir,
                            status_path=generation_job_dir / "generation.status.json",
                        )
                    )
    return jobs


def _stage_set(stage: str) -> set[str]:
    normalized = str(stage).strip().lower()
    if normalized == "all":
        return {"preflight", "train", "index", "retrieval", "generation", "summary"}
    if normalized == "retrieval":
        return {"retrieval", "summary"}
    if normalized == "generation":
        return {"generation", "summary"}
    return {normalized}


def _run_training_stage(
    cfg: DictConfig,
    *,
    training_mode: str,
    output_dir: Path,
    cwd: Path,
    dry_run: bool,
) -> dict[str, Any]:
    status = {
        "training_mode": training_mode,
        "dry_run": bool(dry_run),
        "commands": [],
    }
    if training_mode == "colab_artifacts":
        status["status"] = "skipped_colab_artifacts_expected"
        _write_json(output_dir / "training.status.json", status)
        return status
    if training_mode != "local":
        raise ValueError(
            "training_mode must be `colab_artifacts` or `local`, "
            f"got {training_mode!r}."
        )

    local_overrides = [
        "train.limit_train_batches=1.0",
        "train.limit_val_batches=1.0",
        f"run.train_retriever.output_dir={output_dir / 'models' / 'retriever'}",
    ]
    for command in ("prep_retriever_training_data", "train_retriever", "train"):
        status["commands"].append(
            _run_command(command, local_overrides, cwd=cwd, dry_run=dry_run)
        )
    status["status"] = "completed" if not dry_run else "planned"
    _write_json(output_dir / "training.status.json", status)
    return status


def _run_matrix_job(
    job: MatrixJob,
    *,
    cwd: Path,
    dry_run: bool,
    resume: bool,
) -> dict[str, Any]:
    if resume and job.status_path.exists():
        current = _read_json(job.status_path)
        if current.get("status") == "completed":
            return current | {"resumed": True}

    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = {
        "index": "index",
        "retrieval": "eval_retrieval",
        "generation": "eval_generation",
    }[job.stage]
    status: dict[str, Any] = {
        "job_id": job.job_id,
        "method": job.method,
        "stage": job.stage,
        "overrides": list(job.overrides),
        "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "planned" if dry_run else "running",
    }
    if job.stage == "generation":
        overrides = _override_dict(job.overrides)
        status["generation_remote_concurrency"] = _positive_float(
            overrides.get(
                "generation.remote_concurrency",
                os.getenv("GENERATION_REMOTE_CONCURRENCY", "1"),
            )
        )
    _write_json(job.status_path, status)
    try:
        status["command"] = _run_command(
            command, list(job.overrides), cwd=cwd, dry_run=dry_run
        )
        status["status"] = "planned" if dry_run else "completed"
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(job.status_path, status)
        raise
    status["finished_at"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    _write_json(job.status_path, status)
    return status


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_retrieval_metrics(output_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for status_path in (output_dir / "jobs").glob("*/retrieval.status.json"):
        status = _read_json(status_path)
        if status.get("status") != "completed":
            continue
        key = _retrieval_key(status)
        run_path = Path(key) if key else None
        metrics = {}
        if run_path is not None:
            metrics = _read_json(
                run_path.with_suffix(run_path.suffix + ".metrics.json")
            )
        rows[key or str(status_path)] = {"status": status, "metrics": metrics}
    return rows


def _load_generation_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status_path in (output_dir / "jobs").glob("*/generation.status.json"):
        status = _read_json(status_path)
        if status.get("status") != "completed":
            continue
        summary_path = None
        output_jsonl = None
        for override in status.get("overrides", []):
            text = str(override)
            if text.startswith("run.eval_generation.summary_json="):
                summary_path = Path(text.split("=", 1)[1])
            elif text.startswith("run.eval_generation.output_jsonl="):
                output_jsonl = Path(text.split("=", 1)[1])
        summary = _read_json(summary_path) if summary_path is not None else {}
        raw_rows = _read_jsonl(output_jsonl) if output_jsonl is not None else []
        rows.append({"status": status, "summary": summary, "raw_rows": raw_rows})
    return rows


def _summarize_latencies(rows: list[dict[str, Any]], key: str) -> tuple[Any, Any]:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]
    return _percentile(values, 0.50), _percentile(values, 0.95)


def _best_flags(rows: list[dict[str, Any]]) -> None:
    hit_values = [
        row.get("hit@1") for row in rows if isinstance(row.get("hit@1"), float)
    ]
    pass_values = [
        row.get("answer_pass_rate")
        for row in rows
        if isinstance(row.get("answer_pass_rate"), float)
    ]
    best_hit = max(hit_values) if hit_values else None
    best_pass = max(pass_values) if pass_values else None

    best_score = None
    for row in rows:
        hit = row.get("hit@1")
        passed = row.get("answer_pass_rate")
        if isinstance(hit, float) and isinstance(passed, float):
            score = 0.5 * hit + 0.5 * passed
        elif isinstance(hit, float):
            score = hit
        elif isinstance(passed, float):
            score = passed
        else:
            score = 0.0
        row["best_overall_score"] = float(score)
        best_score = score if best_score is None else max(best_score, score)

    for row in rows:
        row["best_by_hit1"] = bool(
            best_hit is not None and row.get("hit@1") == best_hit
        )
        row["best_by_judge_pass_rate"] = bool(
            best_pass is not None and row.get("answer_pass_rate") == best_pass
        )
        row["best_overall"] = bool(
            best_score is not None and row.get("best_overall_score") == best_score
        )


def _build_summary(
    cfg: DictConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    retrieval_by_method = _load_retrieval_metrics(output_dir)
    generation_rows = _load_generation_rows(output_dir)
    e2e_rows: list[dict[str, Any]] = []

    if generation_rows:
        for item in generation_rows:
            status = item["status"]
            method = str(status.get("method"))
            retrieval = retrieval_by_method.get(_retrieval_key(status), {}).get(
                "metrics", {}
            )
            summary = item["summary"]
            raw_rows = item["raw_rows"]
            generation_p50, _generation_p95 = _summarize_latencies(
                raw_rows, "generation_latency_ms"
            )
            e2e_p50, e2e_p95 = _summarize_latencies(raw_rows, "end_to_end_latency_ms")
            generation_remote_concurrency = _generation_remote_concurrency(
                status, summary
            )
            row = _summary_row(
                cfg=cfg,
                method=method,
                status=status,
                retrieval=retrieval,
                generation=summary,
                generation_latency_p50=_normalize_generation_latency(
                    generation_p50, concurrency=generation_remote_concurrency
                ),
                e2e_latency_p50=_normalize_generation_latency(
                    e2e_p50, concurrency=generation_remote_concurrency
                ),
                e2e_latency_p95=_normalize_generation_latency(
                    e2e_p95, concurrency=generation_remote_concurrency
                ),
            )
            e2e_rows.append(row)
    else:
        for item in retrieval_by_method.values():
            method = str(item["status"].get("method"))
            e2e_rows.append(
                _summary_row(
                    cfg=cfg,
                    method=method,
                    status=item["status"],
                    retrieval=item["metrics"],
                    generation={},
                    generation_latency_p50=None,
                    e2e_latency_p50=None,
                    e2e_latency_p95=None,
                )
            )

    _best_flags(e2e_rows)
    _write_csv(output_dir / "e2e_summary.csv", e2e_rows)
    _write_jsonl(output_dir / "e2e_summary.jsonl", e2e_rows)

    best_configs = {
        "best_by_hit1": [row for row in e2e_rows if row.get("best_by_hit1")],
        "best_by_judge_pass_rate": [
            row for row in e2e_rows if row.get("best_by_judge_pass_rate")
        ],
        "best_overall": [row for row in e2e_rows if row.get("best_overall")],
    }
    _write_json(output_dir / "best_configs.json", best_configs)
    _write_markdown_report(output_dir / "comparison_report.md", e2e_rows)
    return {
        "e2e_summary_csv": str(output_dir / "e2e_summary.csv"),
        "e2e_summary_jsonl": str(output_dir / "e2e_summary.jsonl"),
        "num_rows": len(e2e_rows),
        "primary_metrics": list(PRIMARY_RETRIEVAL_METRICS),
        "secondary_metrics": list(SECONDARY_RETRIEVAL_METRICS),
    }


def _override_value(status: dict[str, Any], key: str) -> str:
    prefix = f"{key}="
    for override in status.get("overrides", []):
        text = str(override)
        if text.startswith(prefix):
            return text.split("=", 1)[1]
    return ""


def _summary_row(
    *,
    cfg: DictConfig,
    method: str,
    status: dict[str, Any],
    retrieval: dict[str, Any],
    generation: dict[str, Any],
    generation_latency_p50: float | None,
    e2e_latency_p50: float | None,
    e2e_latency_p95: float | None,
) -> dict[str, Any]:
    retrieval_overrides = _retrieval_overrides(status)
    retrieval_job_id = (
        str(status.get("job_id") or "")
        if status.get("stage") == "retrieval"
        else _job_hash(["retrieval", method, retrieval_overrides])
    )
    generation_job_id = (
        str(status.get("job_id") or "") if status.get("stage") == "generation" else ""
    )
    retrieval_params = _retrieval_params(status)
    return {
        "run_id": str(cfg.run.experiment_matrix.run_id),
        "retrieval_job_id": retrieval_job_id,
        "generation_job_id": generation_job_id,
        "method": method,
        "retrieval_backend": _override_value(status, "retrieval")
        or str(cfg.retrieval.name),
        "retrieval_params": json.dumps(
            retrieval_params, ensure_ascii=False, sort_keys=True
        ),
        "retrieval_run_path": _retrieval_key(status),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION", ""),
        "reranker_enabled": _override_value(status, "generation.use_reranker"),
        "generation_top_k": _override_value(status, "generation.top_k"),
        "generation_retrieve_top_k": _override_value(
            status, "generation.retrieve_top_k"
        ),
        "generation_run_name": _override_value(status, "run.eval_generation.run_name"),
        "generation_remote_concurrency": _generation_remote_concurrency(
            status, generation
        ),
        "judge_mode": _override_value(status, "run.eval_generation.judge_mode"),
        "generator_model": str(cfg.generation.llm_model_name),
        "judge_model": str(cfg.run.eval_generation.judge_model),
        "hit@1": _metric(retrieval, "Hit@1"),
        "hit@3": _metric(retrieval, "Hit@3"),
        "hit@5": _metric(retrieval, "Hit@5"),
        "mrr@10": _metric(retrieval, "MRR@10"),
        "ndcg@10": _metric(retrieval, "NDCG@10"),
        "gold_retention@10": _metric(retrieval, "gold_retention@10"),
        "graph_expansion_size": _metric(retrieval, "graph_expansion_size"),
        "graph_expansion_latency_ms": _metric(retrieval, "graph_expansion_latency_ms"),
        "retrieval_latency_p50_ms": _metric(retrieval, "latency_p50_ms"),
        "retrieval_latency_p95_ms": _metric(retrieval, "latency_p95_ms"),
        "answer_pass_rate": _metric(generation, "pass_rate"),
        "avg_faithfulness": _metric(generation, "avg_faithfulness"),
        "avg_relevance": _metric(generation, "avg_relevance"),
        "avg_completeness": _metric(generation, "avg_completeness"),
        "avg_safety": _metric(generation, "avg_safety"),
        "citation_support_rate": _metric(generation, "avg_supported_citation_rate"),
        "unsupported_citation_rate": _metric(
            generation, "avg_unsupported_citation_rate"
        ),
        "generation_latency_p50_ms": generation_latency_p50,
        "e2e_latency_p50_ms": e2e_latency_p50,
        "e2e_latency_p95_ms": e2e_latency_p95,
    }


def _write_markdown_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Comparison",
        "",
        "Primary retrieval metrics: Hit@1, Hit@3, Hit@5, MRR@10, latency p50/p95.",
        "P@10 and R@10 are intentionally excluded from this summary.",
        "",
        "| method | Hit@1 | Hit@3 | Hit@5 | MRR@10 | pass_rate | e2e p50 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {hit1} | {hit3} | {hit5} | {mrr} | {passed} | {e2e} |".format(
                method=row.get("method", ""),
                hit1=_fmt(row.get("hit@1")),
                hit3=_fmt(row.get("hit@3")),
                hit5=_fmt(row.get("hit@5")),
                mrr=_fmt(row.get("mrr@10")),
                passed=_fmt(row.get("answer_pass_rate")),
                e2e=_fmt(row.get("e2e_latency_p50_ms")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return ""


def run_experiment_matrix(
    cfg: DictConfig,
    *,
    profile: str = "smoke",
    stage: str = "all",
    run_id: str | None = None,
    resume: bool = True,
    dry_run: bool = False,
    training_mode: str = "colab_artifacts",
) -> dict[str, Any]:
    """Run or plan the configured experiment matrix."""
    repo_root = Path.cwd().resolve()
    run_cfg = cfg.run.experiment_matrix
    effective_run_id = run_id or str(run_cfg.run_id)
    cfg.run.experiment_matrix.run_id = effective_run_id
    output_dir = Path(str(run_cfg.output_root)) / effective_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = _stage_set(stage)
    result: dict[str, Any] = {
        "run_id": effective_run_id,
        "profile": profile,
        "stage": stage,
        "training_mode": training_mode,
        "dry_run": bool(dry_run),
        "output_dir": str(output_dir),
        "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

    manifest = {
        "format": "medical-rag-reranker.experiment-matrix",
        "version": 1,
        "run_id": effective_run_id,
        "profile": profile,
        "stage": stage,
        "training_mode": training_mode,
        "dry_run": bool(dry_run),
        "created_at": result["started_at"],
        "primary_retrieval_metrics": list(PRIMARY_RETRIEVAL_METRICS),
        "secondary_retrieval_metrics": list(SECONDARY_RETRIEVAL_METRICS),
        "env": _safe_env(
            [
                "ARTIFACT_DVC_REMOTE",
                "ARTIFACT_REMOTE_URI",
                "QDRANT_URL",
                "QDRANT_API_KEY",
                "QDRANT_COLLECTION",
                "VLLM_BASE_URL",
                "VLLM_API_KEY",
                "GENERATION_LLM_MODEL_NAME",
                "LLM_JUDGE_BASE_URL",
                "LLM_JUDGE_API_KEY",
                "LLM_JUDGE_MODEL",
            ]
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)

    if "preflight" in stages:
        result["preflight"] = _preflight(cfg, output_dir=output_dir, dry_run=dry_run)
    if "train" in stages:
        result["training"] = _run_training_stage(
            cfg,
            training_mode=training_mode,
            output_dir=output_dir,
            cwd=repo_root,
            dry_run=dry_run,
        )

    jobs = _build_jobs(cfg, profile=profile, output_dir=output_dir, stages=stages)
    result["num_jobs"] = len(jobs)
    planned_rows = [
        {
            "job_id": job.job_id,
            "stage": job.stage,
            "method": job.method,
            "overrides": list(job.overrides),
            "status_path": str(job.status_path),
        }
        for job in jobs
    ]
    _write_jsonl(output_dir / "planned_jobs.jsonl", planned_rows)

    statuses = []
    for job in jobs:
        statuses.append(
            _run_matrix_job(job, cwd=repo_root, dry_run=dry_run, resume=resume)
        )
    result["jobs"] = statuses

    if "summary" in stages:
        result["summary"] = _build_summary(cfg, output_dir=output_dir)

    if not dry_run and bool(run_cfg.push_outputs):
        result["artifact_push"] = push_artifacts(
            local_root=".",
            include=f"{output_dir.as_posix()}/**/*",
            remote_name=str(run_cfg.remote.artifact_dvc_remote),
        )

    result["finished_at"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    _write_json(output_dir / "experiment_matrix.result.json", result)
    return result
