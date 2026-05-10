from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


REGISTRY_FORMAT = "medical-rag-reranker.artifact-registry"
DEFAULT_REGISTRY_PATH = "artifacts/index_registry.json"
DEFAULT_DVC_REMOTE = "artifact_s3"

DEFAULT_INCLUDE_PATTERNS = (
    "data/processed/corpus.jsonl",
    "data/processed/eval_queries.jsonl",
    "data/processed/qrels.tsv",
    "data/processed/splits.json",
    "data/processed/medquad_graph.json",
    "artifacts/*.json",
    "artifacts/*.json.gz",
    "artifacts/*.pkl",
    "artifacts/hybrid*/**/*",
    "artifacts/graph*/**/*",
    "artifacts/retriever/**/*",
)

DEFAULT_EXCLUDE_PATTERNS = (
    ".git/**",
    "**/__pycache__/**",
    "reports/**",
    "runs/**",
    "artifacts/retriever_training/**",
)


def _split_patterns(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _norm_rel(path: Path) -> str:
    return path.as_posix().lstrip("/")


def _matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(
    args: list[str],
    *,
    cwd: Path,
    dry_run: bool = False,
) -> list[str]:
    if dry_run:
        return args
    try:
        subprocess.run(args, cwd=str(cwd), check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "DVC is required for artifact sync but was not found."
        ) from exc
    except subprocess.CalledProcessError as exc:
        hint = ""
        if any(str(part).startswith("s3://") for part in args):
            hint = (
                " If this is an S3 remote, install the DVC S3 plugin first: "
                "`poetry add dvc-s3` or `pip install dvc-s3`."
            )
        raise RuntimeError(f"DVC command failed: {' '.join(args)}.{hint}") from exc
    return args


def collect_artifact_files(
    local_root: str | Path,
    *,
    include_patterns: Iterable[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Iterable[str] = DEFAULT_EXCLUDE_PATTERNS,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> list[Path]:
    root = Path(local_root).resolve()
    files: set[Path] = set()
    registry_rel = _norm_rel(Path(registry_path))

    for pattern in include_patterns:
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            rel = _norm_rel(candidate.relative_to(root))
            if rel == registry_rel:
                continue
            if _matches_any(rel, exclude_patterns):
                continue
            files.add(candidate)

    return sorted(files, key=lambda p: _norm_rel(p.relative_to(root)))


def build_registry(
    local_root: str | Path,
    files: Iterable[Path],
    *,
    remote_uri: str,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    root = Path(local_root).resolve()
    rows = []
    for path in files:
        rel = _norm_rel(path.resolve().relative_to(root))
        rows.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    return {
        "format": REGISTRY_FORMAT,
        "version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "remote_uri": remote_uri,
        "registry_path": _norm_rel(Path(registry_path)),
        "files": rows,
    }


def compact_dvc_targets(
    files: Iterable[Path],
    *,
    local_root: str | Path,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> list[str]:
    root = Path(local_root).resolve()
    registry_rel = _norm_rel(Path(registry_path))
    targets: set[str] = set()

    for path in files:
        rel = _norm_rel(path.resolve().relative_to(root))
        if rel == registry_rel:
            continue
        parts = rel.split("/")
        if rel.startswith("data/processed/"):
            targets.add("data/processed")
        elif len(parts) >= 3 and parts[0] == "artifacts":
            if parts[1].startswith("hybrid") or parts[1].startswith("graph"):
                targets.add("/".join(parts[:2]))
            elif parts[1] == "retriever":
                targets.add("artifacts/retriever")
            else:
                targets.add(rel)
        else:
            targets.add(rel)

    return sorted(targets)


def configure_dvc_remote(
    *,
    remote_uri: str,
    local_root: str | Path,
    remote_name: str = DEFAULT_DVC_REMOTE,
    region: str | None = None,
    endpoint_url: str | None = None,
    dry_run: bool = False,
) -> list[list[str]]:
    root = Path(local_root).resolve()
    commands: list[list[str]] = []

    add_cmd = ["dvc", "remote", "add", "--force", remote_name, remote_uri]
    commands.append(_run(add_cmd, cwd=root, dry_run=dry_run))

    effective_region = (
        region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    )
    if effective_region:
        cmd = ["dvc", "remote", "modify", remote_name, "region", effective_region]
        commands.append(_run(cmd, cwd=root, dry_run=dry_run))

    effective_endpoint = endpoint_url or os.getenv("ARTIFACT_S3_ENDPOINT_URL")
    if effective_endpoint:
        cmd = [
            "dvc",
            "remote",
            "modify",
            remote_name,
            "endpointurl",
            effective_endpoint,
        ]
        commands.append(_run(cmd, cwd=root, dry_run=dry_run))

    return commands


def _remote_uri_or_env(remote_uri: str | None) -> str:
    value = remote_uri or os.getenv("ARTIFACT_REMOTE_URI")
    if not value:
        raise ValueError(
            "Artifact remote URI is required. Pass `remote_uri` or set "
            "ARTIFACT_REMOTE_URI, for example s3://bucket/medical-rag/medquad-v1."
        )
    return value


def push_artifacts(
    *,
    remote_uri: str | None = None,
    local_root: str | Path | None = None,
    registry_path: str | None = None,
    include: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    dry_run: bool = False,
    region: str | None = None,
    endpoint_url: str | None = None,
    remote_name: str | None = None,
) -> dict[str, Any]:
    remote = _remote_uri_or_env(remote_uri)
    root = Path(local_root or os.getenv("ARTIFACT_LOCAL_ROOT") or ".").resolve()
    registry_rel = registry_path or os.getenv(
        "ARTIFACT_REGISTRY_PATH", DEFAULT_REGISTRY_PATH
    )
    dvc_remote = remote_name or os.getenv("ARTIFACT_DVC_REMOTE", DEFAULT_DVC_REMOTE)
    include_patterns = _split_patterns(include) or DEFAULT_INCLUDE_PATTERNS
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + _split_patterns(exclude)

    files = collect_artifact_files(
        root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        registry_path=registry_rel,
    )
    registry = build_registry(
        root,
        files,
        remote_uri=remote,
        registry_path=registry_rel,
    )

    registry_local_path = root / registry_rel
    if not dry_run:
        registry_local_path.parent.mkdir(parents=True, exist_ok=True)
        registry_local_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    targets = compact_dvc_targets(files, local_root=root, registry_path=registry_rel)
    commands: list[list[str]] = []
    commands.extend(
        configure_dvc_remote(
            remote_uri=remote,
            local_root=root,
            remote_name=dvc_remote,
            region=region,
            endpoint_url=endpoint_url,
            dry_run=dry_run,
        )
    )
    if targets:
        commands.append(_run(["dvc", "add", *targets], cwd=root, dry_run=dry_run))
    commands.append(_run(["dvc", "push", "-r", dvc_remote], cwd=root, dry_run=dry_run))

    registry["dvc_remote"] = dvc_remote
    registry["dvc_targets"] = targets
    if dry_run:
        registry["dry_run"] = True
        registry["dvc_commands"] = commands
    return registry


def pull_artifacts(
    *,
    remote_uri: str | None = None,
    local_root: str | Path | None = None,
    registry_path: str | None = None,
    region: str | None = None,
    endpoint_url: str | None = None,
    overwrite: bool = True,
    remote_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    remote = remote_uri or os.getenv("ARTIFACT_REMOTE_URI")
    root = Path(local_root or os.getenv("ARTIFACT_LOCAL_ROOT") or ".").resolve()
    registry_rel = registry_path or os.getenv(
        "ARTIFACT_REGISTRY_PATH", DEFAULT_REGISTRY_PATH
    )
    dvc_remote = remote_name or os.getenv("ARTIFACT_DVC_REMOTE", DEFAULT_DVC_REMOTE)

    commands: list[list[str]] = []
    if remote:
        commands.extend(
            configure_dvc_remote(
                remote_uri=remote,
                local_root=root,
                remote_name=dvc_remote,
                region=region,
                endpoint_url=endpoint_url,
                dry_run=dry_run,
            )
        )
    pull_cmd = ["dvc", "pull", "-r", dvc_remote]
    if not overwrite:
        pull_cmd.append("--no-run-cache")
    commands.append(_run(pull_cmd, cwd=root, dry_run=dry_run))

    registry_file = root / registry_rel
    registry: dict[str, Any] = {
        "format": REGISTRY_FORMAT,
        "version": 1,
        "remote_uri": remote or "",
        "registry_path": registry_rel,
        "dvc_remote": dvc_remote,
        "dvc_commands": commands,
    }
    if dry_run:
        registry["dry_run"] = True
    if registry_file.exists():
        loaded = json.loads(registry_file.read_text(encoding="utf-8"))
        if loaded.get("format") == REGISTRY_FORMAT:
            registry.update(loaded)
            registry["dvc_commands"] = commands
            if dry_run:
                registry["dry_run"] = True
    return registry
