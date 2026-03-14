from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import inspect

from medical_rag_reranker.jobs.repositories.postgres import build_postgres_engine

LEGACY_BOOTSTRAP_REVISION = "20260314_0001"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_alembic_config() -> AlembicConfig:
    repo_root = _repo_root()
    alembic_ini = repo_root / "alembic.ini"
    script_location = repo_root / "alembic"

    if not alembic_ini.exists():
        raise FileNotFoundError(f"Alembic config file not found: {alembic_ini}")
    if not script_location.exists():
        raise FileNotFoundError(
            f"Alembic script directory not found: {script_location}"
        )

    cfg = AlembicConfig(str(alembic_ini))
    cfg.set_main_option("script_location", str(script_location))
    return cfg


def upgrade_jobs_schema(dsn: str, revision: str = "head") -> None:
    engine = build_postgres_engine(dsn)
    alembic_cfg = _build_alembic_config()

    with engine.begin() as connection:
        alembic_cfg.attributes["connection"] = connection
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if "alembic_version" not in tables and {
            "inference_jobs",
            "inference_results",
        }.issubset(tables):
            print(
                "Detected legacy jobs schema without alembic versioning; "
                f"stamping revision {LEGACY_BOOTSTRAP_REVISION}."
            )
            command.stamp(alembic_cfg, LEGACY_BOOTSTRAP_REVISION)
        command.upgrade(alembic_cfg, revision)
