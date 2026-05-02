from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def progress_enabled() -> bool:
    value = os.getenv("MEDICAL_RAG_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
) -> Iterable[T]:
    """Wrap an iterable with tqdm when available.

    The fallback keeps commands usable in minimal environments and tests.
    """
    if not progress_enabled():
        return iterable

    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable

    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


@contextmanager
def timed_stage(name: str) -> Iterator[None]:
    started = time.monotonic()
    print(f"{name}: started")
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        print(f"{name}: done in {elapsed:.1f}s")


def count_text_lines(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def path_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for item in p.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total
