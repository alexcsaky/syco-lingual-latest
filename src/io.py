"""File I/O utilities for the SycoLingual pipeline.

Handles atomic JSONL writes (safe for concurrent asyncio writers),
resume scanning, and typed JSONL loading.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TypeVar, Type

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonlWriter:
    """Append-only JSONL writer with asyncio lock for concurrent safety."""

    def __init__(self, path: str):
        self._path = path
        self._lock = asyncio.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    async def write(self, record: BaseModel) -> None:
        line = record.model_dump_json() + "\n"
        async with self._lock:
            self._file.write(line)
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def load_completed_keys(
    path: str,
    key_fields: list[str],
) -> set[tuple]:
    """Scan a JSONL file and return the set of key tuples already present.

    Used for resumability: on restart, skip any records whose keys are
    already in the output file.

    Silently skips corrupt/incomplete lines (e.g. from a crash mid-write).
    """
    keys: set[tuple] = set()
    if not Path(path).exists():
        return keys

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                key = tuple(data[field] for field in key_fields)
                keys.add(key)
            except (json.JSONDecodeError, KeyError):
                continue

    return keys


def load_jsonl(path: str, model_class: Type[T]) -> list[T]:
    """Load a JSONL file into a list of typed Pydantic models.

    Skips corrupt lines with a warning.
    """
    results: list[T] = []
    if not Path(path).exists():
        return results

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(model_class.model_validate_json(line))
            except Exception:
                continue

    return results
