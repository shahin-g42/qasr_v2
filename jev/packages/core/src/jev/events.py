"""Rank-zero append-only event log with an immutable run descriptor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jev_contracts import atomic_write_json, canonical_json

RUN_NAME = "run.json"
EVENTS_NAME = "events.jsonl"


class RunLog:
    """One JSON event per line, fsynced before the next training step begins.

    Nonzero ranks construct the log but never write: event ordering and file
    integrity are owned by rank zero alone. On resume the existing descriptor
    must describe the identical run, or loading fails before any update runs.
    """

    def __init__(self, root: str | Path, *, identity: dict, rank: int):
        if type(identity) is not dict or not identity:
            raise ValueError("identity must be a nonempty mapping")
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.root = Path(root).absolute()
        self.rank = rank
        self._file = None
        if rank != 0:
            return
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"Run directory is not a directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor_path = self.root / RUN_NAME
        if descriptor_path.exists():
            existing = json.loads(descriptor_path.read_text(encoding="utf-8"))
            if existing.get("identity") != identity:
                raise ValueError(
                    f"Run directory {self.root} belongs to a different run identity; "
                    "use a fresh run_dir or resume the matching configuration"
                )
        else:
            atomic_write_json(descriptor_path, {"schema_version": 1, "identity": identity})
        self._file = open(self.root / EVENTS_NAME, "ab")

    def event(self, name: str, payload: dict | None = None) -> None:
        """Append one durable event; failures to serialize are hard errors."""
        if self.rank != 0:
            return
        if type(name) is not str or not name:
            raise ValueError("Event name must be a nonempty string")
        if payload is not None and type(payload) is not dict:
            raise ValueError("Event payload must be a mapping")
        record = {"event": name, **(payload or {})}
        line = (canonical_json(record) + "\n").encode("utf-8")
        self._file.write(line)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
