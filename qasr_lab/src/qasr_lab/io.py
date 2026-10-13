"""Small, dependency-free artifact and provenance utilities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{index}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index}: expected an object")
            yield row


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fresh_directory(path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def environment():
    from . import __version__

    versions = {}
    for name in ("torch", "transformers", "qasr", "qwen-asr", "numpy", "soundfile", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    code_dir = Path(__file__).resolve().parent
    code_hash = digest({p.name: file_hash(p) for p in sorted(code_dir.glob("*.py"))})
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code_dir, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    qasr_module = sys.modules.get("qasr")
    qasr_files = {}
    if qasr_module is not None:
        qasr_dir = Path(qasr_module.__file__).resolve().parent
        qasr_files = {p.name: file_hash(p) for p in sorted(qasr_dir.glob("*.py"))}
    return {
        "qasr_lab": __version__,
        "lab_source_sha256": code_hash,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "qasr_source_sha256": digest(qasr_files) if qasr_files else None,
    }
