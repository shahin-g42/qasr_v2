import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("catalog.json").read_text())
# Keep the report and equal-weight macro ordering identical to the live leaderboard.
DATASETS = ("sada", "commonvoice", "masc_clean", "masc_noisy", "mgb2", "casablanca")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        name = f.name
    os.replace(name, path)


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        name = f.name
    os.replace(name, path)


@contextmanager
def exclusive(directory):
    """Prevent concurrent coordinators; kernel releases the lock after a crash."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_config(path):
    import yaml

    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    # Relative paths are relative to the configuration, never the invoking shell.
    def resolve(value):
        value = str(value)
        expanded_value = os.path.expandvars(os.path.expanduser(value))
        if re.search(r"\$\{?[A-Za-z_][A-Za-z0-9_]*", expanded_value):
            raise ValueError(f"Unset environment variable in benchmark config value: {value}")
        expanded = Path(expanded_value)
        return str((path.parent / expanded).resolve()) if not expanded.is_absolute() else str(expanded)
    for key in ("work_dir", "checkpoint", "output_dir"):
        cfg[key] = resolve(cfg[key])
    for key, entry in cfg["datasets"].items():
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset {key}; expected {DATASETS}")
        for field in ("root", "audio_map", "metadata"):
            if entry.get(field):
                entry[field] = resolve(entry[field])
    if set(cfg["datasets"]) != set(DATASETS):
        raise ValueError("Configuration must declare all six test sets")
    if cfg.get("precision", "bf16") not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    if int(cfg.get("simultaneous_model_loads", 0)) < 1:
        raise ValueError("simultaneous_model_loads must be positive")
    if int(cfg.get("task_size", 0)) < 1:
        raise ValueError("task_size must be positive")
    for field in ("startup_timeout_seconds", "progress_timeout_seconds"):
        if float(cfg.get(field, 1800)) <= 0:
            raise ValueError(f"{field} must be positive")
    return cfg
