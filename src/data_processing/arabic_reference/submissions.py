"""Import complete public benchmark runs, preserving their immutable identities."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .core import sha256
from .benchmark_performance import read_performance
from .leaderboard_inputs import dataset_pin
from .latency import source_binding
from . import local_checkpoint


def read_submissions(path, plan):
    path = Path(path)
    before = sha256(path)
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("manifest_digest") != plan["manifest_digest"]:
        raise ValueError("Public submissions belong to a different frozen plan")
    inputs, details, names = [], {}, set()
    for item in data["models"]:
        name = item["name"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in names:
            raise ValueError("Public submission names must be distinct safe labels")
        names.add(name)
        if item.get("status") != "complete" or item.get("mode") != "arabic":
            raise ValueError("Only complete Arabic-hinted benchmark submissions can be ranked")
        for field in ("predictions", "run_file"):
            relative = Path(item[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Submission paths must stay within their run collection")
            file = path.parent / relative
            if sha256(file) != item[f"{field}_sha256"]:
                raise ValueError(f"Public submission file changed: {name}/{field}")
        run = json.loads((path.parent / item["run_file"]).read_text())
        local = item.get("source_type") == "local_checkpoint"
        public = run.get("benchmark_model" if local else "public_model", {})
        fields = ("source_type", "checkpoint_digest", "training_step", "backend", "display_name") if local else ("repository", "revision", "backend", "display_name")
        for field in fields:
            if public.get(field) != item.get(field):
                raise ValueError(f"Public model provenance mismatch: {name}/{field}")
        if local:
            if name not in local_checkpoint.ENTRIES:
                raise ValueError("Unregistered local checkpoint")
            local_checkpoint.validate_registration(public, run, name)
        else:
            dataset_pin(item["repository"], item["revision"])
        if run.get("modes") != ["arabic"] or run.get("backend") != item["backend"]:
            raise ValueError("Public run decoding protocol differs from the submission")
        display = item["display_name"]
        if (
            not isinstance(display, str)
            or not display.strip()
            or len(display) > 100
            or any(ord(c) < 32 for c in display)
        ):
            raise ValueError("Invalid public model display name")
        inputs.append((name, path.parent / item["predictions"]))
        details[name] = {
            k: item[k]
            for k in (*fields, "run_file_sha256")
        }
        details[name].update(
            performance=read_performance((path.parent / item["run_file"]).parent, item["predictions_sha256"], expected_run=run),
            runtime=run["runtime"],
            max_new_tokens=run["max_new_tokens"],
            precision=run["precision"],
            postprocessing=public["postprocessing"],
            decoding_options=run.get("decoding_options", {}),
            model_url=None if local else f"https://huggingface.co/{item['repository']}/tree/{item['revision']}",
            latency_binding=source_binding(run),
        )
    if not inputs or sha256(path) != before:
        raise ValueError("Empty or changed public submissions manifest")
    return inputs, details, before
