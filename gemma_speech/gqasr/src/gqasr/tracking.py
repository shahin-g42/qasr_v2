"""Rank-zero W&B logging and a JSONL follower for already-running jobs."""
import argparse
import json
import math
import os
import time
import uuid
import warnings
from pathlib import Path

TRACKING_KEYS = {"report_to", "wandb_project", "wandb_entity", "run_name"}


def training_config(cfg):
    """Logging choices must not invalidate an otherwise exact training resume."""
    return {k: v for k, v in cfg.items() if k not in TRACKING_KEYS | {"output_dir", "workers"}}


def scalars(record):
    """Upload numeric metrics only, never transcripts or per-example metadata."""
    result = {"trainer/step": record["step"]}
    for key in ("rank0_last_microbatch_loss", "rank0_gradient_norm", "elapsed_seconds"):
        if key in record:
            result["train/" + key] = record[key]
    for group, value in record.get("learning_rates", {}).items():
        result["lr/" + group] = value
    for group, scores in record.get("validation", {}).items():
        for metric, value in scores.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                result[f"eval/{group}/{metric}"] = value
    rows = record.get("rank0_last_microbatch_augmentation", [])
    if rows:
        for key in ("triggered", "applied", "waveform_changed", "specaugment_changed"):
            result[f"augmentation/rank0_{key}_fraction"] = sum(
                bool(row.get(key)) for row in rows
            ) / len(rows)
    return result


class Tracker:
    def __init__(self, cfg, main=True, resume=None):
        self.cfg = cfg
        self.enabled = main and (
            cfg.get("report_to") == "wandb" or "wandb" in (cfg.get("report_to") or [])
        ) and os.environ.get("WANDB_MODE") != "disabled"
        self.resume = resume
        self.run = None
        self.lock = None
        self.last_step = -1
        self.previous = None
        self.error = None
        self.state_path = Path(cfg["output_dir"]) / "wandb_tracking.json"

    def start(self):
        import fcntl

        import wandb

        if not callable(getattr(wandb, "init", None)) or not callable(getattr(wandb, "Settings", None)):
            raise ImportError(
                "The imported 'wandb' is not the W&B SDK. A local wandb/ log directory "
                "can be imported as a namespace when the SDK is missing. Install into "
                "the same environment: gemma_speech/.venv/bin/python -m pip install "
                "'wandb>=0.19,<1'. Keep your existing wandb/ logs."
            )

        self.lock = (self.state_path.parent / "wandb_follower.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.last_step = state.get("last_step", -1)
        run_id = state.get("id", uuid.uuid4().hex[:12])
        directory = Path(os.environ.get("WANDB_DIR", self.cfg["output_dir"]))
        directory.mkdir(parents=True, exist_ok=True)
        # Deliberately send a small whitelist, not paths/manifests or credentials.
        config = {k: self.cfg[k] for k in (
            "batch_size", "accumulation_steps", "expected_world_size", "max_steps",
            "warmup_steps", "decay_steps", "learning_rate", "decoder_learning_rate",
            "embedding_learning_rate", "acoustic_layers", "decoder_layers",
            "train_embeddings", "precision", "language_batch_counts",
        ) if k in self.cfg}
        config["resumed_training"] = bool(self.resume)
        self.run = wandb.init(
            project=self.cfg.get("wandb_project", "qasr"),
            entity=self.cfg.get("wandb_entity") or None,
            name=self.cfg.get("run_name") or Path(self.cfg["output_dir"]).name,
            id=run_id, resume="allow", dir=str(directory), config=config,
            settings=wandb.Settings(init_timeout=30),
        )
        self.run.define_metric("trainer/step")
        self.run.define_metric("*", step_metric="trainer/step")
        self.save_state()
        print(json.dumps({"event": "wandb_started", "run_id": self.run.id,
                          "url": self.run.url}), flush=True)

    def save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"id": self.run.id, "last_step": self.last_step}))
        os.replace(tmp, self.state_path)

    def log(self, record):
        if not self.enabled:
            return
        try:
            if self.run is None:
                self.start()
            if record["step"] <= self.last_step:
                return
            values = scalars(record)
            if self.previous is not None:
                ds = record["step"] - self.previous["step"]
                dt = record["elapsed_seconds"] - self.previous["elapsed_seconds"]
                if ds > 0 and dt > 0:
                    values["performance/seconds_per_update"] = dt / ds
                    batch = self.cfg.get("batch_size", 0) * self.cfg.get(
                        "accumulation_steps", 1
                    ) * self.cfg.get("expected_world_size", 1)
                    values["performance/examples_per_second"] = batch * ds / dt
            self.run.log(values)
            self.previous = record
            self.last_step = record["step"]
            self.save_state()
        except Exception as exc:
            self.error = exc
            # A tracking outage must not strand the other distributed ranks.
            warnings.warn(f"W&B tracking disabled ({type(exc).__name__}); metrics.jsonl remains authoritative. Check the tracking environment; the standalone follower reports the original traceback.", stacklevel=2)
            self.enabled = False

    def finish(self, exit_code=0):
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as exc:
                warnings.warn(f"W&B finish failed ({type(exc).__name__}); local metrics are retained.", stacklevel=2)
        if self.lock is not None:
            self.lock.close()
            self.lock = None


def follow(output, follow=False, project="qasr", name=None):
    """Read complete JSONL lines; leave an in-progress final line for the next poll."""
    output = Path(output).resolve()
    cfg = json.loads((output / "config.json").read_text())
    cfg.update(output_dir=str(output), report_to="wandb", wandb_project=project)
    if name:
        cfg["run_name"] = name
    tracker = Tracker(cfg)
    try:
        with (output / "metrics.jsonl").open() as source:
            while True:
                position = source.tell()
                line = source.readline()
                if not line.endswith("\n"):
                    source.seek(position)
                    if not follow:
                        break
                    time.sleep(2)
                    continue
                tracker.log(json.loads(line))
                if not tracker.enabled:
                    raise RuntimeError("W&B follower stopped; see the original exception above") from tracker.error
    finally:
        tracker.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--project", default="qasr")
    parser.add_argument("--name")
    args = parser.parse_args()
    follow(**vars(args))


if __name__ == "__main__":
    main()
