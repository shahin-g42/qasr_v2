"""The deterministic global-batch trainer for HSDP decision models.

Every rank derives the identical epoch/window plan from (seed, epoch), takes
its own chunk of each window, and pads its microbatch schedule with zero-weight
dummy passes so all ranks issue the same collectives per optimizer step. The
scaled loss plus FSDP2's mesh-mean reducer yields exactly the gradient of the
mean loss over the window's real decisions; the tail window is trained on its
true size, never dropped or padded with fake data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from jev_contracts import (
    TRAINING_TIERS,
    artifact_path,
    canonical_json,
    validate_example,
    verify_manifest,
)

from .batching import (
    Cursor,
    Window,
    assign_ranks,
    check_rows,
    epoch_order,
    local_lengths,
    plan_windows,
    prepared_rows,
    split_by_budget,
)
from .checkpoint import (
    capture_rng,
    checkpoint_matches,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_rng,
    save_checkpoint,
)
from .config import TrainConfig
from .distributed import (
    assert_no_meta,
    clip_gradients,
    enable_activation_checkpointing,
    hsdp_mesh,
    init_distributed,
    init_sharded_weights,
    materialize_empty,
    parallelize,
    sync_replicated_gradients,
)
from .events import RunLog
from .losses import decision_loss, scaled_loss
from .modeling import build_meta_model
from .processing import PROMPT_VERSION, target_tensor

_TRAINING_SPLITS = {"train", "development"}


def load_release_split(root: str | Path, split: str) -> tuple[list[dict], dict]:
    """Manifest-verified release reader for training and monitoring splits.

    Core consumes the frozen release artifact directly and never imports the
    data package. It enforces manifest integrity, the approval flag, canonical
    split paths, and per-row validity; the approval's audit and rights evidence
    was verified when the release was frozen and is bound to these file hashes.
    The trainer refuses calibration and test splits outright.
    """
    if split not in _TRAINING_SPLITS:
        raise ValueError(
            f"Split {split!r} cannot be loaded by the trainer; "
            "training reads train/development only"
        )
    root = Path(root).absolute()
    manifest = verify_manifest(root)
    meta = manifest.get("metadata")
    if (type(meta) is not dict or meta.get("kind") != "dataset_release"
            or meta.get("approved") is not True
            or meta.get("schema_version") != 2):
        raise ValueError("Release manifest is not an approved dataset_release v2")
    splits = meta.get("splits")
    if type(splits) is not dict or not splits:
        raise ValueError("Release split map is empty or invalid")
    if split not in splits:
        raise ValueError(f"Release does not contain split {split!r}")
    for key, name in splits.items():
        if key not in {"train", "development", "calibration", "test"} or name != key + ".jsonl":
            raise ValueError("Release splits must use canonical safe JSONL paths")
    if "audit.json" not in manifest["files"] or "registry.json" not in manifest["files"]:
        raise ValueError("Approved release must bind its audit and registry")
    name = splits[split]
    if name not in manifest["files"]:
        raise ValueError(f"Split file absent from manifest: {name}")
    text = artifact_path(root, name).read_text(encoding="utf-8")
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank JSONL line in {name} at line {number}")
        row = json.loads(line)
        validated = validate_example(row)
        if validated["split"] != split:
            raise ValueError(
                f"Row {validated['id']!r} is filed under split {split!r} "
                f"but labels {validated['split']!r}"
            )
        if validated["quality"]["tier"] not in TRAINING_TIERS:
            raise ValueError(
                f"Row {validated['id']!r} has tier {validated['quality']['tier']!r}; "
                "only audited gold/silver rows may enter training or monitoring"
            )
        rows.append(validated)
    if not rows:
        raise ValueError(f"Split {split!r} is empty in this release")
    identity = {"manifest": hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()}
    return rows, identity


def snapshot_identity(path: str | Path) -> dict:
    """Strong identity for manifest-sealed snapshots; explicit trust otherwise."""
    root = Path(path).absolute()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "kind": "manifest",
            "sha256": hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest(),
        }
    return {"kind": "caller-trusted-local", "root": str(root)}


def learning_rate(
    *, update: int, total_updates: int, peak: float, warmup_updates: int, min_ratio: float,
) -> float:
    """Linear warmup then cosine decay to min_ratio*peak; a pure function of update."""
    if update < 0 or total_updates < 1 or warmup_updates < 1:
        raise ValueError("update/total/warmup must be nonnegative/positive integers")
    if update < warmup_updates:
        return peak * (update + 1) / warmup_updates
    span = max(1, total_updates - warmup_updates)
    progress = min(1.0, (update - warmup_updates) / span)
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with decoupled decay on matrix weights only; norms and biases excluded.

    FSDP2-managed parameters are DTensors while the replicated readout stays
    dense, and torch's foreach optimizer kernels require each batch to be
    uniformly one kind, so every decay group is subdivided by DTensor
    membership. A fully dense or fully managed model keeps exactly two groups.
    """
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters; vision-only training is not a supported mode")
    groups = []
    for decay, weight_decay in ((True, config.weight_decay), (False, 0.0)):
        selected = [parameter for parameter in trainable if (parameter.ndim >= 2) == decay]
        for managed in (True, False):
            params = [
                parameter for parameter in selected
                if isinstance(parameter, DTensor) == managed
            ]
            if params:
                groups.append({"params": params, "weight_decay": weight_decay})
    return torch.optim.AdamW(
        groups, lr=config.learning_rate, betas=(config.beta1, config.beta2),
        eps=config.epsilon, foreach=config.foreach_optimizer,
    )


class Trainer:
    """One rank's view of the synchronized global training loop."""

    def __init__(
        self, config: TrainConfig, *, model: Any, optimizer: Any, mesh: Any, device: torch.device,
        rank: int, world_size: int, rows_by_id: dict[str, dict], dev_rows: list[dict],
        identity: dict, log: RunLog, cursor: Cursor, resumed_from: str | None = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.mesh = mesh
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.rows_by_id = rows_by_id
        self.dev_rows = dev_rows
        self.dev_by_id = {row["id"]: row for row in dev_rows}
        if len(self.dev_by_id) != len(dev_rows):
            raise ValueError("Duplicate development example IDs")
        self.identity = identity
        self.log = log
        self.cursor = cursor
        self.resumed_from = resumed_from
        self.checkpoint_root = (
            Path(config.checkpoint_dir) if config.checkpoint_dir
            else Path(config.run_dir) / "checkpoints"
        ).absolute()
        self._windows: dict[int, list[Window]] = {}
        self.total_updates = sum(len(self._windows_for(epoch)) for epoch in range(config.epochs))
        if self.total_updates < 1:
            raise ValueError("Configuration produces zero updates")
        self.warmup_updates = max(1, int(round(config.warmup_ratio * self.total_updates)))

    def _windows_for(self, epoch: int) -> list[Window]:
        if epoch not in self._windows:
            order = epoch_order(self.rows_by_id, self.config.seed, epoch)
            self._windows[epoch] = plan_windows(order, self.config.global_batch)
        return self._windows[epoch]

    def _normalize(self, cursor: Cursor) -> Cursor:
        while cursor.epoch < self.config.epochs:
            windows = self._windows_for(cursor.epoch)
            if cursor.window_index < len(windows):
                return cursor
            cursor = Cursor(cursor.epoch + 1, 0, cursor.update, cursor.consumed)
        return cursor

    def _aligned_microbatches(self, count: int) -> int:
        """All ranks run the same microbatch count; zeros pad the difference."""
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        local = torch.tensor([count], dtype=torch.int64, device=self.device)
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
        return max(1, int(local.item()))

    def _current_learning_rate(self) -> float:
        return learning_rate(
            update=self.cursor.update, total_updates=self.total_updates,
            peak=self.config.learning_rate, warmup_updates=self.warmup_updates,
            min_ratio=self.config.lr_min_ratio,
        )

    def _train_window(self, window: Window, epoch: int) -> dict:
        prepared = prepared_rows(self.rows_by_id, window, seed=self.config.seed, epoch=epoch)
        prepared_by_id = {row["id"]: row for row in prepared}
        local_ids = assign_ranks(window, self.world_size, self.config.micro_cap)[self.rank]
        lengths = local_lengths(self.model, self.rows_by_id, local_ids)
        subbatches = (
            split_by_budget(
                local_ids, lengths, micro_cap=self.config.micro_cap,
                token_budget=self.config.token_budget,
            ) if local_ids else []
        )
        aligned = self._aligned_microbatches(len(subbatches))
        dummy = self.rows_by_id[window.ids[0]]
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        real_examples = 0
        for index in range(aligned):
            if index < len(subbatches):
                rows = [prepared_by_id[row_id] for row_id in subbatches[index]]
                batch = self.model.prepare(rows, self.config.max_length)
                logits = self.model(batch)
                targets = target_tensor(rows, device=logits.device)
                per_example = decision_loss(logits, targets, batch.counts, reduction="none")
                loss = scaled_loss(
                    per_example, global_examples=window.size, world_size=self.world_size,
                )
                with torch.no_grad():
                    loss_sum += per_example.detach().double().sum()
                    real_examples += len(rows)
            else:
                # Zero-weight pass keeps every rank's collective schedule aligned.
                batch = self.model.prepare([dummy], self.config.max_length)
                logits = self.model(batch)
                loss = logits.sum() * 0.0
            loss.backward()
        sync_replicated_gradients(self.model, world_size=self.world_size)
        grad_norm = clip_gradients(self.model, self.config.clip_norm, world_size=self.world_size)
        rate = self._current_learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()
        update_index = self.cursor.update
        self.cursor = self._normalize(self.cursor.advance(window))
        stats = torch.stack([
            loss_sum,
            torch.tensor(float(real_examples), dtype=torch.float64, device=self.device),
        ])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if int(stats[1].item()) != window.size:
            raise ValueError(
                f"Window accounting mismatch: {int(stats[1].item())} examples != {window.size}"
            )
        return {
            "update": update_index, "epoch": epoch, "window": window.index,
            "examples": window.size, "loss": stats[0].item() / window.size,
            "learning_rate": rate, "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def _evaluate(self) -> dict:
        """Aligned development monitoring: NLL and top-1 at temperature one."""
        if not self.dev_rows:
            raise ValueError("Evaluation is configured but the release has no development split")
        ids = sorted(self.dev_by_id)
        local_ids = ids[self.rank::self.world_size]
        lengths = local_lengths(self.model, self.dev_by_id, local_ids)
        subbatches = (
            split_by_budget(
                local_ids, lengths, micro_cap=self.config.micro_cap,
                token_budget=self.config.token_budget,
            ) if local_ids else []
        )
        aligned = self._aligned_microbatches(len(subbatches))
        dummy = self.dev_by_id[ids[0]]
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        correct = torch.zeros((), dtype=torch.float64, device=self.device)
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            for index in range(aligned):
                if index < len(subbatches):
                    rows = [self.dev_by_id[row_id] for row_id in subbatches[index]]
                else:
                    rows = [dummy]
                batch = self.model.prepare(rows, self.config.max_length)
                logits = self.model(batch)
                if index >= len(subbatches):
                    continue
                targets = target_tensor(rows, device=logits.device)
                per_example = decision_loss(logits, targets, batch.counts, reduction="none")
                loss_sum += per_example.double().sum()
                for row_index, count in enumerate(batch.counts):
                    predicted = logits[row_index, :count].argmax().item()
                    expected = targets[row_index, :count].argmax().item()
                    correct += float(predicted == expected)
        finally:
            for module, training in modes:
                module.training = training
        stats = torch.stack([loss_sum, correct])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        count = len(ids)
        return {
            "count": count, "nll": stats[0].item() / count,
            "accuracy": stats[1].item() / count,
        }

    def _save_checkpoint(self, *, final: bool) -> None:
        update = self.cursor.update
        directory = self.checkpoint_root / f"step_{update:012d}"
        if checkpoint_matches(directory, update=update, identity=self.identity):
            # A prior pass already sealed this update. Identity binds seed, data,
            # snapshot, and mesh, so update plus identity pins the cursor exactly;
            # resaving the same step is a no-op rather than a collision.
            self.log.event("checkpoint_exists", {
                "update": update, "directory": str(directory),
            })
            dist.barrier()
            return
        run_state = {
            "cursor": asdict(self.cursor), "update": update,
            "learning_rate": self._current_learning_rate(), "final": final,
            "rng": capture_rng(self.device),
        }
        descriptor = save_checkpoint(
            directory, model=self.model, optimizer=self.optimizer, run_state=run_state,
            identity=self.identity, mesh_shape=self.config.mesh_shape, update=update,
        )
        self.log.event("checkpoint", {
            "update": update, "directory": str(directory), "final": final,
            "world_size": descriptor["world_size"],
        })
        if self.rank == 0:
            prune_checkpoints(self.checkpoint_root, keep=self.config.keep_checkpoints)
        dist.barrier()

    def run(self) -> dict:
        """Train to the configured horizon; always leave a sealed final checkpoint."""
        started = self._normalize(self.cursor)
        self.cursor = started
        self.log.event("start", {
            "update": self.cursor.update, "epoch": self.cursor.epoch,
            "total_updates": self.total_updates, "warmup_updates": self.warmup_updates,
            "train_examples": len(self.rows_by_id), "resumed_from": self.resumed_from,
        })
        while self.cursor.epoch < self.config.epochs:
            epoch = self.cursor.epoch
            windows = self._windows_for(epoch)
            # _train_window normalizes the cursor, which rolls to (epoch+1, 0)
            # once the epoch's last window is consumed; the epoch check keeps
            # this loop from re-walking the finished epoch's window list.
            while self.cursor.epoch == epoch and self.cursor.window_index < len(windows):
                window = windows[self.cursor.window_index]
                summary = self._train_window(window, epoch)
                self.log.event("update", summary)
                performed = summary["update"] + 1
                if self.config.eval_every and performed % self.config.eval_every == 0:
                    metrics = self._evaluate()
                    self.log.event("evaluation", {"update": performed, **metrics})
                # The epoch-end checkpoint of the final epoch is the run's terminal
                # update: it is sealed once, by the final save after the loops, with
                # its final marker set. Periodic saves never collide with it. The
                # cursor has already been normalized past the epoch's last window,
                # so epoch end is detected by the epoch roll, not the window index.
                epoch_end = (
                    self.cursor.epoch != epoch
                    or self.cursor.window_index >= len(windows)
                )
                run_end = epoch_end and epoch == self.config.epochs - 1
                if not run_end and (performed % self.config.checkpoint_every == 0 or epoch_end):
                    self._save_checkpoint(final=False)
            self.cursor = self._normalize(self.cursor)
        self._save_checkpoint(final=True)
        self.log.event("done", {"update": self.cursor.update, "examples": self.cursor.consumed})
        self.log.close()
        return {"updates": self.cursor.update, "examples": self.cursor.consumed}


def build_trainer(config: TrainConfig) -> Trainer:
    """Assemble one rank's trainer: world, mesh, sharded model, data, resume."""
    rank, world_size, local_rank = init_distributed(
        device_type=config.device, timeout_minutes=config.timeout_minutes,
    )
    device = (
        torch.device(f"cuda:{local_rank}") if config.device == "cuda" else torch.device("cpu")
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' but CUDA is unavailable on this node")
        torch.cuda.set_device(device)
    torch.manual_seed(config.seed)
    local_world_size = world_size // config.replicate
    mesh = hsdp_mesh(
        device_type=config.device, shard=config.shard, replicate=config.replicate,
        local_world_size=local_world_size,
    )
    model = build_meta_model(Path(config.snapshot), config.initialization)
    if config.gradient_checkpointing:
        enable_activation_checkpointing(model)
    param_dtype = torch.bfloat16 if config.dtype == "bfloat16" else None
    parallelize(model, mesh, param_dtype=param_dtype)
    materialize_empty(model, device)
    model.materialize_runtime_buffers(device)
    assert_no_meta(model)
    init_sharded_weights(model, Path(config.snapshot), initialization=config.initialization)
    optimizer = build_optimizer(model, config)
    train_rows, release_identity = load_release_split(config.release, "train")
    rows_by_id = check_rows(train_rows)
    dev_rows: list[dict] = []
    if config.eval_every:
        dev_rows, _ = load_release_split(config.release, "development")
    identity = {
        "config": config.identity_hash, "release": release_identity["manifest"],
        "snapshot": snapshot_identity(config.snapshot), "prompt_version": PROMPT_VERSION,
        "world_size": world_size, "mesh": list(config.mesh_shape),
    }
    log = RunLog(config.run_dir, identity=identity, rank=rank)
    cursor = Cursor(0, 0, 0, 0)
    resumed_from = None
    if config.resume:
        resume_path = (
            latest_checkpoint(
                Path(config.checkpoint_dir) if config.checkpoint_dir
                else Path(config.run_dir) / "checkpoints"
            ) if config.resume == "latest" else Path(config.resume)
        )
        if resume_path is None:
            raise ValueError(f"No sealed checkpoint found under {config.resume!r}")
        run_state = load_checkpoint(
            resume_path, model=model, optimizer=optimizer,
            identity=identity, mesh_shape=config.mesh_shape,
        )
        raw = run_state.get("cursor")
        if (type(raw) is not dict
                or any(type(raw.get(key)) is not int or raw[key] < 0
                       for key in ("epoch", "window_index", "update", "consumed"))):
            raise ValueError(f"Malformed cursor in checkpoint: {resume_path}")
        cursor = Cursor(raw["epoch"], raw["window_index"], raw["update"], raw["consumed"])
        restore_rng(run_state["rng"], device)
        resumed_from = str(resume_path)
    return Trainer(
        config, model=model, optimizer=optimizer, mesh=mesh, device=device, rank=rank,
        world_size=world_size, rows_by_id=rows_by_id, dev_rows=dev_rows, identity=identity,
        log=log, cursor=cursor, resumed_from=resumed_from,
    )
