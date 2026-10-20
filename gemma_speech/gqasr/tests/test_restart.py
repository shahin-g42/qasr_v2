"""Checkpoint recovery, microbatch safety and preserved optimizer state."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from gqasr.config import load_config
from gqasr.data import StratifiedUpdateSampler
from gqasr.resume import check_checkpoint, validate_resume
from gqasr.train import schedule, train
from test_training import fixture

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "gemma_speech/gqasr/configs"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("WANDB_MODE", "disabled")


def saved_recipe():
    old = load_config(CONFIGS / "03_encoder12_b80_12node.yaml")
    new = load_config(CONFIGS / "03_encoder12_b20_acc4_12node.yaml")
    saved = {"config": old, "index_metadata": {"x": 1}, "step": 7500, "rng": [{}] * 96}
    return saved, new


def test_recovery_config_and_all_rank_sampler_continuation():
    saved, cfg = saved_recipe()
    result = validate_resume(saved, cfg, {"x": 1}, True)
    assert result["next_update"] == 7501 and result["remaining_updates"] == 23946
    assert set(result["changes"]) == {"batch_size", "accumulation_steps"}
    assert cfg["workers"] == 8 and cfg["prefetch_factor"] == 1
    assert not cfg["online_code_switching"]
    counts = dict.fromkeys(cfg["language_batch_counts"], 10000)
    for rank in range(96):
        def sampler(batch):
            return list(StratifiedUpdateSampler(
                counts, cfg["language_batch_counts"], batch, 7502, rank, 96,
                cfg["training_data_seed"], start=7500, augmentation_seeds=True,
            ))
        original, resumed = sampler(80), sampler(20)
        assert len(original) == 2 and len(resumed) == 8
        # Microbatch-dependent augmentation seeds may change; anchor order may not.
        assert [x[:2] for b in original for x in b] == [x[:2] for b in resumed for x in b]


@pytest.mark.parametrize("key,value", [
    ("accumulation_steps", 6), ("max_steps", 33000), ("learning_rate", 1e-3),
    ("acoustic_layers", 0), ("expected_world_size", 8), ("training_data_seed", 999),
    ("augmentation", False), ("online_code_switching", True),
    ("language_batch_counts", {"ar": 39, "en": 25, "hi": 8, "ml": 4, "zh": 4}),
])
def test_microbatch_permission_does_not_allow_recipe_drift(key, value):
    saved, cfg = saved_recipe()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_resume(saved, cfg, {"x": 1}, True)


def test_resume_still_requires_explicit_flag_and_same_index():
    saved, cfg = saved_recipe()
    with pytest.raises(ValueError, match="allow-microbatch-change"):
        validate_resume(saved, cfg, {"x": 1})
    with pytest.raises(ValueError, match="index metadata"):
        validate_resume(saved, cfg, {"x": 2}, True)
    saved["rng"] = [{}]
    with pytest.raises(ValueError, match="RNG"):
        validate_resume(saved, cfg, {"x": 1}, True)


@pytest.mark.parametrize("key,value", [
    ("prefetch_factor", 0), ("prefetch_factor", True), ("dataloader_timeout", -1),
])
def test_invalid_loader_options(tmp_path, key, value):
    _, cfg = saved_recipe()
    cfg[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=key):
        load_config(path)


def test_optimizer_moments_schedule_and_step_survive_rebatch(tmp_path, monkeypatch):
    path, cfg = fixture(tmp_path)
    cfg.update(batch_size=2, accumulation_steps=1, max_steps=3, logging_steps=1,
               acoustic_layers=2, decoder_layers=2, train_embeddings=True)
    path.write_text(yaml.safe_dump(cfg))
    train(path)
    checkpoint = tmp_path / "out/checkpoint-00000001"
    saved = torch.load(checkpoint / "training_state.pt", weights_only=True)
    cfg.update(batch_size=1, accumulation_steps=2, workers=1, prefetch_factor=1,
               dataloader_timeout=60, output_dir=str(tmp_path / "resumed"))
    path.write_text(yaml.safe_dump(cfg))
    assert check_checkpoint(path, checkpoint, True)["passed"]
    with pytest.raises(ValueError, match="allow-microbatch-change"):
        train(path, checkpoint)
    observed = []
    original_step = torch.optim.AdamW.step

    def capture_step(optimizer, *args, **kwargs):
        observed.append(copy.deepcopy(optimizer.state_dict()))
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", capture_step)
    train(path, checkpoint, allow_microbatch_change=True)
    assert len(observed) == 2
    # Inspect immediately before the first resumed update: every Adam moment and
    # per-parameter step must match the saved checkpoint, not a fresh optimizer.
    for key, state in saved["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(observed[0]["state"][key][name], value, rtol=0, atol=0)
    for group in observed[0]["param_groups"]:
        assert group["lr"] == group["base_lr"] * schedule(1, cfg)
    final = torch.load(tmp_path / "resumed/final/training_state.pt", weights_only=True)
    assert final["step"] == 3
    assert all(value["step"].item() == 3 for value in final["optimizer"]["state"].values())
    record = json.loads((tmp_path / "resumed/resume.json").read_text())
    assert record["microbatch_changed"] and record["completed_updates"] == 1
    rows = [json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [2, 3]
    assert all(row["loss_microbatches"] == 2 for row in rows)
    assert all(row["performance"]["rank0_data_wait_seconds_per_update"] >= 0 for row in rows)
    # Existing checkpoints must never be overwritten by recovery.
    assert torch.load(checkpoint / "training_state.pt", weights_only=True)["step"] == 1


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_earlier_microbatch_nonfinite_loss_stops_before_optimizer(tmp_path, monkeypatch, bad):
    from gqasr.modeling import GQASRForSpeechRecognition
    from gqasr.distributed import Runtime

    path, cfg = fixture(tmp_path)
    cfg.update(batch_size=1, accumulation_steps=4, language_batch_counts={"ar": 4})
    path.write_text(yaml.safe_dump(cfg))
    forward = GQASRForSpeechRecognition.forward
    calls, checks, steps = [], [], []

    def poisoned(self, *args, **kwargs):
        result = forward(self, *args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            # Nonfinite loss with finite derivatives: the gradient check alone
            # would miss it. It must still be caught at the accumulation boundary.
            result["loss"] = result["loss"] + bad
        return result

    finite = Runtime.all_finite
    def check(self, value):
        checks.append(value.numel())
        return finite(self, value)

    monkeypatch.setattr(GQASRForSpeechRecognition, "forward", poisoned)
    monkeypatch.setattr(Runtime, "all_finite", check)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: steps.append(1))
    with pytest.raises(FloatingPointError, match="loss"):
        train(path)
    assert len(calls) == 4 and checks == [4] and not steps


def test_recovery_launcher_forwards_config_checkpoint_and_rank():
    env = dict(os.environ, DRY_RUN="1")
    for key in ("GQASR_CONFIG", "GQASR_RESUME_CHECKPOINT", "PROJECT_DIR", "MASTER_ADDR",
                "MASTER_PORT", "NNODES", "NPROC_PER_NODE"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/resume_gqasr_encoder_12node.sh"), "8"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for expected in ("--node_rank=8", "03_encoder12_b20_acc4_12node.yaml",
                     "checkpoint-00007500", "--allow-microbatch-change", "--master_port=29613"):
        assert expected in result.stdout


def distributed_train_worker(rank, config, rendezvous, poison):
    import torch.distributed as dist
    from datetime import timedelta
    from gqasr.distributed import Runtime
    from gqasr.modeling import GQASRForSpeechRecognition
    from gqasr.train import _train

    torch.set_num_threads(1)
    cfg = load_config(config)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    rt = Runtime.__new__(Runtime)
    rt.rank, rt.local_rank, rt.world, rt.precision = rank, rank, 2, "fp32"
    rt.device = torch.device("cpu")
    checks, updates, no_sync_calls, forward_calls = [], [], [], []
    original_forward = GQASRForSpeechRecognition.forward
    original_check = rt.all_finite
    original_step = torch.optim.AdamW.step
    original_no_sync = torch.nn.parallel.DistributedDataParallel.no_sync

    def forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        forward_calls.append(1)
        if poison and rank == 1 and len(forward_calls) == 1:
            result["loss"] = result["loss"] + float("inf")
        return result

    def check(value):
        checks.append(value.numel())
        return original_check(value)

    def update(self, *args, **kwargs):
        updates.append(1)
        return original_step(self, *args, **kwargs)

    def no_sync(self):
        no_sync_calls.append(1)
        return original_no_sync(self)

    GQASRForSpeechRecognition.forward = forward
    torch.optim.AdamW.step = update
    torch.nn.parallel.DistributedDataParallel.no_sync = no_sync
    rt.all_finite = check
    error = None
    try:
        _train(cfg, rt, None)
    except FloatingPointError as exc:
        error = str(exc)
    finally:
        rt.close()
    Path(config).with_name(f"rank-{rank}.json").write_text(json.dumps({
        "checks": checks, "updates": len(updates), "no_sync": len(no_sync_calls), "error": error,
    }))


@pytest.mark.parametrize("poison", [False, True])
def test_two_real_ddp_ranks_accumulate_and_fail_together(tmp_path, poison):
    import time
    import torch.multiprocessing as mp

    path, cfg = fixture(tmp_path)
    cfg.update(batch_size=1, accumulation_steps=4, language_batch_counts={"ar": 4},
               expected_world_size=2, acoustic_layers=2, decoder_layers=2, train_embeddings=True)
    path.write_text(yaml.safe_dump(cfg))
    context = mp.spawn(distributed_train_worker,
                       args=(str(path), str(tmp_path / "rendezvous"), poison),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 55
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("distributed accumulation test timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result["checks"] == ([4] if poison else [4, 1, 4, 1])
        assert result["updates"] == (0 if poison else 2)
        assert result["no_sync"] == (3 if poison else 6)
        assert bool(result["error"]) == poison
