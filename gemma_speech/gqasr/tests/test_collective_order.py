"""Exercise logging collectives through the real resumed DDP training loop."""

import json
import shutil
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from gqasr.config import load_config
from gqasr.data import ManifestIndex
from gqasr.distributed import Runtime
from gqasr.modeling import GQASRForSpeechRecognition
from gqasr.optimization import build_optimizer
from gqasr.train import _train
from test_training import fixture


def worker(rank, config_path, rendezvous, checkpoint):
    torch.set_num_threads(1)
    cfg = load_config(config_path)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    rt = Runtime.__new__(Runtime)
    rt.rank, rt.local_rank, rt.world, rt.precision = rank, rank, 2, "fp32"
    rt.device = torch.device("cpu")
    events, losses, published = [], [], []
    updates = 0
    original_reduce = dist.all_reduce
    original_step = torch.optim.AdamW.step
    original_forward = GQASRForSpeechRecognition.forward

    def reduce(tensor, op=dist.ReduceOp.SUM, *args, **kwargs):
        events.append({"updates": updates, "numel": tensor.numel(),
                       "dtype": str(tensor.dtype),
                       "op": "SUM" if op == dist.ReduceOp.SUM else "MIN"})
        return original_reduce(tensor, op, *args, **kwargs)

    def update(self, *args, **kwargs):
        nonlocal updates
        result = original_step(self, *args, **kwargs)
        updates += 1
        return result

    def forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        if self.training:
            losses.append(float(result["loss"].detach()))
        return result

    class PublicationSpy:
        def log(self, record):
            # Only publication is rank-zero-specific. No W&B/network calls.
            assert rank == 0
            published.append({k: record[k] for k in ("step", "loss", "loss_microbatches")})

    dist.all_reduce = reduce
    torch.optim.AdamW.step = update
    GQASRForSpeechRecognition.forward = forward
    try:
        _train(cfg, rt, checkpoint, PublicationSpy())
    finally:
        rt.close()
    Path(config_path).with_name(f"collectives-{rank}.json").write_text(json.dumps({
        "events": events, "losses": losses, "published": published, "updates": updates,
    }))


def test_resumed_loss_logging_has_same_collective_order_on_every_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path, cfg = fixture(tmp_path)
    cfg.update(batch_size=1, accumulation_steps=4, language_batch_counts={"ar": 4},
               expected_world_size=2, acoustic_layers=2, decoder_layers=2,
               train_embeddings=True, max_steps=7505, logging_steps=5,
               eval_steps=500, save_steps=500, report_to="wandb",
               workers=2, prefetch_factor=1, dataloader_timeout=60)
    path.write_text(yaml.safe_dump(cfg))
    cfg = load_config(path)
    checkpoint = tmp_path / "checkpoint-00007500"
    shutil.copytree(cfg["base_model"], checkpoint)
    model = GQASRForSpeechRecognition.from_pretrained(checkpoint)
    model.configure_training(
        cfg["acoustic_layers"], cfg["decoder_layers"], cfg["train_embeddings"],
        cfg["gradient_checkpointing"], cfg["label_smoothing"],
        train_acoustic_frontend=cfg.get("train_acoustic_frontend"),
    )
    optimizer = build_optimizer(model, cfg)
    # A synthetic tiny checkpoint at update 7500 tests scheduling; it does not
    # pretend these fixture weights underwent 7500 optimization updates.
    torch.save({
        "optimizer": optimizer.state_dict(), "config": cfg, "step": 7500,
        "index_metadata": ManifestIndex(cfg["index_path"], "train").metadata,
        "rng": [{"cpu": torch.get_rng_state()} for _ in range(2)],
    }, checkpoint / "training_state.pt")
    context = mp.spawn(worker, args=(str(path), str(tmp_path / "rendezvous"), str(checkpoint)),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("resumed logging collective test timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    reports = [json.loads((tmp_path / f"collectives-{rank}.json").read_text())
               for rank in range(2)]
    assert reports[0]["events"] == reports[1]["events"]
    for report in reports:
        assert report["updates"] == 5 and len(report["losses"]) == 20
        # Loss-finiteness and gradient-norm checks: one int32 MIN each,
        # on every update, before the update. Logging: two float32 SUM
        # values once, after update 7505. DDP gradient buckets run in C++.
        assert report["events"] == [
            {"updates": step, "numel": 1, "dtype": "torch.int32", "op": "MIN"}
            for step in range(5) for _ in range(2)
        ] + [{"updates": 5, "numel": 2, "dtype": "torch.float32", "op": "SUM"}]
    assert reports[1]["published"] == []
    publication = reports[0]["published"]
    assert len(publication) == 1 and publication[0]["step"] == 7505
    assert publication[0]["loss_microbatches"] == 40
    expected_loss = sum(sum(report["losses"]) for report in reports) / 40
    assert publication[0]["loss"] == pytest.approx(expected_loss, rel=1e-6)
