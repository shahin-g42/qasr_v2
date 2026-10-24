"""Opt-in real CPU/Gloo + persistent-worker test; no cluster or model downloads."""

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from test_audio_types import artifact
from test_training import data_recipe

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.audio_types import upgrade


@pytest.mark.skipif(
    os.environ.get("RUN_AUD_DDP_TESTS") != "1",
    reason="Set RUN_AUD_DDP_TESTS=1 for CPU/Gloo integration",
)
@pytest.mark.parametrize("stage", ["decoder", "full"])
@pytest.mark.parametrize("ctc", [False, True])
def test_two_ranks_persistent_workers_save_resume_and_eval(tmp_path, tokenizer, stage, ctc):
    source = artifact(tmp_path, tokenizer, "audflash")
    upgrade(source, tmp_path / "typed")
    model_path = tmp_path / "typed"
    if ctc:
        from test_ctc import small_tokenizer

        from audm.ctc import export

        token_path = tmp_path / "ctc-tokenizer"
        token_path.mkdir()
        small_tokenizer().save(str(token_path / "tokenizer.json"))
        export(model_path, token_path, tmp_path / "ctc")
        model_path = tmp_path / "ctc"
    negatives = []
    for kind in ("music", "noise", "silence"):
        # Synthetic waves exercise plumbing, not semantic classification quality.
        path = tmp_path / f"{kind}.wav"
        sf.write(
            path,
            np.zeros(3200)
            if kind == "silence"
            else np.random.default_rng(42).normal(0, 0.01, 3200),
            16000,
        )
        negatives.append(dict(audio_filepath=str(path), text="", duration=0.2, audio_type=kind))
    manifest = tmp_path / "negatives.jsonl"
    manifest.write_text("\n".join(map(json.dumps, negatives)) + "\n")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                data_recipe=str(data_recipe(tmp_path)),
                stage=stage,
                ctc={"warmup_steps": 0},
                efficiency={
                    "distributed_wer": True,
                    "duration_bucketing": True,
                    "index_cache": str(tmp_path / "cache"),
                },
                robustness=dict(
                    non_speech=dict(probability=0.5, manifests=[str(manifest)]),
                    augmentation=dict(
                        clean_probability=0.2, effects=dict(gain=dict(p=1, db=[-3, 3]))
                    ),
                ),
                training=dict(
                    max_duration_seconds=30,
                    expected_world_size=2,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=2,
                    max_steps=2,
                    warmup_steps=0,
                    learning_rate=1e-4,
                    eval_steps=1,
                    save_steps=1,
                    save_total_limit=3,
                    eval_log_samples=1,
                    eval_generation_max_new_tokens=8,
                    eval_wer_samples_per_language=1,
                    report_to="none",
                    dataloader_num_workers=1,
                    dataloader_persistent_workers=True,
                    dataloader_pin_memory=False,
                    bf16=False,
                    tf32=False,
                    deepspeed=None,
                    gradient_checkpointing=True,
                    label_smoothing=0.1,
                    ctc_loss_weight=0.1 if ctc else 0.0,
                    augmentation=dict(spec_augment=dict(enabled=True, p=1)),
                ),
            )
        )
    )
    root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=f"{root / 'audm/src'}:{root / 'src'}",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        ACCELERATE_USE_CPU="true",
        WANDB_MODE="disabled",
    )
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "--nnodes=1",
        "--nproc-per-node=2",
        "-m",
        "audm",
        "train",
        "--config",
        str(recipe),
        "--model",
        str(model_path),
        "--output",
        str(tmp_path / "run"),
    ]
    for attempt, suffix in enumerate(([], ["--resume", str(tmp_path / "run/checkpoint-1")])):
        log = tmp_path / f"distributed-{attempt}.log"
        with log.open("w") as handle:
            process = subprocess.Popen(
                cmd + suffix,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
                pytest.fail(f"Distributed rehearsal timed out: {log.read_text()[-20000:]}")
        assert code == 0, log.read_text()[-24000:]
        state = json.loads((tmp_path / "run/trainer_state.json").read_text())
        assert state["global_step"] == 2
        assert any("eval_loss" in row for row in state["log_history"])
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "run")
    assert all(torch.isfinite(p).all() for p in model.parameters())
    proc = AUDProcessor.from_pretrained(tmp_path / "run/checkpoint-2")
    assert proc.config.audio_type_version == 1
    assert (tmp_path / "run/checkpoint-2/optimizer.pt").is_file()
    receipt = json.loads((tmp_path / "run/training_recipe.json").read_text())
    assert receipt["robustness_assets"]
