"""Real tiny QASR save/load/generation and shared-benchmark integration tests."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gqasr.arabic_benchmark.common import sha256
from gqasr.arabic_benchmark.qasr_backend import QASRRecognizer, check_memory, checkpoint_identity
from gqasr.arabic_benchmark import runner


@pytest.fixture
def checkpoint(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config
    from qasr import QASRConfig, QASRFeatureExtractor, QASRForConditionalGeneration, QASRProcessor
    from qasr.configuration import CohereEncoderConfig

    special = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "<|audio_start|>",
               "<|audio_pad|>", "<|audio_end|>", "<asr_text>"]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(["system user assistant language Arabic English hello مرحبا"],
                               trainers.BpeTrainer(vocab_size=128, special_tokens=special))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>",
        additional_special_tokens=special[2:], audio_token="<|audio_pad|>",
        audio_bos_token="<|audio_start|>", audio_eos_token="<|audio_end|>",
    )
    template = "{{ '<|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\\n<|im_start|>assistant\\n' }}"
    processor = QASRProcessor(feature_extractor=QASRFeatureExtractor(return_attention_mask=True),
                              tokenizer=tokenizer, chat_template=template)
    config = QASRConfig(
        audio_config=CohereEncoderConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, num_mel_bins=128, subsampling_factor=8, subsampling_conv_channels=4,
            subsampling_conv_kernel_size=3, subsampling_conv_stride=2, conv_kernel_size=3,
            max_position_embeddings=256),
        text_config=Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id),
        audio_token_id=tokenizer.convert_tokens_to_ids("<|audio_pad|>"), timestamp_token_id=1,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    path = tmp_path / "checkpoint"
    torch.manual_seed(42)
    # Force sharding: production QASR has multiple safetensors files.
    model = QASRForConditionalGeneration(config).eval()
    model.generation_config.forced_bos_token_id = tokenizer.unk_token_id
    model.generation_config.repetition_penalty = 1.5
    model.save_pretrained(path, max_shard_size="20KB")
    processor.save_pretrained(path)
    return path


@pytest.fixture
def recognizer(checkpoint, monkeypatch):
    # Production has one Recognizer per process. Tests construct several in one.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    return QASRRecognizer(checkpoint, "cpu", "fp32")


def samples(tmp_path):
    result = []
    for i, length in enumerate((1600, 4800)):
        path = tmp_path / f"audio-{i}.npy"
        np.save(path, np.sin(np.arange(length, dtype=np.float32) * .03) * .1)
        result.append(dict(id=f"sada:{i:06d}", audio=str(path), audio_sha256=sha256(path),
                           text="REFERENCE MUST NOT ENTER PROMPT", dataset="sada"))
    return result


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_real_qasr_sharded_loading_and_generation(recognizer, tmp_path, precision):
    if precision == "bf16":
        recognizer.model.to(dtype=torch.bfloat16)
        recognizer.precision = precision
    assert recognizer.model.generation_config.forced_bos_token_id is None
    assert recognizer.model.generation_config.repetition_penalty in (None, 1.0)
    rows = samples(tmp_path)
    features = recognizer.features(rows)
    assert features["input_features"].shape[-1] == 128
    assert not features["attention_mask"][0, 0]  # left-padded shorter prompt
    for mode in ("arabic", "auto"):
        predictions, retries = recognizer.safe_infer(rows, features, mode, 2)
        assert retries == 0
        assert [r["id"] for r in predictions] == [r["id"] for r in rows]
        assert all(0 < r["generated_tokens"] <= 2 for r in predictions)
        assert all(isinstance(r["pred_text"], str) for r in predictions)
        singles = [recognizer.infer([row], recognizer.features([row]), mode, 2)[0] for row in rows]
        assert [r["raw_output"] for r in predictions] == [r["raw_output"] for r in singles]
    altered = [{**r, "text": "A DIFFERENT REFERENCE"} for r in rows]
    assert all(torch.equal(v, recognizer.features(altered)[k]) for k, v in features.items())


@pytest.mark.parametrize("mode", ["arabic", "auto"])
def test_prompt_parsing_eos_and_generation_limit(recognizer, tmp_path, monkeypatch, mode):
    rows = samples(tmp_path)
    features = recognizer.features(rows)
    tokenizer = recognizer.processor.tokenizer
    header = tokenizer.encode("language Arabic<asr_text>", add_special_tokens=False)
    generated = tokenizer.encode(("language Arabic<asr_text>" if mode == "auto" else "") + "hello",
                                 add_special_tokens=False)
    budget = len(generated) + 1

    def generate(**kwargs):
        prompt = kwargs["input_ids"]
        assert torch.equal(prompt[:, :features["input_ids"].shape[1]], features["input_ids"])
        if mode == "arabic":
            assert prompt[:, -len(header):].tolist() == [header, header]
        else:
            assert torch.equal(prompt, features["input_ids"])
        settings = kwargs["generation_config"]
        assert settings.do_sample is False and settings.num_beams == 1 and settings.repetition_penalty == 1
        suffix = torch.tensor([generated + [recognizer.eos_id], generated + [generated[-1]]])
        return torch.cat([prompt, suffix], 1)

    monkeypatch.setattr(recognizer.model, "generate", generate)
    result = recognizer.infer(rows, features, mode, budget)
    assert result[0]["pred_text"] == "hello"
    assert result[0]["eos_seen"] and not result[0]["generation_limit_hit"]
    assert result[1]["generation_limit_hit"] and not result[1]["eos_seen"]
    assert result[0]["predicted_language"] == ("Arabic" if mode == "auto" else None)


def test_qasr_fingerprint_all_shards_processor_and_source(checkpoint):
    from gqasr.arabic_benchmark.qasr_backend import weight_memory
    profile = weight_memory(checkpoint, "bf16")
    assert profile["stored_tensor_elements"] > 0
    assert 0 < profile["estimated_weight_gib_per_worker"] < .01
    identity = checkpoint_identity(checkpoint)
    assert "model.safetensors.index.json" in identity
    assert sum(name.endswith(".safetensors") for name in identity) > 1
    assert "processor_config.json" in identity and "tokenizer.json" in identity
    source = runner.backend_implementation_identity("qasr")
    assert {"qasr/modeling.py", "qasr/processing.py", "qasr/feature_extraction.py"} <= source.keys()
    single = checkpoint / "model.safetensors"
    single.touch()
    with pytest.raises(ValueError, match="Ambiguous QASR checkpoint"):
        checkpoint_identity(checkpoint)
    single.unlink()
    target = checkpoint / "tokenizer_config.json"
    target.write_text(target.read_text() + "\n")
    assert checkpoint_identity(checkpoint) != identity
    shard = next(checkpoint.glob("model-*.safetensors"))
    shard.unlink()
    with pytest.raises(FileNotFoundError, match="Missing QASR weights"):
        checkpoint_identity(checkpoint)


def test_memory_guard_rejects_too_many_large_replicas(checkpoint, monkeypatch):
    from gqasr.arabic_benchmark import qasr_backend
    monkeypatch.setattr(qasr_backend, "weight_memory", lambda *args: {"estimated_weight_gib_per_worker": 7})
    with pytest.raises(ValueError, match="Lower WORKERS_PER_GPU"):
        check_memory(checkpoint, "bf16", 80, 11)
    assert check_memory(checkpoint, "bf16", 80, 4)["budget_gib_per_worker"] == 18


def test_qasr_context_and_audio_integrity_fail_closed(recognizer, tmp_path):
    rows = samples(tmp_path)
    features = recognizer.features(rows)
    with pytest.raises(ValueError, match="exceeds context"):
        recognizer.infer(rows, features, "arabic", 256)
    Path(rows[0]["audio"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Prepared audio changed"):
        recognizer.features(rows)


def test_real_qasr_scheduler_roundtrip(checkpoint, tmp_path, monkeypatch):
    audio = samples(tmp_path)
    rows = [{**row, "ordinal": i, "relative_audio": f"clip-{i}.wav",
             "id": f"sada:{i:06d}", "duration": duration, "actual_duration": duration, "metadata": {}}
            for i, (row, duration) in enumerate(zip(audio * 9, (.1, .3) * 9, strict=True))]
    monkeypatch.setattr(runner, "official_rows", lambda *args: rows)
    monkeypatch.setattr(runner, "prepared_rows", lambda *args: rows)
    output = tmp_path / "results"
    cfg = dict(backend="qasr", checkpoint=str(checkpoint), work_dir=str(tmp_path),
               output_dir=str(output), precision="fp32", max_new_tokens=2,
               task_size=4, simultaneous_model_loads=1, startup_timeout_seconds=60)
    runner.run(cfg, datasets=["sada"], gpus="cpu", workers_per_gpu=1, batch_size=16)
    result = json.loads((output / "scores.json").read_text())
    assert result["protocol"]["backend"] == "qasr"
    assert result["protocol"]["task_size"] == 16
    assert result["performance"]["attempts"][0]["new_predictions"] == 36
    tasks = [json.loads(path.read_text()) for path in (output / "tasks").glob("*.json")]
    assert sorted(len(task["predictions"]) for task in tasks) == [4, 32]
    assert all(task["settings"]["batch_size"] == 16 for task in tasks)
    for mode in ("arabic", "auto"):
        assert result["diagnostics"][mode]["utterances"] == 18
    # No model worker needs to start for a fully completed rerun.
    runner.run(cfg, datasets=["sada"], gpus="cpu", workers_per_gpu=1, batch_size=16)
    assert json.loads((output / "progress.json").read_text())["reused_completed_run"]


def test_qasr_report_uses_qasr_identity(tmp_path):
    from gqasr.arabic_benchmark.common import atomic_json, fingerprint
    row = dict(id="sada:000000", dataset="sada", text="مرجع", pred_text="مرجع", mode="arabic",
               relative_audio="clip.wav", duration=1., ordinal=0, audio_sha256="hash",
               generation_limit_hit=False, generated_tokens=1)
    identity = {"backend": "qasr", "modes": ["arabic"], "precision": "bf16"}
    atomic_json(tmp_path / "tasks" / "one.json", dict(run_id=fingerprint(identity), predictions=[row],
        stats=dict(inference_seconds=1, audio_seconds=1, oom_batch_splits=0,
                   peak_allocated_gib=1, peak_reserved_gib=1)))
    runner.report(tmp_path, [row], identity, 1, [])
    assert (tmp_path / "REPORT.md").read_text().startswith("# QASR —")
    result = json.loads((tmp_path / "scores.json").read_text())
    assert "qasr" in result["leaderboard_context"] and "gqasr" not in result["leaderboard_context"]


@pytest.mark.parametrize("stage", ["doctor", "smoke", "run", "tune"])
@pytest.mark.parametrize("custom_settings", [False, True])
def test_qasr_launcher_selects_completed_stage02_and_shared_cache(tmp_path, stage, custom_settings):
    import os
    import shutil
    import subprocess
    import sys

    repo = Path(__file__).parents[3]
    scripts = tmp_path / "gemma_speech" / "cluster"
    scripts.mkdir(parents=True)
    for name in ("qasr_arabic_leaderboard.sh", "gqasr_arabic_leaderboard.sh"):
        shutil.copyfile(repo / "gemma_speech" / "cluster" / name, scripts / name)
    config_dir = tmp_path / "configs" / "v8.1"
    config_dir.mkdir(parents=True)
    shutil.copyfile(repo / "configs" / "v8.1" / "02_full_12node.yaml", config_dir / "02_full_12node.yaml")
    bin_dir = tmp_path / "gemma_speech" / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    (bin_dir / "activate").write_text("export CUDA_VISIBLE_DEVICES=1\n")
    executable = bin_dir / "gqasr-arabic-benchmark"
    executable.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
        "keys = ['GQASR_BENCH_CHECKPOINT', 'GQASR_BENCH_WORK', 'GQASR_BENCH_OUTPUT', 'CUDA_VISIBLE_DEVICES']\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': {key: os.environ[key] for key in keys}}))\n")
    executable.chmod(0o755)
    env = {k: v for k,v in os.environ.items() if not k.startswith(("QASR_BENCH", "GQASR_BENCH"))}
    for key in ("GPU_IDS", "WORKERS_PER_GPU", "BATCH_SIZE", "GEMMA_SPEECH_ENV", "SMOKE_PER_DATASET"):
        env.pop(key, None)
    if custom_settings:
        env.update(QASR_BENCH_CHECKPOINT=str(tmp_path / "output/v81/full_12node"),
                   QASR_BENCH_WORK=str(tmp_path / "gemma_speech/gqasr/verification/arabic-stream-cache"),
                   QASR_BENCH_OUTPUT=str(tmp_path / "verification/qasr-stage02-arabic"),
                   WORKERS_PER_GPU="8", BATCH_SIZE="16")
    result = subprocess.run(["bash", str(scripts / "qasr_arabic_leaderboard.sh"), stage],
                            env=env, capture_output=True, text=True, check=True)
    record = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith("{"))
    assert record["env"]["GQASR_BENCH_CHECKPOINT"].endswith("/output/v81/full_12node")
    assert record["env"]["GQASR_BENCH_WORK"].endswith("/verification/arabic-stream-cache")
    assert record["env"]["GQASR_BENCH_OUTPUT"].endswith("/verification/qasr-stage02-arabic")
    assert record["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    args = record["args"]
    assert args[args.index("--workers-per-gpu") + 1] == ("8" if custom_settings else "4")
    if custom_settings:
        for field in ("CHECKPOINT", "WORK", "OUTPUT"):
            assert record["env"][f"GQASR_BENCH_{field}"] == env[f"QASR_BENCH_{field}"]
    if stage in ("smoke", "run"):
        assert args[args.index("--modes") + 1] == "arabic,auto"
        assert args[args.index("--batch-size") + 1] == ("16" if custom_settings else "1")
    if stage == "smoke":
        limit = int(args[args.index("--limit-per-dataset") + 1])
        task_size = max(4, int(env.get("BATCH_SIZE", "1")))
        assert (6 * limit + task_size - 1) // task_size >= 8 * int(env.get("WORKERS_PER_GPU", "4"))
