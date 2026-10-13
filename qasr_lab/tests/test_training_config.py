import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qasr_lab.cli import main
from qasr_lab.experiments import load_experiment
from qasr_lab.io import digest, file_hash, read_json, write_json, write_jsonl
from qasr_lab.panel import load_panel
from qasr_lab.training_config import inspect_training_config, prepare_from_training_config

yaml = pytest.importorskip("yaml")


@pytest.fixture
def configured(tmp_path, monkeypatch):
    qasr = pytest.importorskip("qasr")

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(range(len(text.split())))

    monkeypatch.setattr(
        qasr.QASRProcessor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=Tokenizer()),
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_type": "qasr"})
    write_json(checkpoint / "processor_config.json", {})
    (checkpoint / "model.safetensors").write_bytes(b"fixture; not loaded by preparation")
    audio_root = tmp_path / "audio"
    audio_root.mkdir()

    def row(name, **extra):
        (audio_root / f"{name}.wav").write_bytes(name.encode())
        return {"wav_path": f"{name}.wav", "text": "hello world", "duration": "1.0", **extra}

    first = tmp_path / "train-en-1.jsonl"
    second = tmp_path / "train-en-2.jsonl"
    ml = tmp_path / "train-ml.jsonl"
    en_eval = tmp_path / "eval-en.jsonl"
    ml_eval = tmp_path / "eval-ml.jsonl"
    write_jsonl(
        first,
        [
            row("long", text="one two three four five six"),
            row("blank", text=""),
            row("duration", duration=9),
            row("a", text=None, transcript="hello world", language="wrong-row-language"),
            row("b"),
        ],
    )
    write_jsonl(second, [row("c"), row("d")])
    write_jsonl(ml, [row("m1"), row("m2")])
    write_jsonl(en_eval, [row("e1"), row("e2")])
    write_jsonl(ml_eval, [row("me1"), row("me2")])
    config = {
        "model_name_or_path": str(checkpoint),
        "audio_root": str(audio_root),
        "train_manifest": {"en": [str(first), str(second)], "ml": str(ml)},
        "eval_manifest": {"en": str(en_eval), "ml": str(ml_eval)},
        "min_duration_seconds": 0.1,
        "max_duration_seconds": 2,
        "max_target_length": 4,
        "learning_rate": 4e-5,
        "llm_lr_factor": 0.2,
        "seed": 42,
        "data_seed": 17,
        "resume_from_checkpoint": "/do/not/resume/this",
        "output_dir": "/do/not/write/production/output",
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config, row


def test_inspect_real_v81_configs_reads_source_paths_without_scanning():
    pytest.importorskip("qasr")
    repo = Path(__file__).resolve().parents[2]
    for filename in ("02_full_10node.yaml", "02_full_12node.yaml"):
        result = inspect_training_config(repo / "configs/v8.1" / filename)
        assert result["train_manifests"] == 22
        assert result["eval_manifests"] == 6
        assert result["languages"] == ["ar", "en", "hi", "ml", "zh"]
        assert result["checkpoint"].endswith("/models/bck/full/v1")
        assert result["test_panel"] is None
        assert result["non_speech_manifests"] == 0
        musan = inspect_training_config(repo / "configs/v8.1" / filename, musan=True)
        assert musan["musan_directory"].endswith("/stt/musan")
        assert musan["musan_partition"] == "noise_only"


def test_config_builds_capped_panels_plans_and_smoke_without_touching_inputs(configured, tmp_path):
    path, config, _ = configured
    originals = {p: file_hash(p) for p in tmp_path.glob("*.jsonl")}
    originals[path] = file_hash(path)
    out = tmp_path / "lab"
    result = prepare_from_training_config(
        path,
        out,
        train_per_language=2,
        eval_per_language=1,
        hash_audio=True,
    )
    assert result["selected_train"] == 4  # cap spans both English manifests
    assert result["selected_validation"] == 2
    assert result["experiments"] == 8
    assert result["seeds"] == [42, 43]
    assert result["test_panel"] is None
    assert not (out / "test-panel").exists()
    assert not (out / "non-speech-panel").exists()
    metadata, train = load_panel(out / "train-panel")
    _, evaluation = load_panel(out / "validation-panel")
    assert all(row["language"] in ("en", "ml") for row in train)
    assert all(0.1 <= row["duration"] <= 2 for row in train)
    assert all(row["text"] == "hello world" for row in train)
    assert all(
        row["audio_sha256"] == file_hash(row["audio_filepath"]) for row in train + evaluation
    )
    assert metadata["provenance"]["selection_seed"] == 17
    base = load_experiment(out / "seed-42/control.json")
    assert base["checkpoint"] == config["model_name_or_path"]
    assert base["encoder_lr"] == base["projector_lr"] == 4e-5
    assert base["decoder_lr"] == pytest.approx(8e-6)
    assert base["min_audio_seconds"] == 0.1
    assert base["max_audio_seconds"] == 2
    assert base["max_target_tokens"] == 4
    assert base["batch_size"] == 1 and base["gradient_accumulation_steps"] == 8
    assert "resume_from_checkpoint" not in base
    assert Path(base["output"]).is_relative_to(out)
    ctc = load_experiment(out / "seed-42/ctc.json")
    assert ctc["comparison_baseline"] == "upper_encoder"
    assert read_json(out / "eval-initial.json")["backend"] == "qasr"
    assert read_json(out / "eval-ctc-42.json")["backend"] == "qasr-lab"
    smoke = load_experiment(out / "smoke.json")
    assert smoke["max_steps"] == 2
    assert smoke["train_panel"] == str(out / "smoke-train-panel")
    for file, before in originals.items():
        assert file_hash(file) == before
    with pytest.raises(FileExistsError):
        prepare_from_training_config(path, out)


def test_validation_overlap_is_checked_against_training_rows_outside_selected_panel(
    configured, tmp_path
):
    path, config, _ = configured
    audio_root = Path(config["audio_root"])
    # With a cap of one, the highest-ranked row is absent from the training panel.
    worst = max(
        ("a", "b", "c", "d"),
        key=lambda name: int(digest([17, digest(str(audio_root / f"{name}.wav"))[:24]]), 16),
    )
    Path(config["eval_manifest"]["en"]).write_text(
        json.dumps(
            {
                "wav_path": f"{worst}.wav",
                "text": "hello world",
                "duration": 1,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="training/validation overlap"):
        prepare_from_training_config(
            path, tmp_path / "overlap", train_per_language=1, eval_per_language=1
        )
    assert (tmp_path / "overlap/failure.json").is_file()
    assert not (tmp_path / "overlap/train-panel").exists()


def test_non_speech_is_only_included_when_configured(configured, tmp_path):
    path, config, row = configured
    silence_manifest = tmp_path / "silence.jsonl"
    write_jsonl(silence_manifest, [row("silence", text="")])
    config["non_speech_manifest"] = {"en": str(silence_manifest)}
    path.write_text(yaml.safe_dump(config))
    result = prepare_from_training_config(path, tmp_path / "lab", seeds=[42])
    assert result["selected_non_speech"] == 1
    assert result["experiments"] == 5
    _, silence = load_panel(tmp_path / "lab/non-speech-panel")
    assert silence[0]["text"] == ""
    assert load_experiment(tmp_path / "lab/seed-42/non_speech.json")["non_speech_panel"]


def test_cli_inspect_and_missing_eval_fail_explicitly(configured, capsys):
    path, config, _ = configured
    assert main(["from-training-config", "--config", str(path), "--inspect"]) == 0
    assert json.loads(capsys.readouterr().out)["train_manifests"] == 3
    assert main(["from-training-config", "--config", str(path)]) == 2
    config["eval_manifest"] = None
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="explicit eval_manifest"):
        inspect_training_config(path)


def test_musan_from_yaml_adds_training_variant_and_separate_noise_diagnostics(configured, tmp_path):
    from .test_musan import make_corpus

    path, config, _ = configured
    root = tmp_path / "musan"
    make_corpus(root)
    config["augmentation"] = {"noise_injection": {"enabled": True, "noise_dir": str(root)}}
    path.write_text(yaml.safe_dump(config))
    original_config = file_hash(path)
    out = tmp_path / "lab"
    result = prepare_from_training_config(
        path, out, seeds=[42], musan=True, non_speech_per_language=2
    )
    assert result["experiments"] == 5
    assert result["selected_non_speech"] == result["selected_non_speech_validation"] == 4
    assert result["non_speech_manifests"] == 0  # Source YAML remains unchanged.
    assert file_hash(path) == original_config
    _, speech = load_panel(out / "validation-panel")
    _, noise = load_panel(out / "non-speech-validation-panel")
    assert all(row["text"] for row in speech)
    assert all(row["text"] == "" for row in noise)
    variant = load_experiment(out / "seed-42/non_speech.json")
    assert variant["comparison_baseline"] == "control"
    assert variant["non_speech_panel"] == str(out / "non-speech-panel")
    assert "non_speech_panel" not in load_experiment(out / "seed-42/control.json")
    assert result["test_panel"] is None
    crop = read_json(out / "musan-audio/provenance.json")["crops"][0]
    with Path(config["train_manifest"]["en"][0]).open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "audio_filepath": crop["source"]["path"],
                    "duration": 0.3,
                    "text": "incorrectly transcribed noise",
                }
            )
            + "\n"
        )
    # Original source paths are checked even though the panels point to derived crops.
    with pytest.raises(ValueError, match="training/validation overlap"):
        prepare_from_training_config(
            path, tmp_path / "overlap", seeds=[42], musan=True, non_speech_per_language=2
        )


def test_musan_requires_source_and_cannot_mix_with_explicit_manifest(configured):
    path, config, _ = configured
    with pytest.raises(ValueError, match="requires augmentation"):
        inspect_training_config(path, musan=True)
    config["non_speech_manifest"] = {"en": "/not/scanned.jsonl"}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="either --musan"):
        prepare_from_training_config(path, path.parent / "out", musan=True)
