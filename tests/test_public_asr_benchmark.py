"""Public model adapters and exact-membership leaderboard submission contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gqasr.arabic_benchmark import public_backend as backend

from data_processing.arabic_reference.core import DATASETS, load_plan, prepare, sha256, write_jsonl
from data_processing.arabic_reference.leaderboard import build_leaderboard
from data_processing.arabic_reference.public_benchmark import (
    REGISTRY,
    bind_prepared,
    download_model,
    export_submissions,
    selected_models,
)


class Batch(dict):
    def to(self, *args):
        return self


def adapter(kind):
    obj = backend.recognizer_class(kind).__new__(backend.recognizer_class(kind))
    obj.torch = torch
    obj.device = "cpu"
    obj.model = SimpleNamespace(
        dtype=torch.float32,
        generation_config=SimpleNamespace(eos_token_id=9),
        config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4096)),
    )
    return obj


def samples(tmp_path):
    rows = []
    for i in range(2):
        path = tmp_path / f"{i}.npy"
        np.save(path, np.full(1600 + i, 0.1, dtype=np.float32))
        rows.append(
            {
                "id": f"sada:{i:06d}",
                "audio": str(path),
                "audio_sha256": sha256(path),
                "actual_duration": (1600 + i) / 16000,
                "text": "REFERENCE NEVER SENT",
                "ideal_text_draft": "DRAFT NEVER SENT",
            }
        )
    return rows


def test_raw_qwen_output_preserves_loops_and_never_prompts_with_references(tmp_path):
    obj = adapter("hf-qwen")
    rows = samples(tmp_path)
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return Batch(input_ids=torch.ones((2, 3), dtype=torch.long))

    obj.processor = SimpleNamespace(
        apply_transcription_request=request,
        tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: "نحن " * 375 + "<|im_end|>"),
    )
    obj.model.generate = lambda **kwargs: torch.tensor([[1, 1, 1, 7, 9, 0], [1, 1, 1, 7, 9, 0]])
    result = obj.infer(rows, obj.features(rows), "arabic", 10)
    assert all(len(r["pred_text"].split()) == 375 for r in result)
    assert all(
        r["eos_seen"] and not r["generation_limit_hit"] and r["generated_tokens"] == 2
        for r in result
    )
    assert calls[0]["prompt"] == "" and calls[0]["language"] == "Arabic"
    assert set(calls[0]) == {"audio", "language", "prompt", "processor_kwargs"}
    assert calls[0]["processor_kwargs"]["truncation"] is False
    assert np.array_equal(calls[0]["audio"][0], np.load(rows[0]["audio"]))


def test_cohere_native_chunk_expansion_returns_one_complete_transcript_per_clip(tmp_path):
    obj = adapter("cohere")
    rows = samples(tmp_path)
    calls = []

    class Processor:
        tokenizer = SimpleNamespace(
            batch_decode=lambda ids, **kwargs: ["مرحبا", "النص الثاني", "بكم"]
        )

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return Batch(
                decoder_input_ids=torch.ones((3, 2), dtype=torch.long),
                audio_chunk_index=[(0, 0), (1, None), (0, 1)],
            )

    obj.processor = Processor()

    def generate(**kwargs):
        assert "audio_chunk_index" not in kwargs
        return torch.tensor([[1, 1, 7, 9], [1, 1, 7, 9], [1, 1, 7, 7]])

    obj.model.generate = generate
    result = obj.infer(rows, obj.features(rows), "arabic", 2)
    assert result[0]["pred_text"] == "مرحبا بكم" and result[0]["audio_chunks"] == 2
    assert result[0]["generation_limit_hit"] and not result[1]["generation_limit_hit"]
    assert result[1]["pred_text"] == "النص الثاني"
    assert set(calls[0]) == {"audio", "sampling_rate", "return_tensors", "language", "punctuation"}
    assert calls[0]["punctuation"] is True and calls[0]["language"] == "ar"


@pytest.mark.parametrize(
    "mapping", [[(0, None), (0, None)], [(0, 0), (0, 2)], [(2, None), (0, None)], [(0, 0)]]
)
def test_invalid_chunk_mapping_cannot_silently_drop_or_reassign_audio(mapping):
    stats = backend.generation_stats([1, 9], 9, 3)
    with pytest.raises(ValueError):
        backend.reassemble_chunks(["a", "b"], [stats, stats], mapping, 2)


def test_empty_chunk_texts_are_preserved_as_empty_predictions():
    stats = backend.generation_stats([9], 9, 10)
    result = backend.reassemble_chunks(["", ""], [stats, stats], [(0, 0), (0, 1)], 1)
    assert result[0]["pred_text"] == "" and result[0]["audio_chunks"] == 2


def test_whisper_keeps_empty_predictions_and_marks_unobservable_token_limits(tmp_path):
    obj = adapter("whisper")
    rows = samples(tmp_path)
    calls = []

    def pipeline(audio, **kwargs):
        calls.append((audio, kwargs))
        return [{"text": ""}, {"text": "مرحبا"}]

    obj.pipeline = pipeline
    result = obj.infer(rows, obj.features(rows), "arabic", 440)
    assert result[0]["pred_text"] == "" and result[1]["pred_text"] == "مرحبا"
    assert all(r["generation_limit_hit"] is None and r["generated_tokens"] is None for r in result)
    assert calls[0][1]["generate_kwargs"]["language"] == "ar"
    assert calls[0][1]["generate_kwargs"]["task"] == "transcribe"
    with pytest.raises(ValueError, match="440"):
        obj.infer(rows, obj.features(rows), "arabic", 768)


@pytest.mark.parametrize("mutation", ["hash", "shape", "nan", "duration"])
def test_audio_identity_and_shape_checked_before_inference(tmp_path, mutation):
    obj = adapter("hf-qwen")
    rows = samples(tmp_path)
    if mutation == "hash":
        rows[0]["audio_sha256"] = "0" * 64
    elif mutation == "duration":
        rows[0]["actual_duration"] = 20
    else:
        np.save(
            rows[0]["audio"],
            np.full(
                (2, 2) if mutation == "shape" else (1600,),
                np.nan if mutation == "nan" else 0,
                dtype=np.float32,
            ),
        )
        rows[0]["audio_sha256"] = sha256(rows[0]["audio"])
    with pytest.raises(ValueError):
        obj.features(rows)


@pytest.fixture
def public_plan(tmp_path):
    source = []
    for ds in DATASETS:
        audio = tmp_path / f"{ds}.npy"
        np.save(audio, np.zeros(1600, dtype=np.float32))
        source.append(
            {
                "id": f"{ds}:000000",
                "dataset": ds,
                "ordinal": 0,
                "text": "مرحبا بكم",
                "pred_text": "مرحبا",
                "audio": str(audio),
                "audio_sha256": sha256(audio),
                "preparation_id": "fixture",
                "actual_duration": 0.1,
                "mode": "arabic",
                "metadata": {},
                "source": {},
                "generation_limit_hit": False,
                "generated_tokens": 1,
            }
        )
    original = tmp_path / "original.jsonl"
    write_jsonl(original, source)
    plan = tmp_path / "plan"
    prepare([("qasr", original), ("gqasr", original)], plan, full_suite=True)
    return plan, source


def make_public_run(root, source):
    from gqasr.arabic_benchmark.common import atomic_json

    name = "qwen3-asr-0.6b"
    folder = root / name / "full"
    folder.mkdir(parents=True)
    public = {**REGISTRY[name], "revision": "a" * 40, "postprocessing": "preserve generated text"}
    atomic_json(
        folder / "run.json",
        {
            "public_model": public,
            "modes": ["arabic"],
            "backend": "hf-qwen",
            "runtime": {"torch": "test", "transformers": "test"},
            "precision": "bf16",
            "max_new_tokens": 768,
        },
    )
    atomic_json(
        folder / "progress.json",
        {"event": "benchmark_complete", "completed": len(source), "total": len(source)},
    )
    # Empty output must be scored as deletions rather than rejecting the complete run.
    write_jsonl(folder / "predictions-arabic.jsonl", [{**r, "pred_text": ""} for r in source])
    return folder


def test_complete_public_submission_imports_empty_predictions_and_provenance(public_plan, tmp_path):
    plan_root, source = public_plan
    root = tmp_path / "public"
    make_public_run(root, source)
    plan, rows = load_plan(plan_root)
    submissions = export_submissions(root, ["qwen3-asr-0.6b"], plan, rows)
    output = tmp_path / "space"
    build_leaderboard(plan_root, output, submissions=submissions)
    data = json.loads((output / "leaderboard.json").read_text())
    assert (
        len(data["models"]) == 3 and data["model_details"]["qwen3-asr-0.6b"]["revision"] == "a" * 40
    )
    scored = [s for p in data["panels"] for s in p["scores"] if s["model"] == "qwen3-asr-0.6b"]
    assert scored and all(s["wer"] == 1 for s in scored)
    assert data["submission_manifest_sha256"] == sha256(submissions)
    assert all(e["predictions"]["qwen3-asr-0.6b"] == "" for e in data["examples"])


@pytest.mark.parametrize("mutation", ["partial", "wrong_audio", "missing", "duplicate"])
def test_incomplete_or_misaligned_public_results_cannot_enter_leaderboard(
    public_plan, tmp_path, mutation
):
    plan_root, source = public_plan
    root = tmp_path / "public"
    folder = make_public_run(root, source)
    plan, rows = load_plan(plan_root)
    if mutation == "partial":
        (folder / "progress.json").write_text(
            json.dumps({"event": "benchmark_progress", "completed": 5, "total": 6})
        )
    else:
        predictions = [{**r} for r in source]
        if mutation == "wrong_audio":
            predictions[0]["audio_sha256"] = "0" * 64
        if mutation == "missing":
            predictions.pop()
        if mutation == "duplicate":
            predictions.append(predictions[0])
        write_jsonl(folder / "predictions-arabic.jsonl", predictions)
    with pytest.raises(ValueError):
        export_submissions(root, ["qwen3-asr-0.6b"], plan, rows)


@pytest.mark.parametrize("mutation", ["predictions", "run", "revision", "plan"])
def test_public_submission_hash_and_model_binding_rechecked(public_plan, tmp_path, mutation):
    from data_processing.arabic_reference.submissions import read_submissions

    plan_root, source = public_plan
    root = tmp_path / "public"
    folder = make_public_run(root, source)
    plan, rows = load_plan(plan_root)
    manifest = export_submissions(root, ["qwen3-asr-0.6b"], plan, rows)
    if mutation == "predictions":
        (folder / "predictions-arabic.jsonl").write_text("changed")
    if mutation == "run":
        (folder / "run.json").write_text("{}")
    if mutation in ("revision", "plan"):
        data = json.loads(manifest.read_text())
        if mutation == "revision":
            data["models"][0]["revision"] = "b" * 40
        else:
            data["manifest_digest"] = "0" * 64
        manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_submissions(manifest, plan)


@pytest.mark.parametrize("layout", ["separate", "combined"])
def test_download_pins_revision_and_reuses_only_unchanged_snapshot(tmp_path, layout):
    calls = []

    def download(repository, **kwargs):
        root = __import__("pathlib").Path(kwargs["local_dir"])
        root.mkdir(parents=True)
        calls.append(kwargs)
        (root / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))
        if layout == "separate":
            (root / "preprocessor_config.json").write_text('{"sampling_rate":16000}')
        else:
            (root / "processor_config.json").write_text(
                json.dumps(
                    {
                        "processor_class": "Qwen3ASRProcessor",
                        "feature_extractor": {
                            "feature_extractor_type": "Qwen3ASRFeatureExtractor",
                            "sampling_rate": 16000,
                            "feature_size": 128,
                        },
                    }
                )
            )
        (root / "tokenizer.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"fixture weights")

    api = SimpleNamespace(model_info=lambda repo: SimpleNamespace(sha="a" * 40))
    first = download_model("qwen3-asr-0.6b", tmp_path, api=api, download=download)
    assert first["revision"] == "a" * 40 and calls[0]["revision"] == "a" * 40
    assert download_model("qwen3-asr-0.6b", tmp_path, api=api, download=download) == first
    assert len(calls) == 1
    (tmp_path / "qwen3-asr-0.6b/snapshot/model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        download_model("qwen3-asr-0.6b", tmp_path, api=api, download=download)


@pytest.mark.parametrize("kind", backend.BACKENDS)
@pytest.mark.parametrize("layout", ["separate", "combined"])
def test_checkpoint_accepts_native_frontend_layouts_without_rewriting(tmp_path, kind, layout):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": backend.MODEL_TYPES[kind]}))
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    frontend = {"sampling_rate": 16000, "feature_size": 128}
    config = tmp_path / (
        "processor_config.json" if layout == "combined" else "preprocessor_config.json"
    )
    config.write_text(
        json.dumps({"feature_extractor": frontend} if layout == "combined" else frontend)
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    identity = backend.checkpoint_identity(tmp_path, kind)
    assert identity[config.name] == sha256(config)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    frontend["feature_size"] = 80
    config.write_text(
        json.dumps({"feature_extractor": frontend} if layout == "combined" else frontend)
    )
    assert backend.checkpoint_identity(tmp_path, kind) != identity


@pytest.mark.parametrize("frontend", [None, {}, [], "frontend.json", {"sampling_rate": 8000}])
def test_invalid_embedded_frontend_cannot_bypass_validation(tmp_path, frontend):
    (tmp_path / "processor_config.json").write_text(json.dumps({"feature_extractor": frontend}))
    with pytest.raises(ValueError, match="frontend"):
        backend.frontend_config(tmp_path)


def test_checkpoint_errors_identify_missing_component_and_snapshot(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    with pytest.raises(ValueError, match="safetensors weights missing") as exc:
        backend.checkpoint_identity(tmp_path, "hf-qwen")
    assert str(tmp_path) in str(exc.value)
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    # A processor class alone is not a saved audio frontend.
    (tmp_path / "processor_config.json").write_text('{"processor_class":"Qwen3ASRProcessor"}')
    with pytest.raises(ValueError, match="frontend missing") as exc:
        backend.checkpoint_identity(tmp_path, "hf-qwen")
    assert str(tmp_path) in str(exc.value)


def test_prepared_plan_binding_detects_wrong_cache(public_plan, monkeypatch):
    import data_processing.arabic_reference.public_benchmark as module

    plan_root, source = public_plan
    rows = load_plan(plan_root)[1]
    monkeypatch.setattr(module, "prepared_rows", lambda cfg: source)
    assert len(bind_prepared({}, rows, check_audio=True)) == 6
    source[0]["preparation_id"] = "foreign"
    with pytest.raises(ValueError, match="identity"):
        bind_prepared({}, rows)


@pytest.mark.parametrize("value", ["", "unknown", "cohere-arabic cohere-arabic"])
def test_invalid_model_selection_is_explicit(value):
    with pytest.raises(ValueError):
        selected_models(value)


@pytest.mark.parametrize("fail_inference", [False, True])
def test_all_runs_smoke_then_full_and_publishes_only_after_success(
    public_plan, tmp_path, monkeypatch, fail_inference
):
    from data_processing.arabic_reference import cluster_leaderboard
    from data_processing.arabic_reference import public_benchmark as module

    plan_root, _ = public_plan
    events = []
    environment = {
        "ARABIC_REF_PLAN": str(plan_root),
        "ARABIC_REF_ANNOTATIONS_BUNDLE": str(tmp_path / "annotations"),
        "PUBLIC_MODELS": "qwen3-asr-0.6b whisper-large-v3",
        "PUBLIC_MODEL_CACHE": str(tmp_path / "cache"),
        "PUBLIC_BENCH_ROOT": str(tmp_path / "runs"),
        "PUBLIC_PREPARED_WORK": str(tmp_path / "prepared"),
        "PUBLIC_GPUS": "0,1,2,3,4,5,6,7",
        "PUBLIC_WORKERS_PER_GPU": "1",
        "PUBLIC_BATCH_SIZE": "16",
        "ARABIC_REF_SPACE_BUNDLE": str(tmp_path / "board"),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(module, "annotation_drafts", lambda *args: None)
    monkeypatch.setattr(module, "check_runtime", lambda *args: events.append("runtime") or {})
    monkeypatch.setattr(module, "bind_prepared", lambda *args, **kw: events.append("audio"))
    monkeypatch.setattr(
        module,
        "download_model",
        lambda name, cache: events.append(("download", name))
        or {"repository": REGISTRY[name]["repository"], "revision": "a" * 40},
    )
    monkeypatch.setattr(module, "model_config", lambda name, *args: {"name": name})

    def run(cfg, **kwargs):
        assert kwargs["modes"] == ["arabic"] and kwargs["workers_per_gpu"] == 1
        assert kwargs["batch_size"] == 16 and kwargs["gpus"] == environment["PUBLIC_GPUS"]
        events.append((cfg["name"], kwargs["limit_per_dataset"]))
        if fail_inference:
            raise ValueError("Failed GPU smoke run")

    monkeypatch.setattr(module, "run", run)
    submissions = tmp_path / "submissions.json"
    monkeypatch.setattr(
        module, "export_submissions", lambda *args: events.append("export") or submissions
    )
    monkeypatch.setattr(cluster_leaderboard, "main", lambda args: events.append(("board", args)))
    monkeypatch.setattr(module, "write_comparison_report", lambda *args: events.append("report"))
    if fail_inference:
        with pytest.raises(ValueError, match="Failed GPU smoke"):
            module.main(["all"])
        assert "export" not in events and not any(e[0] == "board" for e in events)
    else:
        module.main(["all"])
        assert events[-7:] == [
            ("qwen3-asr-0.6b", 32),
            ("qwen3-asr-0.6b", 0),
            ("whisper-large-v3", 32),
            ("whisper-large-v3", 0),
            "export",
            ("board", ["all"]),
            "report",
        ]
        import os

        assert os.environ["ARABIC_REF_SUBMISSIONS"] == str(submissions)
    assert events[:2] == ["runtime", "audio"]
