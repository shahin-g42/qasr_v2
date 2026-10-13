from types import SimpleNamespace

import pytest

from qasr_lab.backends import evaluate_backend, validate_model_config
from qasr_lab.io import file_hash, write_jsonl
from qasr_lab.panel import load_panel, prepare_panel


def test_model_config_rejects_unknown_fields_and_unsafe_defaults():
    with pytest.raises(ValueError, match="Unknown"):
        validate_model_config({"id": "a", "checkpoint": "/a", "backend": "qasr", "typo": 1})
    with pytest.raises(ValueError, match="explicit language"):
        validate_model_config(
            {"id": "a", "checkpoint": "/a", "backend": "cohere", "language_hints": False}
        )


def test_audio_hashes_are_computed_and_fail_changed_audio_before_inference(tmp_path):
    file = tmp_path / "audio.wav"
    file.write_bytes(b"mock audio")
    manifest = tmp_path / "input.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "sample",
                "audio_filepath": str(file),
                "text": "hello",
                "duration": 1,
                "language": "en",
            }
        ],
    )
    prepare_panel([manifest], tmp_path / "panel", hash_audio=True)
    _, samples = load_panel(tmp_path / "panel")
    assert samples[0]["audio_sha256"] == file_hash(file)
    file.write_bytes(b"changed audio")
    backend = SimpleNamespace(
        config={"warmup_samples": 0}, sync=lambda: None, reset_memory=lambda: None
    )
    predictions = list(evaluate_backend(backend, samples))
    assert predictions[0]["status"] == "error"
    assert "checksum" in predictions[0]["error"]


def test_backend_failure_remains_an_explicit_prediction(monkeypatch):
    import qasr_lab.backends as module

    def bad_audio(_):
        raise RuntimeError("broken decoder")

    monkeypatch.setattr(module, "load_audio", bad_audio)
    backend = SimpleNamespace(
        config={"warmup_samples": 0}, sync=lambda: None, reset_memory=lambda: None
    )
    result = list(evaluate_backend(backend, [{"id": "a", "audio_filepath": "/missing"}]))
    assert result == [{"id": "a", "status": "error", "error": "RuntimeError: broken decoder"}]
