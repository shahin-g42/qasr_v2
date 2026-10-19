"""Regression coverage for inherited Brotli/httpx2 decoder incompatibility."""

import importlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from data_processing.arabic_reference import hub_transport


@pytest.mark.parametrize("codec_name", ["brotli", "brotlicffi"])
def test_real_http_streaming_decoder_round_trip(monkeypatch, codec_name):
    from packaging.version import Version

    client = pytest.importorskip("httpx2")
    codec = pytest.importorskip(codec_name)
    if Version(codec.__version__) < Version("1.2.0"):
        pytest.skip("Use the public inference requirements for the real codec check")
    from huggingface_hub.utils import _http

    decoder = importlib.import_module("httpx2._decoders")
    monkeypatch.setattr(_http, "httpx2", client, raising=False)
    monkeypatch.setattr(decoder, "brotli", codec)
    result = hub_transport.probe_transport()
    assert result["brotli_probe"] == "passed"
    assert result["brotli_backend"] == codec_name
    assert result["decoded_bytes"] == 2 * 1024 * 1024


def test_incompatible_decoder_reports_real_import_path(monkeypatch):
    client = pytest.importorskip("httpx2")
    from huggingface_hub.utils import _http

    decoder = importlib.import_module("httpx2._decoders")

    class LegacyDecompressor:
        def decompress(self, data):
            return b"old decoder does not accept output_buffer_limit"

    old = SimpleNamespace(
        __name__="brotli",
        __version__="1.1.0",
        __file__="/base-env/brotli.py",
        Decompressor=LegacyDecompressor,
        compress=lambda data: b"compressed",
        error=RuntimeError,
    )
    monkeypatch.setattr(_http, "httpx2", client, raising=False)
    monkeypatch.setattr(decoder, "brotli", old)
    with pytest.raises(ValueError, match="output_buffer_limit") as exc:
        hub_transport.probe_transport()
    assert "/base-env/brotli.py" in str(exc.value)


@pytest.mark.parametrize("exit_code", [1, -11])
def test_failed_probe_names_current_interpreter_repair(monkeypatch, exit_code):
    monkeypatch.setattr(hub_transport.sys, "executable", "/inference env/bin/python")
    monkeypatch.setattr(
        hub_transport.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=exit_code, stdout="", stderr="decoder failed"),
    )
    with pytest.raises(ValueError, match="Keep the model cache") as exc:
        hub_transport.check_hub_transport()
    assert "'/inference env/bin/python' -m pip install" in str(exc.value)
    assert f"exited {exit_code}" in str(exc.value)


def test_probe_timeout_does_not_hang_download_startup(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(args[0], 30)

    monkeypatch.setattr(hub_transport.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="timed out"):
        hub_transport.check_hub_transport()


def test_successful_child_probe_returns_diagnostics(monkeypatch):
    result = {"status": "passed", "backend": "httpx2", "brotli_probe": "passed"}
    monkeypatch.setattr(
        hub_transport.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr=""),
    )
    assert hub_transport.check_hub_transport() == result


@pytest.mark.parametrize("stage", ["transport", "download", "check", "publish", "all"])
def test_transport_failure_stops_before_plan_audio_or_hub_access(monkeypatch, stage):
    from data_processing.arabic_reference import public_benchmark

    def failed():
        raise ValueError("HTTP decoder incompatible")

    def unexpected(*args):
        pytest.fail("Must reject the broken decoder before reading benchmark inputs")

    monkeypatch.setattr(public_benchmark, "check_hub_transport", failed)
    monkeypatch.setattr(public_benchmark, "load_plan", unexpected)
    monkeypatch.delenv("ARABIC_REF_PLAN", raising=False)
    with pytest.raises(ValueError, match="HTTP decoder incompatible"):
        public_benchmark.main([stage])


def test_transport_stage_needs_no_plan_or_gpu(monkeypatch):
    from data_processing.arabic_reference import public_benchmark

    monkeypatch.setattr(public_benchmark, "check_hub_transport", lambda: {"status": "passed"})
    monkeypatch.delenv("ARABIC_REF_PLAN", raising=False)
    public_benchmark.main(["transport"])
