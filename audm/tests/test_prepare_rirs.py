import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audm import prepare_rirs as prep
from audm.robustness import RobustAugmenter


def wav(values=None, rate=16000):
    if values is None:
        values = np.zeros((1600, 2), dtype=np.float32)
        values[10] = [0.5, -0.5]
        values[100] = [0.1, -0.1]
    buffer = io.BytesIO()
    sf.write(buffer, values, rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def archive(tmp_path, monkeypatch, entries):
    path = tmp_path / "raw/rirs_noises.zip"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as out:
        for name, value in entries.items():
            out.writestr(name, value)
    monkeypatch.setattr(prep, "ARCHIVE_MD5", hashlib.md5(path.read_bytes()).hexdigest())
    return path


def test_prepare_rirs_preserves_channels_and_excludes_noise(tmp_path, monkeypatch):
    archive(
        tmp_path,
        monkeypatch,
        {
            "RIRS_NOISES/simulated_rirs/smallroom/a.wav": wav(),
            "RIRS_NOISES/simulated_rirs/mediumroom/a.wav": wav(),
            "RIRS_NOISES/simulated_rirs/largeroom/a.wav": wav(),
            "RIRS_NOISES/real_rirs_isotropic_noises/example_rir.wav": wav(),
            "RIRS_NOISES/real_rirs_isotropic_noises/example_noise.wav": wav(),
            "RIRS_NOISES/pointsource_noises/example.wav": wav(),
            "RIRS_NOISES/README": b"original attribution",
        },
    )
    result = prep.prepare(tmp_path)
    rows = [json.loads(s) for s in Path(result["manifest"]).read_text().splitlines()]
    assert len(rows) == 8
    assert {r["kind"] for r in rows} == {"real", "smallroom", "mediumroom", "largeroom"}
    assert len({r["audio_filepath"] for r in rows}) == 8
    for row in rows:
        audio, rate = sf.read(row["audio_filepath"])
        assert audio.ndim == 1 and rate == 16000
        assert audio[10] == (0.5 if row["source_channel"] == 0 else -0.5)
    assert (
        tmp_path / "prepared-v1/source_metadata/RIRS_NOISES/README"
    ).read_bytes() == b"original attribution"
    assert prep.verify(tmp_path)["verified_rirs"] == 8
    augmenter = RobustAugmenter(
        {"clean_probability": 0, "effects": {"rir": {"p": 1, "manifest": result["manifest"]}}}, {}
    )
    assert augmenter.scan_assets()["rirs"] == 8
    wave = np.random.default_rng(42).normal(0, 0.01, 16000).astype(np.float32)
    output = augmenter.augment_waveform(wave, max_audio_seconds=2)
    assert np.isfinite(output).all() and len(output) > len(wave)
    with pytest.raises(FileExistsError):
        prep.prepare(tmp_path)
    Path(rows[0]["audio_filepath"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prep.verify(tmp_path)


@pytest.mark.parametrize(
    "member", ["../bad.wav", "/bad.wav", "RIRS_NOISES/../bad.wav", "a\\bad.wav"]
)
def test_unsafe_zip_paths_rejected(tmp_path, monkeypatch, member):
    archive(tmp_path, monkeypatch, {member: wav()})
    with pytest.raises(ValueError, match="Unsafe"):
        prep.prepare(tmp_path)
    assert not (tmp_path / "prepared-v1").exists()


@pytest.mark.parametrize("audio", [wav(rate=8000), wav(np.zeros((100, 1))), b"corrupt"])
def test_invalid_audio_never_publishes_manifest(tmp_path, monkeypatch, audio):
    archive(tmp_path, monkeypatch, {"RIRS_NOISES/simulated_rirs/smallroom/a.wav": audio})
    with pytest.raises((ValueError, RuntimeError)):
        prep.prepare(tmp_path)
    assert not (tmp_path / "prepared-v1").exists()


def test_download_checksum_and_resume_arguments(tmp_path, monkeypatch):
    payload = b"test archive payload"
    monkeypatch.setattr(prep, "ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    calls = []

    def run(cmd, check):
        calls.append(cmd)
        assert check and "--continue-at" in cmd and "--fail" in cmd
        Path(cmd[cmd.index("--output") + 1]).write_bytes(payload)

    monkeypatch.setattr(prep.subprocess, "run", run)
    report = prep.download(tmp_path)
    assert report["sha256"] == hashlib.sha256(payload).hexdigest()
    prep.download(tmp_path)
    assert len(calls) == 1
    (tmp_path / "raw/rirs_noises.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prep.download(tmp_path)


def test_completed_partial_is_published_without_network(tmp_path, monkeypatch):
    payload = b"finished download"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "rirs_noises.zip.part").write_bytes(payload)
    monkeypatch.setattr(prep, "ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    monkeypatch.setattr(prep.subprocess, "run", lambda *a, **kw: pytest.fail("No network needed"))
    prep.download(tmp_path)
    assert (raw / "rirs_noises.zip").read_bytes() == payload
    assert not (raw / "rirs_noises.zip.part").exists()
