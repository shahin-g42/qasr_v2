"""Exercise remote preparation with real WAV, tar and Parquet byte streams."""
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from gqasr.arabic_benchmark.common import DATASETS, load_config, read_rows

from gqasr.arabic_benchmark import data, remote


def wav_bytes(seconds=.1):
    f = io.BytesIO()
    sf.write(f, np.sin(np.arange(round(seconds * 16000)) / 10) * .1, 16000, format="WAV")
    return f.getvalue()


def row(ds, ordinal, name, duration=.1):
    return {"id": f"{ds}:{ordinal:06d}", "dataset": ds, "ordinal": ordinal,
            "relative_audio": name, "text": "المرجع الرسمي", "duration": duration,
            "upstream_id": name.removesuffix(".wav"), "metadata": {}}


def fixture_sources(tmp_path, monkeypatch):
    originals = [row("sada", 0, "recording-seg_0_100-0_200.wav"),
                 row("commonvoice", 0, "common_voice_ar_1.wav"),
                 row("masc_clean", 0, "video_a_0.1.wav"),
                 row("masc_noisy", 0, "video_b_0.2.wav"),
                 row("masc_noisy", 1, "video_b_0.2.wav"),  # Official duplicate retained.
                 row("mgb2", 0, "segment_one.wav"),
                 row("casablanca", 0, "casablanca_one.wav")]
    locations = {}

    def put(source, path, content):
        spec = remote.SOURCES[source]
        locations[(spec["repository"], path)] = content

    put("sada", "batch_1/recording.wav", wav_bytes(.3))
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        info = tarfile.TarInfo("clips/common_voice_ar_1.mp3")
        payload = wav_bytes()
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    put("commonvoice", "audio/ar/test/ar_test_0.tar", tar_buffer.getvalue())
    mapping = io.StringIO()
    writer = csv.DictWriter(mapping, fieldnames="video_id,start,end,duration,text,type,file_path".split(","))
    writer.writeheader()
    for video, start, kind, filename in (("video_a", .1, "c", "aaa.wav"),
                                        ("video_b", .2, "n", "bbb.wav"),
                                        ("video_b", .2, "n", "ccc.wav")):
        writer.writerow(dict(video_id=video, start=start, end=start + .1, duration=.1,
                             text="المرجع الرسمي", type=kind, file_path=filename))
    spec = remote.SOURCES["masc"]
    locations[(spec["metadata_repository"], spec["metadata_path"])] = mapping.getvalue().encode()
    for source, records, path in (
        ("masc", [{"audio": {"path": p, "bytes": wav_bytes()}}
                  for p in ("aaa.wav", "bbb.wav", "ccc.wav")], "data/test-00000.parquet"),
        ("mgb2", [{"segment_id": "segment_one", "audio": {"bytes": wav_bytes()}}], "data/test-00000.parquet"),
        ("casablanca", [{"seg_id": "casablanca_one", "gender": "male",
                         "audio": {"bytes": wav_bytes()}}], "Egypt/test-00000.parquet"),
    ):
        f = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(records), f, row_group_size=1)
        put(source, path, f.getvalue())
    monkeypatch.setattr(remote, "fetch_references", lambda *args: None)
    monkeypatch.setattr(remote, "official_rows", lambda work, datasets: [r for r in originals if r["dataset"] in datasets])
    monkeypatch.setattr(data, "official_rows", lambda work, datasets: [r for r in originals if r["dataset"] in datasets])
    monkeypatch.setattr(remote, "open_remote", lambda spec, path: io.BytesIO(locations[(spec["repository"], path)]))
    monkeypatch.setattr(remote, "source_files", lambda spec: sorted(
        path for repo, path in locations if repo == spec["repository"] and
        remote.fnmatch.fnmatch(path, spec["pattern"])))
    return originals, locations


def test_all_six_streamed_sources_exact_coverage_and_resume(tmp_path, monkeypatch):
    originals, _ = fixture_sources(tmp_path, monkeypatch)
    result = remote.stream_prepare(tmp_path, workers=3)
    assert result["samples"] == 7
    assert result["counts"] == {"sada": 1, "commonvoice": 1, "masc_clean": 1,
                                "masc_noisy": 2, "mgb2": 1, "casablanca": 1}
    assert result["full_selected_datasets_prepared"]
    assert not result["model_weights_loaded"]
    prepared = data.prepared_rows({"work_dir": str(tmp_path)})
    for original, actual in zip(originals, prepared, strict=True):
        assert all(actual[k] == v for k, v in original.items() if k != "metadata")
        assert np.load(actual["audio"]).shape == (1600,)
        assert "repository" in actual["source"]
    duplicated = [r for r in prepared if r["dataset"] == "masc_noisy"]
    assert duplicated[0]["audio_sha256"] == duplicated[1]["audio_sha256"]
    monkeypatch.setattr(remote, "_decode_payload", lambda *args: pytest.fail("Resume decoded cached audio"))
    repeated = remote.stream_prepare(tmp_path, workers=3)
    assert repeated["reused"] == 7


def test_stream_smoke_separate_from_full_inputs_and_cache_repair(tmp_path, monkeypatch):
    fixture_sources(tmp_path, monkeypatch)
    result = remote.stream_prepare(tmp_path, workers=2, limit_per_dataset=1)
    assert result["samples"] == 6
    assert not result["full_selected_datasets_prepared"]
    assert not (tmp_path / "prepared").exists()
    manifest = Path(result["prepared_directory"]) / "mgb2.jsonl"
    item = read_rows(manifest)[0]
    Path(item["audio"]).write_bytes(b"corrupt")
    rerun = remote.stream_prepare(tmp_path, workers=2, limit_per_dataset=1)
    assert rerun["reused"] == 5
    assert np.load(item["audio"]).shape == (1600,)


@pytest.mark.parametrize("invalid", ["missing", "duplicate"])
def test_remote_parquet_id_errors_are_fatal(tmp_path, monkeypatch, invalid):
    _, locations = fixture_sources(tmp_path, monkeypatch)
    key = (remote.SOURCES["mgb2"]["repository"], "data/test-00000.parquet")
    records = [{"segment_id": "wrong" if invalid == "missing" else "segment_one",
                "audio": {"bytes": wav_bytes()}}] * (2 if invalid == "duplicate" else 1)
    f = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), f)
    locations[key] = f.getvalue()
    with pytest.raises(RuntimeError, match="No dataset was silently omitted"):
        remote.stream_prepare(tmp_path, datasets=["mgb2"], workers=2)
    assert not (tmp_path / "prepared/mgb2-provenance.json").exists()


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "symlink"])
def test_tar_errors_remain_fatal_even_after_partial_cache(tmp_path, monkeypatch, invalid):
    _, locations = fixture_sources(tmp_path, monkeypatch)
    f = io.BytesIO()
    with tarfile.open(fileobj=f, mode="w") as tar:
        for _ in range(2 if invalid == "duplicate" else 1):
            info = tarfile.TarInfo("other.mp3" if invalid == "missing" else "common_voice_ar_1.mp3")
            if invalid == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(info)
            else:
                payload = wav_bytes()
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    locations[(remote.SOURCES["commonvoice"]["repository"], "audio/ar/test/ar_test_0.tar")] = f.getvalue()
    for _ in range(2):
        with pytest.raises(ValueError, match="Duplicate or invalid|missing"):
            remote.stream_prepare(tmp_path, datasets=["commonvoice"], workers=1)
        assert not (tmp_path / "prepared/commonvoice-provenance.json").exists()


def test_masc_mapping_rejects_ambiguous_boundaries_and_changed_reference(tmp_path, monkeypatch):
    _, locations = fixture_sources(tmp_path, monkeypatch)
    spec = remote.SOURCES["masc"]
    key = (spec["metadata_repository"], spec["metadata_path"])
    original = locations[key]
    for bad in (original.replace(b"0.30000000000000004,0.1", b"0.4,0.1", 1),
                original.replace("المرجع الرسمي".encode(), b"wrong text", 1)):
        locations[key] = bad
        with pytest.raises(RuntimeError):
            remote.stream_prepare(tmp_path, datasets=["masc_clean", "masc_noisy"], workers=1)


def test_remote_access_failure_is_explicit_and_redacted(tmp_path, monkeypatch):
    fixture_sources(tmp_path, monkeypatch)

    def inaccessible(spec, path):
        raise PermissionError("Authorization: Bearer SECRET -- signed URL")

    monkeypatch.setattr(remote, "open_remote", inaccessible)
    result = remote.check_sources(tmp_path, datasets=["commonvoice"])
    assert not result["ready"]
    assert "SECRET" not in json.dumps(result)
    assert "hf auth login" in result["sources"][0]["action"]


def test_streaming_config_needs_no_local_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv("GQASR_BENCH_DATA", raising=False)
    for key in ("WORK", "CHECKPOINT", "OUTPUT"):
        monkeypatch.setenv(f"GQASR_BENCH_{key}", str(tmp_path / key))
    path = Path(__file__).parents[1] / "configs/arabic_leaderboard_streaming.yaml"
    cfg = load_config(path)
    assert cfg["audio_input"] == "stream"
    assert set(cfg["datasets"]) == set(DATASETS)
    assert all("root" not in entry for entry in cfg["datasets"].values())


def test_stream_save_rejects_bad_duration_and_nonfinite_audio(tmp_path):
    item = {"row": row("mgb2", 0, "a.wav"), "source": {}, "metadata": {}}
    for wave in (np.zeros(16000), np.full(1600, np.nan), np.empty(0)):
        with pytest.raises(ValueError):
            remote._save(item, wave, tmp_path, .01)
    assert not list(tmp_path.rglob("*.npy"))


@pytest.mark.parametrize("stage", ["stream-check", "stream-smoke", "stream-prepare", "run"])
@pytest.mark.parametrize("custom_settings", [False, True])
def test_stream_launcher_selects_config_and_online_preparation(tmp_path, monkeypatch, stage, custom_settings):
    import os
    import shutil
    import subprocess
    import sys

    root = tmp_path / "gemma_speech"
    (root / "cluster").mkdir(parents=True)
    script = root / "cluster/gqasr_arabic_leaderboard.sh"
    shutil.copyfile(Path(__file__).parents[2] / "cluster" / script.name, script)
    binary = root / ".venv/bin"
    binary.mkdir(parents=True)
    (binary / "activate").write_text("")
    cli = binary / "gqasr-arabic-benchmark"
    cli.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                   "print(json.dumps({'args': sys.argv[1:], 'offline': os.environ['HF_HUB_OFFLINE']}))\n")
    cli.chmod(0o755)
    # Reproduce a caller with exported tuning settings (including QASR's test stage).
    settings = {"WORKERS_PER_GPU": "8", "BATCH_SIZE": "16", "STREAM_WORKERS": "3"}
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    env = {**os.environ, "GQASR_BENCH_INPUT": "stream", "GQASR_BENCH_WORK": str(tmp_path / "work"),
           "GEMMA_SPEECH_ENV": str(root / ".venv")}
    # Default assertions must not inherit the caller's benchmark configuration.
    for name in ("GQASR_BENCH_DATA", "GQASR_BENCH_CONFIG", *settings):
        env.pop(name, None)
    if custom_settings:
        env.update(settings)
    proc = subprocess.run(["bash", str(script), stage], env=env, capture_output=True, text=True, check=True)
    event = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    args = event["args"]
    if stage == "run":
        assert event["offline"] == "1"
        assert args[args.index("--config") + 1].endswith("arabic_leaderboard_streaming.yaml")
        assert args[args.index("--workers-per-gpu") + 1] == ("8" if custom_settings else "11")
        assert args[args.index("--batch-size") + 1] == ("16" if custom_settings else "1")
    else:
        assert event["offline"] == "0"
        assert args[args.index("--work-dir") + 1] == str(tmp_path / "work")
        assert args[args.index("--workers") + 1] == ("3" if custom_settings else "16")
