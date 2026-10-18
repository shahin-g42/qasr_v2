"""New rows are validated while the existing GQASR index stays byte-identical."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from gqasr.data import ManifestIndex, scan
from tokenizers import Tokenizer, models, pre_tokenizers


def setup(tmp_path):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(backend_tokenizer=backend))
    cfg = {"index_path": str(tmp_path / "old.sqlite"), "train_manifest": {}, "eval_manifest": {},
               "audio_root": None, "audio_path_rewrites": {}, "min_duration_seconds": .1,
               "max_duration_seconds": 35, "max_target_length": 512, "language_batch_counts": {"ar": 1, "en": 1}}
    for split in ("train", "eval"):
        for lang in ("ar", "en"):
            audio = tmp_path / f"{split}-{lang}.wav"
            sf.write(audio, np.ones(8000, dtype=np.float32) * .1, 16000)
            manifest = tmp_path / f"{split}-{lang}.jsonl"
            manifest.write_text(json.dumps({"audio_filepath": str(audio), "text": "hello", "duration": .5,
                                               "recording_id": f"{split}-{lang}"}) + "\n")
            cfg[split + "_manifest"][lang] = [str(manifest)]
    scan(cfg, processor, verify_audio=True)
    old = cfg["index_path"]
    cfg["index_path"] = str(tmp_path / "new.sqlite")
    audio = tmp_path / "addition.wav"
    sf.write(audio, np.ones(8000, dtype=np.float32) * .2, 16000)
    manifest = tmp_path / "addition.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": str(audio), "text": "new dialect", "duration": .5,
                                       "recording_id": "new-video", "dialect": "DZ"}) + "\n")
    cfg["train_manifest"]["ar"].append(str(manifest))
    return cfg, processor, old


def test_append_preserves_old_rows_ids_and_bytes_without_old_sources(tmp_path):
    cfg, processor, old = setup(tmp_path)
    before = hashlib.sha256(Path(old).read_bytes()).hexdigest()
    old_row = ManifestIndex(old, "train")[("ar", 1)]
    for prefix in ("train", "eval"):
        for path in tmp_path.glob(prefix + "-*"):
            path.unlink()  # a rescan/old-audio validation would fail
    result = scan(cfg, processor, verify_audio=True, append_from=old)
    assert result["counts"] == {"train": {"ar": 2, "en": 1}, "eval": {"ar": 1, "en": 1}}
    updated = ManifestIndex(cfg["index_path"], "train")
    assert updated[("ar", 1)] == old_row
    assert updated[("ar", 2)]["dialect"] == "DZ"
    assert updated[("ar", 2)]["text"] == "new dialect"
    assert hashlib.sha256(Path(old).read_bytes()).hexdigest() == before
    assert result["audio_headers_checked"] is True
    assert result["data_contract"]["train_manifest"] == cfg["train_manifest"]
    with pytest.raises(FileExistsError):
        scan(cfg, processor, append_from=old)


@pytest.mark.parametrize("change", ["eval", "filters", "order", "tokenizer"])
def test_append_refuses_incompatible_contract(tmp_path, change):
    cfg, processor, old = setup(tmp_path)
    if change == "eval":
        cfg["eval_manifest"]["ar"] = []
    elif change == "filters":
        cfg["max_target_length"] = 256
    elif change == "order":
        cfg["train_manifest"]["ar"].reverse()
    else:
        processor = copy.deepcopy(processor)
        processor.tokenizer.backend_tokenizer.add_tokens(["new"])
    with pytest.raises(ValueError):
        scan(cfg, processor, append_from=old)
    assert not (tmp_path / "new.sqlite").exists()
    assert not (tmp_path / "new.sqlite.scan.lock").exists()


def test_append_checks_recording_leakage_against_retained_eval(tmp_path):
    cfg, processor, old = setup(tmp_path)
    path = tmp_path / "addition.jsonl"
    row = json.loads(path.read_text())
    row["recording_id"] = "eval-en"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="recording leakage"):
        scan(cfg, processor, verify_audio=True, append_from=old)
    assert not (tmp_path / "new.sqlite").exists()
    assert ManifestIndex(old, "train").counts["ar"] == 1
