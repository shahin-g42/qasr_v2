import hashlib
import json

import pytest
from gqasr.data import ManifestIndex, scan
from gqasr.processing import GQASRProcessor
from gqasr.rescan import rescan_language
from test_training import fixture


def setup_index(tmp_path):
    _, cfg = fixture(tmp_path)
    cfg["language_batch_counts"]["ml"] = 1
    for split in ("train", "eval"):
        audio = tmp_path / f"ml-{split}.wav"
        audio.write_bytes((tmp_path / f"{split}.wav").read_bytes())
        manifest = tmp_path / f"ml-{split}.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": str(audio), "text": "old", "duration": 0.3}) + "\n"
        )
        if split == "train":
            extra = tmp_path / "quarantined.wav"
            extra.write_bytes(audio.read_bytes())
            with manifest.open("a") as stream:
                stream.write(
                    json.dumps({"audio_filepath": str(extra), "text": "damaged", "duration": 0.3})
                    + "\n"
                )
        cfg[split + "_manifest"]["ml"] = [str(manifest)]
    cfg["index_path"] = str(tmp_path / "all.sqlite")
    processor = GQASRProcessor(tmp_path / "base")
    scan(cfg, processor, True)
    source = cfg["index_path"]
    cfg["index_path"] = str(tmp_path / "new.sqlite")
    return cfg, processor, source


def test_language_replacement_retains_other_records_without_reading_them(tmp_path):
    cfg, processor, source = setup_index(tmp_path)
    source_hash = hashlib.sha256(open(source, "rb").read()).hexdigest()
    old = ManifestIndex(source, "train")[("ar", 1)]
    replacement = tmp_path / "corrected.jsonl"
    replacement.write_text(
        json.dumps(
            {"audio_filepath": str(tmp_path / "ml-train.wav"), "text": "corrected", "duration": 0.3}
        )
        + "\n"
    )
    cfg["train_manifest"]["ml"] = [str(replacement)]
    # Other-language sources are deliberately unavailable: their completed index
    # entries are reused, not revalidated. Full preflight still checks file presence.
    (tmp_path / "train.jsonl").unlink()
    result = rescan_language(cfg, processor, source, workers=1)
    assert result["counts"]["train"] == {"ar": 1, "ml": 1}
    assert result["language_rescan"]["previous_counts"]["train"]["ml"] == 2
    with pytest.raises(IndexError):
        ManifestIndex(cfg["index_path"], "train")[("ml", 2)]
    assert ManifestIndex(cfg["index_path"], "train")[("ar", 1)] == old
    assert ManifestIndex(cfg["index_path"], "train")[("ml", 1)]["text"] == "corrected"
    assert hashlib.sha256(open(source, "rb").read()).hexdigest() == source_hash
    assert result["audio_headers_checked"]
    assert str(replacement) in {s["path"] for s in result["sources"]}
    assert str(tmp_path / "ml-train.jsonl") not in {s["path"] for s in result["sources"]}
    with pytest.raises(FileExistsError):
        rescan_language(cfg, processor, source, workers=1)


def test_rescan_refuses_changed_global_contract(tmp_path):
    cfg, processor, source = setup_index(tmp_path)
    cfg["max_target_length"] += 1
    with pytest.raises(ValueError, match="full scan"):
        rescan_language(cfg, processor, source, workers=1)
    assert not (tmp_path / "new.sqlite").exists()


def test_rescan_detects_overlap_with_retained_language_and_cleans_up(tmp_path):
    cfg, processor, source = setup_index(tmp_path)
    (tmp_path / "ml-train.jsonl").write_text(
        json.dumps(
            {"audio_filepath": str(tmp_path / "train.wav"), "text": "bad overlap", "duration": 0.3}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="overlap"):
        rescan_language(cfg, processor, source, workers=1)
    assert not (tmp_path / "new.sqlite").exists()
    assert not (tmp_path / "new.sqlite.scan.lock").exists()
    assert ManifestIndex(source, "train")[("ml", 1)]["text"] == "old"
