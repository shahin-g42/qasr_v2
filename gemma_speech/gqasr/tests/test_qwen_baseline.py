import json
import sqlite3

import pytest
from gqasr.qwen_baseline import select_samples, summary


def fixture(tmp_path):
    audio = tmp_path / "a.wav"
    audio.touch()
    index = tmp_path / "index.sqlite"
    with sqlite3.connect(index) as db:
        db.execute("CREATE TABLE samples (split,language,ordinal,audio,text,duration,source,line)")
        db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)",
                   ("eval", "zh", 7, str(audio), "你好。", 2.0, "manifest.jsonl", 9))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(dict(id="eval:zh:7", language="zh",
                                           reference="你好。", hypothesis="Hello.")) + "\n")
    return index, predictions


def test_exact_eval_selection(tmp_path):
    index, predictions = fixture(tmp_path)
    rows = select_samples(index, predictions)
    assert rows[0]["id"] == "eval:zh:7"
    assert rows[0]["gqasr_hypothesis"] == "Hello."
    assert rows[0]["source_line"] == 9


def test_reference_mismatch_rejected(tmp_path):
    index, predictions = fixture(tmp_path)
    row = json.loads(predictions.read_text())
    row["reference"] = "错误"
    predictions.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="reference mismatch"):
        select_samples(index, predictions)


def test_duplicate_ids_rejected(tmp_path):
    index, predictions = fixture(tmp_path)
    predictions.write_text(predictions.read_text() * 2)
    with pytest.raises(ValueError, match="unique"):
        select_samples(index, predictions)


def test_punctuation_diagnostic_does_not_replace_strict_score():
    result = summary([dict(id="eval:zh:1", language="zh", reference="你好。", hypothesis="你好")])
    assert result["strict"]["cer"] > 0
    assert result["punctuation_insensitive"]["cer"] == 0
    assert result["no_han_predictions"] == 0
