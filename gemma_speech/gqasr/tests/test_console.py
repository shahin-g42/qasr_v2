import copy
import json

from gqasr.console import format_record, print_record


def sample():
    return {
        "step": 1042, "rank0_last_microbatch_loss": 1.798, "elapsed_seconds": 6016,
        "rank0_last_microbatch_augmentation": [
            {"triggered": True, "applied": True, "reason": "mixed", "pair": "ar-en"},
            {"triggered": False, "applied": False, "reason": "gate"},
        ],
        "validation": {
            "strict/all": {"utterances": 460, "wer": .2343, "cer": .1231},
            "strict/language:zh": {"utterances": 100, "wer": 1.2277, "cer": .3459,
                                   "word_errors": 124, "words": 101},
            "arabic_normalized/language:ar": {"utterances": 100, "wer": .281, "cer": None},
        },
    }


def test_validation_table_preserves_values_and_record():
    record = sample()
    original = copy.deepcopy(record)
    output = format_record(record)
    for expected in ("VALIDATION COMPLETE", "460 utterances", "122.77%", "34.59%",
                     "124/101", "28.10%", "n/a", "ar-en=1", "mixed=1/2", "1,042"):
        assert expected in output
    assert record == original
    assert "prioritize CER" in output


def test_training_without_validation(monkeypatch, capsys):
    record = sample()
    del record["validation"]
    record["learning_rates"] = {"decoder": 1e-5}
    record["rank0_gradient_norm"] = 2.0
    monkeypatch.delenv("GQASR_CONSOLE_FORMAT", raising=False)
    print_record(record)
    output = capsys.readouterr().out
    assert "VALIDATION COMPLETE" not in output
    assert "decoder=1e-05" in output
    assert "before clipping" in output


def test_raw_json_mode(monkeypatch, capsys):
    monkeypatch.setenv("GQASR_CONSOLE_FORMAT", "json")
    print_record(sample())
    assert json.loads(capsys.readouterr().out) == sample()
