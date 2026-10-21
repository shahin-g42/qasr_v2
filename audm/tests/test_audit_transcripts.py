import json

import pytest
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from audm.audit_transcripts import audit


def setup(tmp_path):
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    model = tmp_path / "model"
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>").save_pretrained(model)
    source = tmp_path / "source.jsonl"
    rows = []
    for n, duration in [(512, 1), (513, 1), (1024, 1), (1025, 31), (1025, None)]:
        rows.append(
            dict(
                audio_filepath=f"relative/{n}-{duration}.wav",
                text=" ".join(["a"] * n),
                duration=duration,
                custom="preserved",
            )
        )
    rows.insert(1, dict(audio_filepath="skip.wav", text="!!!", duration=1))
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            dict(
                train_manifest={"ml": [str(source)]},
                min_duration_seconds=0.1,
                max_duration_seconds=35,
            )
        )
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(data_recipe=str(data), stage="adapter", training=dict(max_duration_seconds=30))
        )
    )
    return model, source, recipe


@pytest.mark.parametrize("batch_size", [1, 3, 1024])
def test_counts_boundaries_filter_and_source_preservation(tmp_path, batch_size):
    model, source, recipe = setup(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "audit"
    result = audit(recipe, model, output, batch_size=batch_size, filter_above=512)
    counts = result["totals"]
    assert counts["records"] == 6
    assert counts["unusable_text"] == 1
    assert counts["all_transcripts"] == 5
    assert counts["all_above_512"] == 4
    assert counts["all_above_1024"] == 2
    assert counts["duration_eligible_transcripts"] == 4
    assert counts["duration_eligible_above_512"] == 3
    assert counts["duration_eligible_above_1024"] == 1
    assert counts["duration_eligible_above_512_percent"] == 75
    assert counts["missing_duration"] == 1
    assert counts["filtered_out"] == 4
    assert counts["written"] == 2
    assert source.read_bytes() == original
    kept = [json.loads(line) for line in (output / "000-source.jsonl").read_text().splitlines()]
    assert kept[0]["audio_filepath"] == str(tmp_path / "relative/512-1.wav")
    assert kept[0]["custom"] == "preserved"
    assert kept[1]["text"] == "!!!"
    rejected = [json.loads(line) for line in (output / "over_limit.jsonl").read_text().splitlines()]
    assert [r["line"] for r in rejected] == [3, 4, 5, 6]
    assert (output / "report.json").is_file()
    with pytest.raises(FileExistsError):
        audit(recipe, model, output)


def test_audit_only_does_not_write_manifest(tmp_path):
    model, source, recipe = setup(tmp_path)
    report = audit(recipe, model, tmp_path / "audit")
    assert report["manifests"][0]["filtered_path"] is None
    assert report["totals"]["filtered_out"] == 0
    assert not (tmp_path / "audit/000-source.jsonl").exists()
