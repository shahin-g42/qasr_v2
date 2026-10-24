import copy
import json

import pytest
import yaml
from test_training import data_recipe

from audm.evaluation_recipe import prepare
from audm.train import resume_contract, validate_resume_recipe, validate_resume_snapshot
from audm.training_config import load_recipe


def fixture_recipe(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                data_recipe=str(data_recipe(tmp_path)),
                stage="full",
                training=dict(
                    model_name_or_path="/tmp/model",
                    output_dir=str(tmp_path / "run"),
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                ),
            )
        )
    )
    return recipe


def test_evaluation_recipe_preserves_training_and_resume_contract(tmp_path):
    recipe = fixture_recipe(tmp_path)
    old, _, receipt = load_recipe(recipe)
    run = tmp_path / "run"
    run.mkdir()
    (run / "training_recipe.json").write_text(json.dumps(receipt))
    output = tmp_path / "eval32.yaml"
    prepare(run, output, 32, ["en"])
    new, _, updated = load_recipe(output)
    assert new.per_device_eval_batch_size == 32
    assert set(new.eval_manifest) == {"ar"}
    assert new.train_manifest == old.train_manifest
    assert new.language_batch_counts == old.language_batch_counts
    assert resume_contract(updated) == resume_contract(receipt)
    validate_resume_recipe(receipt, updated)
    with pytest.raises(FileExistsError):
        prepare(run, output, 32, ["en"])
    for key, value in [
        ("learning_rate", 0.01),
        ("eval_steps", 17),
        ("train_manifest", {"ar": "/different"}),
        ("gradient_accumulation_steps", 8),
        ("label_smoothing", 0.2),
    ]:
        changed = copy.deepcopy(updated)
        changed["resolved"][key] = value
        assert resume_contract(changed) != resume_contract(receipt)
    changed = copy.deepcopy(updated)
    changed["resolved"]["eval_manifest"] = {"zh": old.eval_manifest["en"]}
    with pytest.raises(ValueError, match="adding/relabeling"):
        validate_resume_recipe(receipt, changed)


def test_only_unchanged_legacy_robustness_is_resume_compatible():
    previous = dict(
        resolved={},
        robustness_version=4,
        robustness={"augmentation": {"effects": {"gain": {"p": 0.2}}}},
    )
    current = copy.deepcopy(previous)
    current["robustness_version"] = 5
    assert resume_contract(previous) == resume_contract(current)
    current["robustness"]["augmentation"]["effects"]["silence"] = {"p": 0.2}
    assert resume_contract(previous) != resume_contract(current)
    current = copy.deepcopy(previous)
    current["robustness_version"] = 3
    assert resume_contract(previous) != resume_contract(current)


def test_resume_regrouping_preserves_update_size(tmp_path):
    recipe = fixture_recipe(tmp_path)
    _, _, receipt = load_recipe(recipe)
    receipt["resolved"].update(per_device_train_batch_size=10, gradient_accumulation_steps=8)
    run = tmp_path / "run"
    run.mkdir()
    (run / "training_recipe.json").write_text(json.dumps(receipt))
    output = tmp_path / "b40.yaml"
    prepare(run, output, 40, ["en"], train_microbatch=40, accumulation=2)
    config, _, updated = load_recipe(output)
    assert config.per_device_train_batch_size == 40
    assert config.gradient_accumulation_steps == 2
    validate_resume_recipe(receipt, updated)
    with pytest.raises(ValueError, match="preserve examples"):
        prepare(run, tmp_path / "wrong.yaml", 40, ["en"], train_microbatch=40, accumulation=1)


@pytest.mark.parametrize("excluded", [["typo"], "en", [3], ["ar", "en"]])
def test_invalid_evaluation_exclusions(tmp_path, excluded):
    recipe = fixture_recipe(tmp_path)
    content = yaml.safe_load(recipe.read_text())
    content["evaluation"] = {"exclude_languages": excluded}
    recipe.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError):
        load_recipe(recipe)


def test_snapshot_allows_removal_but_rejects_changes_and_new_files():
    old = {"train": {"size": 100}, "eval_ar": {"size": 10}, "eval_zh": {"size": 20}}
    validate_resume_snapshot(old, {k: v for k, v in old.items() if k != "eval_zh"})
    for current in [
        {**old, "train": {"size": 99}},
        {**old, "eval_ar": {"size": 9}},
        {**old, "new_eval": {"size": 1}},
    ]:
        with pytest.raises(ValueError, match="manifests changed"):
            validate_resume_snapshot(old, current)
