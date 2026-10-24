"""Adjust batching for an existing run without importing newer training defaults."""

import argparse
import copy
import json
from pathlib import Path

import yaml


def prepare(
    run, output, batch_size=40, exclude_languages=("zh",), train_microbatch=None, accumulation=None
):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Evaluation batch size must be a positive integer")
    receipt = json.loads((Path(run) / "training_recipe.json").read_text())
    training = copy.deepcopy(receipt["resolved"])
    manifests = training.get("eval_manifest")
    if not isinstance(manifests, dict):
        raise ValueError("Expected a language-keyed eval_manifest")
    unknown = set(exclude_languages) - set(manifests)
    if unknown:
        raise ValueError(f"Unknown evaluation languages: {sorted(unknown)}")
    training["eval_manifest"] = {
        language: paths
        for language, paths in manifests.items()
        if language not in exclude_languages
    }
    if not training["eval_manifest"]:
        raise ValueError("Keep at least one evaluation language")
    training["per_device_eval_batch_size"] = batch_size
    if (train_microbatch is None) != (accumulation is None):
        raise ValueError("Specify both train_microbatch and accumulation")
    if train_microbatch is not None:
        if any(type(n) is not int or n < 1 for n in (train_microbatch, accumulation)):
            raise ValueError("Training microbatch and accumulation must be positive integers")
        previous_batch = (
            training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
        )
        if train_microbatch * accumulation != previous_batch:
            raise ValueError("Resume must preserve examples per rank per optimizer update")
        training["per_device_train_batch_size"] = train_microbatch
        training["gradient_accumulation_steps"] = accumulation
    training["resume_from_checkpoint"] = None
    recipe = dict(data_recipe=receipt["data_recipe"], stage=receipt["stage"], training=training)
    if receipt.get("robustness"):
        recipe["robustness"] = receipt["robustness"]
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        yaml.safe_dump(recipe, stream, sort_keys=False, allow_unicode=True)
    return dict(
        config=str(output),
        eval_batch_size=batch_size,
        eval_languages=list(training["eval_manifest"]),
        train_languages=list(training["train_manifest"]),
        train_microbatch=training["per_device_train_batch_size"],
        accumulation=training["gradient_accumulation_steps"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing run containing training_recipe.json")
    parser.add_argument("--output", required=True, help="New YAML file; refuses to overwrite")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--exclude-language", action="append", default=None)
    parser.add_argument("--train-microbatch", type=int)
    parser.add_argument("--accumulation", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.run,
                args.output,
                args.batch_size,
                args.exclude_language or ["zh"],
                args.train_microbatch,
                args.accumulation,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
