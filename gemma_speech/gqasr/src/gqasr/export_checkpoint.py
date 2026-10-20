"""Publish a trained GQASR checkpoint as a standalone, weights-only artifact."""

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def progress(phase, **values):
    print(json.dumps({"event": "checkpoint_export_progress", "phase": phase, **values}), flush=True)


def copy_weights(source, destination):
    """Copy without dtype conversion or loading the optimizer/model into RAM."""
    before = source.stat()
    hasher = hashlib.sha256()
    copied, last = 0, time.monotonic()
    progress("copying_weights", copied_bytes=0, total_bytes=before.st_size)
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(16 * 1024 * 1024):
            writer.write(chunk)
            hasher.update(chunk)
            copied += len(chunk)
            now = time.monotonic()
            if now - last >= 5:
                progress("copying_weights", copied_bytes=copied, total_bytes=before.st_size)
                last = now
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ) or copied != before.st_size:
        raise ValueError("Source weights changed during export; use a completed checkpoint")
    return hasher.hexdigest()


def validate_tensors(path):
    """Bound memory to one CPU tensor; no GPU or full model construction."""
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        if not keys or not all(key.startswith("gqasr_") for key in keys):
            raise ValueError("Expected a nonempty GQASR tensor namespace")
        key_set, unbounded_clips = set(keys), []
        for i, key in enumerate(keys, 1):
            value = tensors.get_tensor(key)
            if not torch.isfinite(value).all().item():
                # ClippableLinear stores scalar +/-inf bounds to disable clipping.
                # These are intentional buffers, not learned weights. Preserve
                # them exactly; NaN, reversed infinities and weight inf still fail.
                parent, _, suffix = key.rpartition(".")
                expected = {
                    "input_min": -float("inf"), "output_min": -float("inf"),
                    "input_max": float("inf"), "output_max": float("inf"),
                }.get(suffix)
                if not (
                    key.startswith("gqasr_acoustic.core.")
                    and parent + ".linear.weight" in key_set
                    and value.ndim == 0 and expected is not None
                    and value.item() == expected
                ):
                    raise ValueError(f"Nonfinite model tensor: {key}")
                unbounded_clips.append(key)
            del value
            if i % 100 == 0 or i == len(keys):
                progress("checking_finite_weights", checked=i, total=len(keys))
    return len(keys), unbounded_clips


def export_checkpoint(checkpoint, output):
    """Copy trained weights/assets atomically; never read training_state.pt."""
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Export already exists; choose a new output: {output}")
    if output.is_relative_to(checkpoint) or checkpoint.is_relative_to(output):
        raise ValueError("Export and source checkpoint must be separate directories")
    required_files = ("model.safetensors", "config.json", "preprocessor_config.json",
                      "provenance.json", "MODEL_CARD.md")
    for name in required_files:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    for name in ("tokenizer", "licenses"):
        if not (checkpoint / name).is_dir():
            raise FileNotFoundError(checkpoint / name)
    if not (checkpoint / "tokenizer/tokenizer.json").is_file():
        raise FileNotFoundError(checkpoint / "tokenizer/tokenizer.json")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "gqasr" or config.get("format_version") != 1:
        raise ValueError("Expected GQASR format_version=1")
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    source_step = provenance.get("training", {}).get("completed_optimizer_steps")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A competing exporter must not publish into the same destination.
    lock = output.parent / f".{output.name}.export.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    staging = None
    try:
        if output.exists():
            raise FileExistsError(output)
        staging = Path(tempfile.mkdtemp(prefix=".gqasr-checkpoint-export-", dir=output.parent))
        artifact = staging / "artifact"
        artifact.mkdir()
        weights_hash = copy_weights(
            checkpoint / "model.safetensors", artifact / "model.safetensors"
        )
        progress("copying_assets")
        for name in ("config.json", "preprocessor_config.json", "provenance.json"):
            shutil.copyfile(checkpoint / name, artifact / name)
        for name in ("tokenizer", "licenses"):
            shutil.copytree(checkpoint / name, artifact / name)
        shutil.copyfile(checkpoint / "MODEL_CARD.md", artifact / "SOURCE_MODEL_CARD.md")
        if (checkpoint / "training_config.json").is_file():
            shutil.copyfile(checkpoint / "training_config.json",
                            artifact / "source_training_config.json")
        progress("checking_copied_weights")
        if digest(artifact / "model.safetensors") != weights_hash:
            raise ValueError("Copied model checksum differs from source")
        tensor_count, unbounded_clips = validate_tensors(artifact / "model.safetensors")
        record = {
            "kind": "weights-only export of a trained checkpoint; not speech quality evaluation",
            "source_checkpoint": str(checkpoint),
            "source_optimizer_updates": source_step,
            "weights_sha256": weights_hash,
            "tensor_count": tensor_count,
            "finite_weights": True,
            "expected_infinite_clip_bounds": unbounded_clips,
            "weights_reencoded": False,
            "optimizer_state_included": False,
            "rng_state_included": False,
        }
        provenance["checkpoint_export"] = record
        (artifact / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (artifact / "checkpoint_export.json").write_text(json.dumps(record, indent=2) + "\n")
        (artifact / "MODEL_CARD.md").write_text(
            "# GQASR trained checkpoint export\n\n"
            f"Standalone model exported from `{checkpoint}`"
            + (f" after {source_step} optimizer updates" if source_step is not None else "")
            + ". Model weights, tokenizer and audio frontend are preserved without conversion.\n\n"
            "This artifact contains no optimizer, scheduler or RNG state. Training from it "
            "without `--resume` starts a new optimizer and update schedule. Export integrity "
            "does not establish speech quality or cluster health.\n\n"
            "Upstream attribution and applicable terms remain in provenance.json and licenses/. "
            "SOURCE_MODEL_CARD.md preserves the earlier artifact's model card.\n"
        )
        checksums = {
            str(path.relative_to(artifact)): (
                weights_hash if path.name == "model.safetensors" else digest(path)
            )
            for path in artifact.rglob("*") if path.is_file()
        }
        (artifact / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(artifact, output)
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink()
    return {**record, "output": str(output), "checksums_written": True}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output")
    destination.add_argument("--config", help="Export to this training config's base_model path")
    args = parser.parse_args()
    if args.config:
        from .config import load_config
        args.output = load_config(args.config)["base_model"]
    # This command has no GPU work and should not oversubscribe a cluster node.
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
