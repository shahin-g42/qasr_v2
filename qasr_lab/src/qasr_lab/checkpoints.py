"""Read-only checkpoint fingerprints and post-conversion tensor comparisons."""

from __future__ import annotations

import re
import sys
from contextlib import ExitStack
from pathlib import Path

from .io import digest, file_hash, read_json


def use_qasr_source(path=None):
    if path is not None:
        root = Path(path).expanduser().resolve()
        source = root / "src" if (root / "src" / "qasr").is_dir() else root
        if not (source / "qasr" / "__init__.py").is_file():
            raise ValueError(f"No QASR source package under {path}")
        existing = sys.modules.get("qasr")
        if existing and Path(existing.__file__).resolve().parent != source / "qasr":
            raise RuntimeError("A different QASR checkout is already imported; use a fresh process")
        sys.path.insert(0, str(source))
    try:
        import qasr
    except ImportError as exc:
        raise ImportError("Install QASR separately or set qasr_source to its checkout") from exc
    return str(Path(qasr.__file__).resolve())


def checkpoint_identity(checkpoint, *, revision=None, hash_weights=True):
    path = Path(checkpoint).expanduser()
    if not path.exists():
        if path.is_absolute() or checkpoint.startswith("."):
            raise FileNotFoundError(checkpoint)
        if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("Hub models require a pinned 40-character commit revision")
        return {"kind": "hub", "repository": checkpoint, "revision": revision}
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"Expected a checkpoint directory with config.json: {path}")
    files = {}
    for entry in sorted(path.iterdir()):
        if entry.is_file() and entry.suffix in (
            ".json",
            ".jinja",
            ".safetensors",
            ".model",
            ".txt",
        ):
            is_weight = entry.suffix == ".safetensors"
            files[entry.name] = (
                {"sha256": file_hash(entry), "size": entry.stat().st_size}
                if hash_weights or not is_weight
                else {"size": entry.stat().st_size, "mtime_ns": entry.stat().st_mtime_ns}
            )
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("This package requires safetensors checkpoints")
    return {
        "kind": "local",
        "path": str(path.resolve()),
        "files": files,
        "weights_sha256_verified": hash_weights,
        "fingerprint": digest(files),
    }


def _tensor_files(path):
    from safetensors import safe_open

    result = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                result[name] = file
    if not result:
        raise ValueError(f"No safetensors found in {path}")
    return result


def audit_transfer(target, *, encoder=None, decoder=None, qasr_source=None):
    """Compare final destination tensors to source, including the tied head alias."""
    import torch
    from safetensors import safe_open

    target = Path(target)
    config = read_json(target / "config.json")
    if config.get("model_type") != "qasr":
        raise ValueError("Transfer audits expect a QASR destination")
    destination = _tensor_files(target)
    report = {
        "schema": "qasr-lab-transfer-audit-v1",
        "target": str(target.resolve()),
        "checks": [],
        "mismatches": [],
        "verified_tensors": 0,
    }
    with ExitStack() as stack:
        handles = {}

        def tensor(index, key):
            path = index[key]
            if path not in handles:
                handles[path] = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            return handles[path].get_tensor(key)

        def compare(index, source_key, target_key, component):
            report["verified_tensors"] += 1
            if target_key not in destination:
                report["mismatches"].append(
                    {
                        "component": component,
                        "source": source_key,
                        "target": target_key,
                        "reason": "missing",
                    }
                )
                return
            original, converted = tensor(index, source_key), tensor(destination, target_key)
            # Conversion intentionally casts to the destination dtype.
            if original.shape != converted.shape or not torch.equal(
                original.to(converted.dtype), converted
            ):
                report["mismatches"].append(
                    {
                        "component": component,
                        "source": source_key,
                        "target": target_key,
                        "reason": "value_or_shape",
                    }
                )

        if encoder:
            use_qasr_source(qasr_source)
            from qasr.convert_weights import _convert_encoder_state

            source = _tensor_files(encoder)
            prefix = next(
                (
                    p
                    for p in ("model.encoder.", "encoder.", "model.audio_tower.", "audio_tower.")
                    if any(k.startswith(p) for k in source)
                ),
                None,
            )
            if prefix is None:
                raise ValueError("Cannot identify source encoder tensor prefix")
            mappings = {}
            for key in source:
                if key.startswith(prefix):
                    renamed = next(iter(_convert_encoder_state({key[len(prefix) :]: None})))
                    if renamed in mappings:
                        raise ValueError("Encoder source key collision")
                    mappings[renamed] = key
            for renamed, key in mappings.items():
                compare(source, key, "model.audio_tower." + renamed, "encoder")
            expected = {
                k.removeprefix("model.audio_tower.")
                for k in destination
                if k.startswith("model.audio_tower.")
            }
            for missing in expected - mappings.keys():
                report["mismatches"].append(
                    {"component": "encoder", "reason": "unverified_destination", "target": missing}
                )
            report["checks"].append("encoder")
        if decoder:
            source = _tensor_files(decoder)
            prefix = next(
                (
                    p
                    for p in (
                        "thinker.model.",
                        "model.language_model.",
                        "language_model.",
                        "model.",
                    )
                    if p + "embed_tokens.weight" in source
                ),
                None,
            )
            if prefix is None:
                raise ValueError("Cannot identify source decoder tensor prefix")
            mappings = {k[len(prefix) :]: k for k in source if k.startswith(prefix)}
            expected = {
                k.removeprefix("model.language_model.")
                for k in destination
                if k.startswith("model.language_model.")
            }
            for renamed, key in mappings.items():
                compare(source, key, "model.language_model." + renamed, "decoder")
            for missing in expected - mappings.keys():
                report["mismatches"].append(
                    {"component": "decoder", "reason": "unverified_destination", "target": missing}
                )
            head = next(
                (key for key in ("thinker.lm_head.weight", "lm_head.weight") if key in source), None
            )
            source_config = read_json(Path(decoder) / "config.json")
            nested = source_config.get("thinker_config", source_config)
            source_tied = nested.get(
                "tie_word_embeddings",
                nested.get("text_config", {}).get("tie_word_embeddings", False),
            )
            if head is None:
                if not source_tied:
                    raise ValueError("Source has no LM head and does not declare tied embeddings")
                head = prefix + "embed_tokens.weight"
            destination_head = "lm_head.weight"
            if destination_head not in destination and config.get("tie_word_embeddings"):
                destination_head = "model.language_model.embed_tokens.weight"
            compare(source, head, destination_head, "lm_head")
            report["checks"].append("decoder_and_final_head")
    report["passed"] = bool(report["checks"]) and not report["mismatches"]
    report["scope"] = (
        "Final tensor values after dtype conversion; forward numerical parity is a separate check."
    )
    return report
