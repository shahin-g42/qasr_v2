"""Versioned output-prefix extension; native ASR targets remain intact after speech."""

import json
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AddedToken

TOKENS = {name: f"<|audio_{name}|>" for name in ("speech", "music", "noise", "silence")}


def extend_audio_types(model, processor):
    if model.config.audio_type_version:
        raise ValueError("This model already has audio-type tokens")
    tokenizer = processor.tokenizer
    original_vocab = tokenizer.get_vocab()
    if any(token in original_vocab for token in TOKENS.values()):
        raise ValueError("Audio-type token collision in source tokenizer")
    # Seed new rows from the pretrained representation of each category word.
    seeds = {name: tokenizer.encode(name, add_special_tokens=False) for name in TOKENS}
    if any(not ids for ids in seeds.values()):
        raise ValueError("Category names must tokenize before vocabulary extension")
    original_size = model.aud_decoder.get_input_embeddings().num_embeddings
    tokenizer.add_special_tokens(
        {
            "extra_special_tokens": [
                AddedToken(t, special=True, normalized=False) for t in TOKENS.values()
            ]
        },
        replace_extra_special_tokens=False,
    )
    ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in TOKENS.items()}
    size = max(original_size, max(ids.values()) + 1)
    # Never shrink a padded pretrained vocabulary. Existing assigned IDs are stable.
    model.aud_decoder.resize_token_embeddings(size, mean_resizing=False)
    with torch.no_grad():
        for weights in (
            model.aud_decoder.get_input_embeddings().weight,
            model.aud_decoder.get_output_embeddings().weight,
        ):
            seed_rows = {
                name: weights[index].float().mean(0).to(weights.dtype)
                for name, index in seeds.items()
            }
            for name, index in ids.items():
                weights[index].copy_(seed_rows[name])
    model.config.text_config.vocab_size = size
    model.config.audio_type_version = 1
    model.config.audio_type_token_ids = ids
    processor.config = model.config
    model.config.requires_alignment = True
    return dict(
        audio_type_version=1,
        tokens=TOKENS,
        token_ids=ids,
        original_vocab_size=original_size,
        vocab_size=size,
        initialization="mean of pretrained token rows for the category word",
    )


def upgrade(model_path, output):
    from .export import sha256, write_json
    from .modeling import AUDForConditionalGeneration
    from .processing import AUDProcessor

    source, output = Path(model_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    model = AUDForConditionalGeneration.from_pretrained(source, local_files_only=True, dtype="auto")
    processor = AUDProcessor.from_pretrained(source)
    report = extend_audio_types(model, processor)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        model.save_pretrained(staging)
        processor.save_pretrained(staging)
        for name in ("licenses", "NOTICE.md"):
            if (source / name).is_dir():
                shutil.copytree(source / name, staging / name)
            elif (source / name).is_file():
                shutil.copy2(source / name, staging / name)
        source_provenance = source / "provenance.json"
        write_json(
            staging / "provenance.json",
            dict(
                kind="aud_audio_type_upgrade",
                source=str(source),
                extension=report,
                source_provenance=json.loads(source_provenance.read_text())
                if source_provenance.exists()
                else None,
                source_weights={p.name: sha256(p) for p in source.glob("*.safetensors")},
                scope="Weights-only extension. Requires decoder training; no optimizer state imported.",
            ),
        )
        write_json(
            staging / "checksums.json",
            {
                p.relative_to(staging).as_posix(): sha256(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        )
        # Verify both tensor and tokenizer reload before publishing the directory.
        AUDForConditionalGeneration.from_pretrained(staging, local_files_only=True, dtype="auto")
        AUDProcessor.from_pretrained(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(output=str(output), **report)
