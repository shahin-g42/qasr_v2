"""Auxiliary CTC at the native Gemma encoder rate; the causal decoder is unchanged."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def eligibility(labels, lengths):
    present = labels.ne(-100)
    count = present.sum(-1)
    repeats = ((labels[:, 1:] == labels[:, :-1]) & present[:, 1:] & present[:, :-1]).sum(-1)
    return (count > 0) & (lengths >= count + repeats)


def auxiliary_loss(hidden, head, mask, labels, denominator=None, blank_id=0):
    """Mean target-length-normalized CTC, chunked by example to bound vocabulary memory.

    Infeasible alignments and non-speech targets are excluded from CTC only.
    The denominator can span the entire distributed accumulation window.
    """
    lengths = mask.sum(-1)
    eligible = eligibility(labels, lengths)
    if type(blank_id) is not int or not 0 <= blank_id < head.out_features:
        raise ValueError("CTC blank ID is outside the head vocabulary")
    if (
        (labels != -100) & ((labels < 0) | (labels == blank_id) | (labels >= head.out_features))
    ).any():
        raise ValueError("CTC transcripts must exclude blank and out-of-vocabulary IDs")
    frame_lengths = lengths.tolist()
    # Keep a zero gradient for the head on ranks/windows with no CTC targets.
    total = hidden.sum() * 0 + head.weight.flatten()[0] * 0
    for index in eligible.nonzero().flatten().tolist():
        target = labels[index][labels[index] != -100]
        frames = hidden[index, : frame_lengths[index]]

        def loss_one(states, weight, target=target):
            with torch.autocast(device_type=states.device.type, enabled=False):
                log_probs = F.linear(states.float(), weight.float()).log_softmax(-1)
                value = F.ctc_loss(
                    log_probs[:, None],
                    target,
                    (states.shape[0],),
                    (target.numel(),),
                    blank=blank_id,
                    reduction="sum",
                    zero_infinity=False,
                )
                return value / target.numel()

        total = total + (
            checkpoint(loss_one, frames, head.weight, use_reentrant=False)
            if torch.is_grad_enabled()
            else loss_one(frames, head.weight)
        )
    divisor = eligible.sum() if denominator is None else denominator
    loss = total / torch.as_tensor(divisor, device=total.device).clamp_min(1)
    stats = torch.stack(
        (
            total.detach(),
            eligible.sum(),
            labels.ne(-100).any(-1).sum(),
            lengths.numel() * lengths.new_ones(()),
        )
    )
    return loss, stats


def train_tokenizer(config, output, vocab_size=16000):
    from qasr.data import parse_record
    from tqdm import tqdm

    from .prepare_non_speech import new_directory
    from .training_config import load_recipe

    if type(vocab_size) is not int or vocab_size < 258:
        raise ValueError("CTC vocabulary needs at least 256 byte symbols, blank and unknown")
    recipe, _, _ = load_recipe(config)
    counts = Counter()
    sources = []
    tokenizer = Tokenizer(models.BPE(unk_token="<ctc_unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    def texts():
        seen = set()
        for path, language in recipe.train_manifest_specs:
            path = Path(path).resolve()
            if path in seen:
                continue
            seen.add(path)
            digest = hashlib.sha256()
            identity = path.stat()
            with path.open("rb") as stream:
                for line_number, line in enumerate(
                    tqdm(stream, desc=f"CTC {language}: {path.name}", unit="records"), 1
                ):
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    try:
                        record = parse_record(
                            row,
                            manifest_path=path,
                            line_number=line_number,
                            audio_root=recipe.audio_root,
                            language=language,
                        )
                    except ValueError:
                        counts[f"{language}:invalid_text_or_metadata"] += 1
                        continue
                    duration = record["duration"]
                    if (
                        duration is not None
                        and not recipe.min_duration_seconds
                        <= duration
                        <= recipe.max_duration_seconds
                    ):
                        continue
                    counts[f"{language}:transcripts"] += 1
                    yield record["text"]
            final = path.stat()
            if (identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns) != (
                final.st_size,
                final.st_mtime_ns,
                final.st_ctime_ns,
            ):
                raise ValueError(f"Tokenizer source changed during scan: {path}")
            sources.append(dict(path=str(path), sha256=digest.hexdigest()))

    with new_directory(output) as (stage, destination):
        tokenizer.train_from_iterator(
            texts(),
            trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=["<ctc_blank>", "<ctc_unk>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=True,
            ),
        )
        if not sum(v for k, v in counts.items() if k.endswith(":transcripts")):
            raise ValueError("No usable CTC tokenizer transcripts")
        tokenizer.save(str(stage / "tokenizer.json"))
        report = dict(
            vocab_size=tokenizer.get_vocab_size(),
            blank_id=0,
            counts=counts,
            sources=sources,
            scope="Training transcripts only, no record cap; no audio decoded; separate from ASR tokenizer",
        )
        (stage / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return dict(output=str(destination), **report)


def decoder_tokenizer_digest(tokenizer):
    """Fingerprint segmentation/IDs, ignoring transient batch padding/truncation."""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    for key in ("padding", "truncation"):
        backend.pop(key, None)
    payload = dict(backend=backend, special_ids=sorted(tokenizer.all_special_ids))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_training_tokenizer(training_config, model_config):
    expected = training_config.aud_ctc.get("tokenizer")
    if training_config.ctc_loss_weight and expected == "decoder" and model_config.ctc_version != 2:
        raise ValueError("This recipe requires decoder-tokenizer CTC; run the new export first")


def attach_head(model, processor, tokenizer=None):
    config = model.config
    if (config.variant, config.flash_encoder, config.flash_audio_stride) != (
        "audflash",
        "gemma4",
        1,
    ):
        raise ValueError("Auxiliary CTC currently requires stride-one Gemma AudFlash")
    if config.ctc_vocab_size or config.ctc_version:
        raise ValueError("Source already contains a CTC head; do not overwrite it")
    if tokenizer is None:
        # Preserve every native decoder ID (including unused padded rows).
        # Blank is an extra CTC output class, never added to the decoder tokenizer.
        config.ctc_version = 2
        config.ctc_blank_id = config.text_config.vocab_size
        config.ctc_vocab_size = config.ctc_blank_id + 1
        config.ctc_tokenizer_sha256 = decoder_tokenizer_digest(processor.tokenizer)
        tokenizer = processor.tokenizer
    else:
        if tokenizer.token_to_id("<ctc_blank>") != 0:
            raise ValueError("CTC tokenizer must assign blank ID 0")
        config.ctc_version = 1
        config.ctc_blank_id = 0
        config.ctc_vocab_size = tokenizer.get_vocab_size()
        config.ctc_tokenizer_sha256 = hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
    reference = next(model.aud_encoder.parameters())
    model.aud_ctc_head = torch.nn.Linear(
        config.audio_config.output_proj_dims, config.ctc_vocab_size, bias=False
    ).to(reference)
    torch.nn.init.normal_(model.aud_ctc_head.weight, std=0.02)
    processor.config = config
    processor.ctc_tokenizer = tokenizer


def export(model_path, tokenizer_path=None, output=None):
    from .export import sha256
    from .modeling import AUDForConditionalGeneration
    from .processing import AUDProcessor

    if output is None:
        raise ValueError("Export requires a new output directory")
    source = Path(model_path).resolve()
    model = AUDForConditionalGeneration.from_pretrained(source, local_files_only=True, dtype="auto")
    from .verify import validate_model_tensors

    validate_model_tensors(model)
    processor = AUDProcessor.from_pretrained(source)
    tokenizer = (
        Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
        if tokenizer_path
        else None
    )
    torch.manual_seed(1729)
    attach_head(model, processor, tokenizer)
    report = dict(
        kind="aud_auxiliary_ctc",
        source=str(source),
        stride=1,
        ctc_version=model.config.ctc_version,
        ctc_tokenizer="decoder" if tokenizer is None else "separate_legacy",
        ctc_blank_id=model.config.ctc_blank_id,
        ctc_vocab_size=model.config.ctc_vocab_size,
        added_parameters=model.aud_ctc_head.weight.numel(),
        source_weights={p.name: sha256(p) for p in source.glob("*.safetensors")},
        optimizer_resume=False,
        quality_validated=False,
    )
    return _save_export(model, processor, source, output, report)


def inference_export(model_path, output):
    """Strip training-only CTC without modifying any inference-path tensor."""
    from .export import sha256
    from .modeling import AUDForConditionalGeneration
    from .processing import AUDProcessor
    from .verify import validate_model_tensors

    source = Path(model_path).resolve()
    model = AUDForConditionalGeneration.from_pretrained(source, local_files_only=True, dtype="auto")
    validate_model_tensors(model)
    processor = AUDProcessor.from_pretrained(source)
    if not model.config.ctc_version or model.aud_ctc_head is None:
        raise ValueError("Source has no auxiliary CTC head to remove")
    removed = sum(p.numel() for p in model.aud_ctc_head.parameters())
    model.aud_ctc_head = None
    model.config.ctc_version = 0
    model.config.ctc_vocab_size = 0
    model.config.ctc_blank_id = None
    model.config.ctc_tokenizer_sha256 = None
    processor.config = model.config
    processor.ctc_tokenizer = None
    report = dict(
        kind="aud_inference_export",
        source=str(source),
        removed_ctc_parameters=removed,
        parameters=sum(p.numel() for p in model.parameters()),
        inference_weights_unchanged=True,
        optimizer_resume=False,
        quality_validated=False,
        source_weights={p.name: sha256(p) for p in source.glob("*.safetensors")},
    )
    return _save_export(model, processor, source, output, report)


def _save_export(model, processor, source, output, report):
    from .export import sha256, write_json
    from .modeling import AUDForConditionalGeneration
    from .prepare_non_speech import new_directory
    from .processing import AUDProcessor

    with new_directory(output) as (stage, destination):
        model.save_pretrained(stage)
        processor.save_pretrained(stage)
        for name in ("NOTICE.md", "licenses"):
            if (source / name).is_dir():
                shutil.copytree(source / name, stage / name)
            elif (source / name).is_file():
                shutil.copy2(source / name, stage / name)
        write_json(stage / "provenance.json", report)
        write_json(
            stage / "checksums.json",
            {p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*") if p.is_file()},
        )
        reloaded, info = AUDForConditionalGeneration.from_pretrained(
            stage, local_files_only=True, dtype="auto", output_loading_info=True
        )
        if any(
            info.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"Non-strict CTC export reload: {info}")
        if model.state_dict().keys() != reloaded.state_dict().keys():
            raise ValueError("CTC export changed tensor names")
        for name, tensor in model.state_dict().items():
            if not torch.equal(tensor, reloaded.state_dict()[name]):
                raise ValueError(f"CTC export round-trip mismatch: {name}")
        AUDProcessor.from_pretrained(stage)
    return dict(output=str(destination), **report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tok = commands.add_parser("tokenizer")
    tok.add_argument("--config", required=True)
    tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=16000)
    ex = commands.add_parser("export")
    ex.add_argument("--model-path", required=True)
    ex.add_argument("--tokenizer-path", help="Legacy separate CTC tokenizer; omit to reuse decoder")
    ex.add_argument("--output", required=True)
    inference = commands.add_parser(
        "inference-export", help="Remove CTC from a deployment artifact"
    )
    inference.add_argument("--model-path", required=True)
    inference.add_argument("--output", required=True)
    values = vars(parser.parse_args())
    command = values.pop("command")
    from .console import worker_logging

    with worker_logging():
        result = {
            "tokenizer": train_tokenizer,
            "export": export,
            "inference-export": inference_export,
        }[command](**values)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
