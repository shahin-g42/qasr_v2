"""Read-only GQASR automatic versus Chinese-prefilled decoding audit."""
import argparse
import json
import re
import time
from pathlib import Path

import torch

from ._qasr.audio import load_mono_audio
from .modeling import GQASRForSpeechRecognition
from .processing import GQASRProcessor
from .qwen_baseline import read_rows, select_samples, summary


def append_language(embeds, mask, decoder, tokenizer):
    ids = tokenizer.encode("language Chinese<asr_text>", add_special_tokens=False)
    ids = torch.tensor([ids], device=embeds.device).expand(embeds.shape[0], -1)
    extra = decoder.get_input_embeddings()(ids).to(embeds.dtype)
    return torch.cat([embeds, extra], dim=1), torch.cat(
        [mask, torch.ones(ids.shape, device=mask.device, dtype=mask.dtype)], dim=1
    )


def run(args):
    cfg = json.loads(Path(args.run_config).read_text())
    samples = read_rows(args.samples) if args.samples else select_samples(cfg["index_path"], args.predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    processor = GQASRProcessor(args.checkpoint)
    model = GQASRForSpeechRecognition.from_pretrained(args.checkpoint, args.device).eval()
    if args.bridge_from or args.decoder_from:
        from safetensors.torch import load_file
        for component, source in (("gqasr_bridge", args.bridge_from), ("gqasr_decoder", args.decoder_from)):
            if not source:
                continue
            donor_config = json.loads((Path(source) / "config.json").read_text())
            if donor_config != json.loads((Path(args.checkpoint) / "config.json").read_text()):
                raise ValueError("Component swap requires identical model configurations")
            weights = load_file(str(Path(source) / "model.safetensors"))
            prefix = component + "."
            state = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            if component == "gqasr_decoder" and model.gqasr_decoder.config.tie_word_embeddings:
                if "lm_head.weight" not in state:
                    state["lm_head.weight"] = state["model.embed_tokens.weight"]
                if "model.embed_tokens.weight" not in state:
                    state["model.embed_tokens.weight"] = state["lm_head.weight"]
            getattr(model, component).load_state_dict(state, strict=True)
            del weights, state
    (output / "settings.json").write_text(json.dumps(vars(args), indent=2))
    scores = {}
    start = time.monotonic()
    for mode in ("auto", "chinese"):
        rows = []
        with (output / f"predictions-{mode}.jsonl").open("x") as f:
            for start_index in range(0, len(samples), args.batch_size):
                selected = samples[start_index:start_index + args.batch_size]
                source_rows = selected
                if args.audio_control == "shift":
                    source_rows = [samples[(start_index + i + 1) % len(samples)] for i in range(len(selected))]
                waves = [load_mono_audio(r["audio"], 16000) for r in source_rows]
                batch = {k: v.to(args.device) for k, v in processor.batch(waves).items()}
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda" if args.device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16, enabled=args.device.startswith("cuda") and args.precision == "bf16",
                ):
                    audio, valid = model.audio_embeddings(batch["input_features"], batch["input_features_mask"])
                    embeds, mask, _ = model.fuse(audio, valid, left_pad=True)
                    if mode == "chinese":
                        embeds, mask = append_language(embeds, mask, model.gqasr_decoder, processor.tokenizer)
                    ids = model.gqasr_decoder.generate(
                        inputs_embeds=embeds, attention_mask=mask,
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                        eos_token_id=model.eos_id, pad_token_id=model.eos_id,
                    )
                    diagnostics = [None] * len(selected)
                    if args.teacher_forced and mode == "auto":
                        diagnostics = teacher_losses(model, processor, audio, valid, selected)
                for sample, source, token_ids, diagnostic in zip(selected, source_rows, ids, diagnostics, strict=True):
                    tokens = token_ids.tolist()
                    ended = model.eos_id in tokens
                    if ended:
                        tokens = tokens[:tokens.index(model.eos_id) + 1]
                    raw = processor.tokenizer.decode(tokens, skip_special_tokens=False)
                    header = re.match(r"language\s+([^<]+)<asr_text>", raw)
                    row = {**sample, "hypothesis": processor.decode(tokens), "raw_output": raw,
                           "audio_source_id": source["id"], "teacher_forced": diagnostic,
                           "forced_language": "Chinese" if mode == "chinese" else None,
                           "predicted_language": header.group(1) if header else None,
                           "generated_tokens": len(tokens),
                           "generation_limit_hit": len(tokens) >= args.max_new_tokens and not ended}
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({"mode": mode, "done": len(rows), "total": len(samples),
                                  "elapsed_seconds": round(time.monotonic() - start, 1)}), flush=True)
        scores[mode] = summary(rows)
        scores[mode]["by_group"] = {
            group: summary([r for r in rows if r.get("audit_group", "eval") == group])
            for group in sorted({r.get("audit_group", "eval") for r in rows})
        }
        scores[mode]["repeated_uh_outputs"] = sum(r["hypothesis"].lower().count("uh,") >= 5 for r in rows)
        (output / "scores.json").write_text(json.dumps(scores, indent=2))
        print(f"{mode}: CER={scores[mode]['strict']['cer']:.2%}, "
              f"no-Han={scores[mode]['no_han_predictions']}", flush=True)


def teacher_losses(model, processor, audio, valid, rows):
    """Score references separately from free decoding; never feed them to generate."""
    results = []
    for i, row in enumerate(rows):
        ids = torch.tensor([processor.target(row["reference"], "zh")], device=audio.device)
        embeds, mask, labels = model.fuse(audio[i:i+1], valid[i:i+1], ids, torch.ones_like(ids, dtype=torch.bool))
        hidden = model.gqasr_decoder.model(
            inputs_embeds=embeds, attention_mask=mask,
            position_ids=(mask.long().cumsum(-1)-1).clamp_min(0),
            use_cache=False, return_dict=True,
        ).last_hidden_state
        selected = labels[:, 1:] != -100
        logits = model.gqasr_decoder.lm_head(hidden[:, :-1][selected]).float()
        losses = torch.nn.functional.cross_entropy(logits, labels[:, 1:][selected], reduction="none")
        header = len(processor.tokenizer.encode("language Chinese<asr_text>", add_special_tokens=False))
        results.append({"header_nll": losses[:header].mean().item(),
                        "transcript_nll": losses[header:].mean().item(),
                        "header_tokens": header, "transcript_tokens": len(losses)-header})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "run-config", "predictions", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--samples")
    parser.add_argument("--bridge-from")
    parser.add_argument("--decoder-from")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--audio-control", choices=["real", "shift"], default="real")
    parser.add_argument("--teacher-forced", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.batch_size < 1:
        parser.error("max-new-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
