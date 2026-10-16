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
from .qwen_baseline import select_samples, summary


def append_language(embeds, mask, decoder, tokenizer):
    ids = tokenizer.encode("language Chinese<asr_text>", add_special_tokens=False)
    ids = torch.tensor([ids], device=embeds.device).expand(embeds.shape[0], -1)
    extra = decoder.get_input_embeddings()(ids).to(embeds.dtype)
    return torch.cat([embeds, extra], dim=1), torch.cat(
        [mask, torch.ones(ids.shape, device=mask.device, dtype=mask.dtype)], dim=1
    )


def run(args):
    cfg = json.loads(Path(args.run_config).read_text())
    samples = select_samples(cfg["index_path"], args.predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    processor = GQASRProcessor(args.checkpoint)
    model = GQASRForSpeechRecognition.from_pretrained(args.checkpoint, args.device).eval()
    (output / "settings.json").write_text(json.dumps(vars(args), indent=2))
    scores = {}
    start = time.monotonic()
    for mode in ("auto", "chinese"):
        rows = []
        with (output / f"predictions-{mode}.jsonl").open("x") as f:
            for sample in samples:
                wave = load_mono_audio(sample["audio"], 16000)
                batch = {k: v.to(args.device) for k, v in processor.batch([wave]).items()}
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda" if args.device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16, enabled=args.device.startswith("cuda"),
                ):
                    if mode == "auto":
                        ids = model.generate_transcripts(**batch, max_new_tokens=args.max_new_tokens)
                    else:
                        audio, valid = model.audio_embeddings(batch["input_features"], batch["input_features_mask"])
                        embeds, mask, _ = model.fuse(audio, valid, left_pad=True)
                        embeds, mask = append_language(embeds, mask, model.gqasr_decoder, processor.tokenizer)
                        ids = model.gqasr_decoder.generate(
                            inputs_embeds=embeds, attention_mask=mask,
                            max_new_tokens=args.max_new_tokens, do_sample=False,
                            eos_token_id=model.eos_id, pad_token_id=model.eos_id,
                        )
                tokens = ids[0].tolist()
                raw = processor.tokenizer.decode(tokens, skip_special_tokens=False)
                header = re.match(r"language\s+([^<]+)<asr_text>", raw)
                row = {**sample, "hypothesis": processor.decode(tokens), "raw_output": raw,
                       "forced_language": "Chinese" if mode == "chinese" else None,
                       "predicted_language": header.group(1) if header else None,
                       "generated_tokens": len(tokens),
                       "generation_limit_hit": len(tokens) >= args.max_new_tokens and model.eos_id not in tokens}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({"mode": mode, "done": len(rows), "total": len(samples),
                                  "elapsed_seconds": round(time.monotonic() - start, 1)}), flush=True)
        scores[mode] = summary(rows)
        scores[mode]["repeated_uh_outputs"] = sum(r["hypothesis"].lower().count("uh,") >= 5 for r in rows)
        (output / "scores.json").write_text(json.dumps(scores, indent=2))
        print(f"{mode}: CER={scores[mode]['strict']['cer']:.2%}, "
              f"no-Han={scores[mode]['no_han_predictions']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "run-config", "predictions", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
