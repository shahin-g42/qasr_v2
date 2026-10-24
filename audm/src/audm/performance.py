"""Progress and padding measurements without per-step CUDA synchronization."""

import json
import logging
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

LOG = logging.getLogger("audm.performance")


class PerformanceCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.monotonic() - self.started
        model = kwargs["model"]
        wait = getattr(model, "aud_data_wait_seconds", 0.0)
        row = dict(
            step=state.global_step,
            rank=args.process_index,
            update_seconds=elapsed + wait,
            data_wait_seconds=wait,
            timestamp=time.time(),
            examples_per_second=(
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * args.world_size
                / (elapsed + wait)
            ),
        )
        # Local heartbeat remains readable even if another rank or an eval later stalls.
        root = Path(args.output_dir)
        path = root / f"progress-rank-{args.process_index}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row) + "\n")
        temporary.replace(path)
        if state.global_step % max(1, int(args.logging_steps)):
            return
        for name in ("aud_padding_totals", "aud_category_totals", "aud_ctc_totals"):
            values = getattr(model, name, None)
            if values is None:
                continue
            values = values.clone().float()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(values)
            if name == "aud_padding_totals":
                audio, total_audio, text, total_text = values.tolist()
                row["audio_padding_fraction"] = 1 - audio / max(1, total_audio)
                row["decoder_padding_fraction"] = 1 - text / max(1, total_text)
            elif name == "aud_ctc_totals":
                loss, eligible, speech, examples = values.tolist()
                row["ctc"] = dict(
                    weight=getattr(model, "aud_current_ctc_weight", None),
                    loss=loss / max(1, eligible),
                    eligible=int(eligible),
                    ineligible_speech=int(speech - eligible),
                    non_speech=int(examples - speech),
                )
            else:
                total_tokens = max(1, values[:, 3].sum().item())
                row["audio_categories"] = {
                    kind: dict(
                        examples=int(count),
                        category_nll=nll / max(1, count),
                        category_accuracy=correct / max(1, count),
                        supervised_token_fraction=tokens / total_tokens,
                    )
                    for kind, (count, nll, correct, tokens) in zip(
                        ("speech", "music", "noise", "silence"), values.tolist(), strict=True
                    )
                }
            setattr(model, name, None)
        if args.process_index == 0:
            with (root / "performance.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            LOG.info("AUD performance: %s", json.dumps(row))
