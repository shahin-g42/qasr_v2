"""Persistent independent model replicas; no DDP or distributed collectives."""
import concurrent.futures as cf
import contextlib
import os
import re
import time
import traceback
from pathlib import Path

from .common import atomic_json, sha256


class Recognizer:
    def __init__(self, checkpoint, device, precision, memory_fraction=1.0):
        import torch

        from gqasr.modeling import GQASRForSpeechRecognition
        from gqasr.processing import GQASRProcessor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        self.torch, self.device = torch, device
        if device.startswith("cuda"):
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        self.processor = GQASRProcessor(checkpoint)
        # Load on CPU and cast before GPU transfer: never transiently double GPU weight memory.
        self.model = GQASRForSpeechRecognition.from_pretrained(checkpoint, "cpu").eval()
        if precision == "bf16":
            self.model.to(dtype=torch.bfloat16)
        self.model.to(device)
        self.precision = precision

    def features(self, rows):
        import numpy as np

        waves = []
        for row in rows:
            if sha256(row["audio"]) != row["audio_sha256"]:
                raise ValueError(f"Prepared audio changed: {row['id']}")
            waves.append(np.load(row["audio"], allow_pickle=False))
        return self.processor.batch(waves)

    def infer(self, rows, features, mode, max_new_tokens):
        torch, model = self.torch, self.model
        batch = {k: v.to(self.device) for k, v in features.items()}
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") and self.precision == "bf16" else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            audio, valid = model.audio_embeddings(batch["input_features"], batch["input_features_mask"])
            embeds, mask, _ = model.fuse(audio, valid, left_pad=True)
            if mode == "arabic":
                # An explicit language prior is allowed; no reference text is ever supplied.
                ids = self.processor.tokenizer.encode("language Arabic<asr_text>", add_special_tokens=False)
                ids = torch.tensor([ids], device=self.device).expand(len(rows), -1)
                embeds = torch.cat([embeds, model.gqasr_decoder.get_input_embeddings()(ids).to(embeds.dtype)], 1)
                mask = torch.cat([mask, torch.ones_like(ids, dtype=mask.dtype)], 1)
            if embeds.shape[1] + max_new_tokens > model.gqasr_decoder.config.max_position_embeddings:
                raise ValueError("Audio prompt plus generation budget exceeds context; no truncation applied")
            output = model.gqasr_decoder.generate(
                inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True,
                eos_token_id=model.eos_id, pad_token_id=model.eos_id,
            ).tolist()
        predictions = []
        for row, tokens in zip(rows, output, strict=True):
            ended = model.eos_id in tokens
            if ended:
                tokens = tokens[:tokens.index(model.eos_id) + 1]
            raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=False)
            header = re.match(r"language\s+([^<]+)<asr_text>", raw)
            text = self.processor.decode(tokens)
            predictions.append({**row, "pred_text": text, "raw_output": raw, "mode": mode,
                                "forced_language": "Arabic" if mode == "arabic" else None,
                                "predicted_language": header.group(1).strip() if header else None,
                                "generated_tokens": len(tokens), "eos_seen": ended,
                                "generation_limit_hit": not ended and len(tokens) >= max_new_tokens})
        return predictions

    def safe_infer(self, rows, features, mode, max_new_tokens):
        try:
            return self.infer(rows, features, mode, max_new_tokens), 0
        except self.torch.cuda.OutOfMemoryError:
            # Retry smaller batches without dropping examples or changing decoding policy.
            if len(rows) == 1:
                raise RuntimeError("GPU OOM on one utterance. Resume with fewer workers per GPU; "
                                   "completed task files are retained.") from None
        self.torch.cuda.empty_cache()
        midpoint = len(rows) // 2
        predictions, retries = [], 1
        for part in (rows[:midpoint], rows[midpoint:]):
            pred, retry = self.safe_infer(part, self.features(part), mode, max_new_tokens)
            predictions.extend(pred)
            retries += retry
        return predictions, retries


def worker_main(slot, gpu, checkpoint, settings, jobs, events, start_event, load_slots, log_path):
    # Set before importing torch/numpy/BLAS in the spawned interpreter.
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu if gpu != "cpu" else "", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    with Path(log_path).open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        task = None
        try:
            load_started = time.monotonic()
            with load_slots:
                recognizer = Recognizer(checkpoint, "cpu" if gpu == "cpu" else "cuda:0",
                                        settings["precision"], settings["memory_fraction"])
            torch = recognizer.torch
            import transformers
            device_info = "cpu" if gpu == "cpu" else torch.cuda.get_device_name(0)
            events.put({"event": "ready", "slot": slot, "gpu": gpu, "device": device_info,
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "load_seconds": time.monotonic() - load_started})
            start_event.wait()
            with cf.ThreadPoolExecutor(1) as prefetch:
                while (task := jobs.get()) is not None:
                    started = time.monotonic()
                    rows = task["rows"]
                    modes_by_id = task["modes_by_id"]
                    batches = [rows[i:i + settings["batch_size"]] for i in range(0, len(rows), settings["batch_size"])]
                    pending = prefetch.submit(recognizer.features, batches[0])
                    predictions, retries, inference_seconds = [], 0, 0.0
                    for i, batch in enumerate(batches):
                        features = pending.result()
                        if i + 1 < len(batches):
                            pending = prefetch.submit(recognizer.features, batches[i + 1])
                        for mode in task["modes"]:
                            indices = [j for j, row in enumerate(batch) if mode in modes_by_id[row["id"]]]
                            if not indices:
                                continue
                            mode_rows = [batch[j] for j in indices]
                            mode_features = {key: value[indices] for key, value in features.items()}
                            if gpu != "cpu":
                                torch.cuda.synchronize()
                            before = time.monotonic()
                            pred, retry = recognizer.safe_infer(
                                mode_rows, mode_features, mode, settings["max_new_tokens"]
                            )
                            if gpu != "cpu":
                                torch.cuda.synchronize()
                            inference_seconds += time.monotonic() - before
                            predictions.extend(pred)
                            retries += retry
                    stats = {"wall_seconds": time.monotonic() - started, "inference_seconds": inference_seconds,
                             "unique_audio_seconds": sum(r["actual_duration"] for r in rows),
                             "audio_seconds": sum(r["actual_duration"] * len(modes_by_id[r["id"]]) for r in rows),
                             "oom_batch_splits": retries,
                             "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if gpu != "cpu" else 0,
                             "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if gpu != "cpu" else 0}
                    atomic_json(task["output"], {"run_id": task["run_id"], "task_id": task["task_id"],
                                               "settings": settings, "slot": slot, "gpu": gpu,
                                               "modes": task["modes"], "modes_by_id": modes_by_id,
                                               "predictions": predictions, "stats": stats})
                    events.put({"event": "task_complete", "task_id": task["task_id"], "slot": slot,
                                "count": len(rows), "stats": stats})
        except BaseException:
            trace = traceback.format_exc()
            log.write(trace)
            events.put({"event": "worker_error", "slot": slot,
                        "task_id": task["task_id"] if task else None, "traceback": trace})
            raise
