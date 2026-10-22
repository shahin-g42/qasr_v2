"""Per-rank CUDA diagnostics without collectives, including the OOM failure path."""

import json
import logging
from pathlib import Path

import torch
from transformers import TrainerCallback


def write_memory_report(args, output, status, step):
    if args.device.type != "cuda":
        return
    device = args.device
    free, total = torch.cuda.mem_get_info(device)
    report = dict(
        rank=args.process_index,
        status=status,
        global_step=step,
        microbatch=args.per_device_train_batch_size,
        accumulation=args.gradient_accumulation_steps,
        world_size=args.world_size,
        allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
        reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        free_gib=free / 2**30,
        total_gib=total / 2**30,
    )
    path = Path(output) / f"cuda-memory-rank-{args.process_index}.json"
    try:
        path.write_text(json.dumps(report, indent=2) + "\n")
    except OSError as exc:
        logging.getLogger("audm").warning("Could not write memory report %s: %s", path, exc)
    logging.getLogger("audm").info("CUDA memory: %s", json.dumps(report))


class CUDAMemoryCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 2 or state.global_step % max(100, args.logging_steps) == 0:
            write_memory_report(args, args.output_dir, "training", state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        write_memory_report(args, args.output_dir, "completed", state.global_step)
