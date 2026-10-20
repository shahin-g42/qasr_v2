"""Read existing NCCL flight recorders and local logs without loading a model."""

import argparse
import datetime
import getpass
import json
import os
import pickle
import re
import socket
from pathlib import Path


class PrimitiveUnpickler(pickle.Unpickler):
    """Flight-recorder records contain primitive containers, not Python objects."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Refusing executable pickle global: {module}.{name}")


def timestamp(path):
    return datetime.datetime.fromtimestamp(
        path.stat().st_mtime, datetime.timezone.utc
    ).isoformat()


def inspect_trace(path, sequence):
    result = {"file": str(path)}
    try:
        result["modified_utc"] = timestamp(path)
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("Trace exceeds 128 MiB; inspect it with the PyTorch flight-recorder tool")
        with path.open("rb") as stream:
            trace = PrimitiveUnpickler(stream).load()
        if not isinstance(trace, dict) or not isinstance(trace.get("entries"), list):
            raise ValueError("Not a recognized flight-recorder dictionary")
        entries = trace["entries"]
        if any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("Flight-recorder entries must be dictionaries")
        result.update(version=trace.get("version"), pg_status=trace.get("pg_status"),
                      pg_config=trace.get("pg_config"), total_entries=len(entries),
                      comm_lib_version=trace.get("comm_lib_version"),
                      nccl_comm_state=trace.get("nccl_comm_state"))
        # Include the requested collective plus the tail, even when the rank
        # never reached that collective. Keep coalesced operations separately.
        chosen = [i for i, e in enumerate(entries)
                  if e.get("collective_seq_id") == sequence or i >= len(entries) - 2]
        fields = ("record_id", "pg_id", "process_group", "collective_seq_id",
                  "op_id", "profiling_name", "input_sizes", "input_dtypes",
                  "output_sizes", "output_dtypes", "state", "retired", "timeout_ms",
                  "time_created_ns", "time_discovered_started_ns",
                  "time_discovered_completed_ns", "duration_ms", "is_p2p")
        result["entries"] = []
        for i in chosen:
            e = entries[i]
            item = {key: e[key] for key in fields if key in e}
            frames = e.get("frames", [])
            # Python call stacks identify loss checks versus DDP gradients/init.
            item["frames"] = frames[:12]
            result["entries"].append(item)
        result["requested_sequence_present"] = any(
            e.get("collective_seq_id") == sequence for e in entries
        )
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, TypeError, AttributeError) as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_log(path):
    # Bound reads; these logs may include a long preceding training run.
    limit = 2 * 1024 * 1024
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        lines = stream.read(limit).decode("utf-8", errors="replace").splitlines()
    patterns = (
        "Watchdog caught", "Stack trace of the failed", "Traceback",
        "NCCL WARN", "Input/output error", "bus error", "DataLoader timed out",
        "Error", "GQASR stage=", "training_resume", "global_batch",
        "=== GQASR", "torch.OutOfMemoryError",
    )
    selected = set()
    for i, line in enumerate(lines):
        if any(p in line for p in patterns):
            selected.update(range(max(0, i - 2), min(len(lines), i + 13)))
    return {
        "file": str(path), "modified_utc": timestamp(path),
        "tail_truncated": path.stat().st_size > limit,
        "relevant_lines": [lines[i] for i in sorted(selected)][:240],
        "last_lines": lines[-12:],
    }


def collect(log_dir, node_rank=None, sequence=16, trace_prefix=None):
    directory = Path(log_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"No local diagnostic directory: {directory}")
    prefix = Path(trace_prefix) if trace_prefix else directory / "nccl_trace_rank_"
    traces = []
    for path in sorted(prefix.parent.glob(prefix.name + "*")):
        suffix = path.name[len(prefix.name):]
        match = re.match(r"(\d+)(?:\D.*)?$", suffix)
        if not path.is_file() or not match:
            continue
        rank = int(match[1])
        if node_rank is None or rank // 8 == node_rank:
            traces.append({"rank": rank, **inspect_trace(path, sequence)})
    pattern = f"gqasr-encoder-*-{node_rank}.log" if node_rank is not None else "gqasr-encoder-*.log"
    logs = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return {
        "kind": "existing NCCL/log evidence; no training, model loading or network test",
        "hostname": socket.gethostname(), "node_rank": node_rank,
        "requested_sequence": sequence, "local_log_dir": str(directory),
        "traces": traces, "latest_log": inspect_log(logs[-1]) if logs else None,
        "root_cause_confirmed": False,
        "notes": [
            "Reporting a timeout does not identify the faulty rank.",
            "Compare sequence, operation, shape, dtype and call stack across all 96 ranks.",
            "A missing trace is missing evidence, not proof that a rank failed.",
            "Check timestamps: dumps from different attempts must not be combined.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=os.environ.get(
        "GQASR_LOG_DIR", f"/tmp/gqasr-{getpass.getuser()}-logs"))
    parser.add_argument("--node-rank", type=int, choices=range(12))
    parser.add_argument("--sequence", type=int, default=16)
    parser.add_argument("--trace-prefix", default=os.environ.get("TORCH_FR_DUMP_TEMP_FILE")
                        or os.environ.get("TORCH_NCCL_DEBUG_INFO_TEMP_FILE"))
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = collect(**args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(json.dumps({"report": str(output.resolve()), "hostname": report["hostname"],
                          "trace_ranks": [r["rank"] for r in report["traces"]],
                          "log_found": report["latest_log"] is not None,
                          "trace_read_errors": sum("read_error" in r for r in report["traces"])}))
    else:
        print(text)


if __name__ == "__main__":
    main()
