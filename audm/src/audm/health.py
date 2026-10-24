"""Read-only GPU thermal and training-progress checks for direct cluster launch."""

import argparse
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def inspect_gpus(xml):
    records = []
    for gpu in ET.fromstring(xml).findall("gpu"):
        active = any(
            (gpu.findtext(path) or "").strip().lower() == "active"
            for path in (
                "clocks_event_reasons/clocks_event_reason_sw_thermal_slowdown",
                "clocks_throttle_reasons/clocks_throttle_reason_sw_thermal_slowdown",
                "clocks_event_reasons/clocks_event_reason_hw_thermal_slowdown",
                "clocks_throttle_reasons/clocks_throttle_reason_hw_thermal_slowdown",
            )
        )
        records.append(
            dict(
                uuid=gpu.findtext("uuid"),
                temperature=gpu.findtext("temperature/gpu_temp"),
                sm_clock=gpu.findtext("clocks/sm_clock"),
                gpu_utilization=gpu.findtext("utilization/gpu_util"),
                memory_utilization=gpu.findtext("utilization/memory_util"),
                memory_used=gpu.findtext("fb_memory_usage/used"),
                power_draw=gpu.findtext("gpu_power_readings/power_draw")
                or gpu.findtext("power_readings/power_draw"),
                thermal_slowdown=active,
                processes=len(gpu.findall("processes/process_info")),
            )
        )
    if not records:
        raise ValueError("nvidia-smi returned no GPUs")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Inspect progress heartbeats instead of GPUs")
    parser.add_argument("--max-idle-seconds", type=int, default=600)
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--expected-ranks", type=int)
    args = parser.parse_args()
    if args.run:
        paths = sorted(Path(args.run).glob("progress-rank-*.json"))
        if not paths:
            raise SystemExit("No progress heartbeats found")
        records = [json.loads(p.read_text()) for p in paths]
        expected = args.expected_ranks
        receipt = Path(args.run) / "training_recipe.json"
        if expected is None and receipt.is_file():
            expected = json.loads(receipt.read_text())["resolved"].get("expected_world_size")
        missing = sorted(set(range(expected or 0)) - {row["rank"] for row in records})
        stale = any(time.time() - row["timestamp"] > args.max_idle_seconds for row in records)
        print(
            json.dumps(
                dict(
                    ranks=records,
                    stale=stale,
                    missing_ranks=missing,
                    note="Long evaluation/checkpoint pauses also make heartbeats old.",
                ),
                indent=2,
            )
        )
        raise SystemExit(1 if stale or missing else 0)
    result = subprocess.run(
        ["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True, timeout=30
    )
    records = inspect_gpus(result.stdout)
    print(
        json.dumps(
            dict(gpus=records, note="An idle check does not prove cooling under load."), indent=2
        )
    )
    if any(row["thermal_slowdown"] or (args.require_idle and row["processes"]) for row in records):
        raise SystemExit(
            "Resolve thermal slowdown/active GPU jobs before benchmarking or launching"
        )


if __name__ == "__main__":
    main()
