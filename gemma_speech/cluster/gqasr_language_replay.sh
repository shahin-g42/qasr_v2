#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${GEMMA_SPEECH_ENV:-$ROOT/.venv}/bin/activate"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export REPLAY_ROOT="$ROOT"
python - "$@" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
root = Path(os.environ["REPLAY_ROOT"])
run = Path(os.environ.get("GQASR_RUN_DIR", root / "gqasr/outputs/full_b40_frozen"))
gpus = os.environ.get("GPU_IDS", "0,1,2").split(",")
if not all(gpus) or len(gpus) != len(set(gpus)):
    raise ValueError("GPU_IDS must contain unique, nonempty IDs")
if sys.argv[1:]:
    checkpoints = [Path(p).resolve() for p in sys.argv[1:]]
else:
    checkpoints = [root / "gqasr/outputs/bridge/final"]
    early = run / "checkpoint-00001042"
    if early.is_dir():
        checkpoints.append(early)
    else:
        print("Step 1042 checkpoint is no longer present; skipping it.", flush=True)
    latest = json.loads((run / "latest.json").read_text())["checkpoint"]
    checkpoints.append(run / latest)
checkpoints = list(dict.fromkeys(checkpoints))
for p in checkpoints:
    if not (p / "model.safetensors").is_file():
        raise FileNotFoundError(p)
predictions = run / ("predictions-" + os.environ.get("EVAL_STEP", "7294") + ".jsonl")
if not predictions.is_file():
    raise FileNotFoundError(predictions)
output = Path(os.environ.get("REPLAY_OUTPUT", root / "gqasr/verification" /
                            time.strftime("gqasr-language-replay-%Y%m%dT%H%M%SZ", time.gmtime())))
output.mkdir(parents=True, exist_ok=False)
jobs = [{"checkpoint": str(p), "output": str(output / f"{i}-{p.parent.name}-{p.name}")}
        for i,p in enumerate(checkpoints)]
(output / "jobs.json").write_text(json.dumps(jobs, indent=2))
print(f"Replay output: {output}", flush=True)
children, logs = [], []
try:
    for start in range(0, len(jobs), len(gpus)):
        children = []
        for gpu, job in zip(gpus, jobs[start:start+len(gpus)]):
            log = open(job["output"] + ".log", "w")
            logs.append(log)
            cmd = [sys.executable, "-m", "gqasr.language_replay",
                   "--checkpoint", job["checkpoint"], "--run-config", str(run / "config.json"),
                   "--predictions", str(predictions), "--output", job["output"]]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}))
            print(f"GPU {gpu}: {job['checkpoint']} -> {job['output']}.log", flush=True)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("Replay failed; inspect per-checkpoint logs")
            print(f"Replay workers complete: {sum(p.poll() == 0 for p in children)}/{len(children)}", flush=True)
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("Replay failed; inspect per-checkpoint logs")
    comparison = {j["checkpoint"]: json.loads(Path(j["output"], "scores.json").read_text()) for j in jobs}
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2), flush=True)
finally:
    for p in children:
        if p.poll() is None:
            p.terminate()
    for p in children:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    for log in logs:
        log.close()
PY
