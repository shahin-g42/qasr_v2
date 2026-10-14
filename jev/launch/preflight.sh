#!/usr/bin/env bash
# JEV cluster preflight: run on a compute node before the multi-node job.
#
#   launch/preflight.sh [config.toml]
#
# Verifies, in order: the strict TOML config loads and its mesh matches the
# local node; the torch runtime has CUDA/NCCL; the node presents exactly the
# configured number of H100 80GB GPUs without MIG; interconnect facts are
# reported; the snapshot and the approved release verify through the real
# manifest machinery; the run directory is writable. Advisory checks warn;
# hard mismatches fail and the script exits nonzero.
#
# JEV_PREFLIGHT_SMOKE=1 additionally runs a single-node NCCL all-reduce over
# all local GPUs and reports achieved bus bandwidth and free device memory.
set -euo pipefail

JEV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$JEV_ROOT/configs/train/h100_8x8.toml}"
export PYTHONPATH="$JEV_ROOT/packages/contracts/src:$JEV_ROOT/packages/core/src\
${PYTHONPATH:+:$PYTHONPATH}"

FAILURES=0
WARNINGS=0
SHARD=0
SNAPSHOT=""
RELEASE=""
RUN_DIR=""
ok()   { echo "PASS  $*"; }
warn() { echo "WARN  $*"; WARNINGS=$((WARNINGS + 1)); }
fail() { echo "FAIL  $*"; FAILURES=$((FAILURES + 1)); }

echo "== config =="
CONFIG_OUT="$(python3 - "$CONFIG" <<'PY'
import sys

from jev.config import TrainConfig

config = TrainConfig.from_toml(sys.argv[1])
print(config.shard)
print(
    f"mesh=(replicate={config.replicate}, shard={config.shard}) "
    f"world={config.shard * config.replicate} dtype={config.dtype} "
    f"device={config.device} global_batch={config.global_batch} "
    f"micro_cap={config.micro_cap} token_budget={config.token_budget} "
    f"peak_lr={config.learning_rate} warmup={config.warmup_ratio} "
    f"epochs={config.epochs} initialization={config.initialization}"
)
print(config.snapshot)
print(config.release)
print(config.run_dir)
PY
)" || fail "config did not load: $CONFIG"
if [ -n "$CONFIG_OUT" ]; then
    SHARD="$(echo "$CONFIG_OUT" | sed -n 1p)"
    SNAPSHOT="$(echo "$CONFIG_OUT" | sed -n 3p)"
    RELEASE="$(echo "$CONFIG_OUT" | sed -n 4p)"
    RUN_DIR="$(echo "$CONFIG_OUT" | sed -n 5p)"
    ok "config: $(echo "$CONFIG_OUT" | sed -n 2p)"
fi

echo "== torch runtime =="
TORCH_OK=0
python3 - <<'PY' && TORCH_OK=1 || fail "torch CUDA/NCCL runtime unusable"
import torch

assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
print(
    f"torch {torch.__version__} cuda {torch.version.cuda} "
    f"nccl {torch.cuda.nccl.version()} devices {torch.cuda.device_count()}"
)
PY
if [ "$TORCH_OK" -eq 1 ]; then ok "torch runtime present"; fi

echo "== gpu node =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found: run preflight on a compute node"
else
    GPU_LIST="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
    GPU_COUNT="$(echo "$GPU_LIST" | wc -l | tr -d ' ')"
    if [ "${GPU_COUNT:-0}" -ne "${SHARD:-0}" ]; then
        fail "expected $SHARD GPUs on the node, found $GPU_COUNT"
    else
        ok "gpu count: $GPU_COUNT"
    fi
    if echo "$GPU_LIST" | grep -ivq "H100"; then
        fail "not all GPUs are H100: $GPU_LIST"
    else
        ok "all GPUs are H100"
    fi
    if echo "$GPU_LIST" | grep -ivq "8[0-9]\{4\} MiB"; then
        warn "not every GPU reports ~80GiB VRAM: $GPU_LIST"
    else
        ok "80GiB-class VRAM on every GPU"
    fi
    MIG="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null || true)"
    if echo "$MIG" | grep -qi "enabled"; then
        fail "MIG is enabled; the qualified topology is one process per full H100"
    fi
    TOPO_SUMMARY="$(nvidia-smi topo -m 2>/dev/null | head -n 12 || true)"
    echo "$TOPO_SUMMARY"
    echo "$TOPO_SUMMARY" | grep -q "NVLink" \
        && ok "NVLink present in topology" || warn "no NVLink rows in topology output"
fi

echo "== fabric (advisory) =="
if [ -d /sys/class/infiniband ] && [ -n "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
    NIC_COUNT="$(ls /sys/class/infiniband | wc -l | tr -d ' ')"
    ok "$NIC_COUNT InfiniBand device(s): $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')"
else
    warn "no InfiniBand devices under /sys/class/infiniband; inter-node fabric unverified"
fi

echo "== artifacts =="
ARTIFACT_OK=0
if [ -z "$RUN_DIR" ]; then
    fail "config did not load; artifact paths are unknown"
elif python3 - "$SNAPSHOT" "$RELEASE" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

from jev_contracts import verify_manifest

snapshot, release, run_dir = (Path(value) for value in sys.argv[1:4])
missing = [
    str(path) for path in (
        snapshot / "config.json", snapshot / "model.safetensors",
        snapshot / "model.safetensors.index.json", release / "manifest.json",
        release / "train.jsonl", release / "development.jsonl",
    ) if not path.exists()
]
if "model.safetensors" in missing and "model.safetensors.index.json" not in missing:
    missing.remove("model.safetensors")  # sharded snapshot: the index suffices
if missing:
    raise SystemExit(f"missing artifact files: {missing}")
manifest = verify_manifest(release)
meta = manifest.get("metadata") or {}
if meta.get("kind") != "dataset_release" or meta.get("approved") is not True:
    raise SystemExit("release manifest is not an approved dataset_release")
if (release / "manifest.json").exists():
    print(f"release verified: {len(manifest['files'])} sealed files, approved")
if (snapshot / "manifest.json").exists():
    verify_manifest(snapshot)
    print("snapshot manifest verified")
else:
    print("snapshot has no manifest.json: caller-trusted local directory")
run_dir.mkdir(parents=True, exist_ok=True)
probe = run_dir / ".jev_preflight_probe"
probe.write_text("ok", encoding="utf-8")
probe.unlink()
print(f"run_dir writable: {run_dir}")
PY
then
    ARTIFACT_OK=1
fi
if [ "$ARTIFACT_OK" -eq 1 ]; then
    ok "artifacts verified through the manifest machinery"
else
    fail "artifact verification failed (see messages above)"
fi

if [ "${JEV_PREFLIGHT_SMOKE:-0}" = "1" ]; then
    echo "== nccl smoke (single node, $SHARD GPUs) =="
    torchrun --nnodes=1 --nproc-per-node="$SHARD" --max-restarts=0 \
      launch/preflight_smoke.py
fi

echo "== summary =="
if [ "$FAILURES" -ne 0 ]; then
    echo "preflight FAILED with $FAILURES failure(s), $WARNINGS warning(s)"
    exit 1
fi
echo "preflight passed with $WARNINGS warning(s)"
