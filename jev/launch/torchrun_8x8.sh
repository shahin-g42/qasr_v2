#!/usr/bin/env bash
# Multi-node torchrun launcher for the JEV trainer.
#
# Start this SAME command once on every node of the job (directly, or through
# your cluster's node launcher such as srun in slurm_8x8.sbatch). Each
# invocation runs one local torchrun agent, and every agent joins a single
# c10d rendezvous; ranks are then assigned node-major, which is the order the
# trainer's mesh verification requires (each shard group is exactly one node).
#
#   launch/torchrun_8x8.sh [config.toml] [nodes] [-- extra jev.train args]
#
# Examples:
#   JEV_RDZV_ENDPOINT=10.0.0.1:29500 ./launch/torchrun_8x8.sh            # each node
#   ./launch/torchrun_8x8.sh configs/train/h100_8x8.toml 8 -- --resume latest
#
# Environment:
#   JEV_RDZV_ENDPOINT  rendezvous host:port; every node must use the FIRST
#                      node's address (default: this host's primary IP)
#   JEV_RDZV_ID        rendezvous run id, unique per attempt (default: jev-<nodes>x8)
#
# The job must see the same JEV_ROOT, snapshot, and release on every node.
set -euo pipefail

JEV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$JEV_ROOT/configs/train/h100_8x8.toml}"
NODES="${2:-8}"
NPROC_PER_NODE=8
shift "$(( $# > 2 ? 2 : $# ))" || true
if [ "${1:-}" = "--" ]; then shift; fi

# The trainer runs from the workspace sources exactly like the test suite.
export PYTHONPATH="$JEV_ROOT/packages/contracts/src:$JEV_ROOT/packages/core/src\
${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PRIMARY_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
[ -n "$PRIMARY_IP" ] || PRIMARY_IP="$(hostname)"
RDZV_ENDPOINT="${JEV_RDZV_ENDPOINT:-$PRIMARY_IP:29500}"
RDZV_ID="${JEV_RDZV_ID:-jev-${NODES}x8}"

# Fail before rendezvous when the config's mesh cannot match this launch.
python3 - "$CONFIG" "$NODES" "$NPROC_PER_NODE" <<'PY'
import sys

import torch

from jev.config import TrainConfig

config_path, nodes, procs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
config = TrainConfig.from_toml(config_path)
world = config.shard * config.replicate
if config.device != "cuda":
    sys.exit(f"Refusing to launch on GPUs with device={config.device!r}")
if procs != config.shard:
    sys.exit(f"shard={config.shard} must equal the {procs} GPUs launched per node")
if nodes != config.replicate:
    sys.exit(f"replicate={config.replicate} must equal the {nodes} nodes launched")
visible = torch.cuda.device_count()
if visible != config.shard:
    sys.exit(f"Expected {config.shard} visible GPUs on this node, found {visible}")
print(
    f"config ok: world={world} mesh=(replicate={config.replicate}, shard={config.shard})"
    f" dtype={config.dtype} global_batch={config.global_batch}"
)
PY

# One process per GPU; an agent crash fails the whole job instead of silently
# replaying mid-run. Recovery is the tested path: resume from a sealed
# checkpoint with --resume latest after re-launching.
exec torchrun \
  --nnodes="$NODES" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --max-restarts=0 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$RDZV_ENDPOINT" \
  --rdzv-id="$RDZV_ID" \
  -m jev.train --config "$CONFIG" "$@"
