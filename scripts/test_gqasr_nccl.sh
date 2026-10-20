#!/usr/bin/env bash
# Run on every participating node with a unique NODE_RANK.
set -euo pipefail
usage() {
  cat <<'USAGE'
Usage: bash scripts/test_gqasr_nccl.sh NODE_RANK [MASTER_ADDR] [test options]

Defaults: NNODES=4, NPROC_PER_NODE=8, NCCL_TEST_PORT=29691.
MASTER_ADDR must be supplied as an argument or environment variable.
Test options: --timeout 60 --iterations 10 --tensor-mib 32
Use --backend gloo for a CPU-only comparison (does not test NCCL).

Overrides:
  NNODES, NPROC_PER_NODE       Test topology; use NNODES=1 for a local test.
  GPU_IDS                     Optional explicit CUDA_VISIBLE_DEVICES.
  NCCL_TEST_PYTHON             Python executable (default: gemma_speech/.venv/bin/python).
  NCCL_TEST_RDZV_TIMEOUT       Seconds to wait for node launchers (default: 180).
  NCCL_TEST_LOG_DIR            Local log parent (default: /tmp/gqasr-nccl-smoke).
  DRY_RUN=1                   Print the command without starting workers.

Existing NCCL/Gloo network settings are preserved. Start every node with
matching options. This creates a new process group; it cannot inspect or
repair an already-running training process group.
USAGE
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
SMOKE_NODE_RANK=$((10#$1))
shift
SMOKE_MASTER="${MASTER_ADDR:-}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  SMOKE_MASTER="$1"
  shift
fi
[[ -n "$SMOKE_MASTER" ]] || { echo 'Supply MASTER_ADDR.' >&2; exit 2; }
SMOKE_NNODES="${NNODES:-4}"
SMOKE_NPROC="${NPROC_PER_NODE:-8}"
SMOKE_PORT="${NCCL_TEST_PORT:-29691}"
SMOKE_RDZV_TIMEOUT="${NCCL_TEST_RDZV_TIMEOUT:-180}"
for value in "$SMOKE_NNODES" "$SMOKE_NPROC" "$SMOKE_PORT" "$SMOKE_RDZV_TIMEOUT"; do
  [[ "$value" =~ ^[0-9]+$ ]] && (( 10#$value > 0 )) || {
    echo 'Node/GPU counts, test port and rendezvous timeout must be positive integers.' >&2
    exit 2
  }
done
SMOKE_NNODES=$((10#$SMOKE_NNODES))
SMOKE_NPROC=$((10#$SMOKE_NPROC))
SMOKE_PORT=$((10#$SMOKE_PORT))
SMOKE_RDZV_TIMEOUT=$((10#$SMOKE_RDZV_TIMEOUT))
(( SMOKE_NODE_RANK < SMOKE_NNODES && SMOKE_PORT <= 65535 && SMOKE_NNODES * SMOKE_NPROC >= 2 )) || {
  echo 'Check node rank, port, and topology (at least two workers required).' >&2
  exit 2
}
SMOKE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SMOKE_PYTHON="${NCCL_TEST_PYTHON:-${GEMMA_SPEECH_ENV:-$SMOKE_ROOT/gemma_speech/.venv}/bin/python}"
SMOKE_LOG_PARENT="${NCCL_TEST_LOG_DIR:-/tmp/gqasr-nccl-smoke}"
SMOKE_LOG_DIR="$SMOKE_LOG_PARENT/$(date -u +%Y%m%dT%H%M%SZ)-node-$SMOKE_NODE_RANK-$$"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-$SMOKE_LOG_DIR/nccl.%h.%p.log}"
# Match run_gqasr_12node.sh defaults, while preserving supplied overrides.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
if [[ -n "${GPU_IDS:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_IDS"; fi
CMD=(
  "$SMOKE_PYTHON" -m torch.distributed.run
  "--nnodes=$SMOKE_NNODES" "--nproc_per_node=$SMOKE_NPROC"
  "--node_rank=$SMOKE_NODE_RANK" "--master_addr=$SMOKE_MASTER" "--master_port=$SMOKE_PORT"
  --rdzv_backend=static "--rdzv_endpoint=$SMOKE_MASTER:$SMOKE_PORT"
  "--rdzv_conf=timeout=$SMOKE_RDZV_TIMEOUT" --max_restarts=0
  "--log-dir=$SMOKE_LOG_DIR" --tee=3
  "$SMOKE_ROOT/gemma_speech/gqasr/src/gqasr/nccl_smoke.py" "$@"
)
echo "Communication test: node=$SMOKE_NODE_RANK/$SMOKE_NNODES workers=$SMOKE_NPROC master=$SMOKE_MASTER:$SMOKE_PORT"
echo "Logs: $SMOKE_LOG_DIR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
[[ -x "$SMOKE_PYTHON" ]] || {
  echo "Missing Python: $SMOKE_PYTHON (set NCCL_TEST_PYTHON to your training environment's Python)." >&2
  exit 2
}
mkdir -p "$SMOKE_LOG_DIR"
exec "${CMD[@]}"
