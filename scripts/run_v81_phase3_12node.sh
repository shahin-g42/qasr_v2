#!/usr/bin/env bash
# QASR v8.1 Stage 3. Run on each node: NODE_RANK (0-11), optional MASTER_ADDR (defaults to inception-H100-hpc-051).
# NNODES=1 ... 0 localhost --smoke-test is the one-node rehearsal.
# Automatically activates CONDA_ENV (default: qasr3) before preflight/training.
# DRY_RUN=1 prints the command without activating conda or launching training.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 NODE_RANK [MASTER_ADDR] [--smoke-test | --preflight [--scan]] [train.py args]" >&2
  exit 1
fi
[[ "$1" =~ ^[0-9]+$ ]] || { echo 'NODE_RANK must be an integer' >&2; exit 1; }
NODE_RANK=$((10#$1))
shift
MASTER_ADDR="${MASTER_ADDR:-inception-H100-hpc-051}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  MASTER_ADDR="$1"
  shift
fi
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${CONFIG:-configs/v8.1/03_hq_12node.yaml}"
NNODES="${NNODES:-12}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29583}"
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3}"
[[ "$NNODES" =~ ^[1-9][0-9]*$ && "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || {
  echo 'NNODES and NPROC_PER_NODE must be positive integers' >&2; exit 1;
}
(( NODE_RANK < NNODES )) || { echo "NODE_RANK must be 0-$((NNODES - 1))" >&2; exit 1; }
[[ -n "$MASTER_ADDR" ]] || { echo 'MASTER_ADDR is required' >&2; exit 1; }
SMOKE=0
PREFLIGHT=0
RESUME=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --smoke-test) SMOKE=1 ;;
    --preflight) PREFLIGHT=1 ;;
    --resume-from-checkpoint)
      (( i + 1 < ${#ARGS[@]} )) || { echo 'Missing resume checkpoint' >&2; exit 1; }
      RESUME="${ARGS[$((i + 1))]}" ;;
    --resume-from-checkpoint=*) RESUME="${ARGS[$i]#*=}" ;;
  esac
done
if (( ! SMOKE && ! PREFLIGHT )) && (( NNODES != 12 || NPROC_PER_NODE != 8 )); then
  echo 'Production v8.1 requires 12 nodes x 8 GPUs. Use --smoke-test for a smaller rehearsal.' >&2
  exit 1
fi
cd "$PROJECT_DIR"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
CMD=(torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK"
     --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
     --rdzv_backend=static --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_conf=timeout=7200
     train.py --config "$CONFIG" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
echo "Startup: node rank $NODE_RANK/$NNODES; initializing logs"
mkdir -p "$PROJECT_DIR/output/v81/logs_p3_12node"
LOG_FILE="$PROJECT_DIR/output/v81/logs_p3_12node/$(date +%Y%m%d-%H%M%S)-rank${NODE_RANK}-$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "QASR v8.1 Stage 3: rank $NODE_RANK/$NNODES, master $MASTER_ADDR:$MASTER_PORT, config $CONFIG"
echo "Startup log: $LOG_FILE"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_CACHE:-${HF_HOME:-$PROJECT_DIR/.cache}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_BUFFSIZE=8388608
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_EXTENSIONS_DIR="/tmp/${USER}_qasr_v81_torch_ext"
export TRITON_CACHE_DIR="/tmp/${USER}_qasr_v81_triton"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
echo "Startup: before conda, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "Startup: activating conda environment $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate "$CONDA_ENV"
# Full-node launchers default to all eight GPUs, applied AFTER conda activation.
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3,4,5,6,7}"
echo "Startup: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}; NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-<unset>}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Startup: checking Python/Transformers ($(command -v python))"
python -c 'import transformers; assert transformers.__version__.startswith("5.14."), transformers.__version__'
if (( ! PREFLIGHT )); then
echo "Startup: checking visible GPUs"
python - "$NPROC_PER_NODE" <<'PY'
import os
import sys
import torch
expected = int(sys.argv[1])
actual = torch.cuda.device_count()
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, visible GPUs: {actual}/{expected}", flush=True)
if actual != expected:
    raise RuntimeError(
        f"Expected {expected} visible GPUs, got {actual}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<unset>')}. "
        "Check the GPU allocation, driver and GPU_IDS selection. "
        "The override is applied after conda activation."
    )
for i in range(expected):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    assert torch.cuda.is_bf16_supported(), "BF16 support is required"
PY
fi
if (( PREFLIGHT )); then
  CHECK_ARGS=()
  for arg in "$@"; do [[ "$arg" == --preflight ]] || CHECK_ARGS+=("$arg"); done
  echo "Startup: running requested preflight"
  exec python -u scripts/preflight_v81_phase3.py --config "$CONFIG" "${CHECK_ARGS[@]}"
fi
CHECK_ARGS=(--config "$CONFIG")
if (( SMOKE )); then CHECK_ARGS+=(--smoke-test); fi
if [[ -n "$RESUME" ]]; then CHECK_ARGS+=(--resume-from-checkpoint "$RESUME"); fi
echo "Startup: running preflight (checkpoint, manifests, processor and index)"
python -u scripts/preflight_v81_phase3.py "${CHECK_ARGS[@]}"

echo "Startup: executing torchrun ($NNODES nodes, node rank $NODE_RANK, endpoint $MASTER_ADDR:$MASTER_PORT)"
echo "Startup: torchrun executable: $(command -v torchrun)"
echo "Startup: Worker entered messages confirm rendezvous completed and training processes started"
exec "${CMD[@]}"
