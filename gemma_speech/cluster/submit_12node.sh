#!/usr/bin/env bash
#SBATCH --job-name=gemma-speech
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
set -euo pipefail
# Submit from the repository root, or export GEMMA_SPEECH_ROOT explicitly.
ROOT="${GEMMA_SPEECH_ROOT:-${SLURM_SUBMIT_DIR:?}/gemma_speech}"
MODEL="${1:?Usage: sbatch submit_12node.sh gqasr|gqtts STAGE [--resume CHECKPOINT]}"
STAGE="${2:?Provide bridge/full for gqasr or speaker-train/train for gqtts}"
shift 2
case "$MODEL:$STAGE" in
  gqasr:bridge|gqasr:full|gqtts:speaker-train|gqtts:train) ;;
  *) echo "Unsupported distributed model/stage: $MODEL:$STAGE" >&2; exit 2 ;;
esac
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="$SLURM_NNODES"
srun --kill-on-bad-exit=1 bash -c 'export NODE_RANK="$SLURM_PROCID"; exec "$@"' \
  bash "$ROOT/cluster/${MODEL}_stage.sh" "$STAGE" "$@"
