#!/usr/bin/env bash
# Fresh adaptation and full training from checkpoint-3700. Direct torchrun only.
set -euo pipefail
AUD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$AUD_ROOT/audm/src:$AUD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export AUD_PYTHON="${AUD_PYTHON:-python}"
export AUD_NEXT_SOURCE="${AUD_NEXT_SOURCE:-$AUD_ROOT/output/audflash-audio-types-v1/full/checkpoint-3700}"
export AUD_NEXT_ROOT="${AUD_NEXT_ROOT:-$AUD_ROOT/output/audflash-stride1-ctc-v1}"
export AUD_USE_SLURM=0 AUD_NNODES=1 AUD_NODE_RANK=0 AUD_GPUS_PER_NODE=8 AUD_COMMAND=train
export AUD_MASTER_ADDR="${AUD_MASTER_ADDR:-127.0.0.1}"
export AUD_MASTER_PORT="${AUD_MASTER_PORT:-29579}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${AUD_NEXT_TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/audflash-ctc-triton-${USER:-user}}"
export PYTHONUNBUFFERED=1

command="${1:-help}"
shift || true
stage_model() {
  case "$1" in
    adaptation) AUD_STAGE_MODEL="$AUD_NEXT_ROOT/export" ;;
    full) AUD_STAGE_MODEL="$AUD_NEXT_ROOT/adaptation" ;;
    *) echo 'Stage must be adaptation or full' >&2; exit 2 ;;
  esac
}
case "$command" in
  prepare)
    exec "$AUD_PYTHON" -m audm.next_run --source "$AUD_NEXT_SOURCE" --output "$AUD_NEXT_ROOT" "$@" ;;
  tokenizer)
    exec "$AUD_PYTHON" -m audm.ctc tokenizer --config "$AUD_NEXT_ROOT/adaptation.yaml" --output "$AUD_NEXT_ROOT/ctc-tokenizer" "$@" ;;
  export)
    exec "$AUD_PYTHON" -m audm.ctc export --model-path "$AUD_NEXT_SOURCE" --tokenizer-path "$AUD_NEXT_ROOT/ctc-tokenizer" --output "$AUD_NEXT_ROOT/export" "$@" ;;
  verify)
    directory="${1:-export}"; shift || true
    case "$directory" in export|adaptation|full) ;; *) echo 'Verify export, adaptation or full' >&2; exit 2 ;; esac
    exec "$AUD_PYTHON" -m audm verify "$AUD_NEXT_ROOT/$directory" --device "${AUD_VERIFY_DEVICE:-cuda:0}" "$@" ;;
  health)
    exec "$AUD_PYTHON" -m audm.health "$@" ;;
  fused-check)
    exec "$AUD_PYTHON" -m audm.fused_check "$@" ;;
  distributed-check)
    export AUD_COMMAND=distributed-check
    exec bash "$AUD_ROOT/audm/scripts/train_distributed.sh" --expected-world-size 8 "$@" ;;
  preflight)
    stage="${1:?Specify adaptation or full}"; shift
    stage_model "$stage"
    exec "$AUD_PYTHON" -m audm preflight --config "$AUD_NEXT_ROOT/$stage.yaml" --model "$AUD_STAGE_MODEL" --output "$AUD_NEXT_ROOT/$stage" "$@" ;;
  probe)
    stage="${1:?Specify adaptation or full}"; shift
    stage_model "$stage"
    probe="${AUD_PROBE_ROOT:-$AUD_NEXT_ROOT/probe-$stage}"
    "$AUD_PYTHON" -m audm.memory_probe --config "$AUD_NEXT_ROOT/$stage.yaml" --model "$AUD_STAGE_MODEL" --output "$probe" "$@"
    exec bash "$AUD_ROOT/audm/scripts/train_distributed.sh" --config "$probe/recipe.yaml" --model "$AUD_STAGE_MODEL" --output "$probe/run" ;;
  smoke)
    stage="${1:?Specify adaptation or full}"; shift
    stage_model "$stage"
    exec bash "$AUD_ROOT/audm/scripts/train_distributed.sh" --config "$AUD_NEXT_ROOT/$stage.yaml" --model "$AUD_STAGE_MODEL" --output "$AUD_NEXT_ROOT/$stage" --smoke "$@" ;;
  adaptation|full)
    stage_model "$command"
    exec bash "$AUD_ROOT/audm/scripts/train_distributed.sh" --config "$AUD_NEXT_ROOT/$command.yaml" --model "$AUD_STAGE_MODEL" --output "$AUD_NEXT_ROOT/$command" "$@" ;;
  *)
    cat <<'HELP'
Usage: bash audm/scripts/audflash_next.sh COMMAND
  prepare                    Prepare fresh stride-one CTC recipes from checkpoint-3700
  tokenizer                  Train compact CTC tokenizer on training transcripts (no audio)
  export                     Add new CTC head; retain existing speech weights/tokenizer
  verify [export|adaptation|full]
  health [--require-idle]     GPU/thermal check; does not terminate jobs
  distributed-check          NCCL collective test on eight GPUs
  preflight adaptation|full --scan
  probe adaptation|full [--steps 3]   Synthetic test; discard its trained weights
  smoke adaptation|full      Real-data training rehearsal in a separate smoke-test directory
  adaptation                 New 2000-update decoder/bridge/CTC warm-up, encoder frozen
  full                       New 3-epoch all-parameter training from adaptation output
  fused-check                Optional Liger CUDA equivalence check

Defaults: batch 40 x accumulation 2 x 8 GPUs = 640; eval/WER batch 40; 4 workers/GPU.
Full eval/save every 3700 updates. CTC weight 0.1, warmed up over 100/500 updates.
Run from the audm environment. See audm/NEXT_RUN_CLUSTER.md for the complete sequence.
HELP
    [[ "$command" == help ]] || exit 2 ;;
esac
