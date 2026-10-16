#!/usr/bin/env bash
# One GPU, sequential fresh processes, same sampled batch and unchanged checkpoint.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
failed=0
for mode in bridge decoder acoustic full; do
  echo "=== Backward isolation: $mode, BF16, configured checkpointing/augmentation ==="
  bash gemma_speech/cluster/gqasr_stage.sh backward-audit "$@" --mode "$mode" --precision bf16 || failed=1
done
echo '=== Full backward: no checkpointing ==='
bash gemma_speech/cluster/gqasr_stage.sh backward-audit "$@" --mode full --precision bf16 --no-checkpointing || failed=1
echo '=== Full backward: FP32, no checkpointing ==='
bash gemma_speech/cluster/gqasr_stage.sh backward-audit "$@" --mode full --precision fp32 --no-checkpointing || failed=1
echo '=== Full backward: no augmentation ==='
bash gemma_speech/cluster/gqasr_stage.sh backward-audit "$@" --mode full --precision bf16 --no-augmentation || failed=1
exit "$failed"
