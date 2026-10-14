#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE="${1:?Stages: doctor download export verify scan preflight test bridge full transcribe}"
shift
ENV_DIR="${GEMMA_SPEECH_ENV:-$ROOT/.venv}"
source "$ENV_DIR/bin/activate"
mkdir -p "$ROOT/logs" "$ROOT/gqasr/verification"
exec > >(tee "$ROOT/logs/gqasr-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ)-${NODE_RANK:-local}.log") 2>&1
export TOKENIZERS_PARALLELISM=false
BRIDGE="$ROOT/gqasr/configs/01_bridge_12node.yaml"
FULL="$ROOT/gqasr/configs/02_full_12node.yaml"
echo "GQASR stage=$STAGE host=$(hostname) time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
case "$STAGE" in
  doctor)
    nvidia-smi
    python - <<'PYCODE'
import os
import torch
import transformers
expected = int(os.environ.get("NPROC_PER_NODE", "8"))
assert torch.__version__.split("+")[0] == "2.11.0", torch.__version__
assert transformers.__version__ == "5.19.0", transformers.__version__
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
for i in range(expected):
    with torch.cuda.device(i):
        assert torch.cuda.is_bf16_supported(), f"GPU {i} lacks BF16"
        print({"gpu": i, "name": torch.cuda.get_device_name(i), "bf16": True})
print({"torch": str(torch.__version__), "transformers": transformers.__version__,
       "cuda_runtime": torch.version.cuda, "nccl": torch.cuda.nccl.version()})
PYCODE
    ;;
  download)
    export HF_HUB_OFFLINE=0
    gemma-speech prefetch --output "$ROOT/models/embeddinggemma2"
    gemma-speech prefetch --qwen-asr --output "$ROOT/models/qwen-asr"
    ;;
  export)
    export HF_HUB_OFFLINE=1
    gqasr export-base --acoustic-path "$ROOT/models/embeddinggemma2" \
      --decoder-path "$ROOT/models/qwen-asr" --output "$ROOT/gqasr/models/gqasr-base" "$@"
    ;;
  verify)
    export HF_HUB_OFFLINE=1
    gqasr verify-base --model "$ROOT/gqasr/models/gqasr-base" \
      --report "$ROOT/gqasr/verification/base-$(date -u +%Y%m%dT%H%M%SZ).json" "$@"
    ;;
  scan)
    export HF_HUB_OFFLINE=1
    gqasr preflight --config "${GQASR_CONFIG:-$BRIDGE}" --scan --verify-audio \
      --report "$ROOT/gqasr/verification/scan-$(date -u +%Y%m%dT%H%M%SZ).json" "$@"
    ;;
  preflight)
    export HF_HUB_OFFLINE=1
    gqasr preflight --config "${GQASR_CONFIG:-$BRIDGE}" "$@"
    ;;
  test)
    export HF_HUB_OFFLINE=1
    python -m pytest "$ROOT/gqasr/tests" "$@"
    ;;
  bridge|full)
    CONFIG="$BRIDGE"
    if [[ "$STAGE" == full ]]; then CONFIG="$FULL"; fi
    CONFIG="${GQASR_CONFIG:-$CONFIG}"
    export HF_HUB_OFFLINE=1
    python - "$CONFIG" <<'PY'
import sys
from gqasr.cli import preflight
from gqasr.config import load_config
result = preflight(load_config(sys.argv[1]))
if not result['ready_to_train']:
    raise SystemExit(f"Preflight incomplete: {result}")
PY
    "$ROOT/gqasr/scripts/torchrun_node.sh" "$CONFIG" "$@"
    ;;
  transcribe)
    export HF_HUB_OFFLINE=1
    gqasr transcribe "$@"
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
