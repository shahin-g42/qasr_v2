#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE="${1:?Stages: doctor download export verify scan rescan-ml preflight augmentation-smoke backward-audit test bridge full encoder transcribe}"
shift
ENV_DIR="${GEMMA_SPEECH_ENV:-$ROOT/.venv}"
source "$ENV_DIR/bin/activate"
# Preserve the launcher selection even if environment activation changes visibility.
if [[ -n "${GQASR_CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GQASR_CUDA_VISIBLE_DEVICES"
fi
LOG_DIR="${GQASR_LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR" "$ROOT/gqasr/verification"
exec > >(tee "$LOG_DIR/gqasr-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ)-${NODE_RANK:-local}.log") 2>&1
export TOKENIZERS_PARALLELISM=false
BRIDGE="$ROOT/gqasr/configs/01_bridge_12node.yaml"
FULL="$ROOT/gqasr/configs/02_full_b40_12node.yaml"
ENCODER="$ROOT/gqasr/configs/03_encoder12_b80_12node.yaml"
echo "GQASR stage=$STAGE host=$(hostname) time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
case "$STAGE" in
  backward-audit)
    export HF_HUB_OFFLINE=1
    python -m gqasr.backward_audit --config "${GQASR_CONFIG:-$FULL}" "$@"
    ;;
  doctor)
    nvidia-smi
    python - <<'PYCODE'
import os
import sys
import torch
import transformers
expected = int(os.environ.get("NPROC_PER_NODE", "8"))
from gqasr.env_check import check_training_profile
check_training_profile()
visible = torch.cuda.device_count()
print({"python": sys.executable, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
       "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
       "visible_gpus": visible, "expected_gpus": expected}, flush=True)
if visible != expected:
    raise SystemExit(
        f"PyTorch sees {visible} GPUs; this launcher requires {expected}. "
        "Check CUDA_VISIBLE_DEVICES and your GPU allocation. For a dedicated eight-GPU node, "
        "launch with GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_gqasr_12node.sh YOUR_RANK. "
        "If visibility is already correct, check container/scheduler GPU access."
    )
assert torch.cuda.is_available(), "CUDA is unavailable"
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
  rescan-ml)
    export HF_HUB_OFFLINE=1
    gqasr rescan-language --config "${GQASR_CONFIG:-$BRIDGE}" --language ml \
      --report "$ROOT/gqasr/verification/rescan-ml-$(date -u +%Y%m%dT%H%M%SZ).json" "$@"
    ;;
  augmentation-smoke)
    export HF_HUB_OFFLINE=1
    gqasr augmentation-smoke --config "${GQASR_CONFIG:-$FULL}" \
      --model "$ROOT/gqasr/models/gqasr-base" \
      --report "$ROOT/gqasr/verification/augmentation-$(date -u +%Y%m%dT%H%M%SZ).json" "$@"
    ;;
  preflight)
    export HF_HUB_OFFLINE=1
    gqasr preflight --config "${GQASR_CONFIG:-$BRIDGE}" "$@"
    ;;
  test)
    export HF_HUB_OFFLINE=1
    python -m pytest "$ROOT/gqasr/tests" "$@"
    ;;
  bridge|full|encoder)
    CONFIG="$BRIDGE"
    if [[ "$STAGE" == full ]]; then CONFIG="$FULL"; fi
    if [[ "$STAGE" == encoder ]]; then CONFIG="$ENCODER"; fi
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
