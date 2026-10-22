#!/usr/bin/env bash
set -euo pipefail
AUD_REPO_ROOT="${AUD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$AUD_REPO_ROOT/audm/src:$AUD_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
AUD_RIR_ROOT="${AUD_RIR_ROOT:-/lustrefs/shared/shahin.konadath/workspace/train/stt/rirs-noises}"
ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
exec "${AUD_PYTHON:-python}" -m audm.prepare_rirs "$ACTION" --root "$AUD_RIR_ROOT" "$@"
