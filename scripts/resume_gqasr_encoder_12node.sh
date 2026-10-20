#!/usr/bin/env bash
# Explicit legacy 12-node recovery; the generic recovery defaults to four nodes.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_DIR GQASR_LAUNCH_NNODES=12
export GQASR_CONFIG="${GQASR_CONFIG:-$PROJECT_DIR/gemma_speech/gqasr/configs/03_encoder12_b40_resume_12node.yaml}"
exec bash "$PROJECT_DIR/scripts/resume_gqasr_encoder.sh" "$@"
