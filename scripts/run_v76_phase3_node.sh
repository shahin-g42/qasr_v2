#!/bin/bash
# QASR v7.6 — PHASE 3 (HQ polish) per-node launcher.
# Thin wrapper over run_v76_phase1_node.sh with the Phase 3 config pinned.
#
#   bash scripts/run_v76_phase3_node.sh <rank 0-7> <master-hostname>
set -euo pipefail
export CONFIG="${CONFIG:-configs/v7.6/03_hq_8node.yaml}"
exec bash "$(dirname "$0")/run_v76_phase1_node.sh" "$@"
