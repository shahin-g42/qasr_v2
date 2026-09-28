#!/bin/bash
# QASR v7.6 — PHASE 2 (full fine-tune) per-node launcher.
# Thin wrapper over run_v76_phase1_node.sh with the Phase 2 config pinned.
#
#   Node 0 (master):  bash scripts/run_v76_phase2_node.sh 0 <master-hostname>
#   Node N:           bash scripts/run_v76_phase2_node.sh N <master-hostname>
#
# BEFORE the 64-GPU launch, run the batch-20 memory rehearsal on one node:
#   CONFIG=configs/v7.6/02_full_8node.yaml NNODES=1 \
#     bash scripts/run_v76_phase1_node.sh 0 localhost --smoke-test
set -euo pipefail
export CONFIG="${CONFIG:-configs/v7.6/02_full_8node.yaml}"
exec bash "$(dirname "$0")/run_v76_phase1_node.sh" "$@"
