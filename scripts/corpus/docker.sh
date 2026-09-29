#!/bin/bash
# Run any corpus command inside the data-tools container, against the shared
# venv built by scripts/corpus/docker_build_env.sh. `python3 -m ...` becomes
# `scripts/corpus/docker.sh -m ...`:
#
#   scripts/corpus/docker.sh -m data_processing.build_corpus preflight \
#       --probe-fields --json --root "$INTERNAL_ROOT"
#
# This is env.sh's docker twin: it feeds the SAME path defaults (edit the two
# together), mounts the three filesystems the stages touch, and keeps the HF
# cache node-local on /scratch so nine nodes never race one cache. It runs as
# your uid (nothing on the shared filesystem ends up root-owned) and uses
# --network host, which keeps localhost:8010 (the co-located corrector)
# reachable for stage 2 and gives the container the host's outbound internet.
#
# Source scripts/corpus/env.sh on the host first as usual: its conda activation
# is irrelevant (python runs in the container), but the path vars it exports
# are forwarded into the container by this wrapper.

set -euo pipefail
BASE="${BASE:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
IMAGE="${IMAGE:-python:3.12-slim}"
PY="$BASE/.hfenv/bin/python"

[ -x "$PY" ] || { echo "no shared venv at $PY -- run scripts/corpus/docker_build_env.sh once" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker not on PATH" >&2; exit 1; }

# Node-local dirs (ledgers: SQLite-WAL must never sit on Lustre; HF cache:
# per-node so concurrent nodes never share one download cache).
mkdir -p /scratch/corpus/hf /scratch/corpus/ledgers

exec docker run --rm \
    --user "$(id -u):$(id -g)" \
    --network host \
    -e HOME=/tmp \
    -e PYTHONPATH="$BASE/src" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/scratch/corpus/hf \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e POOL_DIR="${POOL_DIR:-$BASE/corpus/pool}" \
    -e AUDIO_ROOT="${AUDIO_ROOT:-$BASE/corpus/audio}" \
    -e OUT_DIR="${OUT_DIR:-$BASE/training_manifests/v8.0}" \
    -e LOGS="${LOGS:-$BASE/logs/corpus}" \
    -e LEDGER_DIR="${LEDGER_DIR:-/scratch/corpus/ledgers}" \
    -e INTERNAL_ROOT="${INTERNAL_ROOT:-$BASE/training_manifests/v7.6}" \
    -e QASR_SFT_ROOT="${QASR_SFT_ROOT:-$BASE/q3asr_sft_manifests}" \
    -e QASR_EMILIA_ROOT="${QASR_EMILIA_ROOT:-/vast/audio/data/tts/44k/Emilia-Dataset-extracted}" \
    -e CORRECTOR_URL="${CORRECTOR_URL:-http://localhost:8010/v1}" \
    -v "$BASE":"$BASE" \
    -v /scratch:/scratch \
    -v /vast:/vast \
    -w "$BASE" \
    "$IMAGE" \
    "$PY" "$@"
