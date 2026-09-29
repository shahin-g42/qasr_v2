#!/bin/bash
# ONE-TIME (any single node): build the shared data-tools venv at $BASE/.hfenv.
#
# The training conda env cannot take the HF `datasets` stack (it fights the
# pinned torch/transformers), so corpus stages run in a plain python container
# instead. The venv lives on the shared filesystem: built once here, reused
# read-only by all nine nodes through scripts/corpus/docker.sh.
#
# .hfenv/bin/python symlinks the IMAGE's interpreter, so IMAGE here and in
# docker.sh are a matched pair -- bump both together and rebuild.

set -euo pipefail
BASE="${BASE:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
IMAGE="${IMAGE:-python:3.12-slim}"

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$BASE":"$BASE" \
    -w "$BASE" \
    "$IMAGE" \
    bash -c 'python -m venv .hfenv && \
             .hfenv/bin/pip install --upgrade pip && \
             .hfenv/bin/pip install "datasets>=3.0" "huggingface-hub[hf_transfer]" \
                 pyyaml numpy scipy soundfile httpx'

echo "shared venv ready: $BASE/.hfenv (used by scripts/corpus/docker.sh)"
