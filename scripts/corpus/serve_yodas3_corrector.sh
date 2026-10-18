#!/usr/bin/env bash
# Run once on each of the 11 corrector nodes; existing services may be reused.
set -euo pipefail
COMMAND="${1:-print}"
IMAGE="${LLM_IMAGE:-vllm/vllm-openai:v0.26.0-x86_64-cu129}"
MODEL="${LLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"
PORT="${LLM_PORT:-8010}"
CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# The cu129 image's compatibility driver is used on the cluster's older CUDA driver.
PRELUDE='for d in /usr/local/cuda-*/compat; do if [ -d "$d" ]; then export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi; done; exec vllm serve "$@"'
CMD=(docker run -d --name "${LLM_CONTAINER:-yodas3-corrector}" --restart unless-stopped
     --gpus all --ipc=host --network host
     -v /lustrefs:/lustrefs:ro -v "$CACHE:/root/.cache/huggingface"
     -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 --entrypoint bash "$IMAGE"
     -c "$PRELUDE" bash "$MODEL" --served-model-name corrector
     --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "${LLM_TP:-8}"
     --max-model-len "${LLM_CONTEXT:-16384}" --max-num-seqs "${LLM_MAX_NUM_SEQS:-256}"
     --gpu-memory-utilization "${LLM_GPU_UTIL:-0.90}")
case "$COMMAND" in
  print) printf '%q ' "${CMD[@]}"; printf '\n' ;;
  start) exec "${CMD[@]}" ;;
  *) echo 'Usage: serve_yodas3_corrector.sh {print|start}' >&2; exit 1 ;;
esac
