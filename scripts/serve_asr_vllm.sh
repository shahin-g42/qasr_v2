#!/usr/bin/env bash
# Serve a trained QASR checkpoint with vLLM, reachable from every cluster node.
#
#   scripts/serve_asr_vllm.sh preflight        # GPUs free? image? checkpoint layout?
#   scripts/serve_asr_vllm.sh serve            # start the container (detached)
#   scripts/serve_asr_vllm.sh smoke [MANIFEST] # health, prompt parity, a few transcripts
#   scripts/serve_asr_vllm.sh print            # print the docker command, run nothing
#   scripts/serve_asr_vllm.sh stop
#
# Stock vLLM cannot load model_type "qasr" (Parakeet/Cohere Conformer welded to
# Qwen3), so the container pip-installs vllm_plugin/ (entry point
# vllm.general_plugins) before `vllm serve`. The plugin keeps the trained HF
# Parakeet encoder (bit-exact audio embeddings, see tests/test_qasr_vllm_audio.py)
# and runs the Qwen3 decoder on vLLM's kernels.
#
# The plugin is written against vLLM v0.26.0 -- keep IMAGE pinned to it.
#
# One engine replica per GPU (data parallel, ~7.3 GB weights each) behind one
# OpenAI-compatible endpoint on the host network:
#   http://inception-H100-hpc-029:8020/v1/chat/completions      (file:// audio)
#   http://inception-H100-hpc-029:8020/v1/audio/transcriptions  (upload)

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

IMAGE="${IMAGE:-vllm/vllm-openai:v0.26.0-x86_64-cu129}"
# Phase-2 (full fine-tune) output. Point at a checkpoint-N dir to serve a snapshot.
MODEL_DIR="${MODEL_DIR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/output/v76/full}"
CONTAINER="${CONTAINER:-qasr-asr}"
PORT="${PORT:-8020}"                 # corrector owns 8010
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPU_UTIL="${GPU_UTIL:-0.85}"
# 35 s audio -> 438 encoder tokens + ~30 prompt tokens + transcript.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
# Mel extraction and audio decoding run in the API-server processes.
API_SERVERS="${API_SERVERS:-8}"
MEDIA_ROOT="${MEDIA_ROOT:-/lustrefs}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
ngpus() { tr ',' '\n' <<<"$GPUS" | grep -c .; }

docker_cmd() {
  # Plugin source is mounted read-only and copied before install (pip writes
  # build artifacts next to the source).
  cat <<CMD
docker run -d --name ${CONTAINER} --restart unless-stopped \\
  --gpus '"device=${GPUS}"' --ipc=host --network host \\
  -v ${MODEL_DIR}:/model:ro \\
  -v ${REPO}/vllm_plugin:/opt/qasr_vllm_src:ro \\
  -v ${MEDIA_ROOT}:${MEDIA_ROOT}:ro \\
  -v ${HF_CACHE}:/root/.cache/huggingface \\
  --entrypoint bash ${IMAGE} -c '
    set -e
    cp -r /opt/qasr_vllm_src /tmp/qasr_vllm
    pip install --no-deps --no-build-isolation -q /tmp/qasr_vllm
    exec vllm serve /model \\
      --served-model-name qasr \\
      --host 0.0.0.0 --port ${PORT} \\
      --dtype bfloat16 \\
      --data-parallel-size $(ngpus) \\
      --api-server-count ${API_SERVERS} \\
      --max-model-len ${MAX_MODEL_LEN} \\
      --max-num-seqs ${MAX_NUM_SEQS} \\
      --gpu-memory-utilization ${GPU_UTIL} \\
      --limit-mm-per-prompt "{\"audio\": 1}" \\
      --allowed-local-media-path ${MEDIA_ROOT} \\
      --generation-config vllm'
CMD
}

preflight() {
  local fail=0
  log "=== preflight: QASR vLLM on $(hostname) ==="
  [ -f "$MODEL_DIR/config.json" ] || { log "  !! no config.json in $MODEL_DIR"; fail=1; }
  if [ -f "$MODEL_DIR/config.json" ]; then
    python3 - "$MODEL_DIR" <<'PY' || fail=1
import json, sys, glob, os
d = sys.argv[1]
c = json.load(open(os.path.join(d, "config.json")))
ok = c.get("model_type") == "qasr" and c.get("architectures") == ["QASRForConditionalGeneration"]
print(f"  checkpoint          : model_type={c.get('model_type')} arch={c.get('architectures')}")
print(f"  projector           : depth={c.get('projector_depth', 1)} pool_stride={c.get('projector_pool_stride', 1)}")
if c.get("projector_pool_stride", 1) != 1:
    print("  !! projector_pool_stride > 1 is not supported by the plugin"); ok = False
w = glob.glob(os.path.join(d, "*.safetensors"))
tok = [f for f in ("tokenizer_config.json", "tokenizer.json", "vocab.json") if os.path.exists(os.path.join(d, f))]
tmpl = [f for f in ("chat_template.jinja", "chat_template.json") if os.path.exists(os.path.join(d, f))]
print(f"  weights             : {len(w)} safetensors file(s)")
print(f"  tokenizer files     : {tok or 'MISSING'}")
print(f"  chat template       : {tmpl or 'MISSING (pass --chat-template vllm_plugin/qasr_vllm/chat_template.jinja)'}")
sys.exit(0 if ok and w and tok else 1)
PY
  fi
  if command -v nvidia-smi >/dev/null; then
    log "  GPUs requested      : ${GPUS}"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | while IFS=', ' read -r idx used total; do
          case ",${GPUS}," in *",${idx},"*) ;; *) continue ;; esac
          if [ "$used" -gt 2000 ]; then
            log "  !! GPU ${idx} already has ${used} MiB in use (corrector?) -- set GPUS to free ones"
          else
            log "  GPU ${idx}               : free (${used}/${total} MiB)"
          fi
        done
  else
    log "  !! nvidia-smi not found"; fail=1
  fi
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    log "  !! port ${PORT} already serving"; fail=1
  fi
  if command -v docker >/dev/null; then
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
      && log "  image               : ${IMAGE} (cached)" \
      || log "  image               : ${IMAGE} (will be pulled)"
  else
    log "  !! docker not found"; fail=1
  fi
  return $fail
}

serve() {
  preflight || die "preflight failed"
  log "=== starting ${CONTAINER} on GPUs ${GPUS}, port ${PORT} ==="
  eval "$(docker_cmd)"
  local tries=0
  until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    tries=$((tries + 1))
    [ "$tries" -gt 120 ] && die "not healthy after 20 min -- docker logs ${CONTAINER}"
    docker ps -q -f name="^${CONTAINER}$" | grep -q . || die "container exited -- docker logs ${CONTAINER}"
    sleep 10
  done
  log "  ready: http://$(hostname):${PORT}/v1  (model name: qasr)"
}

smoke() {
  local manifest="${1:-}" url="http://localhost:${PORT}"
  curl -sf "${url}/v1/models" | python3 -c 'import sys,json;print("  served:", [m["id"] for m in json.load(sys.stdin)["data"]])' \
    || die "nothing served on ${PORT}"
  [ -n "$manifest" ] || { log "  pass a manifest (jsonl) to test prompt parity and transcripts"; return 0; }
  local lang first
  lang="${LANG_CODE:-$(basename "$manifest" | grep -oE '_(ar|en|zh|hi|ml)[_.]' | head -1 | tr -d '_.')}"
  [ -n "$lang" ] || die "cannot infer language from $manifest; set LANG_CODE"
  first=$(head -1 "$manifest" | python3 -c 'import sys,json;r=json.loads(sys.stdin.read());print(r.get("audio_filepath") or r["wav_path"])')
  python3 scripts/asr_vllm_client.py check-prompt --url "$url" --lang "$lang" "$first" \
    || die "chat route does not render the training prompt -- restart with --chat-template"
  python3 scripts/asr_vllm_client.py sample --url "$url" --lang "$lang" --manifest "$manifest" -n "${N:-10}"
}

case "${1:-preflight}" in
  preflight) preflight ;;
  serve)     serve ;;
  smoke)     smoke "${2:-}" ;;
  print)     docker_cmd ;;
  stop)      docker rm -f "$CONTAINER" >/dev/null 2>&1 && log "  stopped ${CONTAINER}" || log "  not running" ;;
  *) die "usage: $0 {preflight|serve|smoke [manifest]|print|stop}" ;;
esac
