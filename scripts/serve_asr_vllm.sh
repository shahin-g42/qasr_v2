#!/usr/bin/env bash
# Serve a trained QASR checkpoint with vLLM, reachable from every cluster node.
#
#   scripts/serve_asr_vllm.sh preflight        # GPUs free? image? checkpoint layout?
#   scripts/serve_asr_vllm.sh gpu-test         # image computes on this driver? (1 GPU)
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
# Audio decode threads per API server (vLLM default is 2).
AUDIO_WORKERS="${AUDIO_WORKERS:-8}"
# vLLM's [audio] extra (setup.py v0.26.0), missing from the image.
AUDIO_PKGS="${AUDIO_PKGS:-soundfile av soxr scipy}"
MEDIA_ROOT="${MEDIA_ROOT:-/lustrefs}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
ngpus() { tr ',' '\n' <<<"$GPUS" | grep -c .; }

# CUDA the image was built for vs. what the host driver supports. A -cu129
# image on an older 12.x driver (this cluster reports 12.8) runs through CUDA
# forward compatibility: the image ships /usr/local/cuda-12.9/compat, which is
# supported on datacenter GPUs (H100) with R535/R550/R570 drivers. A CUDA 13
# image cannot be rescued this way on a 12.x driver.
image_cuda()  { case "$IMAGE" in *cu129*) echo 12.9 ;; *) echo 13.0 ;; esac; }
driver_cuda() { nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}'; }
version_lt()  { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]; }
# CUDA_COMPAT=auto|1|0. auto turns it on when the driver is older than the image.
cuda_compat() {
  case "${CUDA_COMPAT:-auto}" in
    1|true) echo 1 ;;
    0|false) echo 0 ;;
    *) local d; d=$(driver_cuda); { [ -n "$d" ] && version_lt "$d" "$(image_cuda)"; } && echo 1 || echo 0 ;;
  esac
}
# Shell snippet run inside the container before python starts: putting the
# compat libcuda first on LD_LIBRARY_PATH covers the API server process too
# (vLLM's own VLLM_ENABLE_CUDA_COMPATIBILITY only reaches its subprocesses).
compat_prelude() {
  [ "$(cuda_compat)" = 1 ] || return 0
  printf '%s' 'for d in /usr/local/cuda-*/compat; do if [ -d "$d" ]; then export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi; done; '
}
compat_env() { [ "$(cuda_compat)" = 1 ] && printf '%s' '-e VLLM_ENABLE_CUDA_COMPATIBILITY=1 ' || true; }

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
  -e VLLM_MAX_AUDIO_PREPROCESS_WORKERS=${AUDIO_WORKERS} \\
  $(compat_env)--entrypoint bash ${IMAGE} -c '
    set -e
    $(compat_prelude)
    # The vllm-openai image ships vLLM WITHOUT its [audio] extra: no soundfile/av
    # to decode files, no scipy for the training-matched resampler.
    pip install -q --no-cache-dir ${AUDIO_PKGS}
    python3 -c "import soundfile, av, scipy.signal, soxr; print(\\"audio deps ok\\")"
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
    # The image's CUDA runtime must not exceed what the driver supports: the
    # default v0.26.0 tag is CUDA 13 and will not start on a 12.9 driver.
    local drv_cuda img_cuda
    drv_cuda=$(driver_cuda); img_cuda=$(image_cuda)
    log "  driver CUDA         : ${drv_cuda:-unknown}  (image built for ${img_cuda})"
    if [ -n "$drv_cuda" ] && version_lt "$drv_cuda" "$img_cuda"; then
      if [ "${drv_cuda%%.*}" != "${img_cuda%%.*}" ]; then
        log "  !! CUDA ${img_cuda} image cannot run on a ${drv_cuda} driver -- use a -cu129 image"; fail=1
      elif [ "$(cuda_compat)" = 1 ]; then
        log "  CUDA compat         : ON (forward-compat libs from the image; verify with 'gpu-test')"
      else
        log "  !! driver ${drv_cuda} < image ${img_cuda} and CUDA_COMPAT=0"; fail=1
      fi
    fi
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

# One-GPU check that the image really computes on this driver: a cuBLAS
# matmul and a freshly compiled Triton kernel (the paths a version mismatch
# breaks first). Triton reads kernel source from a file, so the script is
# written to /tmp inside the container. Runs in ~1 minute once pulled.
gpu_test() {
  log "=== gpu-test: ${IMAGE} on GPU ${GPUS%%,*} (CUDA compat=$(cuda_compat)) ==="
  docker run --rm -i --gpus "\"device=${GPUS%%,*}\"" $(compat_env)--entrypoint bash "$IMAGE" -c \
    "$(compat_prelude)pip install -q --no-cache-dir ${AUDIO_PKGS} && cat > /tmp/gpu_test.py && python3 /tmp/gpu_test.py" <<'PY'
import ctypes, os, torch, triton, triton.language as tl
print(f"  torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}")
print(f"  LD_LIBRARY_PATH head: {os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] or '-'}")
# The libcuda actually loaded: 12090 = compat lib in use, 12080 = host driver.
drv = ctypes.c_int()
ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(drv))
print(f"  device: {torch.cuda.get_device_name(0)}, libcuda API {drv.value}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
ref = (x.float() @ x.float())
err = ((x @ x).float() - ref).abs().max().item() / ref.abs().max().item()
print(f"  cuBLAS bf16 matmul: rel err {err:.2e}")
assert err < 1e-2

@triton.jit
def add(a, b, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.load(a + i, mask=m) + tl.load(b + i, mask=m), mask=m)

a = torch.randn(10_000, device="cuda"); b = torch.randn_like(a); o = torch.empty_like(a)
add[(triton.cdiv(a.numel(), 1024),)](a, b, o, a.numel(), BLOCK=1024)
assert torch.allclose(o, a + b)
print("  triton JIT kernel  : OK")

# Audio path exactly as the server runs it: vLLM's file loader (soundfile at the
# native rate) and the scipy resampler the plugin selects, on a 44.1 kHz stereo wav.
import numpy as np, soundfile
from vllm.multimodal.media.audio import load_audio
from vllm.multimodal.audio import resample_audio_scipy
t = np.arange(44_100 * 2) / 44_100
soundfile.write("/tmp/probe.wav", np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32), 44_100)
y, sr = load_audio("/tmp/probe.wav", sr=None)
y = resample_audio_scipy(y, orig_sr=sr, target_sr=16_000)
assert y.ndim == 1 and abs(len(y) - 32_000) <= 1, (y.shape, sr)
print(f"  vLLM audio loader  : OK ({sr} Hz stereo -> {len(y)} samples mono 16 kHz)")
print("GPU TEST PASSED")
PY
}

case "${1:-preflight}" in
  preflight) preflight ;;
  gpu-test)  gpu_test ;;
  serve)     serve ;;
  smoke)     smoke "${2:-}" ;;
  print)     docker_cmd ;;
  stop)      docker rm -f "$CONTAINER" >/dev/null 2>&1 && log "  stopped ${CONTAINER}" || log "  not running" ;;
  *) die "usage: $0 {preflight|gpu-test|serve|smoke [manifest]|print|stop}" ;;
esac
