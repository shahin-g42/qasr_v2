#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on 8x H100 for corpus transcript correction.
#
# Subcommands:
#   preflight   verify GPU/RAM/disk/image before anything is downloaded
#   serve       launch TEP8 -- one instance across all 8 GPUs (recipe's 8-GPU shape)
#   serve-2x4   launch two independent TP4 instances (the H100-validated shape, x2)
#   smoke       health check + one real Arabic correction from your own data
#   bench       measure actual output tok/s on THIS hardware
#   stop        tear down
#
# Everything here comes from the official vLLM recipe for this checkpoint. Three
# things about it are non-obvious and each one silently breaks a naive launch:
#
#   1. PyPI vLLM cannot serve it. The architecture is Qwen4ExpForConditionalGeneration
#      and the recipe states PyPI installation is not supported -- you must use
#      the dedicated image vllm/vllm-openai:qwen38-flash-next.
#   2. Plain TP8 is INCOMPATIBLE with the FP8 checkpoint (128-wide quantization
#      blocks). On 8 GPUs you need TEP8: --enable-expert-parallel.
#   3. On 80GB GPUs the 51B N-gram/PLE embedding table does not fit, so the
#      engine OOMs at startup unless VLLM_PLE_CPU_OFFLOAD=1 is set.
#
# Do NOT enable MTP speculative decoding: measured 8-36% lower throughput and
# 32-173% higher per-token latency on H100 at a ~36% acceptance rate.

set -euo pipefail

# Image choice matters more than it looks, and the version tag is the trap.
#
#   vllm/vllm-openai:v0.26.0-*              released 2026-07-25 -- CANNOT serve
#                                           this model. Flash-Next shipped
#                                           2026-08-26, a month later, so
#                                           Qwen4ExpForConditionalGeneration is
#                                           simply not in that build. No flag
#                                           fixes a missing architecture.
#   vllm/vllm-openai:qwen38-flash-next      the recipe's default tag.
#   vllm/vllm-openai:qwen38-flash-next-x86_64-cu129
#                                           SAME model support, built on CUDA
#                                           12.9. Use this on clusters with an
#                                           older NVIDIA driver: if v0.26.0-cu129
#                                           already runs there, the driver
#                                           requirement is identical.
#
# Fallback (dense 27B, also cu129): vllm/vllm-openai:qwen38-cu129
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next-x86_64-cu129}"
MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}"
CONTAINER="${CONTAINER:-flashnext}"
PORT="${PORT:-8000}"
# Cap context hard. The checkpoint advertises 262,144 native tokens and vLLM
# reserves KV for whatever you allow; transcript correction needs ~800. Leaving
# this at the default wastes most of your KV cache on context you will never use.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_UTIL="${GPU_UTIL:-0.85}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# Disk needed for the FP8 checkpoint: 172.78 GiB, plus headroom.
NEEDED_GB="${NEEDED_GB:-220}"
# Host RAM for PLE offload: the 51B embedding table lives here.
NEEDED_RAM_GB="${NEEDED_RAM_GB:-80}"

log()  { printf '%s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- common engine arguments ------------------------------------------------
# Shared by both topologies. --limit-mm-per-prompt drops the vision tower: this
# is a vision-language checkpoint and we only ever send text.
common_args() {
  cat <<ARGS
--served-model-name corrector
--max-model-len ${MAX_MODEL_LEN}
--max-num-seqs ${MAX_NUM_SEQS}
--gpu-memory-utilization ${GPU_UTIL}
--enable-prefix-caching
--no-enable-flashinfer-autotune
--moe-backend triton
--reasoning-parser qwen3
--limit-mm-per-prompt {"image":0,"video":0}
ARGS
}

preflight() {
  log "=== preflight: Qwen3.8-Flash-Next on this node ==="
  local fail=0

  # GPUs
  if have nvidia-smi; then
    local n vram
    n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log "  GPUs                : ${n} x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    log "  VRAM per GPU        : ${vram} MiB"
    [ "$n" -ge 8 ] || { log "  !! expected 8 GPUs, found ${n}"; fail=1; }
    # H100 is 80GB (81559 MiB). Anything below means PLE offload is mandatory.
    if [ "${vram:-0}" -lt 90000 ]; then
      log "  80GB-class GPU      : yes -> VLLM_PLE_CPU_OFFLOAD=1 is REQUIRED"
    fi
  else
    log "  !! nvidia-smi not found"; fail=1
  fi

  # Host RAM for the offloaded 51B embedding table
  local ram_gb
  if [ "$(uname)" = "Linux" ]; then
    ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    log "  Host RAM            : ${ram_gb} GiB (need >= ${NEEDED_RAM_GB})"
    [ "${ram_gb:-0}" -ge "$NEEDED_RAM_GB" ] || { log "  !! insufficient host RAM for PLE offload"; fail=1; }
  fi

  # Disk for the 172.78 GiB checkpoint
  local free_gb
  free_gb=$(df -BG --output=avail "$(dirname "$HF_CACHE")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  log "  Free disk at cache  : ${free_gb} GiB (need >= ${NEEDED_GB})"
  [ "${free_gb:-0}" -ge "$NEEDED_GB" ] || { log "  !! insufficient disk for the FP8 checkpoint"; fail=1; }

  # The single most common way to get this wrong: a version-tagged image built
  # before 2026-08-26 cannot contain the architecture, and it fails only after
  # the image pull and the 173 GB weight download.
  case "$IMAGE" in
    *qwen38-flash-next*) : ;;
    *) log "  !! IMAGE=${IMAGE} is not a qwen38-flash-next build."
       log "  !! Flash-Next needs vllm/vllm-openai:qwen38-flash-next-x86_64-cu129"
       log "     (v0.26.0 predates the model by a month and will not serve it)"
       fail=1 ;;
  esac
  case "$IMAGE" in
    *cu129*) log "  CUDA 12.9 build    : yes (compatible with older NVIDIA drivers)" ;;
    *)       log "  CUDA 12.9 build    : NO -- may need a newer driver than the cluster has" ;;
  esac

  # Runtime: PyPI vLLM will not work.
  if have docker; then
    log "  docker              : $(docker --version | cut -d, -f1)"
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      log "  image               : present (${IMAGE})"
    else
      log "  image               : NOT pulled -> docker pull ${IMAGE}"
    fi
  elif have apptainer; then
    log "  apptainer           : $(apptainer --version)  (SLURM/pyxis cluster)"
    log "  image               : will be pulled from docker://${IMAGE} at launch"
  elif have srun; then
    log "  !! SLURM present but no docker/apptainer -- use pyxis: srun --container-image=${IMAGE}"
    fail=1
  else
    log "  !! no container runtime found; PyPI vLLM CANNOT serve this checkpoint"
    fail=1
  fi

  # Host-installed vLLM is a trap: it will import fine and then fail on the arch.
  if python3 -c "import vllm" 2>/dev/null; then
    log "  host vLLM           : $(python3 -c 'import vllm;print(vllm.__version__)')  (do NOT use for this model)"
  fi

  # Gated / auth
  if [ -n "${HF_TOKEN:-}" ]; then
    log "  HF_TOKEN            : set"
  else
    log "  HF_TOKEN            : not set (fine if ${MODEL} is ungated)"
  fi

  log ""
  if [ "$fail" -eq 0 ]; then
    log "=== preflight PASSED ==="
  else
    log "=== preflight FAILED -- fix the !! items above ==="
    return 1
  fi
}

# Launch inside a container. Prefers docker, falls back to apptainer.
launch_container() {
  local name="$1"; shift
  local gpus="$1"; shift
  local port="$1"; shift

  if have docker; then
    docker rm -f "$name" >/dev/null 2>&1 || true
    log "  starting ${name} on GPU(s) ${gpus}, port ${port}"
    docker run -d --name "$name" --gpus "\"device=${gpus}\"" \
      --ipc=host --shm-size 32g \
      -e VLLM_PLE_CPU_OFFLOAD=1 \
      -e HF_HUB_ENABLE_HF_TRANSFER=1 \
      ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
      -v "${HF_CACHE}:/root/.cache/huggingface" \
      -p "${port}:8000" \
      "$IMAGE" \
      --model "$MODEL" --port 8000 "$@"
  elif have apptainer; then
    log "  starting ${name} via apptainer on GPU(s) ${gpus}, port ${port}"
    VLLM_PLE_CPU_OFFLOAD=1 HF_HUB_ENABLE_HF_TRANSFER=1 \
    apptainer run --nv --bind "${HF_CACHE}:/root/.cache/huggingface" \
      "docker://${IMAGE}" \
      --model "$MODEL" --port "$port" "$@" &
  else
    die "no container runtime"
  fi
}

serve() {
  log "=== TEP8: one instance, tensor+expert parallel across all 8 GPUs ==="
  log "    This is the recipe's own answer for 8 GPUs. Plain TP8 is rejected by the"
  log "    FP8 checkpoint's 128-wide quantization blocks; --enable-expert-parallel"
  log "    is what makes 8-way sharding work."
  preflight || die "preflight failed"
  # shellcheck disable=SC2046
  launch_container "$CONTAINER" "0,1,2,3,4,5,6,7" "$PORT" \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    $(common_args | tr '\n' ' ')
  log ""
  log "  waiting for readiness (model load + PLE offload takes several minutes)..."
  wait_ready "$PORT"
}

serve_2x4() {
  log "=== 2x TP4: two independent instances, the H100-validated shape doubled ==="
  log "    TP4 + PLE offload is what the recipe actually measured on 4x H100"
  log "    (~1,430 output tok/s at concurrency 64). Two of them ~= 2x that,"
  log "    and there is zero cross-instance communication."
  log "    Tradeoff vs TEP8: the 172.78 GiB checkpoint is resident twice, so each"
  log "    GPU has less room for KV cache and therefore smaller max batch -- and the"
  log "    95 GiB PLE table is offloaded to host RAM TWICE (~190 GiB vs ~95 for TEP8)."
  log "    Prefer 'serve' (TEP8) unless TEP8 specifically fails."
  preflight || die "preflight failed"
  # shellcheck disable=SC2046
  # --enable-expert-parallel is NOT optional here. moe_intermediate_size is 640
  # and the FP8 checkpoint uses [128,128] weight blocks; 640 = 5*128 is aligned,
  # but slicing it across ranks gives 320/160/80, none of which are. Expert
  # parallelism hands each rank whole experts (512/4 = 128) so the intermediate
  # dimension is never split.
  launch_container "${CONTAINER}-a" "0,1,2,3" "$PORT" \
    --tensor-parallel-size 4 --enable-expert-parallel $(common_args | tr '\n' ' ')
  # shellcheck disable=SC2046
  launch_container "${CONTAINER}-b" "4,5,6,7" "$((PORT + 1))" \
    --tensor-parallel-size 4 --enable-expert-parallel $(common_args | tr '\n' ' ')
  log ""
  wait_ready "$PORT"
  wait_ready "$((PORT + 1))"
  log "  both instances up: http://localhost:${PORT}/v1 and http://localhost:$((PORT+1))/v1"
}

wait_ready() {
  local port="$1" tries=0
  until curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; do
    tries=$((tries + 1))
    [ "$tries" -gt 120 ] && die "engine on port ${port} not ready after 20 min"
    sleep 10
  done
  log "  port ${port}: READY"
}

# A real record from your Arabic training data. Chosen because it exercises four
# things at once: a spelling error (ظمنها -> ضمنها), missing terminal
# punctuation, Gulf/colloquial forms that must NOT be normalized to MSA
# (كل شي, دفعة واحد), and diacritics that must not be added.
SMOKE_TEXT='سبحان الله، كل شي تطور دفعة واحد في كل المجالات، من ظمنها أساليب القصائد'

smoke() {
  local port="${1:-$PORT}"
  log "=== smoke test against port ${port} ==="
  curl -sf "http://localhost:${port}/v1/models" | python3 -c \
    'import sys,json; d=json.load(sys.stdin)["data"][0]; print(f"  model: {d[\"id\"]}  max_len: {d.get(\"max_model_len\")}")' \
    || die "no model served on port ${port}"

  log "  input : ${SMOKE_TEXT}"
  log ""
  # enable_thinking=false is essential: this is a normalization task and chain
  # of thought multiplies output tokens several-fold for no quality gain.
  curl -sf "http://localhost:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @- <<JSON | python3 -c '
import sys, json
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
if m.get("reasoning_content"):
    print("  !! THINKING WAS ON -- reasoning tokens:", len(m["reasoning_content"]))
print("  output:", m["content"])
u = r.get("usage", {})
print(f"  tokens: prompt={u.get(\"prompt_tokens\")} completion={u.get(\"completion_tokens\")}")
'
{
  "model": "corrector",
  "messages": [
    {"role": "system", "content": "You correct ASR transcripts for speech-to-text training data. Fix spelling and add punctuation. Apply inverse text normalization to numbers, dates and times. NEVER convert colloquial dialect to Modern Standard Arabic. NEVER add diacritics. Preserve the speaker's exact words. Reply with JSON only: {\"text\": \"...\", \"dialect\": \"msa|egyptian|levantine|gulf|iraqi|maghrebi|unknown\", \"confidence\": 0.0, \"changes\": []}"},
    {"role": "user", "content": "${SMOKE_TEXT}"}
  ],
  "temperature": 0,
  "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
}

bench() {
  local port="${1:-$PORT}" conc="${2:-64}" n="${3:-256}"
  log "=== throughput benchmark: ${n} requests at concurrency ${conc} ==="
  log "    Recipe's reference point on 4x H100 TP4: ~1,430 output tok/s at"
  log "    concurrency 64, 1024-in/256-out. Compare your number to that."
  log ""
  python3 - "$port" "$conc" "$n" <<'PY'
import json, sys, time, threading, queue, urllib.request

port, conc, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
url = f"http://localhost:{port}/v1/chat/completions"
body = {
    "model": "corrector", "temperature": 0, "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [
        {"role": "system", "content": "Correct the ASR transcript. Fix spelling, add punctuation, apply ITN to numbers. Preserve dialect verbatim. Never add diacritics. Reply JSON only."},
        {"role": "user", "content": "سبحان الله كل شي تطور دفعة واحد في كل المجالات من ظمنها أساليب القصائد"},
    ],
}
data = json.dumps(body).encode()
q, out = queue.Queue(), []
lock = threading.Lock()

def worker():
    while True:
        try: q.get_nowait()
        except queue.Empty: return
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                res = json.load(r)
            dt = time.perf_counter() - t0
            ct = res.get("usage", {}).get("completion_tokens", 0)
            with lock: out.append((dt, ct))
        except Exception as e:
            with lock: out.append((None, str(e)))

for i in range(n): q.put(i)
threads = [threading.Thread(target=worker) for _ in range(conc)]
t0 = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
wall = time.perf_counter() - t0

ok = [x for x in out if x[0] is not None]
errs = len(out) - len(ok)
toks = sum(x[1] for x in ok)
print(f"  completed      : {len(ok)}/{n}   errors: {errs}")
print(f"  wall clock     : {wall:.1f} s")
print(f"  request thrpt  : {len(ok)/wall:.2f} req/s")
print(f"  OUTPUT tok/s   : {toks/wall:.0f}   <-- the number that matters")
if ok:
    lat = sorted(x[0] for x in ok)
    print(f"  latency p50/p95: {lat[len(lat)//2]:.2f}s / {lat[int(len(lat)*0.95)]:.2f}s")
    print(f"  mean out tokens: {toks/len(ok):.0f}")
if errs:
    print(f"  first error    : {next(x[1] for x in out if x[0] is None)}")
print()
hours = 300_000_000 / max(toks/wall, 1) / 3600
print(f"  projection: 300M output tokens (9M triaged samples, batched 12/call)")
print(f"              -> {hours:.1f} hours on this configuration")
PY
}

stop() {
  for c in "$CONTAINER" "${CONTAINER}-a" "${CONTAINER}-b"; do
    have docker && docker rm -f "$c" >/dev/null 2>&1 && log "  stopped $c"
  done
  log "  done"
}

case "${1:-preflight}" in
  preflight)  preflight ;;
  serve)      serve ;;
  serve-2x4)  serve_2x4 ;;
  smoke)      smoke "${2:-$PORT}" ;;
  bench)      bench "${2:-$PORT}" "${3:-64}" "${4:-256}" ;;
  stop)       stop ;;
  *) die "usage: $0 {preflight|serve|serve-2x4|smoke [port]|bench [port] [conc] [n]|stop}" ;;
esac
