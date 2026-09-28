#!/usr/bin/env bash
# Serve Qwen/Qwen3.8-27B-FP8 on 8x H100 for corpus transcript correction.
#
# Subcommands:
#   preflight   verify GPU/disk/image + that the image actually has the arch
#   serve       launch DP=8 / TP=1 -- one replica per GPU, this node
#   smoke       health check + one real Arabic correction from your own data
#   bench       measure actual output tok/s on THIS hardware
#   stop        tear down
#
# This is the DENSE sibling of run_flash_next.sh. Three things differ from
# Flash-Next, and copying Flash-Next's flags here silently breaks the launch:
#
#   1. It is DENSE, not MoE. text_config has NO num_experts, so there are no
#      experts to shard: --enable-expert-parallel and --moe-backend are WRONG
#      here. (The FP8 config's modules_to_not_convert lists vestigial MoE names
#      mlp.gate / mlp.shared_expert_gate -- ignore them, they match no weights.)
#   2. No PLE / n-gram embedding table, so VLLM_PLE_CPU_OFFLOAD is not needed
#      and there is no host-RAM requirement.
#   3. num_key_value_heads is 4 (< 8), so TP=8 would REPLICATE KV heads -- wasted
#      VRAM, zero gain. FP8 weights are only ~32 GB, so they fit ONE H100: the
#      throughput-optimal shape is DP=8 / TP=1 (8 independent replicas per node).
#
# FP8 detail: quant_method fp8, fmt e4m3, activation_scheme dynamic (no
# calibration), weight_block_size [128,128]. embed_tokens, lm_head, every norm,
# linear_attn.in_proj_*, visual and mtp stay BF16 (modules_to_not_convert), so
# weights land ~32 GB (not 27) and the download is ~35 GB (vs 173 for Flash-Next).
#
# Do NOT enable MTP speculative decoding: measured slower on H100.
#
# Image: cu129 is mandatory on this cluster (old NVIDIA driver). qwen38-cu129 is
# the designated dense-27B build -- see run_flash_next.sh:44.

set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-cu129}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
ARCH="${ARCH:-Qwen3_5ForConditionalGeneration}"
CONTAINER="${CONTAINER:-qwen27b}"
PORT="${PORT:-8000}"
# Cap context hard. The checkpoint advertises 262,144 native tokens and vLLM
# reserves KV for whatever you allow; transcript correction needs ~800. Leaving
# this at the default wastes most of your KV cache on context you will never use.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_UTIL="${GPU_UTIL:-0.90}"
# Shared Lustre cache: the 4 nodes download the checkpoint ONCE.
HF_CACHE="${HF_CACHE:-/lustrefs/shared/shahin.konadath/workspace/train/qasr/.cache}"
# Disk for the ~35 GB FP8 checkpoint, plus headroom.
NEEDED_GB="${NEEDED_GB:-60}"

log()  { printf '%s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- engine arguments -------------------------------------------------------
# --limit-mm-per-prompt drops the vision tower: this is a vision-language
# checkpoint and we only ever send text. No --enable-expert-parallel, no
# --moe-backend, no VLLM_PLE_CPU_OFFLOAD -- this model is dense with no PLE table.
common_args() {
  cat <<ARGS
--served-model-name corrector
--max-model-len ${MAX_MODEL_LEN}
--max-num-seqs ${MAX_NUM_SEQS}
--gpu-memory-utilization ${GPU_UTIL}
--enable-prefix-caching
--limit-mm-per-prompt {"image":0,"video":0}
ARGS
}

preflight() {
  log "=== preflight: Qwen3.8-27B-FP8 on this node ==="
  local fail=0

  # GPUs
  if have nvidia-smi; then
    local n vram
    n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log "  GPUs                : ${n} x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    log "  VRAM per GPU        : ${vram} MiB"
    [ "$n" -ge 8 ] || { log "  !! expected 8 GPUs, found ${n}"; fail=1; }
    # FP8 weights are ~32 GB; an 80GB H100 leaves ~40 GB for KV at util 0.90.
    if [ "${vram:-0}" -lt 40000 ]; then
      log "  !! <40GB GPU: the ~32GB weights leave no KV headroom"; fail=1
    fi
  else
    log "  !! nvidia-smi not found"; fail=1
  fi

  # Disk for the ~35 GB checkpoint (shared cache: downloaded once across nodes)
  local free_gb
  free_gb=$(df -BG --output=avail "$(dirname "$HF_CACHE")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  log "  Free disk at cache  : ${free_gb} GiB (need >= ${NEEDED_GB})"
  [ "${free_gb:-0}" -ge "$NEEDED_GB" ] || { log "  !! insufficient disk for the FP8 checkpoint"; fail=1; }

  # cu129 is required for the old cluster driver; a pre-model tag cannot serve it.
  case "$IMAGE" in
    *cu129*) log "  CUDA 12.9 build    : yes (compatible with older NVIDIA drivers)" ;;
    *)       log "  !! IMAGE=${IMAGE} is not a cu129 build -- may need a newer driver than the cluster has"; fail=1 ;;
  esac
  case "$IMAGE" in
    *qwen38*) log "  qwen38 build        : yes" ;;
    *)        log "  !! IMAGE=${IMAGE} is not a qwen38 build; expected vllm/vllm-openai:qwen38-cu129"; fail=1 ;;
  esac

  # Runtime
  if have docker; then
    log "  docker              : $(docker --version | cut -d, -f1)"
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      log "  image               : present (${IMAGE})"
    else
      log "  image               : NOT pulled -> docker pull ${IMAGE}"
    fi
    # The decisive check, run BEFORE the download: does this image's vLLM
    # actually register the architecture? A missing arch cannot be fixed by flags.
    log "  arch support        : checking ${ARCH} in image (pulls if absent)..."
    local ok
    ok=$(docker run --rm "$IMAGE" python3 -c \
      "from vllm import ModelRegistry; print('${ARCH}' in ModelRegistry.get_supported_archs())" 2>/dev/null || echo "ERR")
    if [ "$ok" = "True" ]; then
      log "  arch support        : YES (${ARCH})"
    else
      log "  !! arch support     : ${ok} -- image does NOT serve ${ARCH}; no flag fixes this"; fail=1
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
    docker run -d --name "$name" --restart unless-stopped \
      --gpus "\"device=${gpus}\"" \
      --ipc=host --shm-size 128g \
      -e HF_HUB_ENABLE_HF_TRANSFER=1 \
      ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
      -v "${HF_CACHE}:/root/.cache/huggingface" \
      -p "${port}:8000" \
      "$IMAGE" \
      --model "$MODEL" --port 8000 "$@"
  elif have apptainer; then
    log "  starting ${name} via apptainer on GPU(s) ${gpus}, port ${port}"
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    apptainer run --nv --bind "${HF_CACHE}:/root/.cache/huggingface" \
      "docker://${IMAGE}" \
      --model "$MODEL" --port "$port" "$@" &
  else
    die "no container runtime"
  fi
}

serve() {
  log "=== DP8: one vLLM front-end, 8 independent replicas (one per GPU) ==="
  log "    Dense model, ~32GB FP8 weights fit a single H100, so DP=8/TP=1 maximizes"
  log "    throughput. TP>=8 is rejected on purpose: only 4 KV heads, so TP8 would"
  log "    replicate them -- more VRAM, no speedup. Run this SAME command on each of"
  log "    nodes 0-3 (4 deployments = 32 replicas) and put a load balancer in front."
  preflight || die "preflight failed"
  # shellcheck disable=SC2046
  launch_container "$CONTAINER" "0,1,2,3,4,5,6,7" "$PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    $(common_args | tr '\n' ' ')
  log ""
  log "  waiting for readiness (model load takes a minute or two)..."
  wait_ready "$PORT"
  log "  up: http://localhost:${PORT}/v1  (served-model-name: corrector)"
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
  log "    DP=8 means 8 replicas behind one port; push concurrency well above 64 to"
  log "    keep all 8 busy. Compare output tok/s against your Flash-Next nodes."
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
  have docker && docker rm -f "$CONTAINER" >/dev/null 2>&1 && log "  stopped $CONTAINER"
  log "  done"
}

case "${1:-preflight}" in
  preflight)  preflight ;;
  serve)      serve ;;
  smoke)      smoke "${2:-$PORT}" ;;
  bench)      bench "${2:-$PORT}" "${3:-64}" "${4:-256}" ;;
  stop)       stop ;;
  *) die "usage: $0 {preflight|serve|smoke [port]|bench [port] [conc] [n]|stop}" ;;
esac
