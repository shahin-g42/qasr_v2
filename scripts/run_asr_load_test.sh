#!/usr/bin/env bash
# Cluster launcher for scripts/asr_load_test.py.
#
#   bash scripts/run_asr_load_test.sh                       # foreground, eval split
#   BACKGROUND=1 bash scripts/run_asr_load_test.sh          # detached; survives logout
#   SPLIT=train PER_LANG=500 PROJECT_HOURS=250000 bash scripts/run_asr_load_test.sh
#   URLS="http://hpc-029:8020 http://hpc-030:8020" bash scripts/run_asr_load_test.sh
#
# Before loading the server it checks /health on every URL and that the chat
# route renders the training prompt (asr_vllm_client.py check-prompt). When run
# on the node that hosts the server it also records GPU utilization
# (nvidia-smi) and the container's CPU use (docker stats) and folds both into
# the per-level table, which is how you tell a GPU-bound server from an
# audio-decode (API-server CPU) bottleneck.
#
# Everything lands in $OUT: report.json, requests_c*.jsonl, issues.jsonl
# (every clip flagged stripped_marks / loop / wrong_script / unstable / ...),
# gpu.csv, cpu.csv, run.log, monitors.txt.

set -euo pipefail
cd "$(dirname "$0")/.."

URLS="${URLS:-http://inception-H100-hpc-029.inception.ai:8020}"
CONFIG="${CONFIG:-configs/v7.6/02_full_8node.yaml}"
SPLIT="${SPLIT:-eval}"
LANGS="${LANGS:-}"
PER_LANG="${PER_LANG:-200}"
CONCURRENCY="${CONCURRENCY:-64,256,512,1024,2048}"
# Each level runs at least this long so the numbers are steady state, not ramp-up.
MIN_LEVEL_S="${MIN_LEVEL_S:-60}"
PROJECT_HOURS="${PROJECT_HOURS:-0}"
CONTAINER="${CONTAINER:-qasr-asr}"
PY="${PYTHON:-python3}"
OUT="${OUT:-logs/asr_load_test/$(date +%Y%m%d_%H%M%S)_${SPLIT}}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [ "${BACKGROUND:-0}" = 1 ]; then
  mkdir -p "$OUT"
  BACKGROUND=0 OUT="$OUT" nohup bash "$0" "$@" > "$OUT/run.log" 2>&1 &
  echo "started in background (pid $!)"
  echo "  follow : tail -f $OUT/run.log"
  echo "  results: $OUT/report.json"
  exit 0
fi

mkdir -p "$OUT"
log "load test -> $OUT"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || die "$PY is older than 3.10; set PYTHON=/path/to/python3.10+"

# Hundreds of concurrent HTTP connections plus manifest/audio files.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
log "open-files limit: $(ulimit -n)"

URL_ARGS=()
for url in $URLS; do
  curl -sf "${url}/health" >/dev/null || die "server not healthy: ${url}/health"
  URL_ARGS+=(--url "$url")
  log "healthy: $url"
done

# Prompt parity on the first clip of the first manifest of the chosen split.
first=$("$PY" - "$CONFIG" "$SPLIT" "$LANGS" <<'PY'
import json, os, sys
sys.path.insert(0, "scripts")
from asr_load_test import config_manifests
config, split, langs = sys.argv[1:4]
want = set(langs.split(",")) if langs else None
for lang, paths in config_manifests(config, f"{split}_manifest").items():
    if want and lang not in want:
        continue
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                row = json.loads(fh.readline())
            print(lang, row.get("audio_filepath") or row.get("wav_path"))
            raise SystemExit
PY
)
[ -n "$first" ] || die "no readable manifest for split=$SPLIT in $CONFIG"
for url in $URLS; do
  "$PY" scripts/asr_vllm_client.py check-prompt --url "$url" --lang "${first%% *}" "${first#* }" \
    || die "prompt parity failed on $url -- the server does not render the training prompt"
done

# Monitors, only when the server's GPUs/container are on this node.
MON_PIDS=()
cleanup() { for p in "${MON_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
if command -v docker >/dev/null && docker ps -q -f name="^${CONTAINER}$" 2>/dev/null | grep -q .; then
  log "server container is local: recording GPU and CPU use"
  docker inspect --format '{{join .Args " "}}' "$CONTAINER" \
    | grep -oE -- '--(api-server-count|data-parallel-size) [0-9]+' > "$OUT/topology.txt" || true
  ( while :; do
      ts=$(date +%s)
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
        | sed "s/^/${ts}, /"
      sleep 2
    done ) > "$OUT/gpu.csv" 2>/dev/null &
  MON_PIDS+=($!)
  ( while :; do
      printf '%s, %s\n' "$(date +%s)" "$(docker stats --no-stream --format '{{.CPUPerc}}' "$CONTAINER" | tr -d '%')"
      sleep 3
    done ) > "$OUT/cpu.csv" 2>/dev/null &
  MON_PIDS+=($!)
else
  log "server is remote (or docker unavailable here): no GPU/CPU monitors"
fi

ARGS=("${URL_ARGS[@]}" --config "$CONFIG" --split "$SPLIT" --per-lang "$PER_LANG"
      --concurrency "$CONCURRENCY" --min-level-seconds "$MIN_LEVEL_S" --out "$OUT")
[ -n "$LANGS" ] && ARGS+=(--langs "$LANGS")
[ "$PROJECT_HOURS" != 0 ] && ARGS+=(--project-hours "$PROJECT_HOURS")
# Engine-side queue, from vLLM's own Prometheus gauges (summed over the DP
# engines of the first server): requests the client has in flight but the
# engines do not see are queued in the API-server (front-end) processes.
FIRST_URL="${URLS%% *}"
( while :; do
    printf '%s, %s\n' "$(date +%s)" "$(curl -sf --max-time 2 "${FIRST_URL}/metrics" \
      | awk '/^vllm:num_requests_running/{r+=$NF} /^vllm:num_requests_waiting/{w+=$NF} END{printf "%d, %d", r, w}')"
    sleep 3
  done ) > "$OUT/engine.csv" 2>/dev/null &
MON_PIDS+=($!)

log "$PY scripts/asr_load_test.py ${ARGS[*]}"
"$PY" scripts/asr_load_test.py "${ARGS[@]}" &
CLIENT_PID=$!
# The client's own CPU: one Python process, so ~100% means the GIL is the cap.
( while kill -0 "$CLIENT_PID" 2>/dev/null; do
    printf '%s, %s\n' "$(date +%s)" "$(ps -o %cpu= -p "$CLIENT_PID" | tr -d ' ')"
    sleep 3
  done ) > "$OUT/client.csv" 2>/dev/null &
MON_PIDS+=($!)
wait "$CLIENT_PID"

cleanup
if [ -s "$OUT/report.json" ]; then
  "$PY" - "$OUT" "$(nproc 2>/dev/null || echo 0)" <<'PY' | tee "$OUT/monitors.txt"
import csv, json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
cores = int(sys.argv[2]) or None
report = json.loads((out / "report.json").read_text())


def series(name: str, cols: int) -> list[tuple[float, ...]]:
    path = out / name
    rows = []
    if path.exists():
        for r in csv.reader(open(path)):
            try:
                vals = tuple(float(x) for x in r)
            except ValueError:
                continue
            if len(vals) == cols:
                rows.append(vals)
    return rows


gpu, cpu = series("gpu.csv", 4), series("cpu.csv", 2)
eng, cli = series("engine.csv", 3), series("client.csv", 2)
topo = dict(re.findall(r"--([a-z-]+) (\d+)", (out / "topology.txt").read_text())) if (out / "topology.txt").exists() else {}
api, dp = int(topo.get("api-server-count", 0)), int(topo.get("data-parallel-size", 0))


def mean(rows, col, lv):
    xs = [r[col] for r in rows if lv["t_start"] <= r[0] <= lv["t_end"]]
    return sum(xs) / len(xs) if xs else None


def f(v, fmt):
    return "-" if v is None else fmt.format(v)


print("\n=== where requests are and who is busy, per level ===")
print("(GPU = mean over GPUs; server CPU = container cores; engine run/wait = vLLM gauges summed over engines;")
print(" client CPU = the load-test process, 100% = one core)")
print(f"{'conc':>5} {'RTFx':>7} {'GPU':>5} {'srv cores':>9} {'eng run':>8} {'eng wait':>8} {'client':>7}")
rows = []
for lv in report["levels"]:
    m = dict(gpu=mean(gpu, 2, lv), mem=mean(gpu, 3, lv), cpu=mean(cpu, 1, lv),
             run=mean(eng, 1, lv), wait=mean(eng, 2, lv), cli=mean(cli, 1, lv))
    rows.append((lv, m))
    print(f"{lv['concurrency']:>5} {lv['rtfx']:>7} {f(m['gpu'], '{:.0f}%'):>5} "
          f"{f(m['cpu'] and m['cpu'] / 100, '{:.1f}'):>9} {f(m['run'], '{:.0f}'):>8} "
          f"{f(m['wait'], '{:.0f}'):>8} {f(m['cli'], '{:.0f}%'):>7}")

# Verdict at the highest level, from measurements.
lv, m = rows[-1]
conc = lv["concurrency"]
in_engines = (m["run"] or 0) + (m["wait"] or 0)
rising = len(rows) > 1 and lv["rtfx"] > 1.10 * rows[-2][0]["rtfx"]
fixed_procs = api + dp
print()
if m["gpu"] is not None and m["gpu"] >= 85:
    print(f"verdict: GPU-bound at concurrency {conc} ({m['gpu']:.0f}% util) -- this is the server's capacity")
elif m["cli"] is not None and m["cli"] >= 90:
    print(f"verdict: CLIENT-bound (load-test process at {m['cli']:.0f}% CPU, one GIL) -- run more clients, "
          "e.g. on other nodes, and add their RTFx")
elif m["run"] is not None and in_engines < 0.5 * conc:
    msg = (f"verdict: FRONT-END bound -- the client holds {conc} requests but the engines see only "
           f"{in_engines:.0f}; the rest queue in the {api or '?'} API-server processes (audio decode + mel run there)")
    if m["cpu"] is not None and fixed_procs and m["cpu"] / 100 >= 0.8 * fixed_procs:
        msg += f"; server CPU {m['cpu'] / 100:.0f} cores ~= {api} API servers + {dp} engines each pegged at 1 core"
    print(msg + f". Restart with API_SERVERS={max(2 * (api or 8), 16)} (node has {cores or '?'} cores).")
elif m["cpu"] is not None and cores and m["cpu"] / 100 >= 0.7 * cores:
    print(f"verdict: CPU-bound ({m['cpu'] / 100:.0f} of {cores} cores) -- audio decode/mel")
elif rising:
    print(f"verdict: NOT saturated (RTFx still rising >10% per step) -- raise CONCURRENCY")
else:
    print("verdict: throughput flat but no resource saturated -- check engine.csv / docker logs")
PY
fi
log "done: $OUT/report.json"
