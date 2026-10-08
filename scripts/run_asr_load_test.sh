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
log "$PY scripts/asr_load_test.py ${ARGS[*]}"
"$PY" scripts/asr_load_test.py "${ARGS[@]}"

cleanup
if [ -s "$OUT/gpu.csv" ]; then
  "$PY" - "$OUT" "$(nproc 2>/dev/null || echo 0)" <<'PY' | tee "$OUT/monitors.txt"
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1])
report = json.loads((out / "report.json").read_text())
gpu = [(float(r[0]), float(r[2]), float(r[3])) for r in csv.reader(open(out / "gpu.csv")) if len(r) == 4]
cpu = []
for r in csv.reader(open(out / "cpu.csv")):
    try:
        cpu.append((float(r[0]), float(r[1])))
    except (ValueError, IndexError):
        pass
print("\n=== server utilisation per level (GPU mean over all GPUs; CPU = container, 100% = 1 core) ===")
print(f"{'conc':>5} {'RTFx':>7} {'GPU util':>9} {'GPU mem GiB':>12} {'CPU cores':>10}")
for lv in report["levels"]:
    g = [u for t, u, _ in gpu if lv["t_start"] <= t <= lv["t_end"]]
    m = [mem for t, _, mem in gpu if lv["t_start"] <= t <= lv["t_end"]]
    c = [v for t, v in cpu if lv["t_start"] <= t <= lv["t_end"]]
    fmt = lambda xs, f: f(sum(xs) / len(xs)) if xs else "-"
    print(f"{lv['concurrency']:>5} {lv['rtfx']:>7} {fmt(g, lambda v: f'{v:.0f}%'):>9} "
          f"{fmt(m, lambda v: f'{v / 1024:.1f}'):>12} {fmt(c, lambda v: f'{v / 100:.1f}'):>10}")
# Verdict at the highest level, from what was measured (not a fixed hint).
cores = int(sys.argv[2]) or None
top = report["levels"][-1]
g = [u for t, u, _ in gpu if top["t_start"] <= t <= top["t_end"]]
c = [v / 100 for t, v in cpu if top["t_start"] <= t <= top["t_end"]]
gu, cu = (sum(g) / len(g) if g else None), (sum(c) / len(c) if c else None)
rising = len(report["levels"]) > 1 and top["rtfx"] > 1.10 * report["levels"][-2]["rtfx"]
print()
if gu is not None and gu >= 85:
    print(f"verdict: GPU-bound at concurrency {top['concurrency']} ({gu:.0f}% util) -- this is the server's capacity")
elif cu is not None and cores and cu >= 0.7 * cores:
    print(f"verdict: CPU-bound ({cu:.0f} of {cores} cores) -- audio decode/mel: restart with API_SERVERS=16 AUDIO_WORKERS=16")
elif rising:
    print(f"verdict: NOT saturated (GPU {gu or 0:.0f}%, CPU {cu or 0:.0f}/{cores} cores, RTFx still rising "
          f">10% per step) -- raise CONCURRENCY or run a second client on another node")
else:
    print(f"verdict: throughput flat with GPU {gu or 0:.0f}% and CPU {cu or 0:.0f}/{cores} cores -- the client may be "
          "the limit: run a second client on another node and compare")
PY
fi
log "done: $OUT/report.json"
