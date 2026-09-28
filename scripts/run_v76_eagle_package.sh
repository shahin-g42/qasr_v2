#!/bin/bash
# =============================================================================
# QASR v7.6 — PHASE 4 (ADHOC): package the final model after EAGLE training.
#
# Assembles a self-contained HuggingFace release from:
#   1. the TARGET model  — the Phase-2 mid-checkpoint the head was distilled on
#                          (models/v7.6/full/checkpoint-150000)
#   2. the EAGLE head    — output of run_v76_eagle_1node.sh
#                          (output/v76/eagle_v2_1node_ckpt150k/eagle_head.pt)
#
# Wraps scripts/package_and_push.py with:
#   - conda/PYTHONPATH/transformers-5.14 guard (same as the training launchers)
#   - pre-flight completeness checks on BOTH inputs (fails once, not per-rank)
#   - EAGLE head auto-detection: prefers the final eagle_head.pt at the output
#     root, falls back to the highest checkpoint-* subdir if the run was
#     preempted before trainer.save_model() landed
#   - optional acceptance gate (evaluate.py) BEFORE packaging; both offline and
#     eagle modes are required because exact_transcript_match and eagle_speedup
#     are only emitted when the two ran together
#   - the gate's eval report is then fed to package_and_push.py so the model
#     card publishes numbers measured on THIS head, not on whatever run the
#     card template was last written against
#   - SAFE BY DEFAULT: assembles + verifies locally, does NOT push to the Hub.
#     Pass --push to upload (private repo).
#
# -----------------------------------------------------------------------------
# WARNING — READ BEFORE PUSHING
# -----------------------------------------------------------------------------
# The EAGLE head is bound to EXACT target weights. This package pairs the head
# with the Phase-2 MID-checkpoint (step 150,000 of 451,609) it was trained on.
# If you later ship a different target (Phase-3 final, a later Phase-2 step,
# any re-trained model), THIS HEAD IS INVALID — acceptance collapses toward 0.
# Re-run the EAGLE training against the new target and re-package.
#
# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
#   # Assemble + verify locally (default, no upload):
#   bash scripts/run_v76_eagle_package.sh
#
#   # RECOMMENDED: gate the head, then assemble with the card filled from that
#   # same eval (no eval => the card's metrics render as "_not measured_"):
#   bash scripts/run_v76_eagle_package.sh --acceptance-gate
#
#   # Assemble, verify, and push to a PRIVATE Hub repo:
#   bash scripts/run_v76_eagle_package.sh --acceptance-gate --push
#
#   # Re-use an eval you already ran, without re-running it:
#   bash scripts/run_v76_eagle_package.sh --eval-json logs/eagle_acceptance_ckpt150k.json
#
#   # Override any path:
#   bash scripts/run_v76_eagle_package.sh \
#     --model-dir /path/to/target \
#     --eagle-dir /path/to/eagle_output \
#     --staging-dir /path/to/release \
#     --repo-id audarai/My-Release \
#     --eval-json /path/to/eval.json \
#     --gate-min-pos0 0.4 --gate-min-tpf 1.2 \
#     --gate-min-speedup 1.1 --gate-min-match 0.98 \
#     --gate-modes "offline eagle" --gate-samples 200
#
# Acceptance gate
# -----------------------------------------------------------------------------
# The thresholds quoted in scripts/qasr_train_eagle_8node.slurm:27
# (acceptance >= 0.5, tokens/forward >= 2.5, exact_match 100%) were aspirational
# and are UNREACHABLE by this architecture: they would reject every head ever
# produced here. Calibrated instead on two real evaluate.py reports --
#
#   criterion                  matched (n=50)  ckpt150k (n=512)  MISMATCHED   gate
#   acceptance_by_position[0]      0.7175          0.7136         0.0000    >= 0.40
#   tokens_per_target_forward      1.7620          1.6900         1.0000    >= 1.20
#   eagle_speedup                  1.9800          1.4090         1.0520    >= 1.10
#   exact_transcript_match          49/50         501/512          49/50    >= 0.95
#
# exact_transcript_match is held LOOSE (0.95) on purpose. Both known-good heads
# diverge on ~2% of samples (49/50 = 0.980, 501/512 = 0.9785), so a 0.98 gate
# sits exactly on the good-head rate and rejects them on sampling noise -- it
# fired on the n=512 run above while every discriminating criterion passed.
# Greedy speculation is lossless only in exact arithmetic; in bf16 the verify
# pass runs different kernel shapes than offline decode, so near-tie argmaxes
# flip. WER still improved (0.1447 -> 0.1433), i.e. the flips are not harmful.
# A real decoder bug diverges far more than 2%. To judge a borderline case,
# diff the two hypotheses per sample instead of trusting the aggregate.
#
# Position-0 acceptance is the discriminator. The OVERALL acceptance rate is not:
# the matched head above scores 0.182 overall because depths 2-5 collapse
# (exposure bias -- trained single-step, chained at inference). And
# exact_transcript_match is 49/50 in BOTH cases, because a head that never
# accepts anything just reproduces greedy output. So overall acceptance is
# reported as a warning, never a failure.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — tuned for the adhoc 1-node EAGLE run
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3}"
HF_CACHE="${HF_CACHE:-/lustrefs/shared/shahin.konadath/workspace/train/qasr/.cache}"

MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models/v7.6/full/checkpoint-150000}"
EAGLE_DIR="${EAGLE_DIR:-${PROJECT_DIR}/output/v76/eagle_v2_1node_ckpt150k}"
STAGING_DIR="${STAGING_DIR:-${PROJECT_DIR}/output/v76/hf_release_ckpt150k_adhoc}"
REPO_ID="${REPO_ID:-audarai/Audar-ASR-V1-Pro-ckpt150k-adhoc}"
SRC_DIR="${SRC_DIR:-src/qasr}"
CARD="${CARD:-scripts/HF_MODEL_CARD.md}"

# Acceptance-gate defaults (only used with --acceptance-gate)
GATE_MANIFEST="${GATE_MANIFEST:-${PROJECT_DIR}/training_manifests/v7.6/en/eval_en_inworld.jsonl}"
GATE_LANGUAGE="${GATE_LANGUAGE:-en}"
GATE_SAMPLES="${GATE_SAMPLES:-200}"
GATE_OUTPUT="${GATE_OUTPUT:-${PROJECT_DIR}/logs/eagle_acceptance_ckpt150k.json}"
# Gate thresholds -- see the calibration table in the header. Every one is a
# CLI flag so a future head can be judged without editing this file.
GATE_MODES="${GATE_MODES:-offline eagle}"
GATE_MIN_POS0="${GATE_MIN_POS0:-0.40}"
GATE_MIN_TPF="${GATE_MIN_TPF:-1.20}"
GATE_MIN_SPEEDUP="${GATE_MIN_SPEEDUP:-1.10}"
GATE_MIN_MATCH="${GATE_MIN_MATCH:-0.95}"
# Advisory only: evaluate.py's own "head does not match this checkpoint"
# heuristic. Never gates the build -- see header for why.
GATE_WARN_ACCEPTANCE="${GATE_WARN_ACCEPTANCE:-0.30}"

# Reuse an existing eval report instead of running the gate (card still fills).
EVAL_JSON=""
# Provenance for the card. HEAD_STEPS auto-reads trainer_state.json when unset.
HEAD_STEPS="${HEAD_STEPS:-}"

PUSH=false
ACCEPTANCE_GATE=false

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)            PUSH=true; shift ;;
        --acceptance-gate) ACCEPTANCE_GATE=true; shift ;;
        --model-dir)       MODEL_DIR="$2"; shift 2 ;;
        --eagle-dir)       EAGLE_DIR="$2"; shift 2 ;;
        --staging-dir)     STAGING_DIR="$2"; shift 2 ;;
        --repo-id)         REPO_ID="$2"; shift 2 ;;
        --gate-manifest)   GATE_MANIFEST="$2"; shift 2 ;;
        --gate-language)   GATE_LANGUAGE="$2"; shift 2 ;;
        --gate-samples)    GATE_SAMPLES="$2"; shift 2 ;;
        --gate-modes)        GATE_MODES="$2"; shift 2 ;;
        --gate-min-pos0)     GATE_MIN_POS0="$2"; shift 2 ;;
        --gate-min-tpf)      GATE_MIN_TPF="$2"; shift 2 ;;
        --gate-min-speedup)  GATE_MIN_SPEEDUP="$2"; shift 2 ;;
        --gate-min-match)    GATE_MIN_MATCH="$2"; shift 2 ;;
        --eval-json)         EVAL_JSON="$2"; shift 2 ;;
        --head-steps)        HEAD_STEPS="$2"; shift 2 ;;
        -h|--help)
            # Robust to header edits: print every comment line up to `set -euo`.
            awk 'NR>1 && /^set -euo pipefail/ {exit} NR>1 {sub(/^# ?/, ""); print}' "$0"
            exit 0 ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--push] [--acceptance-gate] [--model-dir DIR] [--eagle-dir DIR]" >&2
            echo "          [--staging-dir DIR] [--repo-id ID] [--eval-json FILE] [--head-steps N]" >&2
            echo "          [--gate-modes 'offline eagle'] [--gate-samples N] [--gate-min-pos0 F]" >&2
            echo "          [--gate-min-tpf F] [--gate-min-speedup F] [--gate-min-match F]" >&2
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment (identical to the training launchers)
# ---------------------------------------------------------------------------
export HF_HOME="${HF_CACHE}"
export TOKENIZERS_PARALLELISM=false

eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src"

python -c "import transformers; assert transformers.__version__.startswith('5.14'), \
    f'transformers {transformers.__version__} != pinned 5.14.x'"

mkdir -p logs "$(dirname "${STAGING_DIR}")"

# ---------------------------------------------------------------------------
# Pre-flight 0: free space. Staging duplicates the target's safetensors AND the
# 1.2 GB head, so it needs roughly the size of the checkpoint again. Failing here
# is cheap; failing halfway through upload_folder is not.
# ---------------------------------------------------------------------------
TARGET_KB="$(du -sk "${MODEL_DIR}" 2>/dev/null | cut -f1 || echo 0)"
FREE_KB="$(df -Pk "$(dirname "${STAGING_DIR}")" | awk 'NR==2 {print $4}')"
NEED_KB=$(( TARGET_KB + TARGET_KB / 5 ))
if [[ "${TARGET_KB}" -gt 0 && "${FREE_KB}" -lt "${NEED_KB}" ]]; then
    echo "FATAL: staging needs ~$(( NEED_KB / 1024 / 1024 )) GB free at $(dirname "${STAGING_DIR}")," >&2
    echo "       only $(( FREE_KB / 1024 / 1024 )) GB available (target is $(( TARGET_KB / 1024 / 1024 )) GB)." >&2
    exit 1
fi
echo " Pre-flight OK: $(( FREE_KB / 1024 / 1024 )) GB free, staging needs ~$(( NEED_KB / 1024 / 1024 )) GB."

echo "============================================================"
echo " QASR v7.6 — package final model (target + EAGLE head)"
echo "============================================================"
echo " Target model:    ${MODEL_DIR}"
echo " EAGLE output:    ${EAGLE_DIR}"
echo " Staging dir:     ${STAGING_DIR}"
echo " Repo id:         ${REPO_ID}"
echo " Push to Hub:     ${PUSH}"
echo " Acceptance gate: ${ACCEPTANCE_GATE} (modes: ${GATE_MODES})"
echo " Started:         $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Pre-flight 1: target checkpoint completeness
# (mirrors validate_local_checkpoint in src/qasr/config.py)
# ---------------------------------------------------------------------------
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "FATAL: target model dir does not exist: ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "FATAL: ${MODEL_DIR}/config.json missing — target checkpoint incomplete." >&2
    exit 1
fi
if ! ls "${MODEL_DIR}"/model*.safetensors >/dev/null 2>&1; then
    echo "FATAL: no model*.safetensors in ${MODEL_DIR}." >&2
    echo "       Every v7.6 phase runs ZeRO-2 (configs/deepspeed_zero2.json), where" >&2
    echo "       parameters are NOT partitioned, so trainer.save_model() writes" >&2
    echo "       consolidated safetensors directly -- there is no ZeRO-3 gather step" >&2
    echo "       to run first. A directory without them means the save never" >&2
    echo "       completed, is still in flight, or this is not the output root (a" >&2
    echo "       checkpoint-* subdir may be). Check the training log." >&2
    echo "       Note: scripts/run_convert_weights.sh is Step 0 (Cohere encoder +" >&2
    echo "       Qwen3 decoder -> the initial checkpoint). It does not consolidate" >&2
    echo "       training checkpoints." >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/preprocessor_config.json" && ! -f "${MODEL_DIR}/processor_config.json" ]]; then
    echo "FATAL: no preprocessor_config.json / processor_config.json in ${MODEL_DIR}." >&2
    exit 1
fi
echo " Pre-flight OK: target checkpoint complete."

# ---------------------------------------------------------------------------
# Pre-flight 2: EAGLE head auto-detection
# Prefer the final head at the output root (written by trainer.save_model()).
# Fall back to the highest checkpoint-* subdir if the run was preempted.
# ---------------------------------------------------------------------------
EAGLE_HEAD_DIR=""
if [[ -f "${EAGLE_DIR}/eagle_head.pt" ]]; then
    EAGLE_HEAD_DIR="${EAGLE_DIR}"
    echo " EAGLE head:      ${EAGLE_HEAD_DIR}/eagle_head.pt (final)"
else
    LATEST="$(find "${EAGLE_DIR}" -maxdepth 2 -type f -name eagle_head.pt \
              -path '*/checkpoint-*/*' 2>/dev/null \
              | sed -E 's|.*/checkpoint-([0-9]+)/eagle_head\.pt|\1 &|' \
              | sort -rn | head -n1 | cut -d' ' -f2-)"
    if [[ -n "${LATEST}" ]]; then
        EAGLE_HEAD_DIR="$(dirname "${LATEST}")"
        echo " EAGLE head:      ${EAGLE_HEAD_DIR}/eagle_head.pt (latest checkpoint)"
        echo " WARNING: packaging an intermediate checkpoint — the run did not finish."
        echo "          The final head lands at ${EAGLE_DIR}/eagle_head.pt when"
        echo "          trainer.save_model() completes."
    else
        echo "FATAL: no eagle_head.pt found in ${EAGLE_DIR} or any checkpoint-* subdir." >&2
        echo "       Did the EAGLE training run? See: bash scripts/run_v76_eagle_1node.sh" >&2
        exit 1
    fi
fi
HEAD_SIZE="$(du -h "${EAGLE_HEAD_DIR}/eagle_head.pt" | cut -f1)"
echo " EAGLE head size: ${HEAD_SIZE}"

# Card provenance, read from the artifact instead of typed in: the trained param
# count (the file is ~1.28 GB because 97% of it is a frozen copy of the target's
# own 151,936-row output projection) and the step count from trainer_state.json.
# Never a hard failure -- packaging is the point, and the card falls back to a
# visible "_not measured_" marker for anything unreadable.
HEAD_META="$(python - "${EAGLE_HEAD_DIR}" "${HEAD_STEPS}" <<'PYMETA'
import json, sys
from pathlib import Path

head_dir, override = Path(sys.argv[1]), sys.argv[2].strip()
params = steps = ""
try:
    import torch
    ck = torch.load(head_dir / "eagle_head.pt", map_location="cpu", weights_only=True)
    params = f"{sum(v.numel() for v in ck['state_dict'].values()):,}"
except Exception:
    pass
if override:
    steps = override
else:
    states = sorted(
        head_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda q: int(q.parent.name.rsplit("-", 1)[-1]),
    )
    if (head_dir / "trainer_state.json").is_file():
        states.append(head_dir / "trainer_state.json")
    for state in reversed(states):
        try:
            steps = f"{json.loads(state.read_text())['global_step']:,}"
            break
        except Exception:
            continue
print(params or "unknown")
print(steps or "unknown")
PYMETA
)"
HEAD_PARAMS="$(printf '%s\n' "${HEAD_META}" | sed -n 1p)"
HEAD_STEPS_RESOLVED="$(printf '%s\n' "${HEAD_META}" | sed -n 2p)"
echo " EAGLE head params: ${HEAD_PARAMS} (trained: ~8.4M; rest is the frozen lm_head copy)"
echo " Head train steps:  ${HEAD_STEPS_RESOLVED}"

# ---------------------------------------------------------------------------
# Optional: acceptance gate BEFORE packaging
# Catches a head/target mismatch (the eval_results_en_commentary.json failure)
# before it lands in a release. Requires 1 GPU.
# ---------------------------------------------------------------------------
if [[ "${ACCEPTANCE_GATE}" == true ]]; then
    echo "------------------------------------------------------------"
    echo " Running acceptance gate (evaluate.py --modes ${GATE_MODES})"
    echo "   manifest: ${GATE_MANIFEST}"
    echo "   language: ${GATE_LANGUAGE}   samples: ${GATE_SAMPLES}"
    echo "   output:   ${GATE_OUTPUT}"
    echo "   pass if:  pos0>=${GATE_MIN_POS0} tpf>=${GATE_MIN_TPF} speedup>=${GATE_MIN_SPEEDUP} match>=${GATE_MIN_MATCH}"
    echo "------------------------------------------------------------"
    if [[ ! -f "${GATE_MANIFEST}" ]]; then
        echo "FATAL: gate manifest not found: ${GATE_MANIFEST}" >&2
        exit 1
    fi
    python evaluate.py \
        --model "${MODEL_DIR}" \
        --eagle "${EAGLE_HEAD_DIR}" \
        --manifest "${GATE_MANIFEST}" \
        --language "${GATE_LANGUAGE}" \
        --num-samples "${GATE_SAMPLES}" \
        --seed 42 \
        --modes ${GATE_MODES} \
        --output "${GATE_OUTPUT}"

    python - "${GATE_OUTPUT}" "${GATE_MIN_POS0}" "${GATE_MIN_TPF}" \
             "${GATE_MIN_SPEEDUP}" "${GATE_MIN_MATCH}" "${GATE_WARN_ACCEPTANCE}" <<'PYEOF'
"""Decide whether this head belongs to this target checkpoint.

See the calibration table in this script's header. Position-0 acceptance
separates a matched head (0.7175) from a mismatched one (0.0) with an enormous
margin; overall acceptance (0.1822 vs 0.0) and exact_transcript_match (49/50 vs
49/50) do not separate them usefully at all.
"""
import sys, json

path, min_pos0, min_tpf, min_speedup, min_match, warn_acc = sys.argv[1:7]
min_pos0, min_tpf = float(min_pos0), float(min_tpf)
min_speedup, min_match, warn_acc = float(min_speedup), float(min_match), float(warn_acc)

with open(path) as handle:
    data = json.load(handle)

stats = data.get("eagle_stats") or {}
comp = data.get("comparison") or {}
by_pos = stats.get("acceptance_by_position") or []
pos0 = by_pos[0] if by_pos else None
acc = stats.get("acceptance_rate")
tpf = stats.get("tokens_per_target_forward")
speedup = comp.get("eagle_speedup")

# evaluate.py writes "N/M", and only when offline AND eagle both ran.
etm_raw = comp.get("exact_transcript_match")
etm = None
if isinstance(etm_raw, str) and "/" in etm_raw:
    num, _, den = etm_raw.partition("/")
    etm = int(num) / int(den) if int(den) else None
elif isinstance(etm_raw, (int, float)):
    etm = float(etm_raw)

print(f"  acceptance_by_position[0]  = {pos0}   (gate >= {min_pos0})")
print(f"  tokens_per_target_forward  = {tpf}   (gate >= {min_tpf})")
print(f"  eagle_speedup              = {speedup}   (gate >= {min_speedup})")
print(f"  exact_transcript_match     = {etm_raw} = {etm}   (gate >= {min_match})")
print(f"  acceptance_rate (overall)  = {acc}   (advisory only, warn < {warn_acc})")
print(f"  mean_accepted_per_round    = {stats.get('mean_accepted_per_round')}")
if by_pos:
    print("  acceptance by depth        = " + " | ".join(f"{p:.3f}" for p in by_pos))

failed = []
if pos0 is None or pos0 < min_pos0:
    failed.append(f"position-0 acceptance {pos0} < {min_pos0} -- MISMATCH signature")
if tpf is None or tpf < min_tpf:
    failed.append(f"tokens_per_target_forward {tpf} < {min_tpf}")
if speedup is None or speedup < min_speedup:
    failed.append(f"eagle_speedup {speedup} < {min_speedup}")
if etm is None:
    failed.append("exact_transcript_match missing -- need offline AND eagle in --gate-modes")
elif etm < min_match:
    failed.append(f"exact_transcript_match {etm_raw} < {min_match:.0%}")

if acc is not None and acc < warn_acc and pos0 is not None and pos0 >= min_pos0:
    print("", file=sys.stderr)
    print(f"  NOTE: overall acceptance {acc:.3f} < {warn_acc} despite a healthy position-0.", file=sys.stderr)
    print("  That is NOT a mismatch signal -- it is the expected exposure-bias", file=sys.stderr)
    print("  collapse at draft depths 2+: the head is trained single-step teacher-", file=sys.stderr)
    print("  forced but chained autoregressively at inference, so every draft", file=sys.stderr)
    print("  compounds the previous one's error. Packaging proceeds.", file=sys.stderr)

if failed:
    print("\nACCEPTANCE GATE FAILED:", file=sys.stderr)
    for item in failed:
        print(f"  - {item}", file=sys.stderr)
    print(f"\n  head:   {data.get('eagle')}", file=sys.stderr)
    print(f"  target: {data.get('model')}", file=sys.stderr)
    mismatch = pos0 is None or pos0 < min_pos0
    if mismatch:
        print("DO NOT PACKAGE. Position-0 acceptance near zero means the head was", file=sys.stderr)
        print("distilled against different target weights, or a different head", file=sys.stderr)
        print("architecture (e.g. a v1 head on a v2 fc1 4096x2048 layout).", file=sys.stderr)
        print("Retrain against THIS target, or pass the matching --eagle-dir.", file=sys.stderr)
        print("If you genuinely mean to ship it, relax the gate explicitly:", file=sys.stderr)
        print("  --gate-min-pos0 0.0 --gate-min-tpf 1.0 --gate-min-speedup 1.0", file=sys.stderr)
    else:
        # Position-0 is healthy, so the head DOES belong to this target. Whatever
        # failed above is a quality/threshold question, and printing the mismatch
        # spiel here would send you off to retrain a perfectly good head.
        print("Position-0 acceptance is HEALTHY -- the head DOES match this target.", file=sys.stderr)
        print("This is NOT a mismatch. Do not retrain; judge the criterion that failed:", file=sys.stderr)
        print("  - exact_transcript_match: greedy speculation is lossless only in exact", file=sys.stderr)
        print("    arithmetic. In bf16 the verify pass uses different kernel shapes than", file=sys.stderr)
        print("    offline decode, so near-tie argmaxes can flip. Inspect the divergences", file=sys.stderr)
        print("    before deciding: 1-2 changed tokens each = tie-breaks, ship it;", file=sys.stderr)
        print("    wholesale rewrites = a real decode bug, fix it first.", file=sys.stderr)
        print("  - tokens_per_target_forward / eagle_speedup: the head is correct but not", file=sys.stderr)
        print("    paying for itself. Ship anyway, or train the head longer.", file=sys.stderr)
        print("Relax ONLY the criterion that failed, e.g. --gate-min-match 0.95.", file=sys.stderr)
    sys.exit(1)
print("\n  ACCEPTANCE GATE PASSED")
PYEOF
fi

# ---------------------------------------------------------------------------
# Which eval report feeds the model card?
# Prefer an explicit --eval-json, else the report this run's gate just wrote.
# With neither, package_and_push.py renders every measured metric as
# "_not measured_" -- the safe outcome, because the alternative is publishing
# the previous head's numbers as if they described this one.
# ---------------------------------------------------------------------------
if [[ -z "${EVAL_JSON}" && "${ACCEPTANCE_GATE}" == true && -f "${GATE_OUTPUT}" ]]; then
    EVAL_JSON="${GATE_OUTPUT}"
fi
if [[ -n "${EVAL_JSON}" && ! -f "${EVAL_JSON}" ]]; then
    echo "FATAL: --eval-json not found: ${EVAL_JSON}" >&2
    exit 1
fi
echo " Card metrics from: ${EVAL_JSON:-<none -- card metrics render as _not measured_>}"

# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------
echo "------------------------------------------------------------"
echo " Assembling release at ${STAGING_DIR}"
echo "------------------------------------------------------------"

PACKAGE_ARGS=(
    --model-dir   "${MODEL_DIR}"
    --eagle-dir   "${EAGLE_HEAD_DIR}"
    --src-dir     "${SRC_DIR}"
    --staging-dir "${STAGING_DIR}"
    --card        "${CARD}"
    --repo-id     "${REPO_ID}"
    --verify
)
if [[ -n "${EVAL_JSON}" ]]; then
    PACKAGE_ARGS+=( --eval-json "${EVAL_JSON}" )
fi
PACKAGE_ARGS+=(
    --card-var "TARGET_CHECKPOINT=$(basename "${MODEL_DIR}")"
    --card-var "HEAD_STEPS=${HEAD_STEPS_RESOLVED}"
    --card-var "HEAD_FILE_SIZE=${HEAD_SIZE}"
    --card-var "HEAD_PARAMS=${HEAD_PARAMS}"
)
if [[ "${PUSH}" == true ]]; then
    PACKAGE_ARGS+=( --private )
else
    PACKAGE_ARGS+=( --no-upload )
fi

python scripts/package_and_push.py "${PACKAGE_ARGS[@]}"

# ---------------------------------------------------------------------------
# Post-flight: show what landed
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Release assembled at: ${STAGING_DIR}"
echo "============================================================"
du -sh "${STAGING_DIR}"
echo ""
ls -lh "${STAGING_DIR}" | sed 's/^/   /'
if [[ -d "${STAGING_DIR}/eagle" ]]; then
    echo ""
    ls -lh "${STAGING_DIR}/eagle" | sed 's/^/   eagle\//'
fi
echo ""
if [[ "${PUSH}" == true ]]; then
    echo " Pushed to: https://huggingface.co/${REPO_ID} (private)"
else
    echo " NOT pushed (--no-upload). To publish:"
    echo "   huggingface-cli login   # once"
    echo "   bash scripts/run_v76_eagle_package.sh --push"
fi
echo ""
echo " REMINDER: this release pairs the EAGLE head with the Phase-2"
echo " mid-checkpoint at ${MODEL_DIR}. The head is INVALID for any"
echo " other target. Re-train + re-package if the target changes."
echo "============================================================"
