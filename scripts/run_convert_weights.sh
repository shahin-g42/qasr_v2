#!/bin/bash
# =============================================================================
# QASR — Step 0: weight conversion (Cohere Conformer encoder + Qwen3 decoder
# -> initial QASR checkpoint with a randomly-initialized projector).
#
# Run ONCE, on ONE node (login or compute). CPU-only — no GPUs needed:
# both source checkpoints are read with device="cpu" and the merged model is
# assembled in bf16 on CPU. Budget ~30 GB RAM and ~15 GB disk for the output.
# Takes ~10 minutes plus model download time on the first run.
#
#   bash scripts/run_convert_weights.sh
#
# Overridables (env):
#   OUTPUT_DIR  where the checkpoint lands
#               (default .../train/stt/qasr/output/initial — the path the
#               v7.6 Phase 1 config documents as the fresh-conversion input)
#   ENCODER     encoder repo/path (default CohereLabs/cohere-transcribe-03-2026)
#   QWEN        decoder repo/path (default audarai/Audar-ASR-V1.2-Turbo)
#   SEED        projector init seed (default 42)
#
# The converter itself refuses a non-empty output dir, transfers encoder and
# decoder tensors with strict=True plus a per-tensor torch.equal re-read, ties
# embeddings and asserts the tie, and --verify-reload (always on here) reloads
# the saved checkpoint and re-asserts the tie.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3}"
PROJECT_DIR="${PROJECT_DIR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
# The original model downloads live in the pre-stt-move cache
# (workspace/train/qasr/.cache) — verified: the Cohere snapshot is fully
# cached there, so conversion needs no Hub access (and no license gate).
HF_CACHE="${HF_CACHE:-/lustrefs/shared/shahin.konadath/workspace/train/qasr/.cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/initial}"
ENCODER="${ENCODER:-CohereLabs/cohere-transcribe-03-2026}"
QWEN="${QWEN:-audarai/Audar-ASR-V1.2-Turbo}"
SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
# Environment
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

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: output dir is not empty: ${OUTPUT_DIR}" >&2
    echo "The converter refuses to overwrite. Move the old checkpoint aside first:" >&2
    echo "  mv '${OUTPUT_DIR}' '${OUTPUT_DIR}.bak.$(date +%Y%m%d%H%M%S)'" >&2
    exit 1
fi

# Source repos: a complete local cache snapshot needs no Hub access at all;
# only probe the Hub for repos we cannot serve from cache, and explain gated
# repos correctly (being logged in is NOT the same as having accepted the
# model's license gate).
python - <<PYEOF
import os, sys
from huggingface_hub import auth_check, snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError

for repo in ("${ENCODER}", "${QWEN}"):
    if os.path.isdir(repo):
        continue  # local checkpoint directory, nothing to authorize
    try:
        snapshot_download(repo, local_files_only=True)
        print(f"cache OK: {repo} (fully cached, no Hub access needed)")
        continue
    except LocalEntryNotFoundError:
        pass  # not (fully) cached -> must be downloadable
    try:
        auth_check(repo)
        print(f"Hub access OK: {repo}")
    except GatedRepoError:
        print(f"ERROR: '{repo}' is a GATED repo and this account has not been", file=sys.stderr)
        print("granted access. Logging in again does not help. Fix:", file=sys.stderr)
        print(f"  1. Open https://huggingface.co/{repo} in a browser,", file=sys.stderr)
        print("     log in as THIS account, and accept the license/gate.", file=sys.stderr)
        print("  2. If using a fine-grained token: enable 'Read access to", file=sys.stderr)
        print("     contents of all public gated repos you can access'.", file=sys.stderr)
        print("  3. Check the account: hf auth whoami", file=sys.stderr)
        print("Or avoid the Hub entirely: pass a local copy via ENCODER=/path", file=sys.stderr)
        print("or reuse an existing converted checkpoint (see script header).", file=sys.stderr)
        raise SystemExit(1)
    except RepositoryNotFoundError:
        print(f"ERROR: '{repo}' not found (private repo or typo).", file=sys.stderr)
        print("Check the id, or log in with a token that can read it.", file=sys.stderr)
        raise SystemExit(1)
PYEOF

echo "============================================================"
echo " QASR weight conversion"
echo " Encoder:  ${ENCODER}"
echo " Decoder:  ${QWEN}"
echo " Output:   ${OUTPUT_DIR}"
echo " Seed:     ${SEED}"
echo " HF cache: ${HF_CACHE}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Convert (with reload verification)
# ---------------------------------------------------------------------------
python -m qasr.convert_weights \
    --encoder "${ENCODER}" \
    --qwen "${QWEN}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --verify-reload

# ---------------------------------------------------------------------------
# Post-flight: the artifacts every downstream phase needs must exist
# ---------------------------------------------------------------------------
MISSING=0
if [[ ! -f "${OUTPUT_DIR}/config.json" ]]; then
    echo "MISSING: ${OUTPUT_DIR}/config.json" >&2
    MISSING=1
fi
# transformers <5 wrote preprocessor_config.json; 5.x folds the feature
# extractor into processor_config.json. Either satisfies loading.
if [[ ! -f "${OUTPUT_DIR}/preprocessor_config.json" && ! -f "${OUTPUT_DIR}/processor_config.json" ]]; then
    echo "MISSING: ${OUTPUT_DIR}/preprocessor_config.json (or processor_config.json)" >&2
    MISSING=1
fi
if ! ls "${OUTPUT_DIR}"/model*.safetensors >/dev/null 2>&1; then
    echo "MISSING: ${OUTPUT_DIR}/model*.safetensors" >&2
    MISSING=1
fi
if [[ "${MISSING}" -ne 0 ]]; then
    echo "Conversion finished but the checkpoint is incomplete — do NOT train on it." >&2
    exit 1
fi

echo "============================================================"
echo " Checkpoint complete:"
du -sh "${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}" | sed 's/^/   /'
echo ""
echo " Next: point Phase 1 at it —"
echo "   sed -i 's|^model_name_or_path: .*|model_name_or_path: ${OUTPUT_DIR}|' \\"
echo "     configs/v7.6/01_projector_8node.yaml"
echo " then launch (see scripts/run_v76_phase1_node.sh)."
echo "============================================================"
