# Shared environment for the staged corpus drivers (scripts/corpus/*.slurm).
#
# This is the bash mirror of configs/corpus.yaml's `paths:` section -- edit
# both together (the test suite pins the YAML side; this side is ops). Every
# value is overridable from the environment, so a one-off run can redirect a
# single path without editing anything:
#
#   POOL_DIR=/tmp/pool sbatch scripts/corpus/stage1_prepare.slurm
#
# Source, don't execute. Sourcing also activates the conda env and cd's to the
# repo root, so drivers stay three lines of substance each.

# --- repo + python -----------------------------------------------------------
# The repo checkout that ALSO hosts the internal trees (training_manifests/v7.6,
# q3asr_sft_manifests) -- the training checkout, not a separate one.
export QASR="${QASR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
cd "$QASR"
export PYTHONPATH="$QASR/src${PYTHONPATH:+:$PYTHONPATH}"

# Conda environment with data_processing's dependencies (datasets,
# huggingface-hub, pyyaml, numpy/scipy/soundfile).
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- shared build paths (all nodes must mount these) -------------------------
# Mirrors configs/corpus.yaml paths: internal_root / sft_root / pool_dir /
# audio_root / out_dir / logs. Both internal trees live in the TRAINING
# checkout (train/stt/qasr) and are named file-for-file in
# configs/v7.6/internal_ds_sources.yaml -- the corpus registry mirrors that
# list (internal_v76_* + internal_sft_* specs).
export INTERNAL_ROOT="${INTERNAL_ROOT:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/training_manifests/v7.6}"
# SECOND internal tree: raw q3asr SFT envelopes, sibling of the v7.6 manifests.
# The registry reads it (the internal_sft_* specs resolve against it), and
# stages 2 + 4 pass it as --eval-root so its eval_<lang>_q3asr.jsonl files
# join the leak-exclusion set.
export QASR_SFT_ROOT="${QASR_SFT_ROOT:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/q3asr_sft_manifests}"
# The extracted Emilia-ZH tree (LOCAL_AUDIO: used in place, never downloaded).
# The zh spec reads ${QASR_EMILIA_ROOT}/ZH; override if the extract lives elsewhere.
export QASR_EMILIA_ROOT="${QASR_EMILIA_ROOT:-/vast/audio/data/tts/44k/Emilia-Dataset-extracted}"
export POOL_DIR="${POOL_DIR:-$QASR/corpus/pool}"
export AUDIO_ROOT="${AUDIO_ROOT:-$QASR/corpus/audio}"
export OUT_DIR="${OUT_DIR:-$QASR/training_manifests/v8.0}"
export LOGS="${LOGS:-$QASR/logs/corpus}"

# SQLite-WAL ledgers: exactly one writer per ledger file, ever. Node-local
# disk is preferred when the node has it (override LEDGER_DIR); this cluster
# cannot create /scratch, so the default scratch area lives in the workspace
# (one writer per file is what keeps shared storage safe here).
export LEDGER_DIR="${LEDGER_DIR:-$QASR/scratch/corpus/ledgers}"

mkdir -p "$POOL_DIR" "$AUDIO_ROOT" "$OUT_DIR" "$LOGS" "$LEDGER_DIR"

# --- Hub access for the external sources --------------------------------------
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set -- gated sources (hi/ml especially) will fail" >&2
fi

# The corrector: a persistent vLLM container that owns every node's GPUs. No
# driver ever launches it or requests GPUs; where a stage needs it, the driver
# gates on this health endpoint instead.
export CORRECTOR_URL="${CORRECTOR_URL:-http://localhost:8010/v1}"
