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
export QASR="${QASR:-/lustrefs/shared/shahin.konadath/workspace/train/qasr}"
cd "$QASR"
export PYTHONPATH="$QASR/src${PYTHONPATH:+:$PYTHONPATH}"

# Conda environment with data_processing's dependencies (datasets,
# huggingface-hub, pyyaml, numpy/scipy/soundfile).
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- shared build paths (all nodes must mount these) -------------------------
# Mirrors configs/corpus.yaml paths: internal_root / pool_dir / audio_root /
# out_dir / logs.
export INTERNAL_ROOT="${INTERNAL_ROOT:-$QASR/training_manifests/v7.6}"
export POOL_DIR="${POOL_DIR:-$QASR/corpus/pool}"
export AUDIO_ROOT="${AUDIO_ROOT:-$QASR/corpus/audio}"
export OUT_DIR="${OUT_DIR:-$QASR/training_manifests/v8.0}"
export LOGS="${LOGS:-$QASR/logs/corpus}"

# NODE-LOCAL: the per-language SQLite-WAL ledgers must never live on
# Lustre/NFS (WAL corrupts under concurrent writers). /scratch is node-local.
export LEDGER_DIR="${LEDGER_DIR:-/scratch/corpus/ledgers}"

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
