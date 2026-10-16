#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${GEMMA_SPEECH_ENV:-$ROOT/.venv}/bin/activate"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Downloads happen on the cluster; reuse pinned local snapshots on subsequent runs.
HF_HUB_OFFLINE=0 python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
for size in ('0.6', '1.7'):
    repo = f'Qwen/Qwen3-ASR-{size}B-hf'
    target = Path(sys.argv[1]) / f'models/qwen3-asr-{size}b-hf'
    provenance = target / 'baseline_source.json'
    if provenance.exists():
        source = json.loads(provenance.read_text())
        if source['repository'] != repo:
            raise ValueError(f'Unexpected baseline source: {source}')
        revision = source['revision']
    else:
        revision = HfApi().model_info(repo).sha
    snapshot_download(repo, revision=revision, local_dir=str(target))
    provenance.write_text(json.dumps(dict(repository=repo, revision=revision), indent=2))
PY
export HF_HUB_OFFLINE=1
ARGS=()
if [[ -n "${GQASR_RUN_DIR:-}" ]]; then ARGS+=(--run-dir "$GQASR_RUN_DIR"); fi
python -m gqasr.comparison_audit --root "$ROOT" --gpus "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --eval-step "${EVAL_STEP:-7294}" "${ARGS[@]}" "$@"
