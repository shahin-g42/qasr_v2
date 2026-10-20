"""Recovery launcher CPU thread budget."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_recovery_launcher_replaces_inherited_thread_oversubscription(tmp_path):
    # Only the --check path is executed with a fake Python that prints its env.
    envdir = tmp_path / "env/bin"
    envdir.mkdir(parents=True)
    python = envdir / "python"
    python.write_text('#!/bin/bash\nprintf "%s %s %s %s %s\\n" "$OMP_NUM_THREADS" '
                      '"$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$NUMBA_NUM_THREADS" '
                      '"${GQASR_WORKER_START_METHOD:-unset}"\n')
    python.chmod(0o755)
    env = dict(os.environ, GEMMA_SPEECH_ENV=str(envdir.parent),
               OMP_NUM_THREADS="24", OPENBLAS_NUM_THREADS="32", MKL_NUM_THREADS="32")
    env.pop("GQASR_CPU_THREADS", None)
    env.pop("GQASR_WORKER_START_METHOD", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/resume_gqasr_encoder_12node.sh"), "check"],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "1 1 1 1 unset"
