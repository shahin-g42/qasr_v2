#!/usr/bin/env python3
"""Stage 4 CLI: group the five languages' bN batches into shippable bundles.

Thin wrapper around ``data_processing.bundle`` (the importable, unit-tested
implementation), mirroring how the root ``train.py`` shims ``qasr.train``.
Run it beside the staged README's other stages:

    PYTHONPATH=src python3 scripts/assemble_bundles.py \\
        --out-dir "$OUT_DIR" --audio-root "$AUDIO_ROOT" --root "$INTERNAL_ROOT" \\
        --report "$LOGS/bundle.json"

Exit code 0 = every hard gate passed and ``$OUT_DIR/MANIFEST.json`` was
written; 1 = the audit failed (or nothing is shippable yet), and a previously
good MANIFEST is left untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_processing.bundle import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
