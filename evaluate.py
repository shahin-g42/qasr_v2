"""Root entry point for the QASR inference evaluation suite.

Usage:
    PYTHONPATH=src python evaluate.py \
        --model /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/output/full \
        --manifest /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/validated_manifests/train_en_commentary.json \
        --eagle /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/output/eagle/checkpoint-5000 \
        --language en --num-samples 50
"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qasr.evaluate import main

if __name__ == "__main__":
    main()
