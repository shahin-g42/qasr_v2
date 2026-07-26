"""Root entry point for EAGLE-2 draft head training.

Usage:
    PYTHONPATH=src python train_eagle.py --config configs/train_eagle_8node_filtered.yaml
    torchrun --nproc_per_node=8 train_eagle.py --config configs/train_eagle_8node_filtered.yaml
"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qasr.train_eagle import main

if __name__ == "__main__":
    main()
