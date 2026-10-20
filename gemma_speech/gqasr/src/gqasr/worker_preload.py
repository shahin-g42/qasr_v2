"""CPU imports shared by DataLoader children of a fresh forkserver.

Do not construct processors, tokenize, run tensor kernels, open datasets or
initialize CUDA here. The server must never inherit the training process.
"""

import os

import torch
import soundfile  # noqa: F401
from scipy import signal  # noqa: F401

from . import augmentation, data, processing  # noqa: F401


PRELOAD_PID = os.getpid()
if torch.cuda.is_initialized():
    raise RuntimeError("DataLoader forkserver must not initialize CUDA")
