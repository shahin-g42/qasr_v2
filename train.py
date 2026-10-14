#!/usr/bin/env python3
"""Convenience entry point for the QASR trainer."""

import os
import socket


def main():
    print(
        f"Worker entered: host={socket.gethostname()} pid={os.getpid()} "
        f"rank={os.environ.get('RANK', 'standalone')} "
        f"local_rank={os.environ.get('LOCAL_RANK', 'standalone')} "
        f"world_size={os.environ.get('WORLD_SIZE', 'standalone')}; importing training modules",
        flush=True,
    )
    from qasr.train import main as train_main
    print('Worker: training module imports complete', flush=True)
    train_main()


if __name__ == "__main__":
    main()
