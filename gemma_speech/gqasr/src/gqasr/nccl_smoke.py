"""Bounded, model-free distributed communication check; launch with torchrun."""

import argparse
import faulthandler
import json
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import timedelta


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


class Progress:
    def __init__(self, backend, timeout):
        self.backend = backend
        self.timeout = timeout
        self.phase = "startup"
        self.identity = {
            "rank": os.environ.get("RANK"),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }

    def emit(self, status, **fields):
        print(json.dumps({
            "event": "communication_smoke",
            **self.identity,
            "backend": self.backend,
            "phase": self.phase,
            "status": status,
            **fields,
        }), flush=True)

    @contextmanager
    def stage(self, name, **fields):
        self.phase = name
        started = time.monotonic()
        self.emit("start", hard_timeout_seconds=self.timeout + 15, **fields)
        # This C-level watchdog still fires when a native CUDA/NCCL call holds
        # the GIL. Leave time for the process group's own timeout diagnostics.
        faulthandler.dump_traceback_later(self.timeout + 15, exit=True)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
        self.emit("ok", elapsed_seconds=round(time.monotonic() - started, 3), **fields)


def run(args, progress):
    with progress.stage("import_torch"):
        import torch
        import torch.distributed as dist

    with progress.stage("local_device"):
        missing = [key for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if key not in os.environ]
        if missing:
            raise RuntimeError(f"Launch with torchrun; missing environment variables: {missing}")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        if world < 2 or not 0 <= rank < world or local_rank < 0:
            raise RuntimeError("Use at least two workers with valid, unique ranks")
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("PyTorch needs Gloo for the CPU readiness check")
        torch.set_num_threads(1)
        device = torch.device("cpu")
        info = {
            **progress.identity,
            "rank": rank,
            "local_rank": local_rank,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "settings": {
                key: os.environ[key] for key in (
                    "CUDA_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
                    "NCCL_IB_HCA", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL",
                    "NCCL_BUFFSIZE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_NET",
                    "CUDA_DEVICE_MAX_CONNECTIONS",
                ) if key in os.environ
            },
        }
        if args.backend == "nccl":
            if not torch.cuda.is_available() or not dist.is_nccl_available():
                raise RuntimeError("CUDA/NCCL unavailable; this test will not fall back to CPU")
            visible = torch.cuda.device_count()
            local_world = int(os.environ.get("LOCAL_WORLD_SIZE", local_rank + 1))
            if local_rank >= visible or local_world > visible:
                raise RuntimeError(
                    f"{local_world} workers but only {visible} visible GPUs; "
                    "check NPROC_PER_NODE, GPU_IDS and CUDA_VISIBLE_DEVICES"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            probe = torch.ones(1, device=device)
            if (probe + 1).item() != 2:
                raise RuntimeError("Local GPU arithmetic failed")
            torch.cuda.synchronize(device)
            info.update(gpu=torch.cuda.get_device_name(device), nccl=torch.cuda.nccl.version())
        progress.emit("device_ready", **info)

    timeout = timedelta(seconds=args.timeout)
    with progress.stage("cpu_rendezvous"):
        dist.init_process_group("gloo", timeout=timeout)

    with progress.stage("rank_inventory"):
        peers = [None] * world
        # Include the test parameters so mismatched per-node commands fail here,
        # instead of issuing mismatched collectives later.
        dist.all_gather_object(peers, {"worker": info, "test": vars(args)})
        if any(peer["test"] != vars(args) for peer in peers):
            raise RuntimeError("Test arguments differ across ranks; use identical options on all nodes")
        assignments = [(peer["worker"]["host"], peer["worker"]["local_rank"]) for peer in peers]
        if len(set(assignments)) != world:
            raise RuntimeError("Duplicate host/local-rank assignments; check node ranks and launchers")
        if rank == 0:
            progress.emit("inventory", world_size=world, workers=[peer["worker"] for peer in peers])

    with progress.stage("cpu_ready_barrier"):
        dist.monitored_barrier(timeout=timeout, wait_all_ranks=True)

    group = dist.group.WORLD
    if args.backend == "nccl":
        with progress.stage("nccl_init"):
            # Eagerly initialize the communicator on this worker's GPU.
            group = dist.new_group(backend="nccl", timeout=timeout, device_id=device)

    def barrier():
        kwargs = {"device_ids": [local_rank]} if args.backend == "nccl" else {}
        dist.barrier(group=group, **kwargs)

    with progress.stage("initial_barrier"):
        barrier()

    with progress.stage("scalar_all_reduce"):
        scalar = torch.tensor([rank + 1], dtype=torch.int64, device=device)
        dist.all_reduce(scalar, group=group)
        expected = world * (world + 1) // 2
        actual = scalar.item()  # Synchronizes CUDA before reporting success.
        if actual != expected:
            raise RuntimeError(f"Scalar all-reduce returned {actual}; expected {expected}")

    with progress.stage("allocate_payload", tensor_mib=args.tensor_mib):
        payload = torch.empty(args.tensor_mib * 1024 * 1024 // 4, dtype=torch.float32, device=device)
        if args.backend == "nccl":
            torch.cuda.synchronize(device)

    for iteration in range(1, args.iterations + 1):
        with progress.stage("payload_all_reduce", iteration=iteration, tensor_mib=args.tensor_mib):
            # Alternating values catch stale results as well as missing contributions.
            value = rank % 2 + iteration % 2
            expected = world // 2 + world * (iteration % 2)
            payload.fill_(value)
            dist.all_reduce(payload, group=group)
            if not payload.eq(expected).all().item():
                raise RuntimeError(f"Payload all-reduce mismatch at iteration {iteration}")

    with progress.stage("broadcast", tensor_mib=args.tensor_mib):
        payload.fill_(7 if rank == 0 else -1)
        dist.broadcast(payload, src=0, group=group)
        if not payload.eq(7).all().item():
            raise RuntimeError("Broadcast payload mismatch")

    with progress.stage("final_barrier"):
        barrier()
    with progress.stage("all_ranks_verified"):
        dist.monitored_barrier(timeout=timeout, wait_all_ranks=True)
    with progress.stage("shutdown"):
        if args.backend == "nccl":
            dist.destroy_process_group(group)
        dist.destroy_process_group()
    progress.phase = "complete"
    progress.emit(
        "PASS",
        world_size=world,
        iterations=args.iterations,
        tensor_mib=args.tensor_mib,
        nccl_tested=args.backend == "nccl",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl",
                        help="gloo is an explicit CPU-only comparison; it does not test NCCL")
    parser.add_argument("--timeout", type=positive_int, default=60,
                        help="per-operation timeout in seconds; hard exit follows 15 seconds later")
    parser.add_argument("--tensor-mib", type=positive_int, default=32)
    parser.add_argument("--iterations", type=positive_int, default=10)
    args = parser.parse_args(argv)
    faulthandler.enable()
    progress = Progress(args.backend, args.timeout)
    try:
        run(args, progress)
    except Exception as exc:
        progress.emit("FAIL", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        # A failed communicator can hang in destruction. This diagnostic worker
        # exits immediately; torchrun terminates its remaining local siblings.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
