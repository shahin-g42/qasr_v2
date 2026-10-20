# NCCL communication check

This model-free test checks local CUDA arithmetic, CPU/Gloo connectivity,
NCCL communicator initialization, barriers, scalar and repeated 32 MiB
all-reduces, and broadcast. It checks the returned values on every rank.
No checkpoints or datasets are loaded. Each worker prints its host, rank,
GPU, versions, settings, current phase, and a final `PASS` on success.
Failures produce a `FAIL` record, a native error, or a watchdog stack dump.

Run from the repository root in each node's terminal. For the four hosts
in the startup log, launch these four commands, one on each indicated host:

```bash
# inception-H100-hpc-020.inception.ai
bash scripts/test_gqasr_nccl.sh 0 inception-H100-hpc-020.inception.ai
# inception-H100-hpc-034.inception.ai
bash scripts/test_gqasr_nccl.sh 1 inception-H100-hpc-020.inception.ai
# inception-H100-hpc-047.inception.ai
bash scripts/test_gqasr_nccl.sh 2 inception-H100-hpc-020.inception.ai
# inception-H100-hpc-007.inception.ai
bash scripts/test_gqasr_nccl.sh 3 inception-H100-hpc-020.inception.ai
```

Defaults are four nodes, eight visible GPUs per node, the
`gemma_speech/.venv/bin/python` environment, and test port **29691**.
The launcher preserves GPU visibility; `GPU_IDS=0,1,2,3,4,5,6,7` explicitly
selects all GPUs on a dedicated node. Use the same allocation and network
environment as training. Existing training processes are not stopped by this
test, and competing GPU workloads can affect its results.

Start all launchers within 180 seconds, or increase
`NCCL_TEST_RDZV_TIMEOUT` on every node. Collectives have a 60-second timeout.
A separate watchdog dumps Python stacks and exits after 75 seconds in any
one worker phase, including initialization and shutdown. `--timeout N`
changes these to N and N+15 seconds. No automatic restarts occur.

Logs are saved on each node under the printed
`/tmp/gqasr-nccl-smoke/...` directory: torchrun keeps separate worker stdout
and stderr logs, and `nccl.HOST.PID.log` files contain NCCL diagnostics.
Use `NCCL_TEST_LOG_DIR` to change the local log parent.

| Last unfinished phase | What to investigate |
| --- | --- |
| No worker records | Missing node launcher, master address/port, or torchrun rendezvous failure |
| `local_device` | GPU visibility, CUDA initialization, local GPU health or memory availability |
| `cpu_rendezvous` / `rank_inventory` / `cpu_ready_barrier` | Missing workers or CPU/TCP connectivity; NCCL has not been tested yet |
| `nccl_init` | NCCL communicator setup, selected network interface/HCA, fabric connectivity |
| `initial_barrier` / `scalar_all_reduce` / `payload_all_reduce` / `broadcast` | GPU communication or a peer that failed; inspect all nodes' logs |
| `shutdown` | Process-group teardown |
| All workers `PASS`, with `nccl_tested: true` | These NCCL collectives passed on this topology at this time |

A passing standalone test does not prove an existing training communicator
is healthy or rule out training collective-order bugs, memory pressure,
or intermittent fabric faults. The Gloo control group also uses TCP paths
that may differ from NCCL's selected transport.

To isolate local GPU communication, run on each host independently:

```bash
NNODES=1 bash scripts/test_gqasr_nccl.sh 0 127.0.0.1
```

Then repeat the four-node commands with `NPROC_PER_NODE=1` on every node to
test connectivity between hosts with one GPU per host. For a CPU-only
comparison, append `--backend gloo`; its `PASS` explicitly says
`nccl_tested: false`.

If the normal NCCL test fails, repeat with `NCCL_IB_DISABLE=1` on every node.
A pass only with that setting implicates the IB/RDMA path or its configuration,
but is not proof of a particular faulty adapter. Keep this override scoped to
the diagnostic run. The test preserves `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`,
`GLOO_SOCKET_IFNAME` and other exported fabric settings.

Other examples:

```bash
# Preview without launching or creating logs.
DRY_RUN=1 bash scripts/test_gqasr_nccl.sh 0 inception-H100-hpc-020.inception.ai
# Use a different environment, on every node.
NCCL_TEST_PYTHON=/path/to/training/python bash scripts/test_gqasr_nccl.sh 0 MASTER
# Longer/larger exercise: use the same options on every participating node.
bash scripts/test_gqasr_nccl.sh 0 MASTER --iterations 100 --tensor-mib 64 --timeout 120
```

The CPU readiness check uses PyTorch's
[Gloo monitored barrier](https://docs.pytorch.org/docs/2.11/distributed.html#torch.distributed.monitored_barrier).
The hard deadline uses Python's
[faulthandler watchdog](https://docs.python.org/3.11/library/faulthandler.html#faulthandler.dump_traceback_later).
