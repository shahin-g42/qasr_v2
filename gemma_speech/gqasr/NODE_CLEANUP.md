# Stop leftover GQASR training processes

Run these commands from the cluster checkout. No administrator access or model
loading is needed. The helper uses Python's standard library and the Linux
process table.

The host file contains the actual 12 hosts from the October 20 diagnostics:
051, 020, 034, 047, 007, 013, 014, 041, 009, 056, 029, and 048, in node-rank
order. Review the file if your allocation changes. All hosts must have the same
checkout path and working SSH authentication.

Preview all nodes in parallel from one cluster node:

```bash
git pull origin main
bash scripts/cleanup_gqasr_nodes.sh \
  --hosts-file gemma_speech/cluster/gqasr_hosts_20261020.txt
```

Stop the listed GQASR processes:

```bash
bash scripts/cleanup_gqasr_nodes.sh \
  --hosts-file gemma_speech/cluster/gqasr_hosts_20261020.txt \
  --apply --force --grace-seconds 30
```

`--apply` sends SIGTERM. `--force` sends SIGKILL only to survivors after the
grace period. This stops any matching training run, including an active one;
do not launch a new run during cleanup. Omit `--force` for SIGTERM only.

If SSH between nodes is unavailable, run the local form on each actual node:

```bash
bash scripts/cleanup_gqasr_nodes.sh
bash scripts/cleanup_gqasr_nodes.sh --apply --force --grace-seconds 30
```

## Scope and reports

The helper selects only processes owned by the invoking user that belong to
GQASR training in this checkout: launchers, training ranks, and their descendants.
Recognizable orphaned Python workers and W&B sidecars require both the inherited
project configuration and distributed rank environment. Unproven orphaned
processes are left alone. The helper protects itself and its ancestor processes
and rechecks process identity before signalling.

Before stopping processes it saves available NCCL/log diagnostics into
`$GQASR_LOG_DIR/cleanup-HOST-TIMESTAMP-PID.json`, defaulting to
`/tmp/gqasr-$USER-logs/` on each node. The existing traces and logs remain in
place. Checkpoints, datasets, caches, and shared-memory files are not deleted.
The script does not invoke a GPU reset or reboot.

Each node reports `matched_processes_stopped`, remaining PIDs and states, GPU
processes reported by `nvidia-smi`, and the report path. A surviving or newly
spawned matching process makes cleanup exit nonzero; SSH failures also make the
parallel controller exit nonzero. Processes in uninterruptible I/O may survive
signals and need the underlying storage or node problem resolved.

`matched_processes_stopped: true` confirms the selected training processes are
gone. It is not an InfiniBand, storage, or cluster health test. Other GPU jobs may
still appear in the report. Keep the reports when investigating a failed run.
