# v8.1 Stage 2 on twelve machines

Use twelve machines with eight H100 GPUs each (96 ranks). The master/rank-0
machine is `inception-H100-hpc-051`; use eleven distinct other machines for node
ranks 1–11. Their hostnames need not be listed in the launcher. All machines must
reach the master on port 29582 and share the same code, checkpoint, manifests and
assembled index.

The additional config is `configs/v8.1/02_full_12node.yaml`. The original
10-node config and launcher are unchanged. This run initializes weights from
`models/bck/full/v1`, not from an optimizer checkpoint of the 10-node run.

| Setting | Twelve-node run |
|---|---|
| Global batch | 1,920 (20 per GPU, accumulation 1) |
| Steps | 62,891; confirm filtered counts with the scan below |
| Warmup / final decay | 833 / 6,250 steps |
| Save / evaluate interval | 2,083 steps |
| Learning rate | 0.00002, unchanged |
| Code switching | p=0.4; existing cs_v1 index |
| Acoustic augmentations | Existing p=0.37 settings |
| Outputs | output/v81/full_12node |
| Logs | output/v81/logs_12node |
| Rendezvous port | 29582 |

On each node, start in the shared repository:

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
```

The launcher deactivates the current conda environment and automatically activates
`CONDA_ENV`, defaulting to `/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3`.
The Transformers version check remains enabled. Before
launching, run the full manifest scan once on the master. This can take time;
it confirms the computed coverage budget matches the config and fails on mismatch.
It does not rebuild or reverify the partner index's source audio.

```bash
bash scripts/run_v81_phase2_12node.sh 0 --preflight --scan
```

Run the loader diagnostic if it has not passed for the assembled index:

```bash
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3
bash scripts/check_v81_dataloader.sh --config configs/v8.1/02_full_12node.yaml --batches 100 --workers 8
```

First rehearse all twelve nodes: append `--smoke-test` to EACH command below,
launch all twelve, and wait for all to finish successfully. This is a two-step
forward/backward rehearsal, not a throughput benchmark. Then run the commands
without that flag for production. Run ONE corresponding command on EACH machine;
do not paste all commands into one machine. The other eleven machines can have
any hostnames, but their node ranks must be unique.

```bash
# Master: inception-H100-hpc-051
bash scripts/run_v81_phase2_12node.sh 0 inception-H100-hpc-051
# Worker 1: a different machine
bash scripts/run_v81_phase2_12node.sh 1 inception-H100-hpc-051
# Worker 2: a different machine
bash scripts/run_v81_phase2_12node.sh 2 inception-H100-hpc-051
# Worker 3: a different machine
bash scripts/run_v81_phase2_12node.sh 3 inception-H100-hpc-051
# Worker 4: a different machine
bash scripts/run_v81_phase2_12node.sh 4 inception-H100-hpc-051
# Worker 5: a different machine
bash scripts/run_v81_phase2_12node.sh 5 inception-H100-hpc-051
# Worker 6: a different machine
bash scripts/run_v81_phase2_12node.sh 6 inception-H100-hpc-051
# Worker 7: a different machine
bash scripts/run_v81_phase2_12node.sh 7 inception-H100-hpc-051
# Worker 8: a different machine
bash scripts/run_v81_phase2_12node.sh 8 inception-H100-hpc-051
# Worker 9: a different machine
bash scripts/run_v81_phase2_12node.sh 9 inception-H100-hpc-051
# Worker 10: a different machine
bash scripts/run_v81_phase2_12node.sh 10 inception-H100-hpc-051
# Worker 11: a different machine
bash scripts/run_v81_phase2_12node.sh 11 inception-H100-hpc-051
```

The master argument can be omitted; it defaults to `inception-H100-hpc-051`.
Keep the same config and world size on every node. For an intentional restart,
pass `--resume-from-checkpoint` with the same complete checkpoint under
`output/v81/full_12node` on all twelve machines. No indexing/assembly rerun is
needed for changing the training machine count. Longer mixed samples still need
cluster GPU-memory validation at batch size 20; local tests cannot establish it.


Startup logging begins before conda activation and is saved under
`output/v81/logs_12node`. It reports activation, Python imports, GPU visibility,
preflight stages, and entry into torchrun. GPU visibility is checked before the
expensive automatic preflight. Explicit `--preflight` remains usable without GPUs.

If a full eight-GPU machine has an inherited `CUDA_VISIBLE_DEVICES=0`, start each
rank with an explicit selection, for example on master:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_v81_phase2_12node.sh 0
```

Use the same prefix with ranks 1–11 on their corresponding full eight-GPU nodes.
`GPU_IDS` is applied after conda activation; without it, the launcher preserves
the resulting environment selection. Do not use this override to exceed a
scheduler-assigned allocation. All twelve launchers must pass the visibility
check before the distributed run can begin.
