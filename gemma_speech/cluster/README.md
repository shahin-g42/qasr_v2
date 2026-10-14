# Cluster workflow

Run these commands **on the cluster**, from the repository root after copying/pulling the code. All model execution belongs on your machines; generated weights, indexes, logs and outputs are ignored by Git. `gqasr` and `gqtts` live inside `gemma_speech` and install together into one environment.

Use shared storage at the same absolute path on all 12 nodes for code, model artifacts, indexes and outputs. Use allocated compute resources for verification, tests and training; do not run heavy model work on a login node. Downloads require network access; later stages set `HF_HUB_OFFLINE=1`.

## 1. Install the common environment

```bash
bash gemma_speech/cluster/setup_env.sh
```

The script installs the CUDA 12.8 wheel of PyTorch 2.11.0 and Transformers 5.19.0. Your CUDA 12.9 toolkit can coexist with that bundled runtime; the NVIDIA driver must support the wheel. `GEMMA_SPEECH_ENV` optionally selects a different environment directory. Use the same environment on all nodes, or reproduce `gemma_speech/logs/environment.lock.txt` on each node. Do not install the upstream `qwen-tts` SDK into this environment: the compatible private port is included.

On every allocated GPU node, check `nvidia-smi` and run:

```bash
bash gemma_speech/cluster/gqasr_stage.sh doctor
```

This requires eight visible CUDA GPUs with BF16 support. Set `NPROC_PER_NODE` only when deliberately testing a smaller allocation; the training config's `expected_world_size` must also match.

## 2. Download and initialize GQASR once

```bash
bash gemma_speech/cluster/gqasr_stage.sh download
bash gemma_speech/cluster/gqasr_stage.sh export
bash gemma_speech/cluster/gqasr_stage.sh verify
bash gemma_speech/cluster/gqasr_stage.sh test -q
```

`download` fetches pinned upstream revisions into `gemma_speech/models/`. `export` creates `gemma_speech/gqasr/models/gqasr-base/` with pretrained core weights and a random bridge. It refuses to overwrite an existing artifact. `verify` is an offline synthetic integration check. The `test` stage uses tiny fixtures and needs no pretrained download.

The public core names and tensor prefixes are GQASR. Provenance and license notices retain the upstream source names. The export is a derived base requiring adaptation, not a trained recognizer.

## 3. Validate the actual manifests once

```bash
bash gemma_speech/cluster/gqasr_stage.sh preflight
bash gemma_speech/cluster/gqasr_stage.sh scan
```

The default configs copy 22 training and 6 evaluation paths from QASR v8.1. Update those paths if the cluster mount differs. `preflight` reports readiness; `scan` reads the manifests and audio headers, filters unusable samples and builds the shared index. A successful scan reports `ready_to_train: true`. It can take substantial time on the full corpus. Do not scan independently from all ranks or mutate the corpus afterward.

## 4. Train the bridge on 96 GPUs

With Slurm, submit **from the repository root**:

```bash
sbatch gemma_speech/cluster/submit_12node.sh gqasr bridge
```

Without Slurm, run the following on each of the 12 nodes, changing `NODE_RANK` from 0 to 11 and using node 0's reachable address:

```bash
export MASTER_ADDR=node0-hostname
export MASTER_PORT=29500
export NODE_RANK=0
bash gemma_speech/cluster/gqasr_stage.sh bridge
```

If needed, configure `NCCL_SOCKET_IFNAME` and other fabric settings for your network. The scripts do not guess interface names. All nodes must use matching rank/world-size and rendezvous settings.

## 5. Train the full ASR model

Wait for a successful bridge run and its `gemma_speech/gqasr/outputs/bridge/final/` artifact, then:

```bash
sbatch gemma_speech/cluster/submit_12node.sh gqasr full
```

For manual launch, run `gqasr_stage.sh full` on every node using the same rank setup. The full stage uses the pretrained bridge-stage artifact and resets the optimizer. Defaults are microbatch 2 × accumulation 10 × 96 GPUs, yielding 1,920 utterances per optimizer update with the source's exact language quotas. The source's online code-switching and augmentations are disabled; see `gqasr/configs/source_mapping.yaml`.

## 6. Resume or transcribe

For resume, copy the stage config **within its existing configs directory** and edit only `output_dir` to a new path. Keep the other settings unchanged. For example:

```bash
export GQASR_CONFIG="$PWD/gemma_speech/gqasr/configs/02_full_resume.yaml"
sbatch gemma_speech/cluster/submit_12node.sh gqasr full \
  --resume /shared/path/to/checkpoint-00002083
```

This config file must be created and edited first. Resume uses the saved optimizer, step and rank RNG states. Output directories are never silently overwritten. For inference after training:

```bash
bash gemma_speech/cluster/gqasr_stage.sh transcribe \
  --model "$PWD/gemma_speech/gqasr/outputs/full/final" \
  --audio /shared/path/sample.wav --device cuda
```

## GQTTS stages in the same environment

GQTTS contains the existing speaker-adapter and talker training implementation. It retains the Qwen codec. The ASR training manifest does not supply the explicit speaker identities and voice requirements needed to automatically prepare a TTS corpus. Its configs remain templates; prepare paired multilingual TTS manifests with speaker IDs and disjoint validation recordings/speakers before running:

```bash
bash gemma_speech/cluster/gqtts_stage.sh download
bash gemma_speech/cluster/gqtts_stage.sh cache \
  --manifest /shared/path/tts-train.jsonl \
  --output "$PWD/gemma_speech/data/speaker-train-cache" --include-codes
bash gemma_speech/cluster/gqtts_stage.sh cache \
  --manifest /shared/path/tts-validation.jsonl \
  --output "$PWD/gemma_speech/data/speaker-val-cache" --include-codes
sbatch gemma_speech/cluster/submit_12node.sh gqtts speaker-train
# After speaker adaptation completes:
sbatch gemma_speech/cluster/submit_12node.sh gqtts train
```

Cache creation supports `--shard` / `--shards`; give each shard a separate output directory and assemble cache manifests before training. Default cache commands run one process on CUDA. `GQTTS_CONFIG` selects a custom stage config. For inference, `gqtts_stage.sh synthesize` accepts `--checkpoint`, `--audio`, `--text`, `--language`, `--output` and optional `--talker-checkpoint`; run `gqtts synthesize --help` for details. GQTTS does not yet have GQASR's standalone renamed base-export format; the existing TTS assets/checkpoints remain its initialization workflow.

## Logs to send back

Stage console logs are saved under `gemma_speech/logs/`; GQASR verification reports are under `gemma_speech/gqasr/verification/`. Training writes `metrics.jsonl`, prediction files and checkpoints under the configured output directory. Send the stage log and any traceback first; for training include the initial parameter/global-batch report, recent metrics and node/GPU diagnostics. Slurm also captures scheduler output in its normal job log.

The scripts and package structure have been checked locally. Cluster data access, real-audio quality, H100 memory consumption, NCCL connectivity and 96-GPU scaling still require your runs.
