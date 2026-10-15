# v8.1 Stage 3 — curated anchors with code-switch replay

Prepared for the existing qasr3 environment, 12 nodes × 8 H100 80 GB GPUs,
master `inception-H100-hpc-051`. Stage 2 and older configs are unchanged.
This is a proposed polish recipe, not a demonstrated accuracy improvement.
The supplied late Stage 2 logs show finite training loss/gradients, but contain
no validation WER/CER. Establish the baseline below before judging Stage 3.

## Recipe and evidence

- Initialize weights from the completed `output/v81/full_12node` export.
  Start a fresh optimizer and scheduler; do not resume Stage 2 optimizer state.
  Save separately to `output/v81/hq_12node`.
- Use current `asr_cleaned_manifests/full_v1` manifests. Remove only Arabic and
  English `train_*_q3asr` from the anchor pool. Retain all other Stage 2 anchors,
  including Arabic el_ar_v1. Hindi and Chinese have no alternative training
  source in this recipe, so retain their cleaned q3asr data. Do not restore
  obsolete Malayalam range slices or empty evaluation manifests.
- These source names do not establish human-verified quality. All references
  passed the cleaning pipeline, which may still make transcription mistakes.
- Keep weighted batch counts ar/en/hi/ml/zh = 10/6/2/1/1, microbatch 20,
  accumulation 1, global batch 1,920. The old v7.6 equal-language Stage 3
  recipe repeated Chinese about 132 times; this recipe reduces that exposure.
- One weighted epoch, 32,679 optimizer steps, 62,743,680 presentations.
  This is the full budget, not a guarantee that the last checkpoint is best.
- Peak LR 4e-6 for audio/projector, 2e-6 for the LLM, 1e-6 for embeddings;
  500-step warmup then cosine decay. This is five times below Stage 2's peak.
  Keep label smoothing, clipping, BF16, SDPA and ZeRO-2 unchanged.
- Preserve code-switch probability 0.4 and all four augmentation probabilities
  at 0.37. Preserve the tested duration/token limits and augmentation strengths.
  No speculative new augmentation is introduced during the stage transition.
- Reuse the existing cs_v1 index, with no plan/run/verify/assemble rebuild.
  **Partner clips still come from the full Stage 2 pool**, including ar/en
  q3asr. This is curated-anchor training with full-pool code-switch replay,
  not a purely curated-audio stage. A strictly curated partner pool would
  require a separate index or a separately tested runtime source filter.
- Evaluate and save every 1,000 steps; retain eight recent checkpoints.
  Keep the same callback sample seed, manifests, 100-per-language budget and
  768-token decode cap for comparison with Stage 2. The independent suite
  below uses more samples. Automatic best-WER checkpoint selection is not
  implemented; preserve a promising checkpoint before retention deletes it.

Budget from the assembled-count snapshot (confirm once on the cluster):

| Language | Anchor records | Approximate presentations per record |
|---|---:|---:|
| Arabic | 21,155,674 | 1.48 |
| English | 18,823,037 | 1.00 |
| Hindi | 5,511,368 | 1.14 |
| Malayalam | 306,175 | 10.25 |
| Chinese | 154,182 | 20.35 |

English determines the epoch length. Online composition/augmentation varies
repeated views but does not remove overfitting risk. Counts exclude neither
runtime audio failures nor overlength-token replacements; the scan validates
the dataset's index-time filters. Do not raise the 512-token limit just to
silence warnings. Audit recurring Malayalam rejects independently.

## 1. Finish Stage 2 and establish a baseline

Wait for step 62,891, final evaluation and final export to finish, and for the
Stage 2 launchers to return. A populated directory or the last progress bar
alone is not proof the save completed. The Stage 3 preflight requires the
root final trainer_state.json, completed step count and finite train loss,
then checks checkpoint shards, processor, manifests, MUSAN and frozen index.
Keep the Stage 2 export unchanged throughout Stage 3.

In the project root, using the existing qasr3 environment:

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
bash scripts/run_v81_phase3_12node.sh 0 --preflight --scan

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/eval_v81_phase3.py \
  --model output/v81/full_12node \
  --output-dir output/v81/evaluation/stage2_final
```

Run the full `--scan` once, not on every node: it reads ~46M anchor records.
It keeps one manifest index at a time in memory and prints per-manifest counts.
Ordinary launches perform metadata/index checks without rescanning all rows
or decoding/re-hashing all partner audio.

The independent evaluation uses 1,000 deterministic examples per manifest,
or all usable examples for a smaller manifest. It runs sequentially on one
GPU; optionally distribute languages across GPUs/nodes with `--language ar`
(likewise en/hi/ml/zh) and distinct GPU assignments. Use the same settings
before and after. Existing result files are not overwritten.

## 2. Rehearse after Stage 2 releases the GPUs

```bash
NNODES=1 bash scripts/run_v81_phase3_12node.sh 0 localhost --smoke-test
```

This checks the model forward/backward, code-switch loader and evaluation
callback using eight GPUs. It writes beneath `hq_12node/smoke-test`, not the
production output. It is not a 96-GPU networking rehearsal. Require the
two-step smoke run to finish without OOM/NaNs before the production launch.

## 3. Launch on twelve nodes

All nodes must use the same checkout, config and master endpoint. The launcher
activates qasr3, selects GPUs 0–7 after activation, validates eight visible
GPUs, then uses torchrun. Master port is **29583**, distinct from Stage 2.
Startup logs are under `output/v81/logs_p3_12node/`.

Use rank 0 on **inception-H100-hpc-051**. Run each remaining line on a
different allocated worker, preserving your Stage 2 rank assignments:

```bash
# master 051
bash scripts/run_v81_phase3_12node.sh 0
# worker assigned rank 1
bash scripts/run_v81_phase3_12node.sh 1
# worker assigned rank 2
bash scripts/run_v81_phase3_12node.sh 2
# worker assigned rank 3
bash scripts/run_v81_phase3_12node.sh 3
# worker assigned rank 4
bash scripts/run_v81_phase3_12node.sh 4
# worker assigned rank 5
bash scripts/run_v81_phase3_12node.sh 5
# worker assigned rank 6
bash scripts/run_v81_phase3_12node.sh 6
# worker assigned rank 7
bash scripts/run_v81_phase3_12node.sh 7
# worker assigned rank 8
bash scripts/run_v81_phase3_12node.sh 8
# worker assigned rank 9
bash scripts/run_v81_phase3_12node.sh 9
# worker assigned rank 10
bash scripts/run_v81_phase3_12node.sh 10
# worker assigned rank 11
bash scripts/run_v81_phase3_12node.sh 11
```

Do not paste all twelve commands into one shell/node. Production requires
96 GPUs; DeepSpeed batch sizes use `auto`, so no separate Accelerate config
is needed. At Stage 2's recent 4.1–4.7 seconds/step this is approximately
37–43 hours of optimizer steps, plus evaluation/checkpoint/startup time;
the changed data mix may change throughput.

If interrupted, resume a complete **Stage 3** checkpoint on every node:

```bash
bash scripts/run_v81_phase3_12node.sh YOUR_RANK \
  --resume-from-checkpoint /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/output/v81/hq_12node/checkpoint-1000
```

Replace the checkpoint number with your actual complete checkpoint. Do not
use `--resume-from-checkpoint` to start Stage 3 from a Stage 2 checkpoint.

## 4. Evaluate and select, rather than automatically promote

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/eval_v81_phase3.py \
  --model output/v81/hq_12node \
  --output-dir output/v81/evaluation/stage3_final
```

Compare per-language WER **and** CER, decoded references/hypotheses, punctuation
and Arabic marks against Stage 2. Scoring preserves diacritics and punctuation.
Use CER especially for Chinese; whitespace WER alone is inadequate there.
Malayalam has only 60 held-out rows and Hindi 115 before filtering; their
estimates are noisy. Do not infer generalization from training loss or a small
macro-average improvement. The standalone sample selection differs from the
training callback: compare standalone to standalone, callback to callback.

The current evaluation manifests are monolingual. A fixed, independently
held-out code-switch set covering all seven pairs is still required to
establish code-switch retention; the training partner index is not a test
set. Inspect real switched speech where available—concatenated clips alone
do not demonstrate natural code-switch recognition. Do not promote a model
with material per-language or code-switch regressions just because it is the
last checkpoint. Streaming and noisy-speech release checks remain separate.
