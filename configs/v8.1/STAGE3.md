# v8.1 Stage 3 — curated anchors with code-switch replay

Prepared for the existing qasr3 environment, 12 nodes × 8 H100 80 GB GPUs,
master `inception-H100-hpc-051`. Stage 2 and older configs are unchanged.
This is a proposed polish recipe, not a demonstrated accuracy improvement.
The supplied late Stage 2 logs show finite training loss/gradients, but contain
no validation WER/CER. Establish the baseline below before judging Stage 3.

## Eight-node batch-50 alternative (code-switching disabled)

Use `configs/v8.1/03_hq_8node_b50_no_cs.yaml` and
`scripts/run_v81_phase3_8node_b50.sh`. This separate recipe leaves the
12-node config unchanged. Master is `inception-H100-hpc-053:29584`.

- 8 nodes × 8 GPUs, batch 50 per GPU, accumulation 1: global batch 3,200.
- Exact budget: 32,679 optimizer steps (104,572,800 presentations).
- DataLoader workers: 10 per GPU process, 80 per node, 640 across the job.
- Code-switch probability 0 and index null: no cs_v1 or refresh dependency.
- Other augmentations and the Stage 3 4e-6 peak LR / cosine schedule remain.
- Quotas ar/en/hi/ml/zh = 25/15/5/3/2. Batch 50 cannot give both smaller
  languages exactly 5%; these integer counts give 50/30/10/6/4 percent.
- The fixed step budget overrides num_train_epochs and spans approximately
  1.67 weighted epochs (19,608 steps per epoch from the count snapshot).
- Same completed Stage 2 initialization; separate output
  `output/v81/hq_8node_b50_no_cs` and startup logs
  `output/v81/logs_p3_8node_b50_no_cs`. Do not resume a code-switched run.

From the project root on master 053 after pulling the new files:

```bash
bash scripts/run_v81_phase3_8node_b50.sh 0 --preflight --scan
NNODES=1 bash scripts/run_v81_phase3_8node_b50.sh 0 localhost --smoke-test
```

The full scan is optional and should run only once. It accepts the requested
multi-epoch budget while rejecting a budget shorter than one full weighted
epoch. The smoke run uses batch 50 with two DataLoader workers per process
(the existing smoke cap), writes to the `smoke-test` subdirectory and must
pass on the cluster before assuming batch 50 fits H100 memory. Local tests
cannot establish GPU memory feasibility.

Then run one command per machine, ranks 0–7 exactly once:

```bash
# Master: inception-H100-hpc-053
bash scripts/run_v81_phase3_8node_b50.sh 0
# Worker 1
bash scripts/run_v81_phase3_8node_b50.sh 1
# Worker 2
bash scripts/run_v81_phase3_8node_b50.sh 2
# Worker 3
bash scripts/run_v81_phase3_8node_b50.sh 3
# Worker 4
bash scripts/run_v81_phase3_8node_b50.sh 4
# Worker 5
bash scripts/run_v81_phase3_8node_b50.sh 5
# Worker 6
bash scripts/run_v81_phase3_8node_b50.sh 6
# Worker 7
bash scripts/run_v81_phase3_8node_b50.sh 7
```

For evaluation use `scripts/eval_v81_phase3.py` with
`--config configs/v8.1/03_hq_8node_b50_no_cs.yaml`, the desired `--model`,
and a new `--output-dir`. The sections below describe the original 12-node
code-switch recipe, not this eight-node alternative.

## Recipe and evidence

- Initialize weights from the completed `output/v81/full_12node` export.
  Start a fresh optimizer and scheduler; do not resume Stage 2 optimizer state.
  Save separately to `output/v81/hq_ml_v2_12node`.
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
- Reuse the unchanged cs_v1 language pools through the Malayalam revision below.
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

## Malayalam revision after the cleanup

The Stage 3 config now uses `code_switch_indexes/cs_v1_ml_v2/assembled/index.json`
and a fresh output directory `output/v81/hq_ml_v2_12node` to avoid mixing index
receipts or optimizer state from an earlier attempt. GQASR and Stage 2 configs
are unchanged.

Run once on one cluster machine in qasr3, from the repository root:

```bash
RUN_ID=cs_v1_ml_v2 JOBS=8 bash scripts/corpus/run_code_switch.sh refresh-ml \
  --source-index "$PWD/code_switch_indexes/cs_v1/assembled/index.json" \
  --quarantine "$PWD/asr_cleaned_manifests/ml_training_v2/quarantine/ml_damaged_transcripts.jsonl"
```

This reads the current Malayalam training manifest at its existing path,
retokenizes and validates its audio, removes missing/deleted and quarantined
partners, and publishes a separate Malayalam pool. It also picks up eligible
new or repaired records. Quarantine is matched by audio basename, as in GQASR.
Only Malayalam audio is reprocessed. Eval sources and other-language source
metadata must still match the original plan; newly overlapping cross-language
partners cause an error. The original `cs_v1` is not modified.

Other language SQLite pools are reused via absolute symlinks, so keep the
original index in place. Their sizes and source metadata are checked; they
are not fully rehashed during this refresh. This assumes the published original
pools remain immutable, as normal training does. The new plan snapshots the
quarantine file and revised source metadata. Publication is atomic, and a
failure removes only private staging files. A hard-killed process may leave
an unpublished `*.building-*` directory; do not use it as an index.

No separate plan/run/verify/assemble command is needed for this revision.
Those lifecycle commands are refused for a derived index: its non-Malayalam
pools are referenced, not locally rebuilt from chunks. Validate it with the
Stage 3 preflight below. For another edit, choose another destination RUN_ID,
refresh from the original full `cs_v1`, and update a copied Stage 3 config's
index and output paths. Never mutate a published index during training.

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
callback using eight GPUs. It writes beneath `hq_ml_v2_12node/smoke-test`, not the
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
  --resume-from-checkpoint /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/output/v81/hq_ml_v2_12node/checkpoint-1000
```

Replace the checkpoint number with your actual complete checkpoint. Do not
use `--resume-from-checkpoint` to start Stage 3 from a Stage 2 checkpoint.

## 4. Evaluate and select, rather than automatically promote

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/eval_v81_phase3.py \
  --model output/v81/hq_ml_v2_12node \
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
