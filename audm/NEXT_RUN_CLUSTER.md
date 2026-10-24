# AudFlash: fresh stride-one CTC adaptation, then full training

This run loads **weights from checkpoint-3700**, creates a new model export, trains a
new auxiliary CTC head, adapts the bridge/decoder, and then trains the entire model.
It does not resume the old optimizer, scheduler, or step counter.

## What changes

- **Existing encoder and stride-one bridge stay intact.** No additional pooling,
  striding, frame stacking, merging or pruning. The encoder's pretrained native
  frontend is unchanged. Roughly 750 audio embeddings for 30 seconds still enter
  the decoder; no claim of raw-waveform information preservation is made.
- An independent `aud_ctc_head` consumes the encoder's native output **before
  the bridge**, using the **existing decoder tokenizer and exact token IDs**.
  There is no tokenizer training, transcript scan for vocabulary building, or
  vocabulary restricted to today's languages. Decoder tokenization stays fixed
  when adding languages. The existing text-normalization/tokenization behavior
  is preserved; CTC targets contain transcript text only.
- CTC blank is a new **CTC-only output class** at `text_config.vocab_size`. It is
  never inserted into the decoder tokenizer or embeddings. Token ID 0 remains
  a valid ordinary CTC target if it is ordinary in the decoder tokenizer.
  With 151,936 decoder rows and encoder width 1536, the new CTC head contains
  **233,375,232 parameters**, bringing training parameters to about **1.135B**.
  Export reports the actual sizes from the source config. Tokenizer fingerprints
  detect changed segmentation/IDs on save and reload.
- Normal autoregressive generation does not execute CTC. The final
  `inference-export` removes its weights and configuration, leaving the trained
  encoder, stride-one bridge, decoder, and tokenizer. This returns the deployment
  artifact to about **901.9M parameters** for this source, without changing any
  retained weight. Keep the full training artifact for further CTC training.
- Objective: `autoregressive_loss + lambda × CTC_loss`. CTC uses transcript text
  only, excludes non-speech and impossible alignments (including repeated-token
  requirements), and never truncates a target to make it fit. Those examples keep
  their normal autoregressive supervision. CTC loss is computed in FP32, one
  example at a time with recomputation; normalization spans the accumulation
  window and ranks. Logs report eligible/ineligible counts and CTC loss.
- `lambda` ramps from zero to **0.1** over 100 adaptation / 500 full-training
  updates. Evaluation uses 0.1, so combined eval loss is not directly comparable
  with old autoregressive-only eval loss. WER/CER remain comparable.
- Duration grouping reorders samples **only inside each optimizer update**,
  preserving its sampled examples and language quota windows. Shared, read-only
  memory-mapped manifest indexes avoid repeated scans and large worker copies.
  Source path/stat/filter changes invalidate the cache.
- Exact autoregressive loss chunks increase from 128 to **512 supervised tokens**.
  Optional Liger fused loss and selective layer checkpointing are available for
  separate benchmarks. Default remains exact chunked loss and checkpointing on
  every encoder/decoder layer.
- Existing validated transcript token IDs are reused for length checks. Loader
  settings are 4 workers/GPU, persistent workers, prefetch 2 and a 300-second
  worker timeout. This makes stalls visible; it does not prove the earlier shared
  memory unlink failure is fixed on the cluster.
- Loss eval and WER decoding use **batch size 40 per GPU**. WER splits the same
  fixed sample set across ranks and combines raw error/reference counts. Small
  subsets/tails naturally use smaller batches. Chinese stays in training and is
  excluded from evaluation.
- Performance logs include update/data-wait time, padding fractions, category
  accuracy/NLL, category share of supervised tokens and CTC eligibility. Category
  loss weights are unchanged. Per-rank progress and CUDA memory reports remain
  available. The health tool reports thermal slowdown and active GPU jobs.
- Existing non-speech sampling, waveform augmentations, **SpecAugment with Gemma
  mean fill**, and fail-before-update numerical guards remain active. The new
  recipes copy data and augmentation settings from checkpoint-3700's receipt,
  adding the previously planned silence-padding effect if absent (p=0.2, 0–100 ms
  at each boundary, within the existing duration bounds/effect budget).

| Setting | Adaptation | Full |
|---|---|---|
| Trainable | Bridge + decoder + CTC head | Encoder + bridge + decoder + CTC head |
| Batch | 40 × 2 × 8 = 640 | 40 × 2 × 8 = 640 |
| Budget | 2,000 updates | 3 coverage epochs |
| Eval/save | Every 500 updates | Every 3,700 updates |
| Eval/WER batch | 40/GPU | 40/GPU |
| DataLoader workers | 4/GPU | 4/GPU |
| Label smoothing | 0 | 0.1 |

The inherited learning rate and parameter-group multipliers are recorded in the
generated YAML. All commands below run directly on one **8 × H100** node. They do
not use `srun` or `sbatch`. Stop an old job using your usual process management
before launching a new job on the same GPUs; these scripts never kill jobs.

## 1. Update and set the run paths

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2-aud-robust
conda activate audm
git pull --ff-only origin main
python -m pip install -e ./audm

export AUD_NEXT_SOURCE="$PWD/output/audflash-audio-types-v1/full/checkpoint-3700"
export AUD_NEXT_ROOT="$PWD/output/audflash-stride1-decoder-ctc-v1"
export AUD_LOG_DIR="$AUD_NEXT_ROOT/logs"
export AUD_MASTER_ADDR=127.0.0.1
export AUD_MASTER_PORT=29579
export PYTHONPATH="$PWD/audm/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
set -o pipefail

bash audm/scripts/audflash_next.sh health --require-idle
```

Keep your existing working FFmpeg configuration. If using `imageio-ffmpeg`, the
augmentation code discovers its bundled executable. No admin installation is
needed. Existing RIR/MUSAN/non-speech assets are reused from the saved recipe;
there is no download or corpus-finalization step here.

Use this **new run root** even if you already prepared the earlier 16k-CTC run.
Its files remain intact. Start from the original checkpoint-3700 above, not an
old CTC export, synthetic probe, or smoke-test model. New recipes explicitly
require decoder-tokenizer CTC and reject legacy CTC artifacts. Run each command
below individually; proceed only after it succeeds.

## 2. Prepare recipes and model (no tokenizer-training step)

```bash
bash audm/scripts/audflash_next.sh prepare
cat "$AUD_NEXT_ROOT/plan.json"

bash audm/scripts/audflash_next.sh export

bash audm/scripts/audflash_next.sh verify export
```

The export checks source tensors, preserves legitimate Gemma infinite clamp
bounds, validates an exact weight round trip, and records source weight hashes.
It refuses to overwrite existing output or an existing CTC head. Expect
`ctc_version: 2`, `ctc_blank_id: 151936` and `ctc_vocab_size: 151937` for this
source. The exact export report is in `export/provenance.json`; full verification
output is also retained in the printed debug-log directory. Do **not** run
`add-audio-types` again: checkpoint-3700 already contains those tokens.

```bash
python - <<'PY'
import os, yaml
from pathlib import Path
root = Path(os.environ['AUD_NEXT_ROOT'])
for stage in ('adaptation', 'full'):
    recipe = yaml.safe_load((root / f'{stage}.yaml').read_text())
    print(stage, 'SpecAugment:', recipe['training']['augmentation']['spec_augment'])
    print(stage, 'waveform policy:', recipe.get('robustness', {}).get('augmentation'))
    print(stage, 'non-speech:', recipe.get('robustness', {}).get('non_speech'))
    print(stage, 'efficiency:', recipe['efficiency'])
PY
```

## 3. Check assets, collectives and adaptation memory

```bash
bash audm/scripts/audflash_next.sh preflight adaptation --scan

bash audm/scripts/audflash_next.sh distributed-check

bash audm/scripts/audflash_next.sh probe adaptation --steps 3

bash audm/scripts/audflash_next.sh smoke adaptation

python -m audm verify "$AUD_NEXT_ROOT/adaptation/smoke-test" --device cuda:0
```

The probe retains the production batch/efficiency settings and exercises CTC at
full weight and maximum audio duration. It uses alternating transcript words to
make near-limit decoder-tokenizer CTC targets feasible when possible; otherwise
it guarantees at least one feasible CTC example alongside near-limit AR targets
in every microbatch. The report records the actual eligible quota. It disables augmentation and evaluation. **Discard probe
weights.** The real-data smoke test covers augmentation, non-speech integration,
evaluation and save/reload, but random sampling does not guarantee every rare
effect/class appears in two updates. Check the per-rank memory reports before
production; previous batch-10 or 16k-CTC results cannot establish memory safety
for this larger head at batch 40. GPU validation must run on your cluster.

A successful smoke test stays in `adaptation/smoke-test`; production writes its
parent without loading smoke weights. For repeat probes, set `AUD_PROBE_ROOT` to
a new scratch directory. Repeat smoke tests need a new scratch `--output` path.

## 4. Run adaptation

```bash
bash audm/scripts/audflash_next.sh adaptation

bash audm/scripts/audflash_next.sh verify adaptation
```

Review WER/CER, category metrics and aggregate CTC eligibility. If many speech
examples are ineligible, inspect token lengths by language before continuing. Large CTC loss or gradient spikes need
investigation before unfreezing. A synthetic probe is not an accuracy test.

## 5. Rehearse and run full training

```bash
bash audm/scripts/audflash_next.sh preflight full --scan

bash audm/scripts/audflash_next.sh probe full --steps 3

bash audm/scripts/audflash_next.sh smoke full

python -m audm verify "$AUD_NEXT_ROOT/full/smoke-test" --device cuda:0

bash audm/scripts/audflash_next.sh full

bash audm/scripts/audflash_next.sh verify full
```

The full run loads **real adaptation output**, not checkpoint-3700 again and not
probe/smoke output. It starts a fresh optimizer and runs three coverage epochs.
On a later interruption of this *new* run, `full --resume auto` is available;
that is separate from the weights-only initialization described above.

## Deployment export after full training

```bash
bash audm/scripts/audflash_next.sh inference-export
bash audm/scripts/audflash_next.sh verify inference
```

Serve `$AUD_NEXT_ROOT/inference`. It contains no `aud_ctc_head` tensors and uses
the original decoder tokenizer. Export verifies exact retained-tensor equality
and strict offline reload; the verify command exercises forward/generation on
CUDA. `inference/provenance.json` records the source and removed parameter count.
Use `$AUD_NEXT_ROOT/full` for further CTC fine-tuning; inference export does not
include optimizer state. Export refuses to overwrite existing destinations.

## 6. Console and troubleshooting logs

The launchers print one timestamped rank-zero progress stream: loss, learning
rate, gradient norm, update time, estimated remaining training time (excluding
evaluation/save pauses), WER/CER, checkpoint paths and errors. The first three
logged warning/error messages on rank zero are summarized; every full message
remains in the diagnostics. Library progress bars, model/optimizer dumps,
example transcripts, repeated dataset notices and per-rank memory JSON stay in
files. W&B metrics, Trainer state and performance reports are unchanged.

Each invocation creates a **new** timestamped directory under
`$AUD_NEXT_ROOT/logs` (override with `AUD_LOG_DIR`). It prints this path at startup:

- `console.log`: the concise console transcript.
- `rank-000.log`, `rank-001.log`, etc.: full training stdout/stderr, Python/native
  messages, warnings, tracebacks and DataLoader subprocess output for that rank.
- `launcher.log`: torchrun startup/shutdown and any output emitted before worker
  capture starts. Export/verification details are in their rank-000.log files.
- `command.json`: the exact command used; environment secrets are not dumped.

No extra `tee` is needed. Failures retain a nonzero exit status and a short error
on the console; Ctrl-C is forwarded to this launch's process group. Logging files
are outside model/checkpoint directories and do not change resume semantics.
Direct `python -m audm ...` calls keep their existing output; use the supplied
shell launchers for the clean console. Preparation/health commands still print
their useful report directly.

To inspect the most recent launch from another shell (a later verification
command updates this pointer, so keep the printed path for a specific run):

```bash
AUD_LOG_RUN="$(cat "$AUD_NEXT_ROOT/logs/latest-node0.txt")"
tail -n 100 -f "$AUD_LOG_RUN/rank-000.log"
# Inspect launcher.log if failure occurred before workers started.
```

For multi-node AudPro/AudTurbo, each node writes a separate timestamped folder
under its run root's `logs`, with `latest-node<N>.txt` pointing to that node's
latest invocation. Only global rank zero prints training metrics.

## 7. Observe performance while it runs

From another shell with the same paths/environment:

```bash
bash audm/scripts/audflash_next.sh health
python -m audm.health --run "$AUD_NEXT_ROOT/full" --max-idle-seconds 900

tail -n 5 "$AUD_NEXT_ROOT/full/performance.jsonl"
cat "$AUD_NEXT_ROOT/full"/cuda-memory-rank-*.json
```

Compare steady-state updates after warm-up, excluding evaluation/checkpoint pauses.
Progress heartbeats can become old during those pauses. Fix thermal slowdown
before judging software speed. A failed worker or nonfinite loss/gradient stops
the run; do not restart from a corrupted checkpoint.

## Optional benchmarks, not enabled defaults

**Liger:** install and validate on the target H100 environment before selecting it:

```bash
python -m pip install -e './audm[fused]'
bash audm/scripts/audflash_next.sh fused-check
```

If the loss/gradient check passes, change `efficiency.loss_backend` to `liger` in
a *separate probe recipe* and compare time, memory and numerical checks. This
checks tied input/output embeddings, BF16/FP32 and label smoothing. Local CPU
tests cannot validate this CUDA kernel.

**Selective checkpointing:** `encoder_checkpoint_every` and
`decoder_checkpoint_every` accept `1` (all layers), `2` (every second layer),
or `0` (off). Benchmark those only after the default batch-40 full probe passes.
They trade additional memory for less recomputation. The production default is 1.

CTC is an additional training cost and an accuracy hypothesis, not a guaranteed
speedup. Distillation, alternative inference decoders, FP8 and compilation are
not enabled. No architectural compression is included in this run.

## Validation for this implementation

Local validation passed 298 tests, including two-rank CPU/Gloo training with
persistent workers, CTC and non-speech integration, distributed WER, save/resume,
exact pretrained-weight preservation, native-rate frame lengths, accumulation
normalization, numerical guards, decoder-tokenizer identity, inference-export
logit/generation equivalence and the direct launcher. Ten dataset-preparation
cases require optional PyArrow and were skipped in the local test environment.
Lint, whitespace checks and shell syntax checks passed. H100/ZeRO-2 memory,
throughput and optional Liger validation must be measured with the supplied
cluster probes; no speedup or ASR accuracy gain is asserted from CPU tests.
