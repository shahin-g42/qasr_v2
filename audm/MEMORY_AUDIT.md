# AudFlash memory and training audit

The single-node default is now **10 × 8 × 8 = 640** sampled slots per update,
in the same adapter, full, audio-type decoder and robust-full YAMLs. Workers
remain four per GPU, transcripts remain limited to 512 tokens, full training
remains three epochs and full evaluation/saving remains every 3,700 updates.
The model architecture, Gemma audio stride of one and speech dataset paths
are unchanged. No alternative production config is required.

## OOM diagnosis

The reported failure had 65.30 GiB allocated, only 197.90 MiB reserved but unused,
and needed 11.65 GiB with 11.29 GiB free. That indicates a live-memory capacity
problem; allocator fragmentation is not the main suspect. The supplied traceback
contains only autograd internals, so the exact failing operator is not proven.

The audit found a concrete source of large allocations: training projected all
decoder positions to the approximately 152k vocabulary, including masked audio
prompts and padding, then upcast logits to FP32. For example,
`20 × 1030 × 151936 × 4 / 2**30` is about **11.66 GiB for one tensor**.
Softmax/backward buffers and the rest of the model add to that. Encoder/decoder
layer checkpointing does not remove these output-head tensors.

A second defect was confirmed in the layer audit: Transformers 5.14.1's
`Gemma4AudioLayer` is a plain module without a checkpoint hook. The outer
checkpoint setting enabled the Qwen decoder, but not the Gemma encoder. The
GQASR implementation had a custom audio-layer checkpoint wrapper; that behavior
was missing from the AUD port. AUD now adds the standard Transformers checkpoint
layer behavior around the unchanged native Gemma forward. Every trainable audio
block recomputes during backward. State-dict names and weights are unchanged;
existing Gemma AUD artifacts do not need re-exporting.

Training and loss-only evaluation now project only supervised next-token
positions, in chunks of at most 128. The projection and cross-entropy are
checkpointed together: simply summing chunk losses would retain every softmax
activation until backward. One FP32 logit chunk at vocabulary 151936 is about
74 MiB; that is not the entire training-memory requirement. All audio context
still passes through the decoder. Label smoothing and token-count normalization
are preserved. Public forward logits and generation retain their native behavior.
Recomputation costs compute; throughput must be measured against the avoided
audio-prompt/padding projections.

| Batch setting | Samples/update | Consequence |
|---|---:|---|
| 20 × 4 × 8 | 640 | Previous setting; larger activation peak |
| 10 × 4 × 8 | 320 | Smaller update batch |
| **10 × 8 × 8** | **640** | Default; preserves the earlier update size with smaller microbatches |
| 10 × 1 × 8 | 80 | Violates the current per-rank 20-example language quota contract |

Accumulation normally runs separate forward/backward passes, keeping gradients
rather than eight simultaneous computation graphs. Microbatch is the direct
activation-memory control. The new default preserves 640 sampled slots per update,
the sample coverage of the 3,700-update evaluation/save interval, and the
three-epoch update budget. Learning rates are unchanged.

See PyTorch's [allocator guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)
and [checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html).
`PYTORCH_ALLOC_CONF=expandable_segments:True` can help variable allocations on
supported builds but does not replace reducing live tensor memory.

## Other findings

| Area | Finding and action |
|---|---|
| Bad speech samples | A replacement at a manifest boundary could enter the next language. Replacements now stay within the original source. An entirely unusable source fails instead of borrowing another language. |
| Real background noise | The inherited loader decoded whole files and always took their beginning. The opt-in policy now reads bounded random windows, tiles short clips and uses polyphase resampling. Decode errors propagate. |
| Checkpoint coverage | Fixed missing Gemma audio-block checkpointing. Tests observe one forward plus one backward recomputation in every encoder/decoder block during full training, under both checkpoint modes. Frozen encoders remain in eval mode; adapter gradients pass through a checkpointed frozen decoder. |
| Layer wiring | Export/reload tests compare all encoder and bridge tensors with their sources; decoder weight sharing, valid audio-token counts, masks, stride and causal targets are covered. No duplicate text decoder or imported Gemma text model was found. |
| Smoke coverage | Ordinary smoke can sample short clips and uses fewer workers. The new longest-shape probe exercises the normal trainer/optimizer at maximum duration and near-maximum target length for two updates. |
| CUDA diagnostics | Each rank writes peak allocated/reserved memory to `cuda-memory-rank-N.json`, including the caught-OOM path. Diagnostic writing requires no collectives after failure. |
| Non-speech data | Only reviewed candidates become empty targets. Digital silence is synthetic; it does not establish accuracy on quiet real-world recordings. Include held-out speech when measuring speech suppression. |
| Category training | New category tokens require decoder warm-up before full training. Typed checkpoints reject adapter-only training. The 3% negative pool replaces slots rather than expanding the batch. |
| Chinese metrics | WER uses whitespace tokens; CER is more informative for Chinese. CER far above 100% still indicates severe insertions/repetition and must not be dismissed as word segmentation. |
| Augmentation cost | Wave effects stay within duration bounds. RIR/codec effects may be skipped near 30 seconds. CPU augmentation cannot explain the old 11.65 GiB GPU allocation, but FFmpeg, resampling and I/O can bottleneck workers. |
| Missing features | Dedicated clipping/compression, packet dropouts and independent pitch shifting are not implemented. RIR downloading and real-room silence curation remain separate asset work. Do not advertise these as enabled. |
| Resume compatibility | Random-window noise advances robustness receipts to version 3. Exact resume rejects changed recipes/assets/policies. |

SpecAugment retains the inherited stage-02 settings. The opt-in policy retains
its clean branch and maximum of two waveform effects. Measure CPU wait time
before increasing augmentation probabilities. Multilingual WER/CER and category
ablations remain necessary; software tests do not establish an accuracy gain.

## Cluster longest-shape rehearsal

Use an updated isolated checkout while other jobs run. The example tests full
training from a completed untyped AudFlash adapter checkpoint. It needs no new
model-library installations. Run setup once, then reuse the checkout on later logins.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
git fetch origin
git worktree add --detach ../qasr_v2-aud-memory origin/main
cd ../qasr_v2-aud-memory
eval "$(conda shell.bash hook)"
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/audm
export PYTHONPATH="$PWD/audm/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ID=cs_v1_ml_v2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/$USER/audflash-memory-$(hostname)"
mkdir -p "$TRITON_CACHE_DIR"

export AUD_PROBE_MODEL=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/output/audflash-gemma4-v1/adapter
export AUD_PROBE_CONFIG="$PWD/audm/configs/audflash_full_1node.yaml"
export AUD_PROBE_ROOT="$PWD/output/audflash-memory-full-v1"

python -m audm.memory_probe --config "$AUD_PROBE_CONFIG" \
  --model "$AUD_PROBE_MODEL" --output "$AUD_PROBE_ROOT" --steps 2

export AUD_USE_SLURM=0 AUD_NNODES=1 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK=0
export AUD_MASTER_ADDR=127.0.0.1 AUD_MASTER_PORT=29615
set -o pipefail
AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_PROBE_ROOT/recipe.yaml" \
  --model "$AUD_PROBE_MODEL" --output "$AUD_PROBE_ROOT/run" \
  2>&1 | tee "$AUD_PROBE_ROOT/train.log"

python - <<'PY'
import json, os
from pathlib import Path
paths = sorted((Path(os.environ['AUD_PROBE_ROOT']) / 'run').glob('cuda-memory-rank-*.json'))
assert len(paths) == 8, f'Expected eight reports, found {len(paths)}'
for path in paths:
    result = json.loads(path.read_text())
    print(result)
    assert result['status'] == 'completed', path
PY
```

Each probe output must be new. To test typed decoder warm-up, select
`audflash_audio_types_decoder.yaml`, your typed export as `AUD_PROBE_MODEL`, and
a separate probe root, then repeat preparation/launch. Test full training again
with `audflash_robust_full.yaml` and the completed decoder checkpoint. Passing
decoder warm-up does not establish that full training fits.

The probe preserves stage, batch, accumulation, BF16, attention backend,
checkpointing, DeepSpeed, learning rates and workers. It uses synthetic
audio/transcripts, disables augmentation, category replacement and evaluation,
and runs two updates to include initialized optimizer state. It needs no RIR or
negative corpus and does not test their throughput. **Discard its trained
weights.** Do not pass `--smoke`, which would change the probe settings.

Check every rank, then run real-data smoke and throughput checks. H100/CUDA/
DeepSpeed behavior cannot be certified by CPU tests. If it fails, retain the
full operator traceback, `probe.json`, logs, installed package versions and
per-rank memory reports.

Startup also logs `AUD architecture` with parameter counts, parameter dtypes,
layer counts, checkpoint coverage, weight tying, decoder attention and cache
settings. For full Gemma training, encoder and decoder checkpoint-enabled counts
must equal their layer counts. `verify` prints the same structure before training,
where checkpoint-enabled counts of zero are expected. Gemma audio attention
uses its native local relative-position computation with FP32 intermediates;
`sdpa` selects the Qwen decoder backend, not a replacement for that audio math.

## Continuing an old run

Changing microbatch 20 to 10 changes the resume contract. Existing checkpoints
remain usable as weights, but `--resume auto` rejects the changed recipe. Keep
the original run and start a weights-only continuation with a new output. The
optimizer, scheduler and nominal three-epoch budget restart; the old epoch
counter is not continued. Set the completed checkpoint path before running:

```bash
export AUD_CONTINUE_FROM=/REPLACE/WITH/completed/checkpoint-3700
export AUD_CONTINUE_OUTPUT="$PWD/output/audflash-full-mb10-v1"

python -m audm preflight --config audm/configs/audflash_full_1node.yaml \
  --model "$AUD_CONTINUE_FROM" --output "$AUD_CONTINUE_OUTPUT" --scan

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config audm/configs/audflash_full_1node.yaml \
  --model "$AUD_CONTINUE_FROM" --output "$AUD_CONTINUE_OUTPUT" --smoke

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config audm/configs/audflash_full_1node.yaml \
  --model "$AUD_CONTINUE_FROM" --output "$AUD_CONTINUE_OUTPUT"
```

For the new typed robustness experiment use [ROBUST_CLUSTER.md](ROBUST_CLUSTER.md),
preserving its typed export → decoder warm-up → full-training sequence.

## Local validation

The release passed 20 loss/gradient tests, 17 training/configuration tests, the
synthetic probe through two real CPU optimizer updates with four loader workers,
and 11 launcher/distributed checks (including two-rank Gloo decoder/full training,
evaluation, save and optimizer resume). The broader model, augmentation, data
preparation and audio-type regression suite also passed. Ruff and runbook Bash
syntax checks passed. Tests use tiny models; no full-size H100/DeepSpeed run has
been performed locally.

The layer follow-up adds seven regression cases for actual per-block backward
recomputation, native-equivalent gradients, adapter/decoder/full freeze behavior,
both reentrant and non-reentrant modes, and reduced encoder saved-tensor volume.
The convolutional audio frontend remains native and is not checkpointed by this
block wrapper. It still contributes activation memory. No GPU memory-fit claim
is made without the eight-H100 rehearsal.

After the audio-layer fix, 69 focused model/loss/training/audio-type tests and
three process-based checks passed: two-rank decoder/full training with
checkpoint save/resume, plus the four-worker probe. Ruff and runbook syntax
checks passed again.
