# Robustness and non-speech audit — 22 October 2026

The audit found and fixed concrete defects. Passing local tests is evidence for
the implementation paths below; it is not a guarantee of pretrained-model quality
or full-size H100 memory use.

| Area | Finding and correction | Validation |
| --- | --- | --- |
| Effect configuration | Typos, non-finite values, fractional effect budgets and empty codec lists could survive initialization. Settings now fail early. | Invalid types, unknown keys, NaN/Inf, reversed ranges, disabled-effect settings, unsupported codecs. |
| Negative labels | Null/numeric fields could be interpreted as empty text. Both transcript aliases now require explicit empty strings. | Missing fields, punctuation, conflicting aliases, unknown categories, duplicate manifests. |
| Negative augmentation | Untyped empty targets could receive real-noise clips containing speech. All negative examples now skip real-noise mixing; music also skips white noise. | Typed/untyped collation across AudPro, AudTurbo and AudFlash; silence features/masks preserved. |
| Code-switching | Replacement occurred after mixing and auditing. It now happens first, and empty targets bypass mixing. | Mixer is never called for negative rows; original speech behavior remains covered. |
| Asset checks | Metadata-only checks missed invalid decoded durations, silent RIRs and codec runtime failures. `preflight --scan` now decodes eligible assets and exercises codecs. | Misleading duration metadata, all-zero RIR, full convolution tails, maximum-duration rejection, actual codec round trips. |
| Asset paths | Relative paths depended on the launch directory and unresolved variables gave obscure file errors. Paths now resolve beside the AUD recipe. | Recipe-relative paths and missing environment variables. |
| FFmpeg | Encoder-name substring checks could accept a different encoder. Checks now use exact names; codec processes have a one-thread setting and timeouts. | AAC versus libfdk_aac mismatch, timeout propagation, Opus/AAC/MP3/G.711 round trips. |
| SpecAugment | Calling the policy with SpecAugment disabled could pop an empty queue. Disabled mode now returns directly; policy decisions are queued only after successful waveform processing. | Clean branch, failed waveform effects, missing policy decisions, repeated mixed batches. |
| Output parsing | Unexpected content after EOS could be treated as a valid negative. It is now preserved in diagnostics and marks the output invalid; padding remains valid. | Post-EOS words, padding, absent/malformed category/language, hallucinated negative content. |
| CPU distributed resume | Accelerate supplied `cpu:0`; PyTorch optimizer loading rejected it. CPU device state is normalized to `cpu`. | Two Gloo ranks, both decoder/full stages, persistent workers, gradient accumulation, WER callbacks, checkpoint save and optimizer resume. CUDA placement is unchanged. |
| Run provenance | Changed negative-augmentation semantics need to be distinguishable from older experiments. | Robustness receipt version 2; resume contract detects earlier policy/changed manifests. |

Validation completed locally:

- 101 model, tokenizer, data, augmentation, training and edge-case tests.
- 9 launcher regression tests, including direct launches inside inherited Slurm environments.
- 2 real CPU/Gloo distributed training/resume tests, with one persistent worker per rank.
- Ruff checks and shell syntax checks for every cluster command block.
- No changes to existing production YAMLs: microbatch, accumulation, workers,
  epochs, evaluation/checkpoint intervals and code-switch defaults remain intact.

The distributed test initially exposed a local `--standalone` rendezvous timeout;
the test now uses the same explicit master address/port approach as the existing
cluster launcher. It then exposed the CPU optimizer-resume issue above, which was
fixed before the successful rerun.

Reproduce the tests in an environment with development dependencies and FFmpeg:

```bash
python -m pytest -c audm/pyproject.toml audm/tests -q
RUN_AUD_DDP_TESTS=1 python -m pytest -c audm/pyproject.toml \
  audm/tests/test_robustness_distributed.py -q
ruff check audm
```

The first command intentionally skips the two opt-in distributed tests; the
second runs them. Without FFmpeg, codec tests skip, so install/check FFmpeg when
using these results as a release gate. Local runs used Transformers 5.14.1 and
Torch 2.13 on CPU; synthetic short audio and tiny models keep the tests fast.

Remaining checks on the cluster:

- Real asset readability and decoded duration, including full corpus category counts.
- Listening review and reliable speech-free labels; code cannot establish those
  from a JSON label. Keep train/eval source recordings disjoint.
- Eight-GPU NCCL and DeepSpeed ZeRO-2 smoke, including full-stage backpropagation
  and a restart from a complete production checkpoint.
- Representative long-audio batches for GPU memory; short synthetic tests cannot
  establish that microbatch 20 fits every full-size batch.
- End-to-end data-loader wait time/throughput at four workers per GPU.
- Multilingual WER/CER regression and held-out category/missed-speech/false-speech
  metrics. No accuracy improvement is claimed before measuring them.

Manifests are hashed; referenced waveform bytes are not. Keep them immutable.
Persistent worker RNG/prefetch state is not checkpointed, so optimizer resume is
not a promise of bit-identical augmentations. Earlier robustness receipts cannot
resume under the new policy; start a weights-only run with a new output instead.

Follow [ROBUST_CLUSTER.md](ROBUST_CLUSTER.md) for the full cluster sequence.
