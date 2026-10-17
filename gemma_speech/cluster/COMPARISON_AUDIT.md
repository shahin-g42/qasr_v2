# Chinese regression isolation

Run on a spare GPU machine from the repository root. This downloads the official
0.6B and 1.7B Transformers-native snapshots on the cluster, then runs offline:

```bash
git pull origin main
GPU_IDS=0,1,2,3,4,5,6,7 bash gemma_speech/cluster/gqasr_comparison.sh
```

Defaults: exact 100 Chinese evaluation samples from step 7294, plus 250 Han-only
and 250 Han+Latin training samples, deterministically sampled once for all jobs.
`--per-group 500` increases this to 1,100 examples. Script strata describe written
labels, not verified spoken languages. This is a diagnostic sample, not a
population-weighted benchmark. `GQASR_RUN_DIR`, `EVAL_STEP`, `--output`, and a smaller
`GPU_IDS` list are supported. Each experiment is split into up to 32 sample
shards (GPU count × workers per GPU). A shared queue runs four workers on each
GPU by default and starts the next shard as soon as a worker finishes.
Set `WORKERS_PER_GPU=1` or `--workers-per-gpu 1` to reduce concurrency; the default
is four for both this suite and `gqasr_recovery_comparison.sh`.
Each worker loads its own model and CUDA context. Higher concurrency increases
memory demand and is not a guarantee of linear speedup. Shell defaults use two
OMP/MKL threads per worker to limit CPU oversubscription with 32 processes. Estimated longer shards run
first to reduce idle GPUs at the end. Model reloads occur between shards.

Shards preserve the original batch-four boundaries and global shifted-audio
pairing. The merger rejects missing/duplicate IDs or changed references, restores
original sample order, and computes metrics from pooled predictions, never an
average of shard percentages. Console progress reports active GPUs, active workers, capacity, completed
prediction rows and overall percentage. Detailed worker logs and settings live
under `shards/` (logs at the run root); merged predictions remain per experiment.

Every experiment uses automatic and Chinese-prefilled greedy decoding, 768-token
limit, identical audio/reference IDs, and no inference augmentation:

- Bridge final, latest full checkpoint, and step 1042 if still retained.
- Full 2x2 bridge/decoder swap matrix (frozen encoder equality checked first).
- Latest model with shifted audio, retaining original reference labels, as a negative control.
- Latest model at batch four and FP32, against batch-one BF16.
- Original Qwen3-ASR 0.6B and 1.7B with their own audio encoders/processors.

GQASR jobs also score reference header and transcript negative log likelihood
without label smoothing. References are used only for this separate diagnostic
forward pass, never for generation. Batch four exercises actual left padding and
per-example EOS handling. Swapped components exist only in worker memory.

Before inference, the suite checks finite/changed weights, frozen encoder equality,
unchanged tokenizer/frontend files, decoded audio hashes, durations, target lengths,
and target round trips. It attempts 100 deterministic code switches with p=1 to
inspect assembled targets, joining and assigned language; this intentionally does
not estimate production augmentation frequency. Source pool audio integrity checks
remain active. No data cleaning or training configuration changes are performed.

Results are under `gemma_speech/gqasr/verification/comparison-TIMESTAMP/`:

- `REPORT.md`: CER comparison table by evaluation/training stratum.
- `comparison.json`: strict and punctuation-insensitive metrics, no-Han rates,
  token-limit hits, paired CER deltas and utterance-bootstrap confidence intervals.
- `weight-drift.json`, `audio-target-audit.jsonl`, `data-summary.json`,
  `code-switch-audit.jsonl`: integrity and target evidence.
- Each job: raw/generated transcripts, automatic language, reference NLL,
  `teacher-summary.json` where applicable, settings and logs.
- `samples.jsonl`, `selection.json`, `checkpoint-sources.json`, `jobs.json`,
  `runtime.json`: reproducibility details.

Checkpoint snapshots are hard-linked when possible, copied across filesystems,
excluding optimizer state. They survive training retention cleanup. Allow roughly
4 GB per retained model if copying is required; hard links also retain that disk
space after the training source is deleted. Remove the audit directory only after
reviewing/exporting its results. A failed worker stops the suite with its log path;
partial output is retained and is not reported as a complete comparison.

Interpretation: forced-language recovery implicates language selection; NLL and
shuffled-audio controls measure reference prediction and audio dependence; the
factorial swap tests isolate component interactions. A bad swap alone is not proof
of forgetting because bridge and decoder coadapt. Good original baselines argue
against widespread label corruption, but do not certify every transcript. These
tests cannot prove that SpecAugment, code switching, or sampling caused a learned
regression; that requires controlled retraining with fixed data/initialization.
