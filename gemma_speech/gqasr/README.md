# GQASR

GQASR is the ASR package inside the `gemma_speech` project. Install the parent project once; `gqasr`, `gqtts` and `gemma-speech` use the same PyTorch 2.11.0 / Transformers 5.19.0 environment. TTS remains a separate model.

The cluster creates `models/gqasr-base/` by loading the pinned pretrained audio encoder and ASR text decoder, then exporting GQASR configuration, tensor names, tokenizer, frontend and provenance. Its public model class is `GQASRForSpeechRecognition`; state keys begin with `gqasr_acoustic`, `gqasr_decoder`, `gqasr_bridge` or `gqasr_prompt`. Local subclasses retain the upstream computations. This is a derived initialization, not a new architecture created by renaming. Upstream Apache notices and attribution remain in the artifact; tokenizer vocabulary is preserved exactly.

The base has approximately 901.9M parameters. The acoustic encoder and text decoder are pretrained; the 0.99M alignment bridge is random and must be trained. Exporting the base does not establish recognition quality. All model downloads, exports, verification and training are performed by the cluster scripts in [the cluster runbook](../cluster/README.md). No generated model weights are included in the code.

## Training recipe

`configs/01_bridge_12node.yaml` trains the bridge for 1,000 optimizer steps with the encoder and decoder frozen. `configs/02_full_12node.yaml` starts from the bridge stage's `final/` artifact and adapts the encoder, decoder and embeddings for 62,891 optimizer steps. Both expect 96 ranks.

The configs use the **22 training and 6 evaluation manifests** from QASR's `configs/v8.1/02_full_12node.yaml`. Their paths remain cluster paths. The source SHA and detailed mapping are in `configs/source_mapping.yaml`.

| Setting | GQASR |
|---|---|
| Machines / GPUs | 12 × 8 |
| Arabic / English / Hindi / Malayalam / Chinese | 50% / 30% / 10% / 5% / 5% |
| Microbatch / accumulation per GPU | 2 / 10 |
| Effective global batch | 1,920 utterances per optimizer step |
| Data workers | 4 per GPU, 32 per machine |
| Precision | BF16 with FP32 parameters, TF32 enabled |
| Distributed strategy | Native PyTorch DDP |
| Audio | Mono 16 kHz, duration 0.1–35 seconds |
| Targets | At most 512 tokens; overlong targets filtered, never truncated |
| Full-stage learning rates | Bridge/audio 2e-5; decoder 1e-5; embeddings 5e-6 |

Language quotas are exact across the ten accumulated microbatches on every rank: 10 Arabic, 6 English, 2 Hindi, 1 Malayalam, 1 Chinese. Sampling uses replacement with deterministic rank/update seeds, so resume does not require a corpus-sized permutation. This is not an epoch traversal. Microbatch accumulation preserves the source example count, but changes loss weighting and stochastic execution; it is not a numerically identical run.

QASR's online code-switching, SpecAugment, speed perturbation, noise and codec augmentation have **not been ported**. These are explicitly disabled. Logging is JSONL instead of W&B, and DDP replaces ZeRO-2. The recipe therefore uses the source data and training budget without claiming to reproduce every source behavior. H100 memory, throughput and scaling need cluster validation.

## Data preparation

`gqasr preflight --config CONFIG --scan --verify-audio` builds one shared SQLite index before distributed training. It streams JSONL manifests rather than copying millions of rows into each rank's Python memory. Accepted audio fields are `audio_filepath`, `wav_path`, or `audio`; text is the first usable `text` or `transcript`. Relative paths use `audio_root` or the manifest directory. `/vast/...` maps to `/lustrefs/taiga/vast40/...`.

The scan uses all CPUs available to the CLI process by default. Worker processes validate audio and tokenize targets; a single ordered writer preserves deterministic sample IDs and duplicate/leakage checks. The scan reads audio headers, filters duration/target limits, rejects conflicting duplicate transcripts and train/eval path or recording-ID leakage, and checks every language has usable train and evaluation data. Identical duplicate paths within a split are counted and skipped. Blank or punctuation-only transcripts are skipped. Training preserves Arabic diacritics. Corpus WER/CER include strict and Arabic-normalized views.

Treat manifests, tokenizer and audio as immutable after scanning. The index records source hashes and tokenizer identity. Rebuilding requires a new `index_path` or explicitly removing the old index yourself; the tool does not overwrite one. Missing data and malformed rows fail with a source filename and line number.

## Artifacts and resume

`models/gqasr-base/` contains config, safetensors, tokenizer, frontend, checksums and attribution. Training needs this local artifact, not upstream model directories. Intermediate `checkpoint-NNNNNNNN/` and `final/` exports also include optimizer state, per-rank RNG state and the training config. Checkpoints use tensors/configs rather than pickled model classes.

For resume, copy the same stage config alongside the original, change **only `output_dir`** to a fresh directory, set `GQASR_CONFIG` to that file and pass `--resume /shared/path/checkpoint-NNNNNNNN`. Workers may also change; the model, batch, world-size, step budget and indexed dataset must match. Starting the full stage from bridge `final/` resets the optimizer intentionally.

The bridge stage always ends at `gqasr/outputs/bridge/final/` under the default config. If you change its output location, update the full-stage `base_model` accordingly.

## Validation scope

The package includes tests for tensor round trips, acoustic/decoder gradient checkpointing, exact multilingual sampling, data leakage, bridge-to-full training, and exact CPU checkpoint resume. Run them on the cluster with the `test` stage. `verify` additionally reloads the actual exported base offline, checks artifact hashes and GQASR names, runs a synthetic forward/backward pass and generates a few tokens. These are integration checks, not a speech-quality benchmark. Evaluate real Arabic dialects and all required languages before accepting the trained model.

## Parallel scan and progress

Stop an older serial scan with Ctrl+C and wait for it to exit. Restart on one allocated 96-core node:

```bash
bash gemma_speech/cluster/gqasr_stage.sh scan \
  --scan-workers 96 --scan-chunk-size 128 --progress-seconds 5 --scan-temp-dir /tmp
```

`--scan-workers 0` (default) detects available CPUs, respecting CPU affinity and `SLURM_CPUS_PER_TASK`. Workers are lightweight spawned processes with independent native tokenizers and single-threaded numerical libraries; they load no neural models. Work queues are bounded to two batches per worker. Results are indexed in source order regardless of worker completion order.

The database is built in node-local scratch (`TMPDIR` by default; explicitly use `/tmp` or your allocated local SSD directory). Ensure that directory has enough space for the completed corpus index. This avoids per-row SQLite writes over Lustre. Only the completed index is copied to a temporary file beside the destination and atomically published without overwriting an existing index.

Progress is emitted every five seconds, including while waiting for a worker batch: worker count, active manifest, checked/kept/skipped rows, rows/sec, elapsed time, processed manifest-byte percentage and an approximate ETA. The percentage tracks manifest bytes, not audio bytes; ETA will vary with file sizes and storage latency. Publishing the completed database also reports copy progress.

A `.scan.lock` file prevents concurrent new scans against the same index. Ctrl+C terminates workers and removes the current temporary files and lock. After an uncatchable kill or node crash, remove a stale lock only after confirming the recorded host/PID is no longer running. A failed or interrupted run never publishes an incomplete index. An existing completed index is preserved; select a fresh `index_path` to intentionally rebuild it.
