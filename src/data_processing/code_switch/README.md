# Online code switching with a staged source index

The stages prepare short, verified source clips. Training joins their waveforms
in memory and then applies the existing speed, noise, codec and SpecAugment
operations. There is no 39-million-file generated audio corpus. Only `peek
--export-audio` writes a small listening preview.

This is opt-in. The v8.0 configs and launcher are unchanged. The separate v8.1
config initializes from the same `models/bck/full/v1` backup. To continue a later
trained model, change `model_name_or_path` in the NEW v8.1 config before starting;
that is weight initialization, not an optimizer-state resume of a v8.0 run.

## Data contract

Each original training example gets an independent Bernoulli trigger with
`code_switch_probability: 0.4`. A successful mix keeps the original waveform and
transcript as its first segment, then appends indexed clips before the ordinary
acoustic augmentations. No extra dataset entries or optimizer steps are added.
A trigger may fall back unchanged if the anchor has no room for another whole
clip within 35 seconds / 512 tokens. Logs and the diagnostic distinguish
`triggered` from `applied`; 40% is an expected trigger rate, not an exact batch
quota or guaranteed successful-mix rate. Evaluation stays clean.

Recipes vary with worker RNG rather than being fixed per dataset position.
Reproducibility assumes the same seeds, worker/rank layout and loader ordering;
an interrupted optimizer resume does not promise exact worker-RNG replay.
These joins do not reproduce natural within-sentence code switching or guarantee
the same speaker. Keep a human-reviewed natural code-switch evaluation panel.

Seven pairs are supported: ar-en, hi-en, ml-en, zh-en, ar-hi, ar-ml, ar-zh.
Conditional partner-selection weights are 40/20/5/5/20/5/5 respectively. Weights are renormalized over pairs containing the anchor language; global pair
shares therefore depend on the training language quotas. Default segment counts are 2/2/2/3/3/4;
optional final segments are removed WHOLE if needed to fit. A successful mix always contains at least two complete segments. Text, including harakat, is joined without linguistic rewriting.

Indexed partners are 0.5–10 seconds; original training anchors retain their usual limits.
Segments are separated by 100 ms gaps. Both decoded mixed audio
and post-augmentation audio must be at most 35 seconds; joined targets must fit
512 transcript tokens. An overlong speed/codec result is skipped, never cropped.
No clean eval collator or eval dataset uses this augmentation.

`language_policy: first` preserves the existing language quotas. `major` assigns
the language with the greatest total decoded speech duration (gaps excluded),
breaking ties by first occurrence. Under `major`, sampler quotas describe the
ANCHOR language; they do not guarantee the distribution of final prompt labels.

## Prepare the index

Use the qasr3 environment, identical code/config on all nodes, and a shared
Lustre output directory. No GPU, ASR server or LLM server is used by these stages.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3
export RUN_ID=cs_v1
```

Before a multi-node production run, verify that this mount implements
**cross-node POSIX flock**, not local-only flock. On machine A:

```bash
RUN_ID=lock_probe bash scripts/corpus/run_code_switch.sh lock-test --hold 30
```

While it holds the lock, run the same command without `--hold` on machine B. B
must report `Lock busy` and exit nonzero; after A exits, B must acquire it. Do not
use multi-node indexing on a mount that fails this test. Locks have persistent
inodes and are never stolen based on elapsed heartbeat time.

For a small preparation pilot, use a separate ID:

```bash
RUN_ID=cs_pilot bash scripts/corpus/run_code_switch.sh plan --samples 1400 --limit 200
RUN_ID=cs_pilot FOREGROUND=1 PROCS=4 bash scripts/corpus/run_code_switch.sh run
RUN_ID=cs_pilot bash scripts/corpus/run_code_switch.sh verify --audio
RUN_ID=cs_pilot bash scripts/corpus/run_code_switch.sh assemble
RUN_ID=cs_pilot bash scripts/corpus/run_code_switch.sh peek -n 2 --export-audio
```

The pilot limits indexed training rows, but planning still fingerprints full
source manifests and excludes ALL evaluation rows. A pilot index is rejected by
production training; it can be used with a smoke config pointing at that index.

Production:

```bash
# Once, any node. Freeze the config, source fingerprints and tokenizer.
RUN_ID=cs_v1 JOBS=8 bash scripts/corpus/run_code_switch.sh plan

# On EACH participating node. Background by default; FOREGROUND=1 for srun.
RUN_ID=cs_v1 PROCS=8 bash scripts/corpus/run_code_switch.sh run

# Any node; reports indexing progress, not generated training audio count.
RUN_ID=cs_v1 bash scripts/corpus/run_code_switch.sh status

# This node only; add --all to request a graceful stop on all active nodes.
RUN_ID=cs_v1 bash scripts/corpus/run_code_switch.sh stop

# Resume: run the same run command. Completed chunks are retained.
RUN_ID=cs_v1 PROCS=8 bash scripts/corpus/run_code_switch.sh run

# After workers exit. Validate frozen manifests, committed hashes and counters.
# No source-audio opens: indexing already decoded/hashes each eligible clip.
RUN_ID=cs_v1 JOBS=16 bash scripts/corpus/run_code_switch.sh verify --metadata-only
RUN_ID=cs_v1 JOBS=5 bash scripts/corpus/run_code_switch.sh assemble
RUN_ID=cs_v1 bash scripts/corpus/run_code_switch.sh peek -n 2 --export-audio
```

`PYTHON`, `ROOT`, `CONFIG`, `PROCS`, `JOBS`, `RUN_ID` and `FOREGROUND` are shell
overrides. Start with modest workers: indexing and online decoding can saturate
shared storage before using all CPU cores. Background supervisor startup is
asynchronous; inspect the printed log for failures, including duplicate launches.
A stopped/offline worker's outstanding chunks become available when its kernel
releases the lock. Restarted workers truncate uncommitted tails before resuming.
No active worker can race `verify --reset` or `assemble`: those require the
exclusive lifecycle lock. Kernel/filesystem lock release after a machine failure
is required; the pipeline does not force-break locks.

`verify --reset` clears only bad/incomplete chunks before publication; it returns
a failing exit code because verification has not yet passed. Run workers and
verify again. Once `assembled/` exists, the index is immutable: repairs/replacement
require a new RUN_ID. Never edit pools/manifests under an active training run.

## Inventory and limits

`run` decodes eligible sources once, records exact resampled frame counts and
waveform hashes, and checks individual token budgets. Eval UUID basenames are
excluded conservatively across path aliases. Assembly deduplicates basenames
within each language and refuses conflicting identities across languages.

`assemble` reuses a current successful verification receipt, or verifies first if
no matching receipt exists, then builds five read-only SQLite pools in parallel,
with exactly one writer per temporary database. Training workers use immutable,
read-only connections. There is no multi-node SQLite writer or shared write ledger.
Assembly checks unique-combination capacity and writes `_reports/inventory.json`.

`preview_count` and `max_source_uses` only control the small deterministic
listening preview generated by `peek`. They do not control training frequency or
bound training reuse. Online partners are sampled with replacement; rare-language
clips can be reused often. Inspect per-pair reports and evaluate rare-language
quality before extending training. The index schema is version 2; rebuild any
unreleased version-1 pilot under a new RUN_ID.

The seven configured weights are starting choices, not measured optima. Header,
decode and token checks do not establish transcript correctness; listen to the
preview and keep a fixed, human-reviewed code-switch evaluation panel.

## Train

Default production index:
`code_switch_indexes/cs_v1/assembled/index.json`.

```bash
bash scripts/run_v81_phase2_node.sh 0 localhost --preflight --scan
NNODES=1 bash scripts/run_v81_phase2_node.sh 0 localhost --smoke-test

# Set your real node-0 hostname on every node. Assign unique ranks 0 through 9.
MASTER=inception-H100-hpc-051
RANK=0
bash scripts/run_v81_phase2_node.sh "$RANK" "$MASTER" --smoke-test
# After all ten smoke jobs finish successfully:
bash scripts/run_v81_phase2_node.sh "$RANK" "$MASTER"
```

Use 051 only if it is actually your rank-0 machine. The launcher uses your active shell environment. Set `ACTIVATE_CONDA=1` to
activate qasr3 automatically instead.
Output goes to `output/v81/full`; logs go to `output/v81/logs`. The config has
`max_steps: 75469`, retaining the original v8.0 weighted epoch budget because
code switching is now an augmentation of existing entries.

The original batch 20 per GPU is only a starting point. Mixed samples are longer,
so check H100 memory and throughput before the full launch. No GPU performance or
memory claim is established by local CPU tests. Resume v8.1 with its own complete
checkpoint and the same index; an index receipt prevents switching recipes in an
existing output directory.

Only audio composition and normal augmentation happen during training. Stage
`status` therefore monitors index preparation; per-worker training logs report
actual code-switch compositions. No 39-million-entry recipe JSONL is created.


## Cluster loader and augmentation diagnostic

After assembling the production index, activate qasr3 and run from the repository:

```bash
bash scripts/check_v81_dataloader.sh --batches 100 --workers 4
```

This reads the first 2,000 rows of each training manifest, then uses the real
JsonlSpeechDataset, resilient decoder, weighted sampler and training collator.
It runs 2,000 examples at the default batch size of 20. Prefix sampling is a fast
integration check, not an unbiased corpus-quality estimate. Use
`--records-per-manifest 0` for the full manifest index and production sampling
population (the initial scan is expensive). No model weights are loaded.

The report `dataloader_checks/rank-0.json` includes the requested 0.4 trigger rate,
observed trigger/applied rates, fallback reasons, pair counts, decoder substitutions,
and actual changed-output counts for speed, noise, channel simulation and SpecAugment.
Speed changes rejected by the final duration guard can still count as candidate
changes. SpecAugment gates once per batch; the other acoustic transforms gate per
sample. Changed-output rates need not equal trigger probabilities.

Independent forced-on/off probes require each enabled transform to produce a
non-noop result and preserve identity when disabled. The test rejects missing or
unreadable MUSAN data that would fall back to Gaussian noise. It checks finite
features, supervised labels, padded-frame integrity, the encoder audio-token
contract, and post-augmentation duration bounds. A five-sigma binomial check
rejects clearly wrong code-switch rates. This does not prove ASR quality.

For concurrent loading on one node (inside qasr3):

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  torchrun --standalone --nproc_per_node=8 scripts/check_v81_dataloader.py \
  --batches 100 --workers 2 --output dataloader_checks/node-048
```

Each rank writes its own report and must print PASS; there are no GPU operations
or cross-rank collectives in this diagnostic. Use a distinct output directory on
each machine. These bounded prefix pools are deliberately oversampled by the
weighted sampler; use the full scan option when assessing source diversity.
Follow with the existing v8.1 `--smoke-test` launcher for GPU forward/backward,
DDP, memory and optimizer validation before starting all ten machines.

Room impulse response (RIR) reverberation is the main missing augmentation worth
an ablation for distant/room-recorded audio. Start with a separate opt-in trial,
then compare clean, noisy and reverberant held-out WER/CER by language; do not
assume it improves every domain. Evidence: [Ko et al., ICASSP 2017](https://www.danielpovey.com/files/2017_icassp_reverberation.pdf).
The existing `codec_augment` simulates bandwidth, quantization and smoothing; it
does not encode/decode actual telephone codecs. Real codec round trips are a
separate candidate if deployment includes telephony. No new acoustic transforms
are enabled by this change. Existing time/frequency masking already covers the
main SpecAugment operations; [Park et al.](https://arxiv.org/abs/1904.08779) supports
this baseline, not a universal guarantee for this model or these languages.


## Verification and assembly cost

Default `verify` checks every eligible source audio header. `verify --audio`
fully decodes and hashes every eligible source again. Either may take hours on
Lustre: more workers can increase metadata-server contention rather than throughput.
`verify --metadata-only` still hashes every frozen source manifest and every
committed part/reject file, validates records/counters, and checks the frozen
tokenizer and eval exclusions. It does not check current audio availability or
contents. Use this mode after indexing when audio is unchanged; use the stronger
modes for an explicit audio audit. Training checks hashes of partners it decodes.

Verification hashes/parses each chunk file in one pass. Manifest hashes retain
the original whole-file SHA256 format, so one huge manifest is still handled by
one worker. Progress now reports files/chunks in completion order, with periodic
byte counts for large manifests; it no longer waits for the first submitted task
before showing other completions.

The success receipt captures size, mtime, ctime and inode for frozen manifests,
plan/tokenizer/exclusions and all part/reject/progress files. Assembly reuses it
only when those signatures still match under the lifecycle lock; changes force
reverification. A failed explicit verification invalidates the old receipt.
Audio auditing is point-in-time: the receipt does not track every source audio
file. Old receipts without signatures need one verification with the updated
code; the existing plan and completed indexing chunks remain compatible.

Assembly has at most five SQLite writers, one per language, regardless of JOBS.
Arabic/English can dominate elapsed time. Each writer reports progress every
roughly 30 seconds at chunk boundaries, followed by database hashing and
cross-language checks. Duplicate checks scan the smaller pool and probe the
larger pool's unique index. The published schema and training reader are unchanged.
Do not restart indexing or delete completed chunks to apply these improvements.


Startup logging includes the initial file-signature scan before manifest hashing.
That scan covers roughly three files per indexing chunk plus frozen inputs, uses
`min(JOBS, 32)` metadata threads, and reports completed signatures/elapsed time
every five seconds while waiting. The final signature recheck uses the same path.
The thread cap limits metadata pressure on Lustre; JOBS still controls the later
verification process pool. The shell wrapper prints before Python starts, helping
distinguish Python startup from plan loading and filesystem work.
