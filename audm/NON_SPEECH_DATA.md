# Download and prepare the non-speech corpus

The cluster preparation script downloads from
[quinnlue/FSD50K-16k](https://huggingface.co/datasets/quinnlue/FSD50K-16k), pinned to
`2a60d475f4e2f0db624a902881af2df79d656145`. This is a 16 kHz mono repack of
FSD50K, with embedded FLAC audio in Parquet, original recording IDs, event labels,
source splits and per-recording license metadata. The raw shards total about
6.85 GB. Reserve **25 GB** for raw files, the default candidate selection, the
final copied WAV files and temporary files. Higher candidate limits need more space.

No corpus was downloaded during local code validation. Run the following on
**one cluster node**, using a shared filesystem. It needs CPU, disk and network;
it does not need a GPU, `torchrun`, Slurm or `srun`. Downloading/preparing data
does not change an existing training job or its configuration.

## 1. Download to the shared data directory

Use the isolated checkout from [ROBUST_CLUSTER.md](ROBUST_CLUSTER.md), or another
checkout containing this script. Leave running jobs' code checkouts untouched.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2-aud-robust
eval "$(conda shell.bash hook)"
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/audm

export AUD_NON_SPEECH_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/aud-data/non-speech-v1
mkdir -p "$AUD_NON_SPEECH_ROOT"

# Only the preparation reader needs this optional dependency. No model-library upgrades.
python -m pip install --no-deps 'pyarrow>=17'
bash audm/scripts/non_speech.sh download --workers 4
```

The download location is:

```text
$AUD_NON_SPEECH_ROOT/
  raw/FSD50K-16k/2a60d475f4e2f0db624a902881af2df79d656145/
    data/*.parquet
    metadata/
    checksums.json
    LICENSE-DATASET.md
    aud_download.json
```

Hub download metadata is kept in that snapshot's `.cache/huggingface` directory.
Rerun `download` after interruption; the Hub client reuses completed files. Every
data shard is checked against the pinned repository's SHA-256 list. Source Python
scripts are neither downloaded nor executed. Do not run two preparation jobs into
the same destination simultaneously.

## 2. Prepare candidates and listen

```bash
bash audm/scripts/non_speech.sh prepare \
  --min-seconds 1 --max-seconds 30 \
  --train-per-class 2000 --eval-per-class 200
```

This writes `$AUD_NON_SPEECH_ROOT/candidates-v1/` with extracted WAV files,
`candidates.jsonl`, `report.json` and `review.html`. It keeps at most 2,000 music
and 2,000 noise chunks for train, and 200 of each for each held-out split. These
are maximums, not guaranteed yields. Selection uses the first eligible clips in
stable shard order; it is a starter corpus, not a representative random benchmark.

Only CC0 and CC BY URLs from `creativecommons.org` are admitted. The output retains
the source page, uploader, license URL, labels, recording ID and source revision
for attribution. Other licenses and unknown label combinations are excluded.

The filters reject known human-vocal labels and voice-related metadata. They admit
instrumental-event labels for music and a conservative list of environmental
events for noise. Mixed music/noise labels are left out. **Weak labels do not
prove that a clip contains no speech**: distant voices and vocals can be unannotated.
The listening step is necessary to create reliable empty-transcript supervision.
No VAD or ASR classifier is claimed to certify these clips.

Open `review.html` with its sibling `audio/` folder accessible. To review without
copying audio to your laptop, run this on the cluster:

```bash
python -m http.server 8765 --bind 127.0.0.1 \
  --directory "$AUD_NON_SPEECH_ROOT/candidates-v1"
```

In another terminal on your laptop, forward port 8765 to that same node using
your usual SSH connection, then open `http://127.0.0.1:8765/review.html`. For example,
replace `CLUSTER_NODE` below with the reachable hostname of the node running the server:

```bash
ssh -N -L 8765:127.0.0.1:8765 shahin.konadath@CLUSTER_NODE
```

Listen to each complete clip you accept. Keep its proposed category only if it is
correct and there is no human speech or singing. Reject uncertain examples.
The page has 20 clips per page and retains decisions while you page through it.
Download `decisions.jsonl` **before refreshing or closing the tab**; decisions
are not automatically persisted. Review examples from train, validation and test.
You may leave candidates pending; only explicitly accepted clips will be used.

Upload the downloaded decisions file to
`$AUD_NON_SPEECH_ROOT/decisions.jsonl`. For programmatic listening workflows, the
same file format is one decision per line, using IDs from the candidate manifest:

```json
{"candidate_id":"fsd-12345-0","decision":"accept"}
{"candidate_id":"fsd-67890-0","decision":"reject"}
```

Do not use these example IDs literally. The finalizer rejects unknown IDs,
duplicate decisions and category edits. Candidates have `text: null`, so supplying
their manifest directly to the negative trainer fails instead of creating false
empty targets. Neither labels nor waveforms should be edited in place.

## 3. Build the training corpus

```bash
bash audm/scripts/non_speech.sh finalize --silence-per-split 100 --seed 1729

export AUD_NON_SPEECH_MANIFEST="$AUD_NON_SPEECH_ROOT/corpus-v1/train.jsonl"
export AUD_NON_SPEECH_VALIDATION_MANIFEST="$AUD_NON_SPEECH_ROOT/corpus-v1/validation.jsonl"
export AUD_NON_SPEECH_TEST_MANIFEST="$AUD_NON_SPEECH_ROOT/corpus-v1/test.jsonl"
cat "$AUD_NON_SPEECH_ROOT/corpus-v1/report.json"
```

The finalizer requires at least one accepted music and noise clip in each source
split. That is a schema minimum, not a sufficient training or evaluation dataset.
Check actual counts/hours with the training preflight. The pool samples uniformly
over rows: the number of accepted clips and synthetic silences sets the category
mix. Choose `--silence-per-split` accordingly; 100 is only a starting count.

Final output:

```text
$AUD_NON_SPEECH_ROOT/corpus-v1/
  audio/{train,validation,test}/*.wav
  train.jsonl
  validation.jsonl
  test.jsonl
  decisions.jsonl
  report.json
```

Accepted audio is hash-checked and copied into the final corpus. Music and noise
retain source-split membership; recording IDs and exact decoded-audio hashes are
deduplicated across candidates, with held-out splits taking priority. Chunks from
one source remain in its split. Short final tails are discarded; chunks are at
most 30 seconds. This is **not** a perceptual duplicate detector and does not
compare against the existing multilingual speech corpus.

Silence is generated as explicitly labeled digital zeros with varied durations.
It is **synthetic**, not a downloaded real-room silence dataset, and its evaluation
does not establish performance on quiet microphones or room ambience. Add reviewed
real quiet-room clips separately before drawing that conclusion. Environmental
audio is never relabeled as silence merely because its volume is low.

Preparation/finalization refuse existing output directories. Downloading is
resumable; interrupted preparation must be rerun. For new decisions or a changed
corpus, pass a new `--output` path and update the manifest environment variables.
Do not mutate a corpus referenced by a running/resumed training job.

Continue with the source-model upgrade, `preflight --scan`, eight-GPU smoke and
training commands in [ROBUST_CLUSTER.md](ROBUST_CLUSTER.md). These manifests supply
the **non-speech** pool. They do not supply RIRs or replace the existing speech
datasets. For four-class evaluation, combine disjoint held-out speech rows with
these held-out negatives; using only this test manifest leaves missed-speech
metrics undefined. Keep validation for tuning and test for final evaluation.
