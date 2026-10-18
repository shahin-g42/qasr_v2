# Arabic reference benchmark initiative

Status: pilot infrastructure implemented; real corrector execution and audio review are pending.
This is an evaluation-only initiative. No existing benchmark reference or training label is overwritten.

## Stages and evidence required

| Stage | Deliverable | Exit evidence |
|---|---|---|
| 1. Reproducible pilot | Frozen original references, anonymous candidate inputs, text-only corrector proposals, listening pack | Source hashes match; cluster doctor and smoke pass; proposal failures and review coverage are visible |
| 2. Annotation validation | Audio-reviewed `verified_text` and formatted `ideal_text`; independent second review of a sample and disagreements | Measured agreement, correction acceptance and remaining uncertainty by dataset/dialect; policy revised and frozen |
| 3. Model comparison | Original/verified scores on identical subsets; additional model baselines; separate reliability and presentation metrics | Every model uses the same frozen reference version; uncertainty and coverage reported; no official full-suite claims from the pilot |
| 4. Public release | Licensed annotation dataset, reproducible scoring package, Hugging Face dataset card and Space | Dataset-specific rights reviewed, attribution retained, review methodology and limitations disclosed, release version frozen |
| 5. Training-data improvement | Separate train/dev correction run following the validated policy | Audio and source-record overlap excluded against evaluation sets; original labels retained; improvement on a fresh holdout |

Stages 2–5 are intentionally not represented as completed by generating LLM proposals.
Single-review pilot records are human-attested, not an independently audited public gold standard.
The existing benchmark has already informed development; final model-quality claims need a fresh holdout.

## Model and node

Reuse the corrector from `bash scripts/corpus/run_yodas3.sh run`:

- Model default: `Qwen/Qwen3.6-35B-A3B`.
- Served alias: `corrector`, endpoint `http://localhost:8010/v1`.
- Existing serving script: `scripts/corpus/serve_yodas3_corrector.sh`.
- Existing default serving configuration: TP=8, context 16,384, vLLM image `v0.26.0-x86_64-cu129`.
- One 8×H100 node. Use up to 90 concurrent text requests for the full pilot (`REF_CONCURRENCY=90`). The reusable launcher default remains 16. The small smoke has only 6–8 requests and cannot measure 90-worker throughput.
- The 96 CPU cores do not require 96 correction workers: the corrector owns the GPUs, and these workers perform lightweight HTTP/I/O tasks. BLAS thread counts are one.

The selected corrector is used as a **text-only proposer**. It does not hear audio. Saved QASR/GQASR predictions are reused; this stage launches no ASR model and downloads no audio or model weights itself.
Use the existing unified `gemma_speech/.venv` for the driver. It already supplies NumPy, pytest and RapidFuzz. Do not install the root project's training pins into it.

The doctor records `/v1/models` metadata and rejects an advertised conflicting Hub checkpoint. When the server exposes only its alias or a local path, identity is explicitly recorded as declaration-only. `LLM_REVISION` should identify the actual serving checkpoint revision when known; it is not an instruction to change the running server. An unrecorded revision is allowed for this internal pilot but must be resolved before public reproducibility claims.

## Where verified_text comes from

The corrector compares the anonymous text candidates and writes only
`proposal.proposed_verified_text` and `proposal.proposed_ideal_text`. It never supplies
the final columns. An Arabic-speaking reviewer listens, enters/corrects the spoken
words and formatted transcript, resolves each flagged issue, and downloads a review
file. `import-reviews` validates that file and populates `verified_text`/`ideal_text`.
The current implementation therefore includes a manual listening stage. Neither
candidate agreement nor text-only LLM judgment is audio verification. Any unresolved
span leaves both final columns null for the whole clip.

## Policy v2 migration

Policy v2 tightens formatting checks and adds anchored proposal edits and per-issue
review notes. It is intentionally incompatible with v1 plans, proposals and review
files. The launcher now defaults to `verification/arabic-reference-pilot-v2`. Keep old
artifacts intact; if `ARABIC_REF_ROOT` is already exported, explicitly point it to the
new directory, then rerun plan, doctor and smoke. Do not update the checkout during a
running proposal job. Old reviews require reassessment under v2; they are not migrated
by changing a hash or renaming a directory.

## First cluster action: plan, doctor, smoke

Use a separate cluster checkout when the main repository is also used by training jobs.
Run this from the existing cluster repository root:

```bash
export ARABIC_REF_SOURCE_REPO="$PWD"
git fetch origin codex/arabic-reference-pilot
git worktree add ../qasr-reference-pilot origin/codex/arabic-reference-pilot
cd ../qasr-reference-pilot
```

The launcher now reads the virtual environment, saved predictions and prepared audio from
`ARABIC_REF_SOURCE_REPO`, and writes results to its `verification/arabic-reference-pilot-v2`.
It runs the new code from this isolated checkout. If the worktree already exists, reuse it
and update it deliberately after the previous run; changing code invalidates proposal resume.

The defaults match the supplied cluster paths. To override them explicitly, use the
following variables (when running directly in the original repository, omit the worktree steps):

```bash
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-$PWD}"
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-pilot-v2"
export QASR_RESULTS="$ARABIC_REF_SOURCE_REPO/verification/qasr-stage02-arabic"
export GQASR_RESULTS="$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-final-streamed"
export ARABIC_REF_AUDIO_ROOT="$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-stream-cache/prepared/audio"
export LLM_URL=http://localhost:8010
export LLM_MODEL=Qwen/Qwen3.6-35B-A3B
export LLM_MODEL_NAME=corrector
export REF_CONCURRENCY=90

bash scripts/corpus/run_arabic_reference.sh test -q &&
bash scripts/corpus/run_arabic_reference.sh plan &&
bash scripts/corpus/run_arabic_reference.sh doctor &&
bash scripts/corpus/run_arabic_reference.sh smoke
```

Reuse an existing healthy corrector. If the available node has none, inspect and start the existing Yodas3 serving recipe:

```bash
bash scripts/corpus/run_arabic_reference.sh llm-print
bash scripts/corpus/run_arabic_reference.sh llm-start
curl -fsS http://localhost:8010/health
```

Wait for model loading and health before `doctor`. These commands do not stop an existing service; container-name or port conflicts should be resolved by identifying the existing service, not automatically replacing it.

Return the doctor output, `smoke/summary.json`, and the smoke records for inspection before the full pilot. Smoke covers each dataset, the longest request and a heavily flagged case (six to eight clips, not all 1,059). Full-model correctness and H100 throughput have not been tested locally.

## After smoke inspection

```bash
export REF_CONCURRENCY=90
bash scripts/corpus/run_arabic_reference.sh run
bash scripts/corpus/run_arabic_reference.sh status
bash scripts/corpus/run_arabic_reference.sh review-pack
```

Rerunning `run` reuses valid completed records and retries failed ones. Each response is saved atomically before progress advances. Truncated responses, schema errors, empty text, and formatting that changes letters or digits fail explicitly. Outputs, input bindings and run identity are hashed. Changed code, prompt, model identity, decoding settings, or selection require a new output run; there is no force-overwrite switch. Concurrent writers to the same proposal directory are rejected.

Copy `$ARABIC_REF_ROOT` back from the cluster. Open `listening/index.html` in a browser with its adjacent `audio/` directory. Audio is prepared-cache audio checked against the original hash and converted to 16 kHz mono PCM16 WAV for listening. The original audio hash and review-WAV hash are both retained.

Listen and transcribe before opening the anonymous candidate/LLM-proposal panels. Read the frozen annotation rules in the page. Enter the reviewer ID, explicitly attest listening, choose Verified or Unresolved, record a reason and complete every required issue-resolution note. An unclear-marker note must distinguish an annotation placeholder from words actually spoken; a foreign-script note must explain the language and spelling decision. Save regularly with **Download completed reviews**; use **Resume review file** to continue. The browser keeps edits only in memory until downloaded. Metadata-only packs deliberately cannot supply audio listening evidence.

Import downloaded reviews, then compare the models:

```bash
bash scripts/corpus/run_arabic_reference.sh import-reviews --reviews /absolute/path/reviews.jsonl
bash scripts/corpus/run_arabic_reference.sh score
```

Partial review is supported and reported. Supply all completed review files in one import; duplicate IDs are rejected rather than silently overwritten. For a revised review, write a new immutable output version with the Python CLI's `--output` option and score it with `--reviewed`.

The scoring code rejects proposals as gold, requires matching audio/policy/input bindings and review evidence, preserves originals, and uses identical verified subsets for original and corrected references. A six-dataset macro value is absent if any dataset has no verified rows. All pilot scores remain explicitly `full_suite: false`.

## Pilot selection and schema

The input prediction files must have identical IDs, exact original reference text, prepared audio hashes, preparation IDs and durations. Each row must be forced-Arabic mode. All six datasets must be present. Pilot sampling ranks unique audio hashes deterministically using seed `20261019`, selecting 100 per dataset independently of model outputs and scores. This balances datasets; it does not guarantee proportional demographic or dialect representation. Coverage must be inspected before scaling.

Diagnostics are a separate overlapping cohort: unclear-reference candidates, token-limit failures from **either** model, predictions containing Latin letters, and duplicate audio. These flags identify review candidates, not proven errors. The Latin flag includes legitimate code switching. Diagnostic results must never be presented as an unbiased population score.

On the supplied results, the local plan contains:

- 46,757 matched source rows per model.
- 600 sampled pilot clips.
- 465 diagnostic clips, six overlapping the pilot, giving **1,059 unique clips**.
- Diagnostic flags overlap: 142 unclear-reference clips, 90 limit hits across either model, 239 Latin-output candidates, eight duplicate-audio rows, and two conflicting duplicate-reference rows.
- The original QASR-only limit-hit count remains 24; the 90 count is the union across QASR and GQASR.

| Field | Meaning |
|---|---|
| `text` | Original reference, immutable |
| `verified_text` | Audio-reviewed, faithful transcription; null until reviewed |
| `ideal_text` | Formatted version of verified text; null until reviewed |
| `proposal.base_candidate`, `proposal.changes` | Anonymous basis and literal before/after edit evidence |
| `proposal.issue_notes`, `review.issue_resolutions` | One explicit note per detected issue; proposal questions and human listening resolutions respectively |
| `proposal.proposed_verified_text` | Text-only corrector suggestion, never gold |
| `proposal.proposed_ideal_text` | Text-only formatting suggestion, never gold |
| `review_status` | pending/proposed/verified/unresolved, depending on artifact stage |
| `binding`, `audio_sha256`, `policy_sha256` | Bind review evidence to the exact source and policy |
| `cohorts`, `flags` | Sampling and diagnostic membership, retained in scoring |
| `split`, `purpose` | `test`, `evaluation_only`; this pipeline has no training export |

## Frozen pilot policy

The executable policy is in `src/data_processing/arabic_reference/core.py`; the proposer instructions are in `prompts.py`. The policy is copied into every plan and hashed. The exact same policy is embedded in the corrector system prompt and displayed in the listening page. Prompt examples illustrate the earlier error categories; competing language/spelling examples are synthetic and do not supply preferred benchmark answers.

- Preserve spoken words, dialect, original language, names, negation, values, hesitations, real repetition and incomplete utterances.
- Propose orthographic corrections explicitly; do not treat every ه/ة or ى/ي difference as harmless.
- Do not replace unclear speech with plausible prose, or restore quotations from memory.
- Format verified words with punctuation, whitespace layout, case and conservative Arabic diacritics. Preserve word boundaries and punctuation inside numbers and identifiers. Do not add base letters, remove foreign accents/Arabic hamza or invent grammatical endings. Punctuation and vowels can change meaning even when the automated check passes; listening review remains necessary.
- Preserve number words/digits in v2. Value-aware ITN requires a separate validated policy and is deferred.
- Both null proposed texts mean unresolved. Automatic agreement/confidence never grants verified status.
- Human-reviewed duplicates must have identical verified and ideal transcripts; conflicting reviews require adjudication.
- Unresolved clips stay visible in coverage. They are not silently replaced by the original label or removed from the selection.

### How the earlier findings are handled

| Earlier finding | Implemented rule and check | Decision requiring listening |
|---|---|---|
| Good punctuation/diacritics can inflate strict error | Separate `ideal_text`; permit conservative Arabic vowel marks and sentence punctuation; protect letters, word boundaries and identifiers | Whether punctuation/vowels preserve the intended meaning; no invented case endings |
| ه/ة, ى/ي and dialect spelling differences | No blanket equivalence; spelling/script/word changes require an anonymous basis and literal edit evidence | Correct lexical form and dialect spelling; no grammatical modernization |
| `غيرواضح` / `غير واضح` references | Detect possible unclear spans; require proposal and human notes; reject bracketed unclear placeholders as fully verified | Whether the words were spoken or indicate missing speech; unresolved means null final text |
| English written phonetically in Arabic | Detect Latin candidates and explicitly discuss script disagreement; neither script wins automatically | Spoken language, actual words, established loanwords and uncertain names/spelling |
| 375-repeat loop and token-limit failures | Flag repetition, caps and clipped inputs; reject loop-shaped proposals and clipped bases; do not automatically collapse repeats | Complete utterance and genuine repetitions; truncated output does not reveal missing words |
| Same Casablanca audio, different references | Group by prepared-audio hash; require issue notes; reject inconsistent verified **or ideal** texts at import and scoring | Resolve wording from audio; duplicate copies are not independent evidence |
| Non-Arabic language headers | Treat headers/metadata as fallible; proposal inputs use the already parsed transcript text | Actual spoken language; never translate the transcript because of a header |
| Negation, values, email/name guesses | Preserve content; formatting rejects examples such as `3.09` → `309`, `-5` → `5`, word merging and altered addresses | Any lexical correction needs acoustic evidence, not plausibility |
| Reference bias toward QASR | Blind candidate identities, retain originals, sample independently of model errors, and score paired reviewed subsets | Independent second review and frozen references before public ranking |

The validators check schema, anchored spans, required notes, formatting invariants,
review evidence, source hashes and duplicate consistency. They cannot prove that a
reason is sound, that every content edit is justified, or that someone actually listened.
That is why human review, independent audit and cluster prompt-quality inspection are
separate requirements. Detected issues are conservative heuristics, not an exhaustive
list of speech errors; reviewers must still inspect every clip.

`original_elm` and `verified_elm` use the existing pinned benchmark scorer. `ideal_strict` is full-string WER/CER, not a standalone punctuation or diacritic quality metric. Dedicated presentation metrics and semantic error annotations belong to stage 3.

## Public release and training boundaries

The existing benchmark documentation records a CC-BY-NC-ND source card for Casablanca, and other datasets have their own conditions. A license and provenance audit is an actual release dependency; do not assume that an annotation overlay automatically resolves redistribution or derivative-work restrictions. No Hugging Face upload or public leaderboard publication is implemented in this pilot.

The eventual dataset card must disclose the reviewed coverage, original versus amended references, corrector identity and prompt, human-review method, dataset-specific permissions, source revisions, duplicate policy, unresolved spans and known limitations. Freeze references before ranking additional model versions, and retain the original-reference evaluation track.

Training cleanup is a later, separate pipeline. Use the validated annotation rules on train/dev records, exclude evaluation overlap using audio hashes and source recording/segment identities, retain original labels and edit provenance, and evaluate retrained models on frozen references plus a fresh holdout. Do not feed this pilot's test recordings or corrected references into training.

## Local verification

```bash
PYTHONPATH="$PWD/src:$PWD/gemma_speech/gqasr/src" python3 -m pytest tests/test_arabic_reference.py -q
python3 -m ruff check src/data_processing/arabic_reference tests/test_arabic_reference.py
bash -n scripts/corpus/run_arabic_reference.sh
```

The tests include a loopback mock HTTP server, crash/resume behavior, invalid model replies, source mismatches, audio verification, review binding and paired scoring. They need no GPU, real corrector, dataset downloads or private audio. The listening page is dependency-free; real browser/audio workflow and real corrector output quality still require the cluster smoke and listening review.
