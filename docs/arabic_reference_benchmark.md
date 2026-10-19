# Arabic reference benchmark initiative

Status: the Flash-Next smoke has been inspected. Prompt/parser v3.2 preserves additional model notes as unverified uncertainties and clarifies retention of useful diacritics. Local replay accepts all 14 saved final replies without changing transcript strings. Full draft generation can follow a successful v3.2 cluster smoke; audio review remains pending.
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

- Client model: the served alias `corrector`; it accepts the checkpoint currently behind that alias. The cluster server reported `Qwen/Qwen3.8-27B-FP8`.
- Served alias: `corrector`, endpoint `http://localhost:8010/v1`.
- Existing serving script: `scripts/corpus/serve_yodas3_corrector.sh`. When starting a new server, `LLM_MODEL` selects its weights (the serving script defaults to `Qwen/Qwen3.6-35B-A3B`). It does not constrain client requests to an existing server.
- Existing default serving configuration: TP=8, context 16,384, vLLM image `v0.26.0-x86_64-cu129`.
- One 8×H100 node. Use up to 90 concurrent text requests for the full pilot (`REF_CONCURRENCY=90`). The reusable launcher default remains 16. The revised smoke has up to 14 clips (14 on the supplied plan) and cannot measure 90-worker throughput.
- The 96 CPU cores do not require 96 correction workers: the corrector owns the GPUs, and these workers perform lightweight HTTP/I/O tasks. BLAS thread counts are one.

The selected corrector is used as a **text-only proposer**. It does not hear audio. Saved QASR/GQASR predictions are reused; this stage launches no ASR model and downloads no audio or model weights itself.
Use the existing unified `gemma_speech/.venv` for the driver. It already supplies NumPy, pytest and RapidFuzz. Do not install the root project's training pins into it.

The doctor checks that `/v1/models` exposes `corrector` and records the advertised root/context for provenance. No checkpoint is pinned by default. Set `ARABIC_REF_CHECKPOINT` (CLI `--checkpoint`) only to opt into an expected-checkpoint check; a conflicting advertised Hub checkpoint then fails. Alias-only responses are recorded as checkpoint-unverified. A reported root is server metadata, not independent proof of the loaded weights. Run resume compares the recorded identity, so an advertised model change requires a new run directory. `LLM_REVISION` should identify the actual serving checkpoint revision when known; it is not an instruction to change the running server. An unrecorded revision is allowed for this internal pilot but must be resolved before public reproducibility claims.

## Where verified_text comes from

The corrector compares the anonymous text candidates and writes only
`proposal.proposed_verified_text` and `proposal.proposed_ideal_text`. It never supplies
the final columns. An Arabic-speaking reviewer listens, enters/corrects the spoken
words and formatted transcript, resolves each flagged issue, and downloads a review
file. `import-reviews` validates that file and populates `verified_text`/`ideal_text`.
The current implementation therefore includes a manual listening stage. Neither
candidate agreement nor text-only LLM judgment is audio verification. Any unresolved
span leaves both final columns null for the whole clip.

## Policy/prompt v3 and first-smoke findings

The copied first smoke used `corrector` backed by the reported `Qwen/Qwen3.8-27B-FP8`.
All eight responses finished with `stop`; none exhausted the 2,048-token response budget.
It returned seven null proposals and one rejected proposal in 12.078 seconds.

- The null responses often treated lack of audio as a reason to withhold any draft, even
  when a complete candidate could be retained with clearly stated uncertainties.
- The rejected SADA response changed `ام` → `أم`, `اتوقع` → `أتوقع`, and `ادريس` → `إدريس`
  only in the ideal rendering, while labelling those spelling edits as formatting.
  Those can be useful ideal-text refinements; the prior guard was too restrictive.
- Some explanations conflated spoken language with written script/hyphenation. Listening
  identifies speech, while script and hyphenation also require editorial policy.
- One explanation introduced `حملت`, which was absent from its supplied candidates.
  Explanations are model suggestions too; they must be checked against the evidence.

V3 explicitly separates provisional drafts from final references. A draft may retain
one complete candidate and flag competing words, without claiming it is correct. Null
remains valid where no useful complete draft exists without invention. No automatic
fallback manufactures a transcript from an unresolved reply.

`ideal_text` is the polished, faithful rendering. Punctuation, conservative diacritics
and case were already supported. V3 additionally allows **declared bare-alif to hamzated
alif refinements** (`ا` → `أ`/`إ`) in the ideal field. The proposer must supply
`ideal_orthography` edits with literal before/after evidence; the human review must
include `ideal_spelling_reason`. The narrow character allowance is not proof of meaning:
every such refinement still needs review. Broader letter substitutions, word changes,
script conversion and number conversion are not automatically licensed by this rule.
Resolve those through lexical review; never globally fold ه/ة or ى/ي as harmless.

The policy is now v3, so use a fresh `verification/arabic-reference-pilot-v3` root and
rerun `plan`, `doctor` and `smoke`. Keep the copied v2 results intact. V3 keeps the same
sampling seed and source data; the supplied selection remains 1,059 clips. Review files
also add the spelling-reason field and cannot be relabelled from v2 by editing hashes.

The revised smoke retains the eight diagnostic/coverage cases and adds one agreement
control per dataset. These controls have matching wording after permitted formatting
normalization; their references are **not** assumed acoustically correct. It requires
zero failed records and a wording-preserving draft for all six controls. Missing controls,
blanket abstention and unsolicited changes to their agreed wording fail the smoke gate.
Passing that gate tests drafting behavior, not audio accuracy. Inspect the difficult-case
drafts before the full run. Controls are not identified to the model in the request.

Complete replies that fail validation receive at most **one** follow-up with the exact
validation error. Both responses, feedback and usage are retained in each record's
`attempts`. A second invalid reply stays failed. Truncated replies, transport failures
and valid null proposals do not trigger this validation retry. The HTTP client retains
its existing bounded network retries. Retry usage is included in aggregate totals.

## Second smoke findings and v3.1 repair

The 14-clip v3 cluster smoke returned 12 proposed, one failed and one unresolved in
18.425 seconds, with four validation retries. Its six agreement controls had five passes
and one validation failure. The remaining failure was a metadata classification error:
`طويلا` → `طويلًا` and `تكلمه` → `تكلّمه` add vowel marks, but the corrector labelled them
`ideal_orthography` and incorrectly called tanwin hamza. In `sada:004898`, the same error
caused the retry to remove the proposed shadda from `ذمّتك`.

V3.1 computes this distinction from characters. When an anchored `ideal_orthography`
edit differs only by formatting already permitted in the frozen policy, it records the
label as `formatting` without changing either transcript. The raw response and original
model explanation are preserved. `validation_adjustments` records the edit index,
before/after spans and category correction in the accepted attempt and record; the
summary counts `normalized_records` and `normalized_edits`. The listening page displays
these adjustments and explicitly identifies model explanations as unverified. Sequential
formatting edits may reference the intermediate text produced by earlier edits.

This does not expand the permitted text transformations: real hamza additions still
require declared spelling edits, and word, numeric, script and unsupported letter changes
remain rejected. No-op edits remain invalid. V3.1 also clarifies tanwin/shadda versus hamza,
omitting no-op edits, preserving useful diacritics during repairs, and avoiding invented
alternative words or morphological explanations. The frozen annotation policy and plan
remain v3; only the proposer/validator changes. New runs use `smoke-v3.1/` so the original
`smoke-v3/` evidence stays intact.

All 14 final replies were replayed locally with the revised parser: 13 drafts and one
abstention, no transcript changes, all six agreement controls passing. The first SADA
reply with shadda is also accepted unchanged. This is parser replay, not a new model run
or audio verification. Source record hashes and bindings were checked before inspection.

The drafts are useful starting points for listening, but explanation quality is still
limited. For example, the Common Voice agreement case invents the rival `شكرا`; an MGB2
control invents a hamza disagreement; the noisy MASC explanation misses some candidate
differences. The Casablanca French draft retains a parenthetical annotation that may
duplicate the spoken content. Review the complete audio and all candidates, not just
issues mentioned by the proposer. Neither agreement nor fluent explanations certify truth.

After the current v3.2 smoke passes, proceed with the 1,059-clip proposal pilot and listening pack.
All final reference columns remain null until review. Inspect failures, abstentions and
representative drafts across all datasets before any later expansion beyond this pilot.

## Flash-Next smoke findings and v3.2 repair

The Flash-Next v3.1 run returned 12 proposed and two failed in 22.222 seconds, using
75,281 tokens and six validation retries. All six agreement controls preserved wording.
Both failures were **extra note categories**, not missing required notes:

- `casablanca:000008` supplied `duration` and `dialect` when `review_issues` was empty.
- `masc_clean:000074` supplied `initial_word_disagreement` when `review_issues` was empty.

The exact-key validator reported both as "Every review issue requires its own note".
V3.2 retains extra model comments verbatim in `uncertainties`, with an explicit unverified
label, and keeps `issue_notes` restricted to the supplied flags. Each relocation is recorded
in `validation_adjustments` with `type="issue_note_relocation"`, the original key/note and
the destination uncertainty index. It does not create a detector flag or invent a required
note. Missing required notes, empty/invalid notes and invalid transcript changes still fail.
Human `issue_resolutions` still require exactly the supplied keys. The listening page shows
the adjustments; final reference columns remain null. `normalized_records` counts records
with any adjustment, `normalized_edits` counts edit-category changes only, and
`relocated_issue_notes` counts moved comments. Existing edit adjustments now carry
`type="edit_category"`.

All 14 saved final replies pass local parser replay, preserving both transcript strings:
seven edit-label corrections across three records, plus three relocated notes across the
two failed records. All six controls pass. The first Casablanca control reply is also
recoverable without retry. Three other first attempts still fail for invalid/no-op edit
spans; those checks are deliberately retained. Source hashes, row bindings and JSONL copies
were validated. This is replay of saved replies, not a new inference run or audio validation.

Text inspection also found substantive limitations:

- `masc_noisy:000031` strips shadda from `الطّيران` and `الأوروبيّة` for unvocalized styling.
- `mgb2:003751` strips three shaddas, calling them non-standard or ASR artifacts without
  evidence. Its explanations invent alternative phrases absent from the candidates.
- `mgb2:000014` removes marks even though its own explanation says to preserve them.
- Several agreement cases invent rival words; Common Voice introduces `ظن` and `توماس`.
- The French Casablanca draft retains parenthetical French alongside phonetic Arabic;
  listening and editorial review must prevent scoring an annotation as extra speech.

V3.2 explicitly forbids stripping marks merely for unvocalized styling, inferring errors
from lack of marks, and choosing longer or more fluent text as proof of completeness.
It reinforces literal candidate comparisons and exact note keys. These are prompt rules;
parser acceptance does not certify their semantic observance. Human review must still
check the entire audio and the supplied drafts, including explanations.

The prior 27B smoke used proposer v3 and Flash-Next used v3.1. They share the same 14 cases,
but this is not a controlled model comparison and neither has adjudicated audio labels.
Flash-Next drafts the difficult SADA case where 27B abstained; this does not establish
higher accuracy. The small smoke times are not 90-worker throughput measurements.
The frozen annotation policy and plan remain v3; use a fresh output root for the new
proposer run, and retain earlier evidence unchanged.

## Retry on the cluster

From the pilot code checkout, after fetching and fast-forwarding the pilot branch:

```bash
export ARABIC_REF_SOURCE_REPO=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-pilot-v3-flash-next-r2"
export LLM_MODEL_NAME=corrector
export REF_CONCURRENCY=90
bash scripts/corpus/run_arabic_reference.sh test -q &&
bash scripts/corpus/run_arabic_reference.sh plan &&
bash scripts/corpus/run_arabic_reference.sh doctor &&
bash scripts/corpus/run_arabic_reference.sh smoke &&
bash scripts/corpus/run_arabic_reference.sh run &&
bash scripts/corpus/run_arabic_reference.sh review-pack
```

The smoke gate stops this chain if it fails. The new smoke writes to `$ARABIC_REF_ROOT/smoke-v3.2/`. Its start event and summary include
the absolute output path. Full drafts are in `proposals/`; the listening pack is in
`listening/`. Return the smoke and full summaries and proposal records for inspection. No serving-model change
or restart is needed for these client-side changes.

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
`ARABIC_REF_SOURCE_REPO`, and writes results to its `verification/arabic-reference-pilot-v3`.
It runs the new code from this isolated checkout. If the worktree already exists, reuse it
and update it deliberately after the previous run; changing code invalidates proposal resume.

The defaults match the supplied cluster paths. To override them explicitly, use the
following variables (when running directly in the original repository, omit the worktree steps):

```bash
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-$PWD}"
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-pilot-v3"
export QASR_RESULTS="$ARABIC_REF_SOURCE_REPO/verification/qasr-stage02-arabic"
export GQASR_RESULTS="$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-final-streamed"
export ARABIC_REF_AUDIO_ROOT="$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-stream-cache/prepared/audio"
export LLM_URL=http://localhost:8010
export LLM_MODEL_NAME=corrector
export REF_CONCURRENCY=90

bash scripts/corpus/run_arabic_reference.sh test -q &&
bash scripts/corpus/run_arabic_reference.sh plan &&
bash scripts/corpus/run_arabic_reference.sh doctor &&
bash scripts/corpus/run_arabic_reference.sh smoke
```

Reuse an existing healthy corrector. `LLM_MODEL_NAME=corrector` controls request routing; `LLM_MODEL` is used only by `llm-start`/`llm-print` to select weights for a new server. If the available node has no corrector, set `LLM_MODEL` to the checkpoint you intend to serve, then inspect and start the existing Yodas3 serving recipe:

```bash
bash scripts/corpus/run_arabic_reference.sh llm-print
bash scripts/corpus/run_arabic_reference.sh llm-start
curl -fsS http://localhost:8010/health
```

Wait for model loading and health before `doctor`. These commands do not stop an existing service; container-name or port conflicts should be resolved by identifying the existing service, not automatically replacing it.

For the current v3 plan, the prior smoke inspection and parser replay support proceeding to the full proposal pilot after the v3.2 smoke gate passes. For a new dataset or plan, inspect the smoke records before scaling. Smoke covers each dataset, the longest request, a heavily flagged case and six wording-agreement controls (14 clips on the supplied plan, not all 1,059). The inspected cluster smoke runs are recorded above; the full pilot will provide the first 90-worker throughput evidence.

## After smoke inspection

```bash
export REF_CONCURRENCY=90
bash scripts/corpus/run_arabic_reference.sh run
bash scripts/corpus/run_arabic_reference.sh status
bash scripts/corpus/run_arabic_reference.sh review-pack
```

Rerunning `run` reuses valid completed records and retries failed ones. Each response is saved atomically before progress advances. Truncated responses, schema errors, empty text, undeclared spelling changes, and unsupported changes to words/values fail explicitly. Outputs, input bindings and run identity are hashed. Changed code, prompt, model identity, decoding settings, or selection require a new output run; there is no force-overwrite switch. Concurrent writers to the same proposal directory are rejected.

Copy `$ARABIC_REF_ROOT` back from the cluster. Open `listening/index.html` in a browser with its adjacent `audio/` directory. Audio is prepared-cache audio checked against the original hash and converted to 16 kHz mono PCM16 WAV for listening. The original audio hash and review-WAV hash are both retained.

Listen and transcribe before opening the anonymous candidate/LLM-proposal panels. Read the frozen annotation rules in the page. Enter the reviewer ID, explicitly attest listening, choose Verified or Unresolved, record a reason and complete every required issue-resolution note. Adding hamza in the ideal rendering also requires a separate ideal-spelling reason. An unclear-marker note must distinguish an annotation placeholder from words actually spoken; a foreign-script note must explain the language and spelling decision. Save regularly with **Download completed reviews**; use **Resume review file** to continue. The browser keeps edits only in memory until downloaded. Metadata-only packs deliberately cannot supply audio listening evidence.

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
| `ideal_text` | Polished rendering of verified wording, including approved spelling refinements; null until reviewed |
| `proposal.base_candidate`, `proposal.changes` | Anonymous basis and literal before/after edit evidence; `ideal_orthography` identifies ideal-only spelling refinements |
| `review.ideal_spelling_reason` | Required explanation when approving ideal spelling refinements |
| `proposal.issue_notes`, `review.issue_resolutions` | One explicit note per detected issue; proposal questions and human listening resolutions respectively |
| `proposal.proposed_verified_text` | Text-only corrector suggestion, never gold |
| `proposal.proposed_ideal_text` | Text-only polished rendering with declared spelling refinements, never gold |
| `review_status` | pending/proposed/verified/unresolved, depending on artifact stage |
| `binding`, `audio_sha256`, `policy_sha256` | Bind review evidence to the exact source and policy |
| `cohorts`, `flags` | Sampling and diagnostic membership, retained in scoring |
| `split`, `purpose` | `test`, `evaluation_only`; this pipeline has no training export |

## Frozen pilot policy

The executable policy is in `src/data_processing/arabic_reference/core.py`; the proposer instructions are in `prompts.py`. The policy is copied into every plan and hashed. The exact same policy is embedded in the corrector system prompt and displayed in the listening page. Prompt examples illustrate the earlier error categories; competing language/spelling examples are synthetic and do not supply preferred benchmark answers.

- Preserve spoken words, dialect, original language, names, negation, values, hesitations, real repetition and incomplete utterances.
- Propose orthographic corrections explicitly; do not treat every ه/ة or ى/ي difference as harmless.
- Do not replace unclear speech with plausible prose, or restore quotations from memory.
- Polish verified wording with punctuation, whitespace layout, case, conservative Arabic diacritics and explicitly documented/reviewed hamza spelling refinements. Preserve word boundaries and punctuation inside numbers and identifiers. Do not otherwise change base letters, remove foreign accents/Arabic hamza or invent grammatical endings. Punctuation and vowels can change meaning even when the automated check passes; listening review remains necessary.
- Preserve number words/digits in v3. Value-aware ITN requires a separate validated policy and is deferred.
- Both null proposed texts mean unresolved. Automatic agreement/confidence never grants verified status.
- Human-reviewed duplicates must have identical verified and ideal transcripts; conflicting reviews require adjudication.
- Unresolved clips stay visible in coverage. They are not silently replaced by the original label or removed from the selection.

### How the earlier findings are handled

| Earlier finding | Implemented rule and check | Decision requiring listening |
|---|---|---|
| Good punctuation/diacritics can inflate strict error | Separate `ideal_text`; permit conservative Arabic vowel marks and sentence punctuation; protect letters, word boundaries and identifiers | Whether punctuation/vowels preserve the intended meaning; no invented case endings |
| ه/ة, ى/ي and dialect spelling differences | No blanket equivalence; spelling/script/word changes require literal edit evidence; ideal-only hamza refinements are explicitly distinguished from lexical changes | Correct lexical form and dialect spelling; no grammatical modernization |
| `غيرواضح` / `غير واضح` references | Detect possible unclear spans; require proposal and human notes; reject bracketed unclear placeholders as fully verified | Whether the words were spoken or indicate missing speech; unresolved means null final text |
| English written phonetically in Arabic | Detect Latin candidates and explicitly discuss script disagreement; neither script wins automatically | Spoken language, actual words, established loanwords and uncertain names/spelling |
| 375-repeat loop and token-limit failures | Flag repetition, caps and clipped inputs; reject loop-shaped proposals and clipped bases; do not automatically collapse repeats | Complete utterance and genuine repetitions; truncated output does not reveal missing words |
| Same Casablanca audio, different references | Group by prepared-audio hash; require issue notes; reject inconsistent verified **or ideal** texts at import and scoring | Resolve wording from audio; duplicate copies are not independent evidence |
| Non-Arabic language headers | Treat headers/metadata as fallible; proposal inputs use the already parsed transcript text | Actual spoken language; never translate the transcript because of a header |
| Negation, values, email/name guesses | Preserve content; formatting rejects examples such as `3.09` → `309`, `-5` → `5`, word merging and altered addresses | Any lexical correction needs acoustic evidence, not plausibility |
| Reference bias toward QASR | Blind candidate identities, retain originals, sample independently of model errors, and score paired reviewed subsets | Independent second review and frozen references before public ranking |

The validators check schema, anchored spans, required notes, formatting/spelling invariants,
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
