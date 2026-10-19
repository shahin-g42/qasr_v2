# Arabic reference benchmark initiative

Status: v4.2 recovered 398 of the original 410 failed responses in 340.644 seconds. The remaining 12 all contain an earlier or final complete saved reply; most were rejected only because of a preferred prose-length budget. Offline `revalidate` now retains those replies as unverified drafts and exposes verbosity as a metadata warning. Linguistic review remains required.
This is an evaluation-only initiative. Original benchmark references and training labels remain immutable. Full-suite selection, private Parquet releases, a private static leaderboard Space and additional-model scoring are now implemented. They do not mark drafts as reviewed.

## Revalidate saved v4.2 replies without further LLM inference

The supplied v4.2 snapshot contains 46,739 proposed, 12 failed and six unresolved records. Eleven final replies are valid complete JSON with overlong explanations. The last failed record, `masc_noisy:005982`, has a complete second reply with a 367-character rationale; the 360-character preferred limit triggered another retry, whose reply was malformed. The complete earlier reply can be retained without reconstructing the malformed one.

Prose budgets remain in the generation prompt, but exceeding them now creates `metadata_warning` entries instead of discarding complete transcript drafts. Notes are preserved verbatim; no explanation or transcript is silently clipped. Structural validation, rejection of `finish_reason=length`, candidate binding and loop-shaped transcript rejection remain unchanged. `revalidate` selects the latest saved attempt that passes those checks; it never combines pieces from different attempts or sends a new model request.

```bash
git pull --ff-only origin codex/arabic-reference-pilot
export ARABIC_REF_SOURCE_REPO=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full"
export ARABIC_REF_PROPOSALS="$ARABIC_REF_ROOT/proposals-repaired-v4.2"
export ARABIC_REF_REVALIDATED_OUTPUT="$ARABIC_REF_ROOT/proposals-revalidated-v4.3"
bash scripts/corpus/run_arabic_reference.sh revalidate
```

This stage runs on CPU, needs no corrector server, and uses the complete `run.json` and `proposals.jsonl` aggregate even if the individual record copy is incomplete. It verifies every record hash, input binding, inherited origin and full ID coverage before processing. It writes a separate complete snapshot with all checkpoint files, records the parent run and aggregate hash, and preserves source failed records under `source-failures/records/`. Successful and unresolved source records retain their exact contents and hashes. Failed records gain a `saved_response_revalidation` audit naming the selected original attempt. All original attempts are retained. `llm_requests=0`, `generation=null` and `new_usage={}` explicitly distinguish this processing from a new LLM generation run. Any record with no complete valid saved reply remains failed.

Use `ARABIC_REF_PROPOSALS="$ARABIC_REF_REVALIDATED_OUTPUT"` for downstream review/export; the normal private dataset and configurable Space workflow remains available. Revalidation preserves full provenance in exports and does not set final `verified_text`/`ideal_text`. The six unresolved SADA clips still require listening. Nor does recovery certify proposed edits: examples in the remaining 12 include shadda removal, number rendering changes and word/grammar substitutions present only in the ideal draft. These remain visible in saved replies, computed differences and review flags; do not treat a zero processing-failure count as zero transcription errors.

## Recover failed responses from the completed v4.1 run

The supplied run has 386 `finish_reason=length` failures (all reached 4,096 completion tokens) and 24 malformed-JSON failures. A conservative repetition check found the same 12-word sequence at least four times within the final 500 words in 346 failed replies. Examples loop over shadda conventions or speculative interpretations after already emitting the draft text. Increasing the token limit alone is not a sufficient remedy.

`repair` retains the linguistic policy and adds bounded final explanations: one short sentence per reason/issue note, up to eight listening uncertainties, no repeated debates or edit lists. It allows 8,192 output tokens without shortening the transcript. Each failed ID gets at most three attempts, with fresh candidate context each time; failed response bodies are never fed back. The first attempt uses temperature 0, subsequent attempts 0.2. Incomplete/invalid replies remain failures, never silently salvaged or padded into valid JSON. Every raw response and its settings are retained. Valid output still needs linguistic/audio review.

Run on the cluster from the `codex/arabic-reference-pilot` checkout, using the existing `corrector` server:

```bash
git pull --ff-only origin codex/arabic-reference-pilot
export ARABIC_REF_SOURCE_REPO=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full"
export ARABIC_REF_PROPOSALS="$ARABIC_REF_ROOT/proposals"
export ARABIC_REF_REPAIR_OUTPUT="$ARABIC_REF_ROOT/proposals-repaired-v4.2"
export LLM_MODEL_NAME=corrector
export REF_CONCURRENCY=90
export REF_REPAIR_MAX_TOKENS=8192

bash scripts/corpus/run_arabic_reference.sh test -q &&
bash scripts/corpus/run_arabic_reference.sh repair
```

This reads the existing full plan; do not rerun `plan-full`, `scan` or `correct`. `REF_REPAIR_MAX_TOKENS` is separate from the old `REF_MAX_TOKENS=4096`. The source directory must contain its original `run.json` and every selected checkpoint in `records/`; the aggregate file alone is not a resumable checkpoint set. Source records are validated before any LLM requests. For this run, only the 410 failed IDs are sent; 46,341 proposed plus six unresolved records are copied byte-for-byte with their original run hashes. New replies use a new run identity recording the parent, selected failure hashes, prompt and generation settings. The six unresolved SADA cases require listening, not retries to force an answer.

For current code, prose-budget overruns are warnings; the original v4.2 run applied a hard check, addressed by `revalidate` above.

The new directory contains all 46,757 records and its combined `summary.json`/`proposals.jsonl`. Source failures are archived under `source-failures/records/`. Repeating the exact same repair command resumes the repair run and retries only remaining failures; prior failed repair records are retained under `attempt-history/`. Settings/code changes require a new repair output directory; the previous repaired directory can become the new source. Never rewrite the original `run.json` to bypass the resume check.

Inspect `summary.json` and its `recovery` section. Set the proposal source for subsequent review/export commands:

```bash
export ARABIC_REF_PROPOSALS="$ARABIC_REF_REPAIR_OUTPUT"
bash scripts/corpus/run_arabic_reference.sh status
# Use a fresh immutable bundle if a previous export already exists:
export ARABIC_REF_DATASET_BUNDLE="$ARABIC_REF_ROOT/dataset-repaired-v4.2"
export ARABIC_REF_RELEASE_ID=v4.2-repaired-drafts
bash scripts/corpus/run_arabic_reference.sh export-dataset
```

To retry remaining failures in the same repair output after this export selection, restore `ARABIC_REF_PROPOSALS="$ARABIC_REF_ROOT/proposals"` first; source and output must be separate. The recovered folder works with review packs and private exports directly; no manual JSON merge is needed. Per-row original run hashes and ancestry remain available, and all final reference columns remain null until listening review. Usage in the combined summary covers current records; archived failed attempts remain available separately.

Recovery addresses execution failures only. The 14,039 proposed records with editorial flags (including potential shadda/diacritic loss) remain separate review work; a well-formed draft is not proof of a better transcript.

## Full correction route: initial cleanup, then contextual LLM review of every clip

The local initial pass processed all 46,757 rows: 7,776 use a shared, mark-preserving rendering and 38,981 retain a conservative source-style baseline. All 46,757 were submitted for LLM contextual review in the completed v4.1 run. The initial draft differs from the original in 10,144 rows (including whitespace/style changes), and 119 drafts contain ﷺ; these are processing counts, not verified accuracy improvements.

Prompt v4.1 incorporates the full scan and the 14-record linguistic audit. The initial pass is followed by the corrector for **all 46,757 clips**, including agreement cases. It does not bypass the LLM based on model agreement. Use the existing served alias `corrector`; the currently selected checkpoint remains the one already running on your node.

1. `plan-full` freezes all original rows, original references and QASR/GQASR candidates.
2. `scan` reads every row, reports recurrent disagreements, writes `corrected-drafts.jsonl`, and queues every ID for contextual review. It reuses an agreed ASR rendering only when all word forms agree, both ASR renderings match, no risk flag is present and original marks/values are preserved. Other rows start from the original text with explicit rendering cleanup; these baselines are labelled as pending, not completed corrections.
3. `correct` supplies the three anonymously labelled candidates plus that unverified initial draft to the LLM. Every row is reviewed in full sentence/dialect context. It uses 90 concurrent workers by default and checkpoints each response. Successful rows resume without another request; failed rows can be retried with the same command and configuration.
4. The saved response retains actual differences, metadata warnings, mark-loss flags and explicit style adjustments. `export-dataset` exports both draft columns and their correction method. Audio verification remains separate.

The contextual rules cover `أ / إ / آ / ؤ / ئ / ء`, `ي / ى`, and `ة / ه`. There is no global replacement table: `ان` can represent `أن` or `إن`; `اكتب` can be a command or `أكتب`; `اسم` and `ابن` retain hamzat al-wasl; `علي` can be a name while `على` is a preposition; `ه` can be a pronoun rather than a misspelling of `ة`. The model must preserve dialect, names, numbers and supported diacritics, and identify unresolved lexical alternatives. It may propose context-supported spelling refinements absent from candidates without inventing different spoken content. Source agreement alone does not establish correctness.

The exact existing phrase `صلى الله عليه وسلم`, including diacritized forms, is rendered as **`ﷺ`** in the ideal draft, both before and after the LLM. The conversion is logged. It never inserts a missing blessing or removes additional words from a longer blessing. Original references and ASR strings are unchanged. Normalized/strict benchmark token counts can differ for this convention; retain the original-reference track alongside editorial comparisons.

Original v4.1 launch commands, already completed for the supplied results (use `repair` above for this existing run):

```bash
git pull --ff-only origin codex/arabic-reference-pilot
export ARABIC_REF_SOURCE_REPO=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full"
export LLM_MODEL_NAME=corrector
export REF_CONCURRENCY=90
export REF_MAX_TOKENS=4096

bash scripts/corpus/run_arabic_reference.sh test -q &&
bash scripts/corpus/run_arabic_reference.sh plan-full &&
bash scripts/corpus/run_arabic_reference.sh scan &&
bash scripts/corpus/run_arabic_reference.sh doctor &&
bash scripts/corpus/run_arabic_reference.sh correct
```

After completion, inspect `proposals/summary.json`, especially `correction_methods`, `initial_cleanup_methods`, `editorial_flag_counts`, failures and unresolved coverage. `candidate_shadda_loss` exposes losses previously hidden by normalized wording checks; `declared_content_change_only_in_ideal` flags inconsistent placement of declared lexical edits. Numeric/identifier changes are flagged separately. These checks expose mechanical risks; they do not certify semantics or acoustic accuracy. Local tests and the initial scan alone do not establish LLM output quality.

```bash
bash scripts/corpus/run_arabic_reference.sh export-dataset
bash scripts/corpus/run_arabic_reference.sh leaderboard
```

The private publishing commands and configurable Space name/title below remain available. The `run` stage remains the plain LLM route without an initial cleanup draft; use `correct` for this two-stage route. Existing earlier results remain immutable.

## Proceed through full enhancement, private datasets and leaderboard

The dataset destination is **private** `shahink/asr_bench`. Set `HF_SPACE_REPO=owner/space-name` independently to choose the private leaderboard Space; it can use the same name or a different one. `HF_SPACE_TITLE` sets its displayed title (default: `Arabic ASR Benchmark`), including the Hub card and page heading. This pipeline creates only private repositories, verifies their visibility before uploading, and refuses an existing public destination. No public release is required for this workflow.

The local source results contain 46,757 matched QASR/GQASR test clips. The full-suite stage uses every original evaluation row, including duplicates, with a `full` cohort; the outcome-selected diagnostic cohort remains separate. The pilot selector stays available. Use a fresh root for full-suite processing and changed code/model settings:

```bash
export ARABIC_REF_SOURCE_REPO=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4-full"
export LLM_MODEL_NAME=corrector
export REF_CONCURRENCY=90
export REF_MAX_TOKENS=4096
export HF_DATASET_REPO=shahink/asr_bench
export HF_SPACE_REPO=shahink/asr_bench  # Or any other owner/space-name you control
export HF_SPACE_TITLE="Arabic ASR Benchmark"

bash scripts/corpus/run_arabic_reference.sh test -q &&
bash scripts/corpus/run_arabic_reference.sh plan-full &&
bash scripts/corpus/run_arabic_reference.sh doctor &&
bash scripts/corpus/run_arabic_reference.sh smoke
```

Inspect the fresh smoke drafts and computed differences, then run all selected clips:

```bash
bash scripts/corpus/run_arabic_reference.sh run
bash scripts/corpus/run_arabic_reference.sh status
bash scripts/corpus/run_arabic_reference.sh export-dataset
bash scripts/corpus/run_arabic_reference.sh leaderboard
```

`run` returns a nonzero status if any request still fails. Completed rows are saved and resumable under the same run identity. Exports retain all rows and their pending/failed/unresolved statuses, so they can also be produced during a partial run. A partial export is a snapshot: subsequent progress requires a new bundle directory/release ID. The launcher automatically includes `proposals/` and `reviewed-v1.jsonl` when present. Existing v4 proposals can be exported with their original plan; do not rewrite older policy bindings or run identities.

The dataset bundle (`dataset-release/`) contains one explicitly typed Parquet file per source dataset, an `all` configuration and six source configurations, all assigned to `test`. Its principal columns are `text`, `asr_text` (QASR), `gqasr_text`, `verified_text_draft`, `ideal_text_draft`, `verified_text`, `ideal_text`, `proposal_status` and `review_status`. Final values remain null until review import; drafts are never substituted for them. Source locators, prediction hashes, policy, edit audits and coverage accompany each release. This is an **annotation-overlay dataset**: audio bytes are not copied. Reuse the pinned source identifiers and existing preparation pipeline to obtain recordings. Prepared cache paths, raw replies, reviewer IDs and host logs are not uploaded.

The leaderboard bundle (`leaderboard-release/`) is a static HTML Space with no server dependency. Original-reference results are available immediately. Reviewed tracks appear only after listening attestations are imported. Original, verified and ideal references each have normalized and strict WER/CER on the exact same reviewed subset. Diagnostics and full/pilot cohorts are shown separately; incomplete six-dataset macros stay unavailable. The page discloses candidate-assisted reference development and the lack of independent audit. These are private research comparisons, not official ELM ranks. Future independent review/fresh-holdout work is unchanged.

Set `HF_SPACE_TITLE` before building `leaderboard`, or pass `leaderboard --title "My Arabic ASR Board"`. Set `HF_SPACE_REPO` before `publish-leaderboard`; changing it selects a destination, not a rename of an existing Space. Changing the title of an already-built bundle requires a new `ARABIC_REF_SPACE_BUNDLE` directory and release ID.

The bundles are immutable and include an allowlist of file hashes in `release.json`. Start with an offline upload preview:

```bash
bash scripts/corpus/run_arabic_reference.sh publish-dataset
bash scripts/corpus/run_arabic_reference.sh publish-leaderboard
```

Authenticate on the publishing machine using its existing Hugging Face CLI (`hf auth login`; never paste a token into chat). Then upload the prepared files:

```bash
bash scripts/corpus/run_arabic_reference.sh publish-dataset --execute
bash scripts/corpus/run_arabic_reference.sh publish-leaderboard --execute
```

For an existing private destination, review its current commit and supply `--expected-head <commit-sha>`; the Hub commit uses that parent to reject concurrent updates. This protects unrelated repositories and older releases without force-pushing or deleting files. Save the returned Hub commit SHA and pin it when loading the dataset. Source attribution is retained; this private workflow does not invent a blanket license.

To build a later reviewed snapshot, import listening reviews as before, then choose new immutable bundle directories:

```bash
export ARABIC_REF_RELEASE_ID=v4-reviewed-001
export ARABIC_REF_DATASET_BUNDLE="$ARABIC_REF_ROOT/dataset-reviewed-001"
export ARABIC_REF_SPACE_BUNDLE="$ARABIC_REF_ROOT/leaderboard-reviewed-001"
bash scripts/corpus/run_arabic_reference.sh export-dataset
bash scripts/corpus/run_arabic_reference.sh leaderboard
```

To add another model, run it on this exact frozen selection and provide its benchmark-format prediction JSONL. IDs, source references, prepared audio hashes, preparation IDs and durations must match; duplicate/missing/extra IDs fail. The submission must use forced-Arabic mode. Reference texts are not regenerated:

```bash
bash scripts/corpus/run_arabic_reference.sh leaderboard \
  --prediction new-model=/absolute/path/predictions-arabic.jsonl \
  --output "$ARABIC_REF_ROOT/leaderboard-with-new-model" \
  --release-id v4-board-002
```

This accepts validated offline submissions. Automatic model serving, arbitrary web uploads and independent annotation adjudication are later stages. Training cleanup remains separate from these test datasets.

## V4 linguistic editing and review

The LLM receives **the latest saved QASR stage02 prediction, the GQASR prediction and the original reference**, anonymously labelled in varying order. QASR is already included; no new ASR inference is needed. The plan retains candidate names, exact texts and source-file hashes for provenance, while the prompt hides identities. The original reference has no privileged status.

V4 allows contextual hamza/maddah, ta marbuta, alif maqsura, support-alif, spelling and clitic/vocative spacing proposals. Examples include `و ندمت → وندمت`, `الحلقه القادمه → الحلقة القادمة` and `المسؤلية → المسؤولية`. These require linguistic judgment, not blanket character replacements. Preserve dialect, actual words, repetitions, names, negation and values; fluency is not permission to paraphrase, repair the speaker's grammar or supply missing speech. Preserve useful punctuation and diacritics without guessing full vocalization or case endings.

Draft validation separates response usability from correctness:

- Malformed/truncated responses, invalid schema/base candidates, missing listening uncertainties or required issue notes, marked placeholders and loops still fail. One bounded repair is attempted for complete invalid responses.
- Unanchored edit explanations, missing edit notes and incorrect formatting labels become explicit `metadata_warning` audit entries. Text is retained unchanged, and raw replies remain saved. An accepted draft is not an accepted correction.
- `proposal_checks.basis_to_wording` and `proposal_checks.wording_to_ideal` independently record actual normalized-token differences, regardless of the LLM's edit list. These ignore only existing formatting normalization; full literal texts remain available. They identify changes, not whether a change preserves meaning. In particular `فالدنيا → فالدار الآخرة` and `قيمته → قيمتها` remain visible even if the LLM claims “spelling.”
- Summary counters distinguish `metadata_warning_records`, `wording_change_records` and `ideal_editorial_change_records`. Counts overlap and are not accuracy measurements.
- Final import still requires human listening and issue resolutions. V4 additionally requires `ideal_content_preserved: true` for a verified review, plus an editorial explanation for non-formatting differences. The reviewer must confirm the same spoken content and dialect; substantive corrections belong in the verified wording first. Numbers and identifiers must remain unchanged between verified and ideal text. The browser clears the content-preservation checkbox when either transcript is edited. Imported `reviewed_ideal_differences` records the final editorial differences.

Offline replay of the 1,059 saved v3.3 final replies under the v4 parser retains 1,021 proposed and five unresolved drafts, with 33 failures (32 truncated replies and one malformed JSON reply). The 103 formerly rejected drafts are reviewable, not verified improvements. The replay reports 162 ideal editorial-change records and 100 records with metadata warnings. Both source prediction checksums and all saved record bindings were checked. This is parser compatibility evidence, not a new-model run or an accuracy result.

No character rule can prove semantic equivalence, and a reviewer attestation is not independent verification. Contextual spelling changes and dialect grammar require real judgment. The listening page displays computed differences and unverified explanations. Its export asks for an editorial note whenever the two text fields differ (including “formatting only” where applicable). Old v3 plans/reviews must not be relabelled as v4; use a new root and fresh review files. No old run is overwritten or upgraded in place.

The policy examples reflect patterns found in the inspected pilot. This pilot has informed development; public model-quality claims still need independent review and a fresh holdout. The historical v3 sections below describe superseded behavior; the V4 section and current run instructions govern new work.

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

- Client model: the served alias `corrector`; it accepts the checkpoint currently behind that alias. The latest cluster server reported `Qwen/Qwen3.8-Flash-Next-FP8`.
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
remain rejected. V3.1 still rejected no-op edit records; v3.3 below handles anchored retention notes separately. V3.1 also clarifies tanwin/shadda versus hamza,
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

That v3.3 smoke subsequently led to the 1,059-clip proposal pilot. New work follows the v4 instructions below.
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
spans in that replay. V3.3 below distinguishes redundant retention notes from absent spans. Source hashes, row bindings and JSONL copies
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

## V3.2 smoke findings and v3.3 no-op normalization

The next Flash-Next run returned 13 proposed and one failed in 17.703 seconds, with
three validation retries. No issue-note relocation was needed. The failed Common Voice
control `commonvoice:008808` preserved its complete selected candidate in both proposed
fields, including punctuation and diacritics. Its edit list nevertheless contained
`أنّه` → `أنّه` and `أنّ` → `أنّ`, explicitly explained as unchanged spelling.
The first reply also included an unchanged whole-sentence formatting record.

V3.3 removes an edit record only when its before/after strings are exactly equal,
nonempty and anchored in both of the stated texts. It preserves the original record and
reason in `validation_adjustments` with `type="no_op_edit_removed"`; `change_index` refers
to the raw response's edit-list index, as it does for category adjustments. The summary
adds `removed_no_op_edits`. These removals contribute to `normalized_records`, while
`normalized_edits` continues to count category changes only. Raw replies stay unchanged.

Retained text is not evidence for a real edit. Removed records cannot satisfy the checks
for changed wording, ideal spelling or formatting. Empty/absent spans still fail, now
with the offending edit index and whether its source or target span is missing. Missing
required issue notes, unsupported transformations and audio-review gates are unchanged.
The system and repair instructions are unchanged from v3.2; this is a validator fix.

After source hashes, bindings and JSONL copies were checked, local replay accepted all
14 final replies and all 17 attempts in this run, without changing either transcript.
All six agreement controls pass. The final Common Voice reply loses two redundant edit
records; the other final proposals retain their existing accepted text and edits.
Regression replay of the two preceding smokes also passes their controls, while their
genuinely absent spans and unsupported script conversion remain rejected.

Content quality still needs review. Noisy MASC `000031` and MGB2 `000014` still strip
supplied shaddas despite the prompt. MGB2 `003751` now adds guessed vowels; its rationale
incorrectly calls vowel marks hamza additions. Several explanations still invent absent
alternatives, and the Casablanca agreement control contains repetitive commentary.
These are text-only drafts, not verified improvements. Parser success must not be treated
as a semantic or audio-accuracy gate. The human listening stage remains necessary before
any scoring against final columns.

## Retry on the cluster

From the pilot code checkout, after fetching and fast-forwarding the pilot branch:

```bash
export ARABIC_REF_SOURCE_REPO=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-pilot-v4-flash-next"
export LLM_MODEL_NAME=corrector
export REF_CONCURRENCY=90
export REF_MAX_TOKENS=4096
bash scripts/corpus/run_arabic_reference.sh test -q &&
bash scripts/corpus/run_arabic_reference.sh plan &&
bash scripts/corpus/run_arabic_reference.sh doctor &&
bash scripts/corpus/run_arabic_reference.sh smoke
```

The smoke gate stops this chain if it fails. The new smoke writes to `$ARABIC_REF_ROOT/smoke-v4/`. Its start event and summary include
the absolute output path. Inspect the smoke drafts and computed differences before using the full-run commands below. Full drafts are in `proposals/`; the listening pack is in `listening/`. Return summaries and records for inspection. No serving-model change
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
`ARABIC_REF_SOURCE_REPO`, and writes results to its `verification/arabic-reference-pilot-v4`.
It runs the new code from this isolated checkout. If the worktree already exists, reuse it
and update it deliberately after the previous run; changing code invalidates proposal resume.

The defaults match the supplied cluster paths. To override them explicitly, use the
following variables (when running directly in the original repository, omit the worktree steps):

```bash
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-$PWD}"
export ARABIC_REF_ROOT="$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-pilot-v4"
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

V4 changes the annotation policy and prompt; run and inspect a fresh smoke before the full proposal pilot. For a new dataset or plan, inspect the smoke records before scaling. Smoke covers each dataset, the longest request, a heavily flagged case and six wording-agreement controls (14 clips on the supplied plan, not all 1,059). The inspected cluster smoke runs are recorded above; the previous v3.3 pilot took 334.415 seconds; v4 throughput and output quality must be measured separately.

## After smoke inspection

```bash
export REF_CONCURRENCY=90
bash scripts/corpus/run_arabic_reference.sh run
bash scripts/corpus/run_arabic_reference.sh status
bash scripts/corpus/run_arabic_reference.sh review-pack
```

Rerunning `run` reuses valid completed records and retries failed ones. Each response is saved atomically before progress advances. Truncated responses, schema errors and unusable text fail explicitly. Editorial differences and deficient edit explanations are retained with computed differences and audit warnings; draft acceptance never establishes accuracy. Outputs, input bindings and run identity are hashed. Changed code, prompt, model identity, decoding settings, or selection require a new output run; there is no force-overwrite switch. Concurrent writers to the same proposal directory are rejected.

Copy `$ARABIC_REF_ROOT` back from the cluster. Open `listening/index.html` in a browser with its adjacent `audio/` directory. Audio is prepared-cache audio checked against the original hash and converted to 16 kHz mono PCM16 WAV for listening. The original audio hash and review-WAV hash are both retained.

Listen and transcribe before opening the anonymous candidate/LLM-proposal panels. Read the frozen annotation rules in the page. Enter the reviewer ID, explicitly attest listening, choose Verified or Unresolved, record a reason and complete every required issue-resolution note. Verified reviews also require the content-preservation attestation and an editorial explanation for spelling or spacing changes. An unclear-marker note must distinguish an annotation placeholder from words actually spoken; a foreign-script note must explain the language and spelling decision. Save regularly with **Download completed reviews**; use **Resume review file** to continue. The browser keeps edits only in memory until downloaded. Metadata-only packs deliberately cannot supply audio listening evidence.

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
| `proposal.base_candidate`, `proposal.changes` | Anonymous basis and model-claimed edit evidence; validate claims against computed differences and audio |
| `proposal_checks` | Independently computed changes and metadata warnings, not correctness judgments |
| `review.ideal_spelling_reason` | Required explanation when approving non-formatting editorial differences |
| `review.ideal_content_preserved` | Explicit human attestation of the same spoken content and dialect |
| `proposal.issue_notes`, `review.issue_resolutions` | One explicit note per detected issue; proposal questions and human listening resolutions respectively |
| `proposal.proposed_verified_text` | Text-only corrector suggestion, never gold |
| `proposal.proposed_ideal_text` | Text-only polished rendering with declared spelling refinements, never gold |
| `review_status` | pending/proposed/verified/unresolved, depending on artifact stage |
| `binding`, `audio_sha256`, `policy_sha256` | Bind review evidence to the exact source and policy |
| `cohorts`, `flags` | Sampling and diagnostic membership, retained in scoring |
| `split`, `purpose` | `test`, `evaluation_only`; this pipeline has no training export |

## Frozen pilot policy

The executable policy is in `src/data_processing/arabic_reference/core.py`; the proposer instructions are in `prompts.py`. The policy is copied into every plan and hashed. The exact same policy is embedded in the corrector system prompt and displayed in the listening page. Prompt examples illustrate observed error patterns; they do not verify any particular recording. The pilot has already informed development.

- Preserve spoken words, dialect, original language, names, negation, values, hesitations, real repetition and incomplete utterances.
- Propose orthographic corrections explicitly; do not treat every ه/ة or ى/ي difference as harmless.
- Do not replace unclear speech with plausible prose, or restore quotations from memory.
- Polish verified wording with punctuation, conservative Arabic diacritics and contextual spelling/spacing refinements. Audit letter and word-boundary changes as editorial decisions; preserve numeric values, identifiers and meaningful foreign accents. Punctuation and vowels can change meaning even when the automated check passes; listening review remains necessary.
- Preserve number words/digits in v4. Value-aware ITN requires a separate validated policy and is deferred.
- Both null proposed texts mean unresolved. Automatic agreement/confidence never grants verified status.
- Human-reviewed duplicates must have identical verified and ideal transcripts; conflicting reviews require adjudication.
- Unresolved clips stay visible in coverage. They are not silently replaced by the original label or removed from the selection.

### How the earlier findings are handled

| Earlier finding | Implemented rule and check | Decision requiring listening |
|---|---|---|
| Good punctuation/diacritics can inflate strict error | Separate `ideal_text`; permit conservative Arabic vowel marks and sentence punctuation; audit letter/word-boundary edits and protect numeric values/identifiers at final import | Whether punctuation/vowels preserve the intended meaning; no invented case endings |
| ه/ة, ى/ي and dialect spelling differences | No blanket equivalence; compute actual spelling/script/word differences independently of model explanations; require editorial review of contextual refinements | Correct lexical form and dialect spelling; no grammatical modernization |
| `غيرواضح` / `غير واضح` references | Detect possible unclear spans; require proposal and human notes; reject bracketed unclear placeholders as fully verified | Whether the words were spoken or indicate missing speech; unresolved means null final text |
| English written phonetically in Arabic | Detect Latin candidates and explicitly discuss script disagreement; neither script wins automatically | Spoken language, actual words, established loanwords and uncertain names/spelling |
| 375-repeat loop and token-limit failures | Flag repetition, caps and clipped inputs; reject loop-shaped proposals and clipped bases; do not automatically collapse repeats | Complete utterance and genuine repetitions; truncated output does not reveal missing words |
| Same Casablanca audio, different references | Group by prepared-audio hash; require issue notes; reject inconsistent verified **or ideal** texts at import and scoring | Resolve wording from audio; duplicate copies are not independent evidence |
| Non-Arabic language headers | Treat headers/metadata as fallible; proposal inputs use the already parsed transcript text | Actual spoken language; never translate the transcript because of a header |
| Negation, values, email/name guesses | Preserve content; draft changes remain visible; final import rejects numeric/identifier changes such as `3.09` → `309`, `-5` → `5` and altered addresses | Any lexical correction needs acoustic evidence, not plausibility |
| Reference bias toward QASR | Blind candidate identities, retain originals, sample independently of model errors, and score paired reviewed subsets | Independent second review and frozen references before public ranking |

The validators check schema, required notes, review attestations, protected numeric/identifier values, source hashes and duplicate consistency. Computed differences and metadata warnings expose editorial changes and unanchored explanations without pretending to resolve them. They cannot prove that a
reason is sound, that every content edit is justified, or that someone actually listened.
That is why human review, independent audit and cluster prompt-quality inspection are
separate requirements. Detected issues are conservative heuristics, not an exhaustive
list of speech errors; reviewers must still inspect every clip.

`original_elm` and `verified_elm` use the existing pinned benchmark scorer. `ideal_strict` is full-string WER/CER, not a standalone punctuation or diacritic quality metric. Dedicated presentation metrics and semantic error annotations belong to stage 3.

## Public release and training boundaries

The existing benchmark documentation records a CC-BY-NC-ND source card for Casablanca, and other datasets have their own conditions. A license and provenance audit is an actual release dependency; do not assume that an annotation overlay automatically resolves redistribution or derivative-work restrictions. The implemented publisher supports private repositories only. A public release would be a separate decision and audit.

The eventual dataset card must disclose the reviewed coverage, original versus amended references, corrector identity and prompt, human-review method, dataset-specific permissions, source revisions, duplicate policy, unresolved spans and known limitations. Freeze references before ranking additional model versions, and retain the original-reference evaluation track.

Training cleanup is a later, separate pipeline. Use the validated annotation rules on train/dev records, exclude evaluation overlap using audio hashes and source recording/segment identities, retain original labels and edit provenance, and evaluate retrained models on frozen references plus a fresh holdout. Do not feed this pilot's test recordings or corrected references into training.

## Local verification

```bash
PYTHONPATH="$PWD/src:$PWD/gemma_speech/gqasr/src" python3 -m pytest tests/test_arabic_reference.py -q
python3 -m ruff check src/data_processing/arabic_reference tests/test_arabic_reference.py
bash -n scripts/corpus/run_arabic_reference.sh
```

The tests include a loopback mock HTTP server, crash/resume behavior, invalid model replies, source mismatches, audio verification, review binding and paired scoring. They need no GPU, real corrector, dataset downloads or private audio. The listening page is dependency-free; real browser/audio workflow and real corrector output quality still require the cluster smoke and listening review.
