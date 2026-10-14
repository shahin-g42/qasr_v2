# Arabic-Centric JEV: Independent Data and 64-H100 Training Specification

Status: research-backed v2 implementation specification; no implementation or GPU validation performed.
Scope: a separate project, provisionally `~/Projects/qoder/jev`; this document remains in QASR for planning only.
Confirmed hardware: **8 nodes × 8 NVIDIA H100 GPUs = 64 GPUs. Multi-node training is required.**
Evidence date: September 30, 2026. Numerical data quotas, thresholds, and experiment budgets below are proposed engineering defaults, not achieved results.

## 1. Executive decisions

1. Build an **Arabic-first, candidate-conditioned decision model**, not a chatbot: one forward pass per question returns probabilities over supplied choices. Support binary propositions and ordinal scores through the same decision contract.
2. **Warm-start from the released AutoJev-27B backbone and trained decision readout**, with a fresh optimizer. Run the corresponding fresh-Qwen initialization as a mandatory control. Do not confuse fine-tuning a release with resuming its historical optimizer.
3. Implement native **FSDP2 multi-node training**. The main 64-GPU profile uses eight local shards per node and eight inter-node replicas, subject to measured memory and network gates. Single-node execution is a qualification stage, not the deliverable.
4. Make **data production a separately installable, CPU-usable pipeline**, with dedicated GPU teacher services when approved. Training consumes only immutable, audited dataset releases.
5. Prioritize **native Arabic, dialects, Arabizi, code-switching, grounded inference, and reliable uncertainty**. Translation is a supplemental source, never a substitute for native evaluation.
6. Scale accepted training data through **100k → 500k → 1M unique decisions**; permit 1.5M only after quality and learning-curve gates. A larger corpus or more GPU-hours is not evidence of a better model.
7. Establish baseline measurements and fresh protected evaluation before optimizing. Do not use AutoJev's published 84.60% as a target on a different Arabic dataset.
8. Retain the vision weights but **freeze them in the first Arabic text campaign**. Multimodal training and claims require a separately qualified data/evaluation phase.
9. No implementation in QASR, shared Python environment, QASR imports, paid API use, dataset publication, or training launch is implied by approval of this research document.

## 2. What Jev is—and what the open release establishes

### 2.1 Original Jev versus AutoJev

Original Jev is TypeSafe AI's typed decision service. Its announcement describes parallel probability outputs and a proprietary method called Reinforcement Learning for Calibrated Decisions (RLCD). **No verified original technical paper specifying Jev's architecture or RLCD objective was found.** Do not invent an RLCD reproduction. See [announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [documentation](https://docs.typesafe.ai/).

AutoJev is an independent open implementation: cross-entropy decision fine-tuning followed by separately fitted temperature scaling. It is not evidence that proprietary Jev uses the same head, loss, or number of internal forward passes. Original Jev's documented text interface and AutoJev's retained vision architecture are also distinct.

A constrained probability output guarantees a valid structure, not a correct answer. “No generated hallucination” does not mean zero classification errors, sound reasoning, or immunity to prompt injection.

| Primitive | Contract for this project | Interpretation and limits |
| --- | --- | --- |
| `choice` | 1–255 supplied options; probability vector, winning stable option ID, optional sharpness statistic | Conditional on the supplied candidate set; add a semantically meaningful `other`/`insufficient` option when the task needs it |
| `noul` | Exactly false/true; return `P(true)` | Probability of a precisely defined proposition, not intensity or severity; false maps to index 0, true to index 1 |
| `score` | 2–10 ordered, independently intelligible level descriptions; full distribution and expected index | `score = sum(i * p_i)` on indices 0 through K−1; not automatically normalized to 0–1 or a physical measurement |

The 2–10 score restriction follows the public TypeSafe interface; AutoJev's lower-level implementation accepts more. Record this as an intentional compatibility boundary.

Multiple binary questions are not a mutually exclusive classification. Separately asked propositions may violate logical complementarity; use a single categorical decision or external constraints when consistency is mandatory.

The AutoJev-style Choice statistic `(K * p_max - 1) / (K - 1)` is distribution sharpness, **not another calibrated probability of correctness**. Handle K=1 separately; TypeSafe describes its public demonstration as approximate. Do not claim exact production confidence-formula parity. Preserve full ordinal distributions because identical expected scores can conceal very different uncertainty.

### 2.2 Audited source pins

| Artifact | Immutable revision |
| --- | --- |
| [AutoJev source](https://github.com/denis-pplx/autojev/tree/ee63c1515980491a742f0bd0685c8dc5ca1f00c3) | `ee63c1515980491a742f0bd0685c8dc5ca1f00c3` |
| [AutoJev-27B weights](https://huggingface.co/denis-pplx/autojev-27b/tree/6f5b557e037f5edb25c7dc92dbc6553e5a19c015) | `6f5b557e037f5edb25c7dc92dbc6553e5a19c015` |
| Base: `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

Pin dataset revisions, tokenizer/processor files, teacher revisions, prompts, and runtime images independently. A mutable model name is not an experiment identity.

### 2.3 Actual architecture

The artifact name says Qwen3.8-27B, but its [configuration](https://huggingface.co/denis-pplx/autojev-27b/raw/6f5b557e037f5edb25c7dc92dbc6553e5a19c015/config.json) and loader use **`Qwen3_5Model`**:

- 64 decoder blocks: 48 GatedDeltaNet linear-attention blocks and 16 full-attention blocks, repeating three linear then one full.
- Text hidden width 5,120; MLP width 17,408; 24 full-attention query heads and four KV heads.
- Retained 27-block vision encoder and merger; approximately 26.0866B retained parameters including the readout.
- Original full vocabulary LM output head is removed. The replacement is `Linear(5120, 255, bias=False)`, only 1,305,600 parameters.
- Fresh-base construction initializes that readout from the corresponding LM-head rows for 255 distinct single-token answer codes. **Warm-start loads the trained readout instead of reinitializing it.**
- One forward pass: candidate descriptions enter the prompt → final non-padding hidden state → 255 logits → valid-option masking → softmax after calibration.
- Left padding, non-thinking chat template, FP32 decision logits, no generated answer tokens. The configured model context is larger, but the initial supported training/input limit remains 8,192 processed tokens.

The answer codes represent **positions in the supplied candidate list**, not a permanent taxonomy of 255 classes. The model can classify an unseen ontology only if it understands its supplied descriptions; this must be evaluated explicitly.

Code-token stability after the exact chat prefix preserves the LM-head initialization contract. Persist and strictly verify all 255 distinct codes/token IDs when loading. A one-pass classifier is mathematically possible without letter tokens; the code invariant is an AutoJev compatibility requirement, not a universal theorem about classifiers.

For a row with K options, compute `z = readout(last_valid_hidden).float()` and `p = softmax(z[:K] / T)`. Training uses raw logits (T=1). Batched masking must not create `0 * -inf` with padded soft targets; compute each loss over valid options or use a validated finite mask. Reject overlength inputs after actual processing instead of truncating. Candidate permutation updates the option-to-code map and every target probability together; persist the stochastic policy for resume.

SDPA handles full attention, not every operation in this hybrid model. FLA, causal convolution, padding, checkpoint recomputation, and recurrent state require separate backward tests. Ordinary packed-sequence attention masks do not automatically reset linear-attention or convolution state.

### 2.4 Limits of the published results

The reference reports 73,000 unique training rows, global batch 256, 286 updates, selected update 200, and fitted temperature approximately 2.2076. The [release manifest](https://huggingface.co/denis-pplx/autojev-27b/raw/6f5b557e037f5edb25c7dc92dbc6553e5a19c015/release-manifest.json) explicitly labels the evaluation **“Repeatedly monitored V3 validation; not an untouched final test.”**

The exact curated corpus is unavailable. The public-source builder, later synthetic continuation code, and published eval generator do not reconstruct that corpus or establish which of their rows contributed to the released result. Historical code hashes also differ from some currently published files.

Author-reported accuracy/ECE/Brier figures are useful context, not independently reproduced estimates of Arabic quality. Fresh evaluation cannot establish that the starting model never saw public benchmarks during its own training; disclose that unknown.

### 2.5 Corrections to v1 of this specification

- Multi-node/64-H100 training is now a core requirement; the single-H200 offload recipe is not the production design.
- BF16 weights **plus gradients** occupy approximately 97.18 GiB before activations. The unchanged full-weight single-GPU loop does not fit an 80-GB H100 merely by enabling activation checkpointing.
- Upstream keeps optimizer tensors in host RAM; transfer buffers, not all masters/moments, are explicitly pinned. FP32 masters do not imply bitwise reproducibility.
- Calibration data must not double as an untouched test. Source-row IDs alone do not prevent cross-dataset or translated-family leakage.
- Unknown licensing cannot be bypassed by a permissive repository license or by choosing our own output-weight license.

## 3. Research foundations translated into engineering choices

| Primary paper | Evidence and limitation | Required implementation consequence |
| --- | --- | --- |
| [Zheng et al., ICLR 2024: Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882) | Option-ID token and position biases across 20 LLMs; not proof of robustness at 255 choices | Deterministic candidate permutation during training, remapped targets, label-renaming and option-order tests; test high-K separately |
| [Hinton et al., 2015: Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Full softened distributions can convey more than argmax labels; teachers can still be wrong | Distill aligned complete distributions only; never manufacture soft labels from verbal self-confidence |
| [Guo et al., ICML 2017: On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Scalar temperature is a strong inexpensive calibration baseline; does not fix rankings | Fit temperature on a separate split after checkpoint selection; publish NLL, Brier, reliability plots |
| [Ovadia et al., NeurIPS 2019: Uncertainty Under Dataset Shift](https://arxiv.org/abs/1906.02530) | In-distribution calibration can fail under shift | Evaluate held-out dialects, domains, priors, candidate counts, distractors, and missing-answer cases |
| [Geifman and El-Yaniv, 2017: Selective Classification](https://arxiv.org/abs/1705.08500) | Rejection can exchange coverage for lower accepted-case risk under sampling assumptions | Measure risk–coverage and escalation load; verify error bounds instead of imposing an arbitrary 0.9 confidence gate |
| [Artetxe et al., EMNLP 2020: Translation Artifacts in Cross-lingual Transfer Learning](https://aclanthology.org/2020.emnlp-main.618/) | Translation changes superficial inference cues; gains on translated evaluations need not transfer to original-language inputs | Separate native/localized/translated provenance; preserve cross-language family identity; native final evaluation |
| [AceGPT, 2023](https://arxiv.org/abs/2309.12053) | Arabic localization combines CPT, SFT, and alignment; not a clean proof that CPT alone is necessary | Direct Arabic decision tuning first; optional controlled CPT ablation, never an automatic enormous pretraining run |
| [On the Importance of Data Scale in Pretraining Arabic Language Models, 2024](https://arxiv.org/abs/2401.07760) | Data quality/scaling effects differ by architecture | Run data-quality and scale ablations; do not extrapolate encoder results into guaranteed 27B decoder gains |
| [ZeRO, 2020](https://arxiv.org/abs/1910.02054) | Sharding optimizer/gradient/parameter state changes memory and communication trade-offs | Explicit state accounting, topology-aware sharding, measured throughput |
| [JevAdvBench, 2026 preprint](https://arxiv.org/abs/2609.31142) | Reports persuasion-driven changes and confidence-gate failures on Jev; consensus labels and one model version limit error claims | Arabic persuasion, quoted-instruction, and escalation-denial-of-service tests; structural validity is insufficient |

Further research tracks, not first-release requirements: [ELECTRA](https://arxiv.org/abs/2003.10555) is a discriminative-pretraining precedent, not Jev's architecture; [RAPS conformal prediction](https://arxiv.org/abs/2009.14193) provides marginal coverage under exchangeability, not per-case certainty or automatic guarantees for variable candidate universes. No claim is made that these papers disclose RLCD.

## 4. Independent project architecture

### 4.1 Three installable packages

Use a standalone monorepo with three distributions. Data workers must not import Torch, model weights, or the training runtime.

```text
jev/
  pyproject.toml                         # uv workspace/tooling; no QASR dependencies
  uv.lock
  packages/
    contracts/pyproject.toml             # distribution jev-contracts
    contracts/src/jev_contracts/
      examples.py                       # versioned decision/target schemas
      artifacts.py                      # dataset/run/checkpoint contracts
      provenance.py                     # source rights and derivation identities
    data/pyproject.toml                  # distribution jev-data; CPU install
    data/src/jev_data/
      cli.py
      registry.py                       # source/version/rights allowlist
      ingest.py                         # public and private approved sources
      planning.py                       # task, ontology, split, quota plans
      annotation.py                     # human task export/import/adjudication
      generate.py                       # grounded/native task authoring
      oracle.py                         # vetted deterministic task solvers
      teachers.py                       # explicitly configured remote/local clients
      verify.py                         # blind verification and disagreement routing
      arabic.py                         # conservative checks, no destructive cleaning
      dedup.py                          # cross-source and multilingual lineage
      curate.py                         # mix, audit, leakage, target validation
      state.py                          # durable shard ownership and publication
      freeze.py                         # immutable releases and signed-off manifests
      active_learning.py                # train-pool-only acquisition
    core/pyproject.toml                  # distribution jev-core, namespace jev
    core/src/jev/
      codes.py                          # checkpoint-specific answer vocabulary
      modeling.py                       # warm/fresh model construction, decision head
      processing.py                     # exact prompt/processor, valid-token checks
      batching.py                       # deterministic global optimizer windows
      losses.py                         # CE, optional distributions/KD/consistency
      distributed.py                    # FSDP2 mesh, wrapping, rank-local loading
      train.py                          # distributed loop and schedules
      checkpoint.py                     # DCP training state, atomic publication
      evaluate.py                       # frozen adapters/metrics/logits
      calibrate.py                      # temperature and policy artifacts
      export.py                         # standalone BF16 inference artifact
      server.py                         # optional bounded decision API
  configs/
    data/{sources,tasks,arabic_mix,quality,teachers,panels}.yaml
    train/{pilot,hsdp_8x8,shard16,shard32,shard64,cpt_ablation}.yaml
    eval/{native_arabic,external,shift,retention}.yaml
  scripts/cluster/{preflight,train,generate,eval}.slurm
  tests/{contracts,data,unit,distributed,integration}/
```

Python 3.12; hatchling, uv, pytest, ruff. Independent environments/locks may be used for teacher servers versus training; core package constraints must not force GPU wheels into a CPU-only data installation. Schema versions, not shared mutable internals, join the pipelines.

Upstream pins Torch 2.14.0, Transformers 5.17.0, and FLA 0.5.2; use these as qualification candidates, not a tested guarantee. Record CUDA, NCCL, Triton, driver, kernel package hashes, and container digest. Freeze the production lock only after qualification.

### 4.2 Commands and pipeline boundaries

- `jev-data sources|ingest|plan|annotate|generate|verify|curate|freeze|audit|retry|acquire`.
- `jev-train`, `jev-eval`, `jev-calibrate`, `jev-export`; `jev-serve` optional.
- Planning and dry-run commands make no LLM calls. Generation needs explicit endpoints/model revisions and a budget configuration.
- `jev-train` accepts an immutable dataset release plus training config; it never runs data generation or repairs labels implicitly.
- External API fallback defaults to disabled. Changing teacher provider, prompt, evidence, or model revision creates a new generation identity and is audited.
- Serving exposes a native `/v1/decide` contract. A `/v1/systemone` compatibility adapter is optional and separately tested; do not describe it as OpenAI-compatible by analogy.

## 5. Strict data, provenance, and supervision contracts

One record represents one decision; related variants share a source-family identity. Example of an unambiguous Arabic decision (illustrative, not benchmark data):

```json
{
  "schema_version": 2,
  "id": "example-ar-policy-0001",
  "family_id": "family-policy-0001",
  "split": "train",
  "task": "numeric_policy",
  "language": "ar",
  "variety": "MSA",
  "state": "السياسة: الطلبات التي تزيد قيمتها على ١٠٠ ريال مؤهلة للشحن المجاني. قيمة هذا الطلب ١٠٠ ريال.",
  "question": {
    "type": "choice",
    "instructions": "هل يستحق الطلب الشحن المجاني وفق السياسة المذكورة فقط؟",
    "options": [
      {"id": "eligible", "description": "يستحق الشحن المجاني"},
      {"id": "ineligible", "description": "لا يستحق الشحن المجاني"},
      {"id": "insufficient", "description": "المعلومات غير كافية"}
    ]
  },
  "target": {"kind": "hard", "option_id": "ineligible"},
  "provenance": {
    "origin": "native_arabic_authored",
    "primary_route": "oracle",
    "license_ref": "commissioned-internal-v1",
    "source_ids": ["policy-template-strict-boundary-v1"],
    "derivation_ids": [],
    "evidence_ref": "audit://example-ar-policy-0001"
  },
  "quality": {"tier": "gold", "audit_ref": "audit://example-ar-policy-0001"}
}
```

Required invariants:

- Ordered option arrays with unique stable IDs; dictionary-order assumptions are confined to the upstream compatibility adapter. Reject duplicate IDs and indistinguishable options.
- Targets are a discriminated union: hard option ID, or a complete finite nonnegative probability mapping over exactly those IDs summing to one within tolerance. Reject missing mass, out-of-range levels, NaN/Inf, and boolean/integer coercion mistakes.
- For score, level indices are consecutive and ordered. Raw rating IDs cannot become ordinal values without an explicit rubric mapping. Scalar fractional labels require a declared conversion, not automatic rounding.
- For `noul`, true and false are both trainable targets. “Not supported by evidence,” “false,” and “unknown” are different predicates/tasks.
- `state` is text or validated structured JSON; images, if later enabled, use content-addressed artifact references. No arbitrary filesystem paths or remote URLs in the public API.
- Prompt rendering sees only state, question, supplied options, and approved images. Labels, provenance, explanations, source hints, reviewer decisions, and quality scores are never rendered.
- Audit artifacts retain raw source identity, content hash, source revision, original language, translation/transliteration lineage, teacher/reviewer versions, prompt hashes, judgments, evidence, and every disposition.
- Track `native_origin`, `human_authored`, `localized`, and `synthetic` separately. Native Arabic synthetic text is not human-authored gold.
- Deduplication and splits operate on linked source families/documents/conversations, not merely the displayed row ID. All translations, option permutations, paraphrases, scripts, and counterfactual variants inherit that family's split.
- Unknown top-level schema fields fail validation; extensible metadata is contained in named versioned objects. Audit logs containing private source text need access controls and retention policy.

## 6. Arabic data strategy: coverage before volume

### 6.1 Initial distribution targets

These are marginal **sampling targets**, not a claim that all combinations exist. Reports show both unique-example counts and effective sampling weights. No secret replacement sampling to pretend a quota was filled.

| Axis | Initial mix |
| --- | --- |
| Language/register | 40% MSA; 30% Arabic-script dialect; 10% code-switching/Arabizi; 20% English retention |
| Primary production route | 30% approved public labeled data; 30% commissioned/native human-grounded; 20% executable-oracle tasks; 20% independently verified synthetic |
| Task family | 25% intent/routing; 25% grounded inference/NLI; 15% sentiment/stance; 15% numeric/temporal/policy; 10% ordinal rubric scoring; 10% safety/answerability |
| Option counts within choice tasks | 10% two; 40% three–five; 25% six–sixteen; 20% seventeen–sixty-four; 5% sixty-five–255 |

Arabic-origin content should comprise at least 70% of Arabic decisions; translated/localized Arabic is capped at 20% initially. The remainder must have documented origin, never `unknown`. Synthetic share can rise from 20% to at most 40% at pilot scale only by an explicit mixture revision backed by a human-audit and quality/scale ablation. Oracle routing is separate but generated surface text still carries a synthetic flag.

English retention is intentional: useful routing often contains product names, mixed terminology, or cross-language instructions. Do not force every option key into Arabic; evaluate Arabic and English descriptions separately.

The eight-GPU-per-node fleet does not justify fabricating 255 irrelevant alternatives for every task. Large candidate sets must come from coherent real ontologies; long descriptions must fit processed token limits. A high-K coverage shortfall is reported, not filled with nonsense options.

### 6.2 Arabic-specific coverage

- MSA plus Egyptian, Levantine, Gulf, Iraqi, Yemeni, Sudanese, Moroccan, Algerian, and Tunisian varieties. Native reviewers identify variety; a country field is not enough.
- Arabic–English and Arabic–French switching; Arabizi digits and variable spellings; Arabic, mixed, and Latin-script renderings. Evaluate script conversion as a paired transformation, not independent data.
- Negation and scope (`ليس`, `لم`, `لن`, `مو`, `مش`, dialectal double negation); exceptions, quantifiers, pronoun reference, clitics, and entity-role reversals.
- Number–noun agreement, grammatical gender, dual/plural forms, Arabic-Indic/Western numerals, decimals, negative values, percentages, units, bounds, and money.
- Temporal tasks explicitly specify chronology, time zone, and calendar conventions. Hijri/Gregorian conversion needs a named calendar/algorithm or authoritative table; never an assumed universally exact conversion.
- Grounded policy decisions with exceptions, conflicting passages, missing information, evidence attribution, and quoted instructions embedded in state.
- Sentiment targets, sarcasm, stance versus sentiment, quoted abuse versus endorsement, identity mentions, and culturally dependent judgments. Explicit rubrics and dissent records replace generic “unsafe Arabic” filters.
- Unseen organizations, source domains, and label ontologies. A strong classifier should read the supplied criteria rather than memorize a banking taxonomy.

### 6.3 Candidate sources and license boundaries

All candidates enter the registry as `pending` until the exact release and intended use are approved. A repository license is not automatic permission to redistribute underlying posts or articles.

| Candidate and primary source | Role | Provenance/rights decision |
| --- | --- | --- |
| [ArBanking77](https://github.com/SinaLab/ArBanking77) | Banking intent seed and within-source diagnostics | 77 intents; English Banking77 localization, not native-origin requests; CC BY-SA 4.0; group all language/dialect variants |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | Broader intent/routing | 60 intents; Arabic locale localized from SLURP; CC BY 4.0; align multilingual source IDs; locale does not prove dialect breadth |
| [SANAD](https://data.mendeley.com/datasets/57zpx667y9/2) | Native MSA documents/topic classification | Depositor declares CC BY 4.0; verify underlying outlet rights and syndicated duplicates; cap easy news tasks |
| [TUNIZI](https://github.com/iCompass-ai/TUNIZI) | Native Tunisian Arabizi sentiment | MIT repository; confirm dataset scope and underlying content rights; dedup versions |
| [MPOLD](https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset) | Arabic platform-shift safety | Apache repository; confirm dataset/content rights, retain agreement information and platform origin |
| [TyDi QA](https://github.com/google-research-datasets/tydiqa) | Native question/evidence decisions | Questions collected without translation; underlying text obligations remain; derive passage-selection/support tasks explicitly; NULL does not mean false |
| [ALARB](https://huggingface.co/datasets/HarethahMo/ALARB) | Optional specialist legal decisions | Saudi court source; card declares Apache 2.0; domain review and facts-only input required; do not leak verdict/reasoning |
| [ArSenTD-LEV](https://huggingface.co/datasets/ramybaly/arsentd_lev), LABR, HARD | Potential ordinal/sentiment extensions | Not approved by this spec; unclear component permissions and review-rating semantics require resolution |

Native Arabic NLI and rigorously anchored ordinal decisions are supply gaps. Commission them; do not silently fill them from translated NLI or unchecked star ratings. ALUE and ARLUE are task collections, not blanket licenses for their component datasets. [AraBench](https://aclanthology.org/2020.coling-main.447/) is a dialect translation suite, not a ready-made decision benchmark.

## 7. High-quality training-data production pipelines

### 7.1 A: rights registry, ingestion, and source quarantine

Store source URL/revision/checksum, legal status, attribution requirements, intended uses, privacy/consent status, original split, collection time, language origin, and derivative permissions. Unknown/restricted rights block training release. Preserve raw bytes in access-controlled immutable storage; never “repair” the only source copy.

Private operational data requires explicit authorization and minimization. Neither existing QASR transcripts nor production logs automatically become JEV training data. Speech/transcript-based decisions can be a later approved text-only dataset, not an implicit audio-model dependency.

### 7.2 B: task/ontology planning and split assignment

Author task cards: precise predicate, mutually exclusive versus multilabel semantics, evidence requirements, ordered-score rubric, abstention meaning, intended deployment priors, locale, risk level, and reviewer qualifications.

Construct source-family groups before transformations or generation. Freeze train/development/calibration/test membership, reserved ontologies, reserved domains, and evaluation source exclusions. Synthetic templates must be grouped at the semantic/template-family level, not by random seed alone. All branches emit planned slots with traceable ancestry.

Plan difficulty deliberately: easy anchors, near-boundary cases, hard distractors, realistic unknowns, long-context distraction, and adversarial phrasing. Do not select only rows on which the current student fails; preserve representative easy and random cases for calibration and generalization.

### 7.3 C: commissioned native examples and annotations

Recruit reviewers fluent in the target variety; domain expertise is additionally required for legal, financial, or scientific task labels. Compensate authors/reviewers and separate authoring from blind labeling.

- Authors create original self-contained situations or use approved source documents with evidence spans.
- Gold training examples receive two independent native-language judgments and adjudication of disagreements/flags.
- Fresh final-test examples receive three independent judgments and adjudication. Evaluation authors do not inspect student outputs.
- Export/import annotation jobs with stable opaque IDs, rubric versions, reviewer IDs, elapsed time, evidence, ambiguity flags, and original votes.
- Include calibration items for annotator qualification and periodic blinded rechecks. Report agreement by task and variety, not a single global score.
- Preserve genuine ambiguity. A task expecting one correct answer must be clarified, assigned an explicit valid `insufficient` label, or quarantined—not force-resolved by majority vote.

Human vote distributions measure a defined annotator population's judgments; they are not automatically objective probabilities of factual truth. Their use as soft supervision is explicit and separately evaluated.

### 7.4 D: executable-oracle and source-grounded generation

For quantitative/rule tasks, create a **typed symbolic world before surface text**: entities, rational quantities, units, dates, predicates, rules, and exceptions. Compute gold decisions with vetted deterministic code, then render Arabic text with native review.

Required controls:

- Independent solver or independently derived invariant for numeric/policy gold; exact rational arithmetic where possible and explicit rounding tolerance where not.
- Property tests: strict versus inclusive thresholds, sign changes, unit conversions, counterfactual flips, impossible/missing data, and distractor uniqueness.
- Verify that Arabic surface text faithfully expresses the symbolic world. Oracle correctness alone does not validate a mistranslated sentence.
- No unrestricted execution of model-generated code. Solvers are maintained code; structured model output is validated data.
- Retain difficult numeral/number–noun examples without normalizing away the phenomenon being learned.

For document-grounded tasks, generation must attach sufficient evidence spans. The verifier checks that the answer follows from exactly the evidence available to the student; external world knowledge cannot repair a missing premise. Counterfactuals must remain coherent with the entire document.

### 7.5 E: synthetic authoring, independent verification, and adjudication

Provider configuration contains explicit model IDs/revisions, endpoints, generation settings, throughput limits, request budgets, and allowed data classes. Local OpenAI-compatible teacher servers are supported; no particular current QASR corrector is assumed suitable. Benchmark candidate teachers on dialect-specific gold data before assignment.

Default sequence:

1. Author receives the approved slot/task card and any allowed source evidence.
2. Deterministic validation checks schema, exact ID coverage, target membership, evidence references, duplicate alternatives, lengths, and source restrictions.
3. Verifier A relabels from **state/question/options only**, plus task-permitted evidence. It does not see the planned label, author's label/explanation, teacher confidence, or identifying slot metadata.
4. Verifier B provides another blind judgment; the two verifiers must use **different model families**, and at least one must differ from the author's family. Record their shared-training-lineage uncertainty: different model names alone do not establish statistical independence. Two replicas of the same checkpoint are not independent evidence.
5. Accept into Silver only if independent labels agree, task-specific/oracle checks pass, and ambiguity is resolved. Any disagreement or high-risk task routes to human adjudication or quarantine.
6. Human adjudication may correct the planned label with an audit trail. Do not repeatedly regenerate until a desired label “wins.” Repeated teacher agreement is not proof of gold truth.

Prefer diversity in teacher training lineage and native-Arabic strength, not just size. If only one teacher family is available, verification runs in an explicit **single-family mode** (revised 2026-10-01 for the one-server deployment): the served model re-labels blindly as in step 3, the mode is recorded verbatim in the row evidence, and row-level independence is absent — such rows must never be described as independently verified. Silver auto-acceptance in this mode is permitted only because the stratified human audit and approval gates in §7.9 remain mandatory; human review supplies the missing check at the sample level rather than per row.

Small structured batches use exact opaque ID matching, strict boolean parsing, and full coverage checks. Truncated/malformed responses may be split and retried for unresolved items only; partial output cannot silently validate missing items. Default maximum three content attempts; transport retry/backoff is tracked separately.

### 7.6 F: Arabic-preserving curation and leakage control

- Keep original text. Unicode normalization/folding and whitespace/case fingerprints are search/dedup views, not automatic replacements of meaningful letters, diacritics, digits, or negation.
- Exact source/content hashes, Arabic-aware near-duplicate candidate retrieval, multilingual semantic matching, and documented human resolution of suspicious pairs.
- Group shared documents, news events, exam questions, reviews, conversation threads, source translations, and semantic template families. Do not merge unrelated examples merely because generic boilerplate matches.
- Scan train/CPT pools against all protected panels and reserved source families; exact leakage blocks release. Near matches require adjudication, not a universally safe similarity threshold.
- Detect answer leakage, equivalent options, multiple correct answers, impossible score rubrics, option-length/grammar shortcuts, and label-correlated generation templates.
- Track source/language/dialect/task/label/count/length distributions; track accepted and rejected distributions to expose filtering bias.
- Check every processed training view after actual tokenization. No silent truncation to force quota completion.

### 7.7 G: dataset freeze, audit, and reproducible recovery

Publish immutable content-addressed shards, a manifest, audit results, source-license ledger, source-family split index, configuration/prompt hashes, annotation versions, accepted counts, and quarantine counts/reasons. Trainer verifies hashes and approval status before starting.

Data job states: planned → authored → verified → adjudicated/accepted or quarantined. Stable job IDs plus request-payload hashes make retries idempotent. A durable response cache freezes the actual LLM result; re-calling a seeded remote model is not assumed byte-reproducible.

Shard-local ownership, bounded in-flight work, heartbeats, bounded retries, and per-stage metrics are mandatory. Use one ledger writer per shard. SQLite is acceptable on validated local storage; do not assume shared-filesystem WAL is safe. Multi-node workers publish immutable artifacts to shared storage; deploy a proper coordination database if ownership cannot be made safe.

Publication order: flush/fsync artifacts → checksum manifest/commit descriptor last → index publication. Resume trusts completed descriptors, ignores orphan temporary output, and rejects corrupt committed artifacts. Retry generations preserve history and never silently mutate accepted training releases.

Reconcile every planned item as accepted, quarantined, explicitly excluded with reason, or pending. Pending work blocks completion. A release may contain an **explicitly approved accepted subset** while quarantines remain outside it; report coverage shortfall. Neither missing rows nor a zero-quarantine counter is a quality guarantee.

### 7.8 H: active learning after the first student

Acquire only from the train-eligible pool: a fixed mixture of uncertain examples, known confidently wrong cases, teacher/student disagreements, underrepresented dialect/tasks, and random diversity controls. Require the same verification and rights gates for newly acquired rows.

Never mine sealed test answers or calibration mistakes into training. Development findings may guide new task design, but copying development items creates leakage. Every active-learning release is a new dataset version; final-test disclosure retires that panel from future untouched-test status.

### 7.9 Human quality budgets and release gates

Proposed pilot obligations:

- Every row: deterministic schema/rights/split checks; required evidence/oracle checks.
- Every Gold row: double independent annotation plus adjudication; three annotations for fresh test.
- Every Silver row: blind verification (dual-family agreement; recorded single-family mode where only one teacher family is served) plus stratified human audit; provenance never relabeled as human gold.
- All designated high-risk/flagged items: human review before acceptance.
- Randomly audit at least 500 accepted examples per preregistered material task/dialect/source stratum, or every row where a stratum is smaller. Overlapping strata may reuse audited rows; record inclusion probabilities.
- For pilot acceptance, aim for a 95% upper confidence bound on audited critical-label error ≤2% overall and ≤5% in material protected cells. Use an independent family-level sample; use Wilson bounds for simple random cell samples and design-aware/bootstrap estimates for weighted aggregate audits.
- These bounds require actual sample support. Small cells are reported as insufficiently validated, not declared reliable; revise allocation before claiming broad dialect coverage.
- Failed gates trigger source/rubric/teacher repair and a new blind audit, not deletion of only discovered bad rows followed by reuse of the same audit result.

Budget human work explicitly. A pilot 100k corpus is not “free” because inference is local: commissioned examples, double review, adjudication, dialect expertise, and the 27k protected-panel pool require staffing. Do not promise delivery dates or money amounts before measuring author/reviewer throughput and rejection rates.

## 8. Corpus sizes, leakage-safe evaluation, and external benchmarks

### 8.1 Internal pools

| Pool | Proposed size | Permitted use |
| --- | --- | --- |
| Train pilot | 100,000 accepted unique decisions | Initial adaptation and quality ablations |
| Train expansion | 500,000 then 1,000,000 cumulative unique decisions | Learning-curve-gated scale; not additive counts |
| Development | 10,000: 8k Arabic-focused + 2k English retention | Hyperparameters, checkpoint selection, curriculum ablations |
| Calibration | 5,000 | 2,500 temperature fit; 1,250 policy-threshold selection; 1,250 independent selective-risk verification |
| Sealed final test | 12,000: 10k freshly authored Arabic + 2k English retention | One frozen release evaluation; no teacher feedback/tuning |

These pools total 27,000 non-training examples. Allocate task/dialect coverage before authoring; source families, authors' reusable templates, and semantic scenario variants cannot cross pools. Final Arabic items must be native-origin; human reviewers validate every item. Record public-benchmark and teacher-contamination uncertainty separately.

Calibrating on balanced data does not guarantee deployment-prevalence calibration. Declare a target deployment mixture; absent real deployment statistics, publish balanced and preregistered shifted-prior diagnostics without claiming production calibration. Separate calibration subpools prevent choosing a rejection threshold and verifying its risk on the same rows.

### 8.2 External panels—evaluation only where terms permit

| Panel | What it tests | Important caveat |
| --- | --- | --- |
| [ArabicMMLU](https://github.com/mbzuai-nlp/ArabicMMLU) | 14,575 native MSA MCQs, 40 tasks | CC BY-NC-SA 4.0; use subject to approved terms; formal knowledge is not dialect competence |
| [AraSTEM](https://data.mendeley.com/datasets/rn4zbzg8z2/5) | Native Arabic STEM MCQs; paper reports 11,637 | Release declares CC BY 4.0; audit common exam/book sources with ArabicMMLU |
| [AraTrust](https://huggingface.co/datasets/asas-ai/AraTrust) | 522 three-choice trustworthiness questions | MIT card; small categories do not establish comprehensive fairness |
| [AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023) | Held-out Algerian/Moroccan native sentiment | CC BY 4.0; reserve entire selected source families if used as transfer evidence |
| [NArabizi](https://github.com/UniversalDependencies/UD_Maghrebi_Arabic_French-Arabizi) | Algerian/French Arabizi and offensive decisions | CC BY-SA 4.0; small sample and revision-specific annotation rights |
| [XNLI](https://github.com/facebookresearch/XNLI), MADAR | Translation-transfer and controlled dialect diagnostics | Translated origins and noncommercial terms; separate from native headline metrics |

MadinahQA explicitly overlaps ArabicMMLU. OALL aggregates existing tasks; it is not another independent dataset or directly comparable decision-model metric. AlGhafa includes translated and reused components despite “Native” in a repository name. ArSarcasm reuses ASTD/SemEval examples. The registry must represent these relations before ingestion.

Evaluate public MCQs using the same decision adapter for all candidate models. Official generative likelihood-normalization or LLM-judge leaderboard scores cannot be mixed directly with this model's one-pass probability scores.

### 8.3 Metrics, statistical discipline, and deployment behavior

- Primary Arabic endpoint: preregistered task-weighted mean of per-family accuracy for choice/noul tasks; use section 6 task weights excluding ordinal and renormalize. Average related views within a family before aggregation. English retention is separate, not allowed to hide Arabic regressions.
- Also report source- and dialect-macro results, ordinary micro accuracy, macro-F1, confusion matrices, option-count and context-length slices.
- NLL and multiclass Brier `sum_k (p_k - y_k)^2`, without dividing by K. Report binary `P(true)` Brier separately; binary two-class summed Brier has a different scale.
- ECE: 15 equal-width bins of top-class probability and accuracy, with bin counts and reliability plots; report binning sensitivity. ECE is a diagnostic, not the optimization or sole release target.
- Score: expected-level MAE, normalized MAE, ordinal ranked-probability score, and full-distribution diagnostics. Do not assume arbitrary ordered rubric levels are equally spaced physical values.
- Option robustness: at least eight deterministic permutations on a frozen representative subset, inverse-map probabilities, measure flip rate and mean JS divergence. Rename option IDs separately; add plausible distractors and missing-answer cases. Changed candidate sets need not preserve normalized probabilities.
- Robustness: native versus translated, held-out domains/dialects, changed class priors, quoted instructions, persuasion, code-switching, long distractors, and semantic counterfactuals. Keep factual ambiguity distinct from abstention.
- Fairness: subgroup errors/calibration and linguistically valid paired counterfactuals; preserve Arabic grammatical agreement. Identity changes apply only where irrelevant to the correct decision.
- Uncertainty: family-clustered paired bootstrap confidence intervals; preregister comparisons and report seeds. Benchmark overlap is not independent replication.
- Selective prediction: error versus coverage and escalation workload. A proposed ≤1% accepted-case error goal is only claimed where independent risk-verification data support it. Do not promise a particular coverage or transfer the claim to shifted populations.

Freeze checkpoint selection before final test. Proposed selection: highest primary development Arabic metric subject to an English regression guard; ties within numerical tolerance use lower uncalibrated development NLL, then earlier checkpoint. Report ordinal quality separately as a gate. Fit temperature only after selecting each model.

Compare untouched released AutoJev, the fresh Qwen readout baseline, fresh-Qwen adaptation, and AutoJev adaptation on identical rows/prompts. Report T=1 and separately calibrated variants. The reference's T≈2.2076 is historical metadata, not this model's calibration.

Proposed promotion gates: a positive paired 95% improvement interval over the preregistered released-AutoJev baseline on the fresh Arabic primary endpoint; no observed material dialect/task or English accuracy regression above two percentage points; ordinal and safety regressions trigger adjudication. Publish intervals and underpowered cells rather than asserting statistical noninferiority from these point-estimate guards. Freeze gates after baseline feasibility review but before tuning, not after seeing final outcomes.

## 9. Training objectives and experiment sequence

### 9.1 Default Arabic decision objective

Start with example-mean hard-label cross-entropy over valid candidates. No label smoothing, RLHF, RLCD, DPO, or focal loss by default; each changes calibration or supervision assumptions and needs an ablation.

For an explicit distribution target `q`, use `-sum(q_k log p_k)` with strict option alignment. Distinguish objective probabilistic-oracle targets, subjective human vote distributions, and teacher predictions in metadata and evaluation.

Optional research losses, off by default:

- Distillation: `(1-alpha) * CE(gold, p) + alpha * Td^2 * KL(q_teacher(Td) || p_student(Td))`; preregister alpha/Td on development, and require complete same-candidate teacher distributions. Verbal confidence is not `q_teacher`.
- Candidate permutation consistency: additional paired training forward passes, inverse-align candidate distributions, add `lambda * JS(p, aligned_p_permuted)`; lambda=0 baseline. This does not require extra deployment passes.
- Ordinal-aware loss only on genuinely ordered rubric tasks, benchmarked against ordinary distributional CE and ranked-probability metrics.

A harder teacher reasoning in private may supply a verified label/evidence, but the student is still a one-pass decision model. Do not claim it acquired the teacher's multi-step reliability without measuring it.

### 9.2 Curriculum and parameter defaults

- Preserve the upstream prompt/processor and answer-code mapping for the baseline. Test an Arabic instruction template as a separate versioned ablation; no unversioned train/inference prompt drift.
- Train all text parameters plus the existing readout; freeze vision for text-only v1. Report this scope accurately rather than “all retained weights.”
- Pilot: one epoch, global batch 256, peak LR 2e-6, AdamW beta=(0.9,0.999), epsilon=1e-8, weight decay 0.01 on matrix weights; exclude biases/norm vectors. Non-reentrant gradient checkpointing; global gradient clipping at 1.0.
- Warmup: 5% of planned updates, minimum one; cosine decay to 10% of peak. These are starting settings, not guaranteed optimal for adaptation.
- Test LR 1e-6 versus 2e-6 on the same pilot release. No automatic linear LR scaling with GPU count. Record optimizer grouping so comparisons do not accidentally change weight decay.
- Dynamic microbatches obey actual processed length plus a starting padded-token budget of 8,192 per rank, microbatch cap eight, and fixed global optimizer-window membership. Lower the token budget only through recorded profiling decisions; retain long-context coverage.
- Mix easy and difficult tasks from the start. After selection, an optional hard-case refinement run may use 50% diverse original training replay and 50% newly verified difficult examples at LR 5e-7, for at most 250 updates initially. It is an experiment, not a mandatory second stage.

### 9.3 Experiments in order

| Experiment | Inputs | Decision unlocked |
| --- | --- | --- |
| E0 | Frozen initial models, no training | Native/dialect/uncertainty baseline and annotation priorities |
| E1 | Same 100k data; AutoJev warm-start versus fresh base; LR 1e-6/2e-6 | Initialization and LR selection |
| E2 | Equal-size quality tiers and native-versus-localized variants | Whether human/native data actually improve fresh Arabic results |
| E3 | Permutation augmentation baseline; optional KD/consistency individually | Robustness benefit versus extra training cost |
| E4 | 100k → 500k → 1M cumulative quality-gated releases | Learning curve and saturation; 1.5M only if justified |
| E5 | Optional Arabic CPT branch | Whether language adaptation fixes deficits beyond decision tuning |
| E6 | Optional image/document decisions | Separate Arabic multimodal capability and distributed qualification |

Use at least two seeds for finalists, reporting variance. Keep unique data, exposures, additional training budget, and prior AutoJev exposure explicit; equal extra fine-tuning budgets do not equal equal lifetime compute.

### 9.4 CPT is a separate full-LM branch

The released decision artifact lacks the full vocabulary output head: **do not run next-token CPT on it as if it were a causal language model**. If E0–E2 expose a language-representation gap, branch from the pinned full Qwen conditional-generation model, retain its LM head for CPT, then initialize/extract the decision head after CPT.

Proposed pilot: 1B rights-cleared Arabic tokens plus 250M English/mixed retention tokens (20% retention of the total), with benchmark-family decontamination. Compare with direct decision tuning and a compute-matched control; then run identical decision adaptation. Preserve dialects rather than rewriting the whole corpus into MSA. CPT remains gated until this test improves the fresh native endpoints without unacceptable retention loss.

Tokenizer expansion, very long contexts, FP8, model-size increases, and training from scratch are outside the first campaign. Measure Arabic tokens per character/example before considering tokenizer surgery; changing the tokenizer breaks warm-start/code/prompt compatibility and requires a separate design.

## 10. Eight-node, 64-H100 distributed implementation

### 10.1 Hardware preflight and launcher

Before launch, record on every node: exact H100 SKU/usable VRAM, driver/CUDA, GPU peer connectivity, NVLink/NVSwitch versus PCIe, NIC count/bandwidth, GPU–NIC affinity, RDMA support, NCCL collective bandwidth, host RAM, local disk, shared-storage bandwidth, and available scheduler/container interfaces.

Slurm launcher design: one launcher per node via `srun`, each starts `torchrun --nproc-per-node=8 --nnodes=8` with a unique job rendezvous ID and configured reachable rendezvous host/port. Validate exactly 64 distinct global ranks and eight local GPU assignments. Do not assume Slurm exists until confirmed; provide an equivalent explicit-host torchrun adapter through the same config.

No hardcoded node addresses, old nine-node QASR roles, or `/scratch` assumptions. Use a configured writable run root and consistent mounts. Fail preflight on incompatible artifacts, missing dataset approval, insufficient scratch/checkpoint space, or rank/fabric mismatch.

### 10.2 Default mesh and precision

- Backend: FSDP2; initial 64-GPU mesh `(replicate=8, shard=8)` maps one shard group to each physical node. All 64 ranks process distinct samples; data-parallel averaging world size is 64.
- Parameters/optimizer state: FP32 sharded original/master parameters and Adam moments; BF16 model computation; FP32 reduction/accumulation initially. Verify real dtypes after the first update, not just configuration strings.
- Wrap complete decoder blocks, the embedding group, and readout appropriately; avoid blindly wrapping child convolution modules whose weights are accessed from a parent. Root sharding follows bottom-up module ownership.
- Shard-aware/meta initialization and incremental distributed loading. Never materialize the entire FP32 model on every GPU; avoid 64 redundant full-model downloads and verify selected tensor values after loading.
- No CPU optimizer offload as the primary path. No sequence packing, TP, CP, speculative kernels, or deferred synchronization until correctness is demonstrated.
- DeepSpeed ZeRO-3 is a qualified fallback if FSDP2 has a demonstrated blocker, not a second obligatory trainer in M0. Re-run all correctness gates before switching. Older MiCS examples and a stage-2 unused-parameter flag are not validated ZeRO-3 solutions.

### 10.3 Memory envelope

Conservative retained-model estimate, including vision for capacity planning, using P≈26.0866B:

| Quantity | Approximate size |
| --- | --- |
| Full BF16 weights | 48.59 GiB |
| Full BF16 weights plus BF16 gradients | 97.18 GiB |
| One full FP32 tensor per parameter | 97.18 GiB |
| FP32 parameter + FP32 gradient + two FP32 Adam moments | 16 bytes/parameter before transient buffers |

| Shard degree | Replica count on 64 GPUs | State-only GiB/rank |
| --- | --- | --- |
| 8 | 8 | 48.59 |
| 16 | 4 | 24.30 |
| 32 | 2 | 12.15 |
| 64 | 1 | 6.07 |

These are not peak VRAM estimates. Add gathered blocks/prefetch, activations, optimizer temporaries, hybrid-attention workspaces, NCCL buffers, and fragmentation. The large input embedding alone is approximately 2.37 GiB when gathered in BF16. Frozen vision reduces trainable-state demand but does not eliminate retained weight memory.

On measured worst-case batches, require at least 15% usable VRAM headroom. If shard-8 fails, qualify shard-16, then shard-32, then shard-64; do not silently change optimizer precision or freeze text layers. Replica count alone does not lower per-rank state memory.

### 10.4 Correct global batching and gradients

Construct a deterministic **global optimizer window** containing N real decisions, normally N=256. Assign length-aware microbatches to ranks while preserving membership and a serialized global cursor.

If collective reduction averages across D ranks, each local microbatch backpropagates:

```text
scaled_loss = (D / N) * sum(loss_of_each_real_local_example)
```

Equivalent local-mean scaling is `D * local_real_count / N`. Sum across local accumulation rounds. Do not average unequal microbatch means equally, normalize by input tokens, or additionally divide by a fixed accumulation count. Verify the backend's actual averaging semantics with a serial-reference test.

All ranks need the same collective-bearing execution/backward schedule. Pad schedules with valid zero-weight dummy examples where necessary; never count dummies as unique data, examples seen, or loss denominator. Handle a partial final window explicitly. Clip the globally reduced accumulated gradient once per optimizer update without double-counting HSDP replicas.

Synchronize every microbatch initially. Full no-sync accumulation can retain unsharded gradients; hybrid deferred inter-node all-reduce is a later optimization with dedicated last-use/unused-parameter tests. [PyTorch issue 196996](https://github.com/pytorch/pytorch/issues/196996) reports relevant mixed-dtype/unused-gradient behavior in a development build; it is a regression-test motivation, not proof the selected release fails or is fixed.

Text-only v1 consistently skips frozen vision. Later multimodal rounds must align conditional module use across the collective group; otherwise one rank may gather vision while another skips it and hangs. Absent gradients and explicit zero gradients differ under AdamW decay/moments. Preserve the intended semantics across dummy and modality-specific rounds.

### 10.5 Optimization batch versus GPU utilization

| Training examples | Updates at global batch 256, one epoch |
| --- | --- |
| 73,000 | 286 |
| 100,000 | 391 |
| 500,000 | 1,954 |
| 1,000,000 | 3,907 |
| 1,500,000 | 5,860 |

At 64 GPUs and batch 256, each rank processes just four real decisions per update. Short inputs can make this communication-bound. Keep batch 256 while qualifying strong scaling; batch 512/1024 are separately tuned optimization experiments, not automatic throughput switches.

Local eight-way sharding keeps parameter gathers within a node, but gradient shards still synchronize across nodes. For the full retained model, an eight-replica ring-style all-reduce of FP32 gradient shards sends roughly 22.8 GB/rank, approximately 183 GB/node per synchronization. This is a planning payload estimate, not measured NIC traffic or step time. NIC count, topology, overlap, frozen parameters, and collective algorithms matter.

No wall-clock training duration is promised before profiling. Record actual processed/padded tokens, decisions/sec, dummy fraction, network time, compute time, peak memory, and GPU-hours. A data row count is not a token budget.

### 10.6 Scaling acceptance

Qualify tiny-config math/kernels first, then the real model on 8 → 16 → 32 → 64 GPUs. The **64-GPU multi-node job must pass** numerical, memory, recovery, and throughput tests even if a later pilot uses fewer nodes for cost efficiency.

For each scale: same starting weights, same global example windows, same global batch and objective, representative and worst-case processed lengths, warmup excluded from throughput timing. Include loading, evaluation, and checkpointing in end-to-end runtime comparisons.

A proposed efficiency warning triggers when doubling GPUs gives less than 1.5× end-to-end speedup. Investigate fabric, microbatch occupancy, and checkpoint overhead; do not conceal poor scaling by changing the learning problem. The final production job remains selected and documented against the multi-node requirement and measured economics.

### 10.7 Durable distributed checkpointing

Use [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) for sharded training state, saved at optimizer boundaries. All participating ranks enter required operations consistently.

Persist model/master and optimizer state, scheduler, RNG states, current epoch/update, consumed real-example count, global data-plan cursor, option-permutation seed/state, mesh metadata, exact config, artifact hashes, code/runtime identity, and parameter-group layout.

Budget approximately 313 GB decimal for unique full FP32 parameters plus Adam moments, before metadata; text-only freezing lowers this somewhat. A separate BF16 inference export is approximately 52 GB decimal. Do not save every HSDP replica's identical states redundantly.

Default eval/checkpoint cadence: every 50 updates and at orderly stage end; profile cost before changing cadence. Bound asynchronous staging memory. Publish a completion descriptor only after all shards are durable and checksummed; incomplete writes are never loadable successes. Never auto-delete the last verified restart checkpoint.

Test interruption versus uninterrupted execution on the same topology. DCP supports resharding, but cross-world-size restart requires separate sampler/RNG/normalization tests; it does not imply bitwise trajectory equivalence. Mid-accumulation checkpoints are out of scope initially.

### 10.8 Teacher/generation use of the same cluster

Generation and training have separate jobs, allocations, runtime images, and resource ownership. A 64-GPU training allocation must not compete with leftover teacher processes.

Initial planning allocation, subject to teacher-fit measurement: up to four nodes for authoring replicas, two for a different-family verifier, one for an additional verifier, and one for held-out teacher qualification/data scoring. Native human review and oracle/ingest stages run on CPU. Adjust replicas/tensor parallelism from memory/throughput profiling; these counts do not assume a particular teacher fits on one GPU.

Do not run student FSDP training inside the teacher-serving environment. Track generated/accepted examples per GPU-hour, per-source rejection rates, review backlog, and cost per accepted independently validated example—not just raw generation throughput.

## 11. Calibration, export, and serving

After selecting a checkpoint, run distributed inference on the temperature-fit split, collecting raw valid-option logits, stable IDs, option counts, and targets. Gather once per real ID; reject duplicates/missing rows from distributed sampler padding. Fit one scalar T on CPU with a deterministic bounded positive optimization, baseline range 0.05–20; flag boundary solutions.

Use FP64 fitting calculations and hard-label NLL by default. Soft-target calibration is a separately declared experiment. Fit globally, not per rank; broadcast the final T. A 2,500×255 FP32 logit array is only about 2.55 MB, so model inference—not the calibration payload—is the bottleneck.

Store calibration as a **new immutable artifact** containing T, checkpoint hash, processor/prompt/option schema version, calibration-data hash, fitting settings, metrics, and fit time. Do not overwrite the training checkpoint or reuse T after changing prompts, quantization, weights, or deployment populations without revalidation.

Construct a new export bundle from checkpoint plus approved calibration, keeping source artifacts immutable. Include backbone, trained readout, processor/tokenizer, answer codes/token IDs, input limits, model scope, schemas, licenses/notices, provenance, and checksums. Strict-load and compare logits before publication.

Serving, if implemented: bounded queue and request sizes, valid-option probability vectors, explicit timeout/backpressure, authenticated access where configured, no raw private-input logs by default. Per-question branching may be batched but does not imply one forward pass answers arbitrarily many questions. Single-choice input yields a trivial probability of one and is excluded from quality headline metrics.

Abstention/escalation is an external calibrated policy with versioned thresholds, task applicability, coverage, and supported risk bounds. A model-generated `insufficient` option is a semantic answer, not the same as abstaining because confidence is low.

## 12. Tests, milestones, and rollout gates

### 12.1 Required test coverage

- CPU package isolation: install and run planning/ingestion/contracts without Torch or GPU dependencies.
- Schema/target validation: false/true mapping, full distribution alignment, duplicate options, unknown fields, ordinal mappings, non-finite values, metadata exclusion.
- Model parity: warm-start preserves readout and answer codes; fresh initialization copies correct LM rows; last-token/padding correctness; exact prompt and processor version checks.
- Tiny tests use a small randomly initialized same-family configuration and fake tokenizer where appropriate; a 9B model is not a “tiny CPU fixture.” Full-kernel and real-tokenizer tests remain separate GPU gates.
- Arabic fidelity: negation, diacritics, alef variants, digit runs, numeral systems, code-switching, score rubrics, semantic versus merely textual duplicates.
- Oracles: independent arithmetic/rule solutions, unit and calendar assumptions, surface-text consistency, no execution of untrusted generated code.
- Generation failures: duplicate/missing IDs, truncation, nonliteral booleans, transport retries, cache hash mismatch, budget stops, verifier correlation/visibility, human corrections, quarantine accounting.
- Data recovery: crash before/after publication, ownership collision, orphan artifacts, corrupt accepted shards, retry idempotency, approved-subset coverage reporting.
- Leakage: cross-language parents, shared reviews/documents, paraphrases, option permutations, template families, benchmark reuse and CPT exclusion.
- Distributed math: compare serial FP32 toy gradients with unequal microbatches across ranks; tail windows; dummy slots; global clipping; actual state dtypes after updates.
- Distributed execution: fast/fallback hybrid kernels, left padding, checkpoint recomputation, long context, new/unused parameters, conditional vision before optional rollout.
- Resume/export/calibration: same-topology interrupted equivalence within predefined numeric tolerances, incomplete-checkpoint rejection, exact real-ID calibration coverage, export/reload parity, changed-world-size tests before claiming support.
- Avoid universal bitwise-reproducibility claims across kernels/rank counts. Freeze tolerances and report numerical deltas.

### 12.2 Implementation order

| Milestone | Deliverable | Exit gate |
| --- | --- | --- |
| M0: contracts and preflight | Three-package skeleton, pinned reference loader, hardware/runtime inventory | CPU isolation and tiny tests; real short-input reference parity |
| M1: protected evaluation and data plans | Rights registry, task cards, family splitting, annotation/import pipeline | Final panels commissioned/secured, baselines recorded, no rights/lineage gaps in enabled sources |
| M2: quality-data pilot | Oracle/grounded/native/synthetic branches, blind review, human audits, durable release | 100k accepted target or explicit approved shortfall; quality bounds and accounting pass |
| M3: distributed trainer | FSDP2 loading/wrapping/batching/checkpointing | 8→16→32→64 qualification; memory and recovery gates |
| M4: controlled Arabic adaptation | E1–E3 on identical pilot data | Initialization/LR/quality ablations; adequate fresh Arabic gains and retention |
| M5: gated scale | 500k/1M release, learning curves, full multi-node campaign | Data quality stays within bounds; 64-GPU implementation validated; scale economics recorded |
| M6: final evaluation/export | Frozen selection, calibration/policy artifacts, untouched test, model/data reports | Preregistered promotion gates, honest unsupported cells, license approval |
| M7: optional extensions | CPT, image decisions, serving, conformal/ensemble research | Independent experiment acceptance; no retroactive expansion of v1 claims |

M1/M2 data work and M3 distributed engineering can proceed in parallel after M0. M4 requires both. No full expensive run proceeds on an unaudited corpus simply because the trainer is ready.

## 13. Remaining facts to confirm before execution

The spec supplies conditional engineering defaults; these are the remaining external inputs, not permission to guess:

1. H100 usable memory/SXM versus PCIe, NVSwitch/NVLink, NIC count and bandwidth/RDMA, shared storage, CPU/RAM, scheduler/container capabilities.
2. Approved teacher model families/endpoints and licenses, private-data handling boundaries, human annotation capacity/budget, and commercial-use requirements.
3. Priority deployment domains and dialect weighting. Until supplied, use the broad language/task mix above; do not claim a specialist legal/medical deployment is covered.
4. Final standalone repo location/name and publication policy; the sibling path remains provisional. No external repository creation/push is part of this document change.

The strongest expected route is **high-quality native and grounded Arabic decisions + robust option conditioning + matched multi-node fine-tuning + independent calibration/evaluation**. It is plausible to surpass the starting model on the intended Arabic tasks, but accuracy, calibrated coverage, and runtime improvements must be demonstrated—not inferred from the 64-GPU allocation.
