"""The corpus registry: every ingestible source, as data.

Verification status
-------------------
Every ``repo_id`` and ``config`` below was resolved against
``https://huggingface.co/api/datasets/<id>`` before being written here, and
``verified=True`` means that call returned 200 with that config present in the
card metadata. That is not a formality. Of 27 candidate identifiers recalled
from memory and checked, **11 did not resolve** -- including ``pkufool/masc``,
``Coqui/MSAD``, ``parakeet-ml/libriheavy``, ``fixie-ai/ultrastudio`` and
``espnet/emilia_dataset``. Two resolved only via redirect (``ai4bharat/shrutilipi``
is ``ai4bharat/Shrutilipi``; ``librispeech_asr`` is ``openslr/librispeech_asr``).
And Mandarin in FLEURS is ``cmn_hans_cn``, not ``zh_cn``.

What ``verified=True`` does NOT cover
-------------------------------------
``FieldMap`` column names. Those cannot be read from the Hub API, and sources
are inconsistent -- Common Voice uses ``sentence``, FLEURS uses
``transcription``, most others use ``text``. Run ``preflight.py --probe-fields``
to stream one row per source and print its actual columns before a full ingest.
Specs below carry the best-known mapping plus a note where it is unconfirmed.

Neither does it cover row counts or hours. ``est_hours`` is for planning only;
the ledger counts what was actually read.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

from .base import DEFAULT_MAX_SAMPLES, DatasetSpec, FieldMap, Kind

LOGGER = logging.getLogger("data_processing.datasets.registry")

LANGUAGES: tuple[str, ...] = ("ar", "en", "zh", "hi", "ml")

#: Roots are env-overridable so the registry stays portable between the laptop,
#: the build node and the training cluster, which mount things differently.
INTERNAL_ROOT = os.environ.get("QASR_INTERNAL_ROOT", "training_manifests/v7.6")
#: The SECOND internal tree: raw q3asr SFT envelopes (verbatim targets,
#: duration-complete), a SIBLING of the v7.6 manifests rather than a subdir.
#: configs/v7.6/internal_ds_sources.yaml -- the training config whose
#: train_manifest lists name every file in both trees -- is the source of
#: truth; the internal specs here mirror it, and a test pins the two together.
SFT_ROOT = os.environ.get("QASR_SFT_ROOT", "q3asr_sft_manifests")
EMILIA_ROOT = os.environ.get(
    "QASR_EMILIA_ROOT", "/vast/audio/data/tts/44k/Emilia-Dataset-extracted"
)

#: Recovery shards are already folded into their parent corpus, and eval shards
#: are held out. A bare `*.jsonl` glob matches both, which is how ml came to be
#: overstated by 41% and how 962 ar eval paths leaked into train.
_INTERNAL_EXCLUDE = ("_still_rejected_", "eval_")

# Canonical field maps, named so specs stay readable and consistent.
_F_CANONICAL = FieldMap()  # internal v7.6 already uses the canonical four
_F_FLEURS = FieldMap(text="transcription", duration=None, path="audio")
_F_CV = FieldMap(text="sentence", duration=None, path="path")
_F_HF_AUDIO = FieldMap(text="text", duration=None, path="audio")


def _internal(lang: str, train_files: int = 0, eval_files: int = 0) -> DatasetSpec:
    """The existing v7.6 manifest tree for one language.

    What lives in the tree is defined by the training config
    ``configs/v7.6/internal_ds_sources.yaml`` (``train_manifest``), which
    stacks the LLM-cleaned v7.6 corpora this spec reads.

    Patterns are RELATIVE TO ``INTERNAL_ROOT``, which callers pass as ``root``.
    They must not also contain the root: ``expand_paths`` joins the two, and an
    already-rooted pattern silently becomes
    ``training_manifests/v7.6/training_manifests/v7.6/ar/*.jsonl`` and matches
    nothing. Preflight caught this; it is the reason ``--root`` is not optional.

    ``*.jsonl*`` rather than ``*.jsonl`` so a gzipped shard is still found. The
    tree is plain ``.jsonl`` today, but the directory branch of ``expand_paths``
    matches on the ``.jsonl`` token anywhere in the name, and the two branches
    disagreeing is how a source quietly returns zero rows.
    """
    return DatasetSpec(
        name=f"internal_v76_{lang}",
        lang=lang,
        kind=Kind.LOCAL_JSONL,
        license="internal",
        paths=(f"{lang}/*.jsonl*",),
        exclude=_INTERNAL_EXCLUDE,
        fields=_F_CANONICAL,
        max_samples=DEFAULT_MAX_SAMPLES,
        verified=True,
        notes=(
            f"Already on disk in canonical form; {train_files} train + {eval_files} eval "
            "file(s) measured. Some shards omit duration entirely -- the q3asr ar "
            "shards were measured at 58.6% missing -- so run duration_coverage() "
            "before trusting a full pass."
        ),
    )


def _sft(lang: str) -> DatasetSpec:
    """The raw q3asr SFT envelopes for one language, in their own tree.

    The tree sits beside -- not under -- the v7.6 manifests, so the spec
    carries ``local_root``: the one ``--root`` the stages pass stays the v7.6
    root and these patterns resolve against ``SFT_ROOT`` instead
    (``local.effective_root`` is the single choke point).

    The audio overlaps the v7.6 q3asr corpora on purpose -- the training
    config stacks a cleaned and a verbatim target on the same clips -- and
    ``audio_filepath`` is the ledger's identity key, so the shared paths
    compete at assemble and one transcript per clip ships; never two.
    """
    return DatasetSpec(
        name=f"internal_sft_{lang}",
        lang=lang,
        kind=Kind.LOCAL_JSONL,
        license="internal",
        paths=(f"{lang}/*.jsonl*",),
        exclude=_INTERNAL_EXCLUDE,
        fields=_F_CANONICAL,
        local_root=SFT_ROOT,
        max_samples=DEFAULT_MAX_SAMPLES,
        verified=True,
        notes=(
            "Raw (verbatim) q3asr envelopes, duration-complete. Same audio as the "
            "v7.6 cleaned q3asr shards by design; the richer v7.6 transcript "
            "usually wins the shared path, and the corrector stage cleans either. "
            "The tree's eval_<lang>_q3asr.jsonl must reach the leak gates via "
            "--eval-root (see env.sh)."
        ),
    )


_SPECS: tuple[DatasetSpec, ...] = (
    # ============================ ARABIC =====================================
    # 29 train files: 11 named corpora + the sharded and range-sliced q3asr
    # passes (train_manifest in configs/v7.6/internal_ds_sources.yaml).
    _internal("ar", train_files=29, eval_files=2),
    _sft("ar"),
    DatasetSpec(
        name="masc_ar", lang="ar", kind=Kind.HF_STREAM, license="cc-by-4.0",
        est_hours=420.0, repo_id="MohamedRashad/MASC-Arabic", config="default",
        fields=_F_HF_AUDIO, gated=False, verified=True,
        notes="Modern Arabic Speech Corpus. Multi-dialect, which is what the "
              "accent tagger needs; config 'default' confirmed present.",
    ),
    DatasetSpec(
        name="fleurs_ar_eg", lang="ar", kind=Kind.HF_STREAM, license="cc-by-4.0",
        est_hours=12.0, repo_id="google/fleurs", config="ar_eg",
        fields=_F_FLEURS, gated=False, verified=True,
        notes="Small but Egyptian-dialect and clean. duration is absent from the "
              "metadata, so Phase A needs a header probe -- expensive.",
    ),
    DatasetSpec(
        name="cv17_ar", lang="ar", kind=Kind.HF_STREAM, license="cc0-1.0",
        est_hours=180.0, repo_id="mozilla-foundation/common_voice_17_0", config="ar",
        fields=_F_CV, gated=False, verified=False,
        notes="Repo resolves, but cardData declares no configs, so 'ar' is "
              "unconfirmed. Read-accented; useful for accent diversity.",
    ),
    DatasetSpec(
        name="arabic_speech_corpus", lang="ar", kind=Kind.HF_STREAM, license="cc-by-4.0",
        est_hours=7.0, repo_id="halabi2016/arabic_speech_corpus",
        fields=_F_HF_AUDIO, gated=False, verified=True,
        notes="Small MSA read corpus. Low priority; included for script coverage.",
    ),

    # ============================ CHINESE ====================================
    # The zh pool is the binding constraint: the internal v7.6 corpus holds only
    # 153,724 unique paths across 7,851 distinct transcripts -- one batch of
    # dubious value, and below the diversity floor on its own. zh therefore
    # depends almost entirely on the external sources below.
    _internal("zh", train_files=1, eval_files=1),
    _sft("zh"),
    DatasetSpec(
        name="emilia_zh_local", lang="zh", kind=Kind.LOCAL_AUDIO, license="cc-by-4.0",
        est_hours=49_900.0, paths=(f"{EMILIA_ROOT}/ZH",),
        exclude=(), fields=FieldMap(text="text", duration="duration", path="audio_filepath"),
        verified=True,
        notes="ALREADY EXTRACTED -- no download needed, by far the largest zh "
              "source available. Confirm the sidecar field names with "
              "--probe-fields before a full pass; the tar-extraction layout "
              "varies between Emilia releases.",
    ),
    DatasetSpec(
        name="aishell1", lang="zh", kind=Kind.HF_STREAM, license="apache-2.0",
        est_hours=178.0, repo_id="AISHELL/AISHELL-1", fields=_F_HF_AUDIO,
        gated=False, verified=True,
        notes="Mandarin read speech, clean transcripts. Small but high quality.",
    ),
    DatasetSpec(
        name="aishell3", lang="zh", kind=Kind.HF_STREAM, license="apache-2.0",
        est_hours=85.0, repo_id="AISHELL/AISHELL-3", fields=_F_HF_AUDIO,
        gated=False, verified=True,
        notes="Multi-speaker TTS-grade recordings; excellent transcript accuracy.",
    ),
    DatasetSpec(
        name="wenetspeech", lang="zh", kind=Kind.HF_GATED, license="cc-by-4.0",
        est_hours=10_000.0, repo_id="wenet-e2e/wenetspeech", fields=_F_HF_AUDIO,
        gated=True, verified=True,
        notes="Largest open Mandarin corpus. Gated: terms must be accepted on "
              "the Hub and a token supplied. Transcripts are partly "
              "auto-labelled, so the LLM stage matters more here than elsewhere.",
    ),
    DatasetSpec(
        name="fleurs_cmn_hans", lang="zh", kind=Kind.HF_STREAM, license="cc-by-4.0",
        est_hours=12.0, repo_id="google/fleurs", config="cmn_hans_cn",
        fields=_F_FLEURS, gated=False, verified=True,
        notes="Config is cmn_hans_cn, NOT zh_cn. yue_hant_hk also exists and "
              "would feed the Cantonese branch of the accent tagger.",
    ),

    # ============================ ENGLISH ====================================
    # v7.6 ships en as well (inworld, q3asr, hifi_tts, expresso, anispeech,
    # commentary -- see internal_ds_sources.yaml), and the SFT tree adds the
    # raw envelopes; external volume still dominates an en batch.
    _internal("en", train_files=6, eval_files=2),
    _sft("en"),
    DatasetSpec(
        name="peoples_speech", lang="en", kind=Kind.HF_STREAM, license="cc-by-2.0",
        est_hours=30_000.0, repo_id="MLCommons/peoples_speech", fields=_F_HF_AUDIO,
        gated=False, verified=True,
        notes="Largest cc-by English pool. Subset configs exist (clean vs "
              "everything); confirm with --probe-fields and prefer the clean "
              "subset, since the rest is weakly labelled.",
    ),
    DatasetSpec(
        name="gigaspeech", lang="en", kind=Kind.HF_GATED, license="apache-2.0",
        est_hours=10_000.0, repo_id="speechcolab/gigaspeech", fields=_F_HF_AUDIO,
        gated=True, verified=True,
        notes="Audiobook/podcast/YouTube. Carries per-utterance confidence "
              "labels -- filter to the high-confidence subset or the transcript "
              "quality will not survive the gates.",
    ),
    DatasetSpec(
        name="librispeech", lang="en", kind=Kind.HF_STREAM, license="cc-by-4.0",
        est_hours=960.0, repo_id="openslr/librispeech_asr", config="all",
        fields=_F_HF_AUDIO, gated=False, verified=True,
        notes="Read speech, gold transcripts. Configs clean/other/all confirmed. "
              "Small, so it is a quality anchor rather than a volume source.",
    ),
    DatasetSpec(
        name="voxpopuli_en", lang="en", kind=Kind.HF_STREAM, license="cc0-1.0",
        est_hours=540.0, repo_id="facebook/voxpopuli", config="en_accented",
        fields=_F_HF_AUDIO, gated=False, verified=True,
        notes="en_accented is the interesting one: non-native English speakers, "
              "which is exactly the accent diversity the tagger is for.",
    ),

    # ============================ HINDI ======================================
    # v7.6 holds only the cleaned q3asr hi shard (the SFT tree carries hi's
    # eval sets); still far short of a 100k batch without the gated Indic
    # sources below.
    _internal("hi", train_files=1, eval_files=0),
    _sft("hi"),
    DatasetSpec(
        name="shrutilipi_hi", lang="hi", kind=Kind.HF_GATED, license="cc-by-4.0",
        est_hours=6_700.0, repo_id="ai4bharat/Shrutilipi", config="hindi",
        fields=_F_HF_AUDIO, gated=True, verified=True,
        notes="Mined broadcast speech; largest Indic pool by far. Transcripts "
              "come from an ASR system, not humans, so expect the LLM stage to "
              "do real work and the gates to reject heavily.",
    ),
    DatasetSpec(
        name="indicvoices_hi", lang="hi", kind=Kind.HF_GATED, license="cc-by-4.0",
        est_hours=1_000.0, repo_id="ai4bharat/IndicVoices", config="hindi",
        fields=_F_HF_AUDIO, gated=True, verified=True,
        notes="Conversational, human-transcribed. Config 'hindi' confirmed. "
              "Much cleaner than Shrutilipi but far smaller.",
    ),
    DatasetSpec(
        name="kathbath_hi", lang="hi", kind=Kind.HF_GATED, license="cc-by-4.0",
        est_hours=600.0, repo_id="ai4bharat/Kathbath", fields=_F_HF_AUDIO,
        gated=True, verified=True,
        notes="Read + conversational. Repo resolves; the per-language config "
              "name is unconfirmed and must be checked with --probe-fields.",
    ),
    DatasetSpec(
        name="fleurs_hi_in", lang="hi", kind=Kind.HF_STREAM, license="cc-by-4.0",
        est_hours=12.0, repo_id="google/fleurs", config="hi_in",
        fields=_F_FLEURS, gated=False, verified=True,
        notes="Small, clean, human-verified. Best used as an eval anchor.",
    ),

    # ============================ MALAYALAM ==================================
    # ml is the thinnest language: the internal pool holds 258,835 unique paths
    # -- about two batches -- and it was measured carrying two different
    # transcripts for the same audio across its two files.
    _internal("ml", train_files=2, eval_files=1),
    _sft("ml"),
    DatasetSpec(
        name="shrutilipi_ml", lang="ml", kind=Kind.HF_GATED, license="cc-by-4.0",
        est_hours=2_900.0, repo_id="ai4bharat/Shrutilipi", config="malayalam",
        fields=_F_HF_AUDIO, gated=True, verified=True,
        notes="Config 'malayalam' confirmed. The main route to ml volume.",
    ),
    DatasetSpec(
        name="indicvoices_ml", lang="ml", kind=Kind.HF_GATED, license="cc-by-4.0",
        est_hours=120.0, repo_id="ai4bharat/IndicVoices", config="malayalam",
        fields=_F_HF_AUDIO, gated=True, verified=True,
        notes="Config 'malayalam' confirmed. Conversational and human-transcribed.",
    ),
    DatasetSpec(
        name="fleurs_ml_in", lang="ml", kind=Kind.HF_STREAM, license="cc-by-4.0",
        est_hours=9.0, repo_id="google/fleurs", config="ml_in",
        fields=_F_FLEURS, gated=False, verified=True,
        notes="Tiny. Useful for accent-tag calibration, not for volume.",
    ),
    DatasetSpec(
        name="cv17_ml", lang="ml", kind=Kind.HF_STREAM, license="cc0-1.0",
        est_hours=25.0, repo_id="mozilla-foundation/common_voice_17_0", config="ml",
        fields=_F_CV, gated=False, verified=False,
        notes="Config unconfirmed (cardData declares none). Read-accented.",
    ),
)

#: name -> spec, for O(1) lookup and for rejecting duplicate names loudly.
_BY_NAME: dict[str, DatasetSpec] = {}
for _spec in _SPECS:
    if _spec.name in _BY_NAME:
        raise ValueError(f"duplicate spec name in registry: {_spec.name}")
    _BY_NAME[_spec.name] = _spec


def all_specs() -> tuple[DatasetSpec, ...]:
    return _SPECS


def by_name(name: str) -> DatasetSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(_BY_NAME)}") from None


def specs_for(lang: str, *, include_gated: bool = True, external_only: bool = False) -> tuple[DatasetSpec, ...]:
    """Specs for one language, in registry (priority) order.

    ``include_gated=False`` yields a build plan that needs no Hub token and no
    accepted terms -- useful for a first end-to-end run.
    """
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language {lang!r}; expected one of {LANGUAGES}")
    out = []
    for spec in _SPECS:
        if spec.lang != lang:
            continue
        if not include_gated and (spec.gated or spec.kind is Kind.HF_GATED):
            continue
        if external_only and spec.kind in (Kind.LOCAL_JSONL, Kind.LOCAL_AUDIO):
            continue
        out.append(spec)
    return tuple(out)


def iter_specs(langs: Iterator[str] | tuple[str, ...] | None = None, **kw) -> Iterator[DatasetSpec]:
    """Flatten specs across languages, preserving registry order per language."""
    for lang in tuple(langs) if langs is not None else LANGUAGES:
        yield from specs_for(lang, **kw)


def summary() -> dict[str, dict]:
    """Per-language planning view: how much is local, gated, verified."""
    out: dict[str, dict] = {}
    for lang in LANGUAGES:
        specs = specs_for(lang)
        out[lang] = {
            "sources": len(specs),
            "local": sum(1 for s in specs if s.kind in (Kind.LOCAL_JSONL, Kind.LOCAL_AUDIO)),
            "gated": sum(1 for s in specs if s.gated or s.kind is Kind.HF_GATED),
            "unverified": [s.name for s in specs if not s.verified],
            "est_hours": round(sum(s.est_hours for s in specs), 1),
            "est_hours_no_gated": round(
                sum(s.est_hours for s in specs if not (s.gated or s.kind is Kind.HF_GATED)), 1
            ),
        }
    return out


__all__ = [
    "EMILIA_ROOT",
    "INTERNAL_ROOT",
    "LANGUAGES",
    "SFT_ROOT",
    "all_specs",
    "by_name",
    "iter_specs",
    "specs_for",
    "summary",
]
