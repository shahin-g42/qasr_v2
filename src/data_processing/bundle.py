"""Stage 4: group the five languages' sequential batches into shippable bundles.

A **bundle N** = ``{ar,en,zh,hi,ml}/bN``, each language exactly
``batch_size`` (100 000) samples. Bundles are numbered ``0..K-1`` where
``K = min`` full-batch count across the five languages, so a bundle ships
only when all five languages have a full ``bN``. A batch below the contract
(a drain-mode trailing shard) is never shipped; a language's surplus full
batches ship in later runs once the laggard languages catch up -- batch
indices continue via the per-language ledger, so no renumbering ever happens.

This stage owns two things:

* **``MANIFEST.json``** -- the shipping document. Per language and batch:
  manifest/sidecar paths (referenced, never copied), row count, hours,
  per-source and per-accent composition, external/internal split, the
  recomputed distinct-transcript fraction, and the quality/richness/diacritics
  means. All statistics are aggregated from the sidecars Stage 2 wrote; no
  re-gating, no re-correction.
* **The audit** -- hard gates whose failure exits 1 so a pipeline can gate on
  this job's exit code: count contract, canonical record shape, no duplicate
  ``audio_filepath`` within or across languages, zero eval leak against the
  v7.6 eval sets, every external clip present at 16 kHz mono, the diversity
  floor recomputed from the manifest texts, and positive durations. Source-cap
  shares and duration bands are reported, not enforced (they fed ranking, not
  admission, by design).

A failed audit never overwrites an existing ``MANIFEST.json``: the manifest
describes what shipped, and what shipped is exactly what passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from .canonical import Sample, _atomic_write, _open_read, iter_shards
from .datasets import registry
from .dedup import key_hash
from .distribute import SeenLedger
from .materialize import DEFAULT_SAMPLE_RATE, verify

LOGGER = logging.getLogger("data_processing.bundle")

#: A batch-labelled manifest: ``train_<lang>_b####_p####.jsonl[.gz]``. Legacy
#: dataset-labelled shards (``train_ar_ar_ae_p0000``) do not match and are
#: ignored: only the staged build's sequential batches are bundle candidates.
_BATCH_RE = re.compile(
    r"^train_(?P<lang>[a-z]{2})_(?P<label>b\d{4})_p\d{4}\.jsonl(?:\.gz)?$"
)

#: Samples per language per batch. The contract the whole build enforces.
DEFAULT_BATCH_SIZE = 100_000

#: Duration band reported (not enforced) by the audit, mirroring the Stage-1
#: gate defaults: duration gates are off by default upstream, so an out-of-band
#: clip is a shipping-report fact, not a failure.
_INFO_DURATION_BAND = (0.5, 30.0)


# --- discovery ---------------------------------------------------------------
def discover_batches(out_dir: str | Path, lang: str) -> dict[str, list[Path]]:
    """``label -> manifest paths`` for this language's sequential batches."""
    out: dict[str, list[Path]] = {}
    for manifest, _sidecar in iter_shards(Path(out_dir) / lang, lang=lang):
        m = _BATCH_RE.match(manifest.name)
        if m and m.group("lang") == lang:
            out.setdefault(m.group("label"), []).append(manifest)
    return out


def _count_rows(path: Path) -> int:
    """Non-blank line count -- rows without paying for JSON parsing."""
    with _open_read(path) as handle:
        return sum(1 for line in handle if line.strip())


def _sidecar_of(manifest: Path) -> Path:
    return manifest.with_name(manifest.name.replace(".jsonl", ".meta.jsonl"))


def _eval_paths(root: str | Path) -> set[str]:
    """Every ``audio_filepath`` in any eval manifest under ``root``.

    Loads all languages, not just the shipped ones: a shipped path leaking
    into *any* eval set is a leak. Uses the tolerant single-field reader
    (``SeenLedger._iter_paths``) because real v7.6 eval files do not meet the
    strict four-key output contract -- the same reason the exclusion loader
    parses for exactly one field.
    """
    out: set[str] = set()
    root = Path(root)
    if not root.is_dir():
        LOGGER.warning("eval root %s is not a directory; leak audit has no exclusions", root)
        return out
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for manifest in sorted(lang_dir.glob("eval_*.jsonl*")):
            try:
                out.update(SeenLedger._iter_paths(manifest))
            except OSError as exc:
                LOGGER.warning("could not read eval manifest %s: %s", manifest, exc)
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentiles(values: list[float]) -> dict[str, float]:
    """p50/p95/max, guarded for the tiny batches tests use."""
    if not values:
        return {}
    out: dict[str, float] = {"p50": round(statistics.median(values), 3),
                             "max": round(max(values), 3)}
    if len(values) >= 2:
        out["p95"] = round(statistics.quantiles(values, n=20)[-1], 3)
    return out


# --- one batch: stream manifest + sidecar in lockstep ------------------------
def _audit_batch(
    lang: str,
    label: str,
    manifests: list[Path],
    *,
    audio_prefix: str,
    sample_rate: int,
    eval_paths: set[str],
    seen_paths: set[str],
    spot_check: int,
    batch_size: int,
    min_distinct_text_fraction: float,
    max_per_source_fraction: float,
) -> tuple[dict, list[str], list[str]]:
    """Stream one shipped batch once, collecting stats and audit verdicts.

    The manifest and sidecar are read in lockstep (they were written 1:1, in
    the same order); an ``audio_filepath`` disagreement between the two files
    is itself a hard failure -- it means the sidecar's provenance describes a
    different sample than the manifest row carries.

    Returns ``(stats, failures, warnings)``; ``failures`` non-empty means the
    batch (and therefore the bundle) must not ship.
    """
    failures: list[str] = []
    warnings: list[str] = []
    rows = 0
    meta_rows = 0
    hours = 0.0
    sources: Counter = Counter()
    accents: Counter = Counter()
    external_paths: list[str] = []
    durations: list[float] = []
    hashes: set[int] = set()
    quality_vals: list[float] = []
    richness_vals: list[float] = []
    diac_vals: list[float] = []
    eval_leaks = 0
    dup_paths = 0

    for manifest in manifests:
        sidecar = _sidecar_of(manifest)
        if not sidecar.exists():
            failures.append(f"{manifest.name}: sidecar missing: {sidecar.name}")
            continue
        with _open_read(manifest) as mfh, _open_read(sidecar) as sfh:
            # strict=False on purpose: a sidecar shorter than its manifest is a
            # reportable count-contract failure, not a reason to crash here.
            for mline, sline in zip(mfh, sfh, strict=False):
                if not mline.strip():
                    continue
                try:
                    sample = Sample.from_dict(json.loads(mline))
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    failures.append(f"{manifest.name}: non-canonical record: {exc}")
                    continue
                rows += 1
                if sample.audio_filepath in seen_paths:
                    dup_paths += 1
                seen_paths.add(sample.audio_filepath)
                if sample.audio_filepath in eval_paths:
                    eval_leaks += 1
                if sample.duration <= 0:
                    failures.append(f"non-positive duration ({sample.duration}) for "
                                    f"{sample.audio_filepath}")
                durations.append(sample.duration)
                # Seconds on disk, hours in the manifest -- same convention as
                # distribute.Batch.hours(), so the shipping document a human
                # reads and the stage-2 report a human reads agree.
                hours += sample.duration / 3600.0
                hashes.add(key_hash(sample.text, lang))
                if sample.audio_filepath.startswith(audio_prefix):
                    external_paths.append(sample.audio_filepath)

                if sline.strip():
                    meta_rows += 1
                    meta = json.loads(sline)
                    if meta.get("audio_filepath") != sample.audio_filepath:
                        failures.append(
                            f"{manifest.name}: sidecar out of sync at row {rows}: "
                            f"{meta.get('audio_filepath')!r} != {sample.audio_filepath!r}")
                    sources[str(meta.get("dataset") or "unknown")] += 1
                    accents[str(meta.get("accent") or "unknown")] += 1
                    if meta.get("quality") is not None:
                        quality_vals.append(float(meta["quality"]))
                    metrics = meta.get("metrics") or {}
                    if metrics.get("richness") is not None:
                        richness_vals.append(float(metrics["richness"]))
                    if meta.get("diacritics_ratio") is not None:
                        diac_vals.append(float(meta["diacritics_ratio"]))

    if meta_rows != rows:
        failures.append(f"sidecar covers {meta_rows} of {rows} manifest rows")
    if rows != batch_size:
        failures.append(f"count contract broken: {rows} of {batch_size} rows")
    if dup_paths:
        failures.append(f"{dup_paths} duplicate audio_filepath (intra- or cross-language)")
    if eval_leaks:
        failures.append(f"{eval_leaks} eval-leaked path(s)")

    distinct_fraction = (len(hashes) / rows) if rows else 0.0
    if rows and distinct_fraction < min_distinct_text_fraction:
        failures.append(
            f"distinct-transcript fraction {distinct_fraction:.3f} < "
            f"{min_distinct_text_fraction:.3f} ({len(hashes):,} of {rows:,})")

    # External audio: full verification by default; ``spot_check`` samples
    # deterministically for a fast re-run mode. Every checked path must be
    # present, mono, at the target rate.
    checked = external_paths
    if spot_check and len(external_paths) > spot_check:
        checked = random.Random(0).sample(external_paths, spot_check)
    bad_audio = [p for p in checked if not verify(p, sample_rate)]
    if bad_audio:
        failures.append(f"{len(bad_audio)} external clip(s) missing or not "
                        f"{sample_rate} Hz mono (e.g. {bad_audio[0]})")

    # Informational: the source cap fed admission upstream, so a breach here is
    # a distribution surprise worth seeing, not a reason to refuse the batch.
    cap = max_per_source_fraction + 1e-9
    for source, count in sources.most_common():
        if rows and count / rows > cap:
            warnings.append(f"source {source!r} holds {count / rows:.1%} of the batch "
                            f"(cap {max_per_source_fraction:.0%})")
    lo, hi = _INFO_DURATION_BAND
    out_of_band = sum(1 for d in durations if d < lo or d > hi)
    if out_of_band:
        warnings.append(f"{out_of_band} of {rows} clips outside the "
                        f"[{lo}, {hi}] s duration band")

    stats = {
        "manifest": str(manifests[0]),
        "sidecar": str(_sidecar_of(manifests[0])),
        "rows": rows,
        "hours": round(hours, 2),
        "sources": dict(sources.most_common()),
        "external_rows": len(external_paths),
        "internal_rows": rows - len(external_paths),
        "accents": dict(accents.most_common()),
        "distinct_text_fraction": round(distinct_fraction, 4),
        "mean_quality": _mean(quality_vals),
        "mean_richness": _mean(richness_vals),
        "diacritics_ratio_mean": _mean(diac_vals),
        "durations": _percentiles(durations),
    }
    return stats, failures, warnings


# --- orchestration -----------------------------------------------------------
def run_bundle(
    *,
    out_dir: str | Path,
    audio_root: str | Path,
    root: str | None = None,
    langs: tuple[str, ...] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    spot_check: int = 0,
    min_distinct_text_fraction: float = 0.50,
    max_per_source_fraction: float = 0.40,
    dry_run: bool = False,
) -> dict:
    """Discover, audit and manifest the shippable bundles. Returns the report.

    ``report["audit"]["ok"]`` is the verdict; ``main`` maps it to the exit
    code. A dry run audits and reports but writes no ``MANIFEST.json``.
    """
    langs = tuple(langs) if langs else registry.LANGUAGES
    root = root if root is not None else registry.INTERNAL_ROOT
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Discovery + row counts (line counts; the audit re-checks with full
    # parsing, which is where a corrupt record actually fails).
    batches_by_lang = {lang: discover_batches(out_dir, lang) for lang in langs}
    rows_by_lang = {
        lang: {label: sum(_count_rows(m) for m in manifests)
               for label, manifests in batches.items()}
        for lang, batches in batches_by_lang.items()
    }
    full = {lang: sorted(label for label, n in counts.items() if n == batch_size)
            for lang, counts in rows_by_lang.items()}
    short = {lang: {label: n for label, n in counts.items() if n != batch_size}
             for lang, counts in rows_by_lang.items()}
    for lang in langs:
        LOGGER.info("%s: %d full batch(es) %s%s", lang, len(full[lang]),
                    full[lang][:4], " ..." if len(full[lang]) > 4 else "")

    k = min((len(full[lang]) for lang in langs), default=0)
    shippable = {lang: full[lang][:k] for lang in langs}
    surplus = {lang: full[lang][k:] for lang in langs}
    if any(surplus.values()):
        LOGGER.info("surplus full batches (ship once laggards catch up): %s",
                    {lang: labels for lang, labels in surplus.items() if labels})

    failures: list[str] = []
    warnings: list[str] = []
    if k == 0:
        failures.append(
            "no shippable bundle: full-batch counts are "
            + ", ".join(f"{lang}={len(full[lang])}" for lang in langs))

    eval_paths = _eval_paths(root)
    seen_paths: set[str] = set()
    audio_prefix = str(audio_root).rstrip("/") + "/"
    entries: dict[str, dict[str, dict]] = {lang: {} for lang in langs}

    for lang in langs:
        for label in shippable[lang]:
            stats, batch_failures, batch_warnings = _audit_batch(
                lang, label, batches_by_lang[lang][label],
                audio_prefix=audio_prefix, sample_rate=sample_rate,
                eval_paths=eval_paths, seen_paths=seen_paths,
                spot_check=spot_check, batch_size=batch_size,
                min_distinct_text_fraction=min_distinct_text_fraction,
                max_per_source_fraction=max_per_source_fraction,
            )
            entries[lang][label] = stats
            failures.extend(f"{lang}/{label}: {f}" for f in batch_failures)
            warnings.extend(f"{lang}/{label}: {w}" for w in batch_warnings)
            verdict = "FAIL" if batch_failures else "ok"
            LOGGER.info("%s/%s audited: %s (%s rows, %.1f h, distinct %.3f)",
                        lang, label, verdict, stats["rows"], stats["hours"],
                        stats["distinct_text_fraction"])

    ok = not failures
    manifest_path = Path(out_dir) / "MANIFEST.json"
    manifest_written = False
    if ok and k > 0 and not dry_run:
        payload = {
            "version": "v8.0",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generator": "data_processing.bundle/1",
            "audio_root": str(Path(audio_root).resolve()),
            "sample_rate_policy": f"internal=native, external={sample_rate} mono flac",
            "bundles": k,
            "bundle_size": {"per_language": batch_size, "languages": list(langs)},
            "batch_labels": dict(shippable),
            "languages": entries,
        }

        def emit(handle) -> None:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        _atomic_write(manifest_path, emit)
        manifest_written = True
        LOGGER.info("MANIFEST.json written: %d bundle(s), %d sample(s) per language",
                    k, batch_size)
    elif not ok:
        LOGGER.error("audit FAILED (%d failure(s)); MANIFEST.json not written", len(failures))

    return {
        "stage": "bundle", "started": started, "out_dir": str(out_dir),
        "audio_root": str(audio_root), "root": str(root), "langs": list(langs),
        "batch_size": batch_size, "k": k,
        "full_batches": {lang: len(full[lang]) for lang in langs},
        "short_batches": {lang: short[lang] for lang in langs if short[lang]},
        "surplus_full_batches": {lang: surplus[lang] for lang in langs if surplus[lang]},
        "shippable": {lang: shippable[lang] for lang in langs},
        "audit": {
            "ok": ok,
            "failure_count": len(failures),
            "failures": failures[:50],
            "warnings": warnings[:50],
            "spot_check": spot_check,
            "eval_paths_loaded": len(eval_paths),
        },
        "manifest_written": manifest_written,
        "manifest_path": str(manifest_path),
        "dry_run": dry_run,
        "languages": entries,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bundle",
        description="Stage 4: group the languages' bN batches into bundles + audit.",
    )
    ap.add_argument("--out-dir", required=True, help="batch manifest root (holds <lang>/)")
    ap.add_argument("--audio-root", required=True,
                    help="root external clips were materialized under")
    ap.add_argument("--root", default=registry.INTERNAL_ROOT,
                    help="eval-manifest root for the leak audit")
    ap.add_argument("--langs", help=f"comma-separated (default {','.join(registry.LANGUAGES)})")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    ap.add_argument("--spot-check", type=int, default=0,
                    help="verify only N random external clips (0 = full pass)")
    ap.add_argument("--min-distinct-text-fraction", type=float, default=0.50)
    ap.add_argument("--max-per-source-fraction", type=float, default=0.40)
    ap.add_argument("--report", help="write the run report JSON here")
    ap.add_argument("--dry-run", action="store_true",
                    help="audit and report; write no MANIFEST.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    langs = tuple(x.strip() for x in args.langs.split(",") if x.strip()) \
        if args.langs else registry.LANGUAGES
    report = run_bundle(
        out_dir=args.out_dir, audio_root=args.audio_root, root=args.root,
        langs=langs, batch_size=args.batch_size, sample_rate=args.sample_rate,
        spot_check=args.spot_check,
        min_distinct_text_fraction=args.min_distinct_text_fraction,
        max_per_source_fraction=args.max_per_source_fraction,
        dry_run=args.dry_run,
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        LOGGER.info("report written to %s", args.report)
    else:
        print(json.dumps({k: report[k] for k in
                          ("k", "full_batches", "shippable", "audit", "manifest_written")},
                         indent=2, ensure_ascii=False, default=str))
    return 0 if report["audit"]["ok"] else 1


__all__ = ["DEFAULT_BATCH_SIZE", "discover_batches", "main", "run_bundle"]


if __name__ == "__main__":
    sys.exit(main())
