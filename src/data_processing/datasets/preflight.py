"""Resolve every registry entry before committing to a build.

A wrong ``repo_id`` fails slowly. It surfaces after the image is pulled, after
the Hub request is made, sometimes after a partial download -- and in the worst
case it does not fail at all, it just resolves to a different corpus than the
one the registry intended.

This module makes that failure immediate and textual. It uses only ``urllib``
for the resolution checks, so it runs in an environment without the ``datasets``
library installed; ``--probe-fields`` is the one option that needs it.

The ambiguity that matters
--------------------------
The Hub answers **401** both for "gated, accept the terms" and for "does not
exist". Without a token those are indistinguishable, so a 401 is reported as
unresolved-with-caveat rather than as a hard failure. With ``HF_TOKEN`` set, a
401 becomes a genuine "this identifier is wrong or its terms are unaccepted",
which is actionable. Of 27 identifiers checked while building the registry, 11
failed here -- that is the case this exists for.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path

from .base import DatasetSpec, Kind
from .local import duration_coverage, expand_paths
from .registry import INTERNAL_ROOT, all_specs, specs_for

LOGGER = logging.getLogger("data_processing.datasets.preflight")

API = "https://huggingface.co/api/datasets/{repo}"
TIMEOUT = 30


@dataclass(slots=True)
class CheckResult:
    """Outcome of resolving one spec."""

    spec_name: str
    lang: str
    kind: str
    ok: bool = True
    #: HTTP status from the Hub, or None for local specs.
    status: int | None = None
    gated: bool | None = None
    license: str | None = None
    downloads: int | None = None
    #: Whether the declared config was found in the card metadata.
    config_ok: bool | None = None
    #: Local specs: files matched after exclusions.
    files: int | None = None
    #: Local specs: measured fraction of rows missing a duration.
    duration_missing_fraction: float | None = None
    #: ``--probe-fields``: the row's actual columns.
    observed_fields: tuple[str, ...] = ()
    problems: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.problems.append(msg)

    def warn(self, msg: str) -> None:
        self.problems.append(msg)

    def as_dict(self) -> dict:
        # slots=True leaves instances without a __dict__; walk the declared
        # fields instead (the JSON report path is the only consumer).
        return {
            f.name: v for f in fields(self)
            if (v := getattr(self, f.name)) not in (None, (), [], "")
        }


def _get(url: str, token: str | None, timeout: int = TIMEOUT) -> tuple[int, dict | None]:
    """GET JSON. Returns ``(status, body)``; body is None on any error."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        LOGGER.debug("%s: %s", url, exc)
        return -1, None


def _declared_configs(body: dict) -> list[str]:
    """Config names from the card metadata. Empty means undeclared, not absent."""
    card = body.get("cardData") or {}
    configs = card.get("configs")
    if not isinstance(configs, list):
        return []
    return [str(c.get("config_name") or c.get("name")) for c in configs if isinstance(c, dict)]


def check_hub(spec: DatasetSpec, token: str | None = None) -> CheckResult:
    """Resolve a Hub spec: does it exist, is it gated, does the config exist?"""
    res = CheckResult(spec_name=spec.name, lang=spec.lang, kind=spec.kind.value)
    if not spec.repo_id:
        res.fail("kind is Hub but repo_id is unset")
        return res

    status, body = _get(API.format(repo=spec.repo_id), token)
    res.status = status

    if status == 200 and body is not None:
        gated = body.get("gated")
        res.gated = gated in (True, "auto", "manual")
        res.downloads = body.get("downloads")
        licenses = [t.split(":", 1)[1] for t in (body.get("tags") or []) if t.startswith("license:")]
        res.license = licenses[0] if licenses else None

        declared = _declared_configs(body)
        if spec.config:
            if declared:
                res.config_ok = spec.config in declared
                if not res.config_ok:
                    res.fail(
                        f"config {spec.config!r} is not among the {len(declared)} declared; "
                        f"close matches: {[c for c in declared if spec.config[:3] in c][:6]}"
                    )
            else:
                # CardData often omits the config list entirely, so absence of
                # evidence is not evidence of absence -- --probe-fields is what
                # confirms the config name for real.
                res.config_ok = None
                res.warn("cardData declares no configs; config name is unconfirmed until --probe-fields")
        if res.gated and not token:
            res.warn("gated and HF_TOKEN is unset: accept terms on the Hub, then export HF_TOKEN")
        if not res.license:
            res.warn("no license tag on the Hub; confirm redistribution terms before publishing")
        return res

    if status == 401:
        if token:
            res.fail(f"401 WITH a token: {spec.repo_id} does not exist or its terms are unaccepted")
        else:
            res.fail(
                f"401 without a token: {spec.repo_id} is gated OR does not exist. "
                "Set HF_TOKEN to disambiguate."
            )
    elif status == 404:
        res.fail(f"404: no dataset {spec.repo_id}")
    elif status == 307 or status == 301:
        res.fail(f"{status} redirect: {spec.repo_id} was renamed; resolve the canonical id")
    elif status == -1:
        res.fail("network error or timeout; cannot resolve (offline?)")
    else:
        res.fail(f"unexpected HTTP {status} for {spec.repo_id}")
    return res


def check_local(spec: DatasetSpec, root: str | None = None, sample: int = 5_000) -> CheckResult:
    """Verify a local spec: do the paths match anything after exclusions?"""
    res = CheckResult(spec_name=spec.name, lang=spec.lang, kind=spec.kind.value)
    files = expand_paths(spec, root)
    res.files = len(files)
    if not files:
        res.fail(f"no files matched {spec.paths} under root={root!r} after exclusions {spec.exclude}")
        return res

    excluded = 0
    for pattern in spec.paths:
        p = Path(pattern)
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
        parent, name = (p.parent, p.name) if any(c in pattern for c in "*?[") else (p, "*")
        if parent.is_dir():
            excluded += sum(1 for m in parent.glob(name) if m.is_file()) - 0
    if excluded and excluded > len(files):
        res.warn(f"{excluded - len(files):,} file(s) removed by exclusions {spec.exclude}")

    # Duration coverage decides whether Phase A stays metadata-only. A source
    # that omits duration forces a header probe, i.e. an audio fetch.
    try:
        cov = duration_coverage(spec, root, sample=sample)
        res.duration_missing_fraction = cov["missing_fraction"]
        if cov["missing_fraction"] > 0.10:
            res.warn(
                f"{cov['missing_fraction']:.1%} of sampled rows have no duration; "
                "Phase A will need header probes (audio fetches) for those"
            )
    except (OSError, ValueError) as exc:
        res.warn(f"duration coverage could not be measured: {exc}")
    return res


def probe_fields(spec: DatasetSpec, root: str | None = None, token: str | None = None) -> CheckResult:
    """Stream one row and report its real columns against the declared FieldMap.

    The only check that can catch a wrong column name, and the cheapest way to
    find one: a single row, no audio decoded.
    """
    from .stream import stream_metadata  # deferred: needs `datasets` for Hub specs

    res = CheckResult(spec_name=spec.name, lang=spec.lang, kind=spec.kind.value)
    try:
        row = next(stream_metadata(spec, root, token=token), None)
    except Exception as exc:
        # Broad on purpose: a dead source must fail THIS spec in the report,
        # never abort the run. EmptyDatasetError(FileNotFoundError) -- raised
        # when a repo is withdrawn from the Hub mid-campaign -- is an OSError,
        # outside any RuntimeError/ValueError tuple, and it once killed a
        # whole 9-node preflight before a single line of report was printed.
        res.fail(f"could not stream a row: {type(exc).__name__}: {exc}")
        return res
    if row is None:
        res.fail("stream produced no rows")
        return res

    res.observed_fields = tuple(sorted(row.keys()))
    fm = spec.fields
    for label, column in (("text", fm.text), ("path", fm.path), ("duration", fm.duration)):
        if column is None:
            continue
        if column not in row:
            res.fail(f"FieldMap.{label}={column!r} is absent from the row")
    if spec.duration_of(row) is None and fm.duration is not None:
        res.warn(f"{fm.duration!r} exists but was null/unparseable in the first row")
    if spec.audio_filepath(row) is None:
        res.fail(f"no audio_filepath derivable from the first row (tried {fm.path!r} and 'audio')")
    if not res.problems:
        LOGGER.info("%s: fields confirmed -> %s", spec.name, res.observed_fields)
    return res


def run(
    specs: tuple[DatasetSpec, ...] | None = None,
    root: str | None = None,
    token: str | None = None,
    probe: bool = False,
    sample: int = 5_000,
) -> list[CheckResult]:
    """Check every spec. Local checks run first: they are free and offline.

    ``root`` defaults to :data:`registry.INTERNAL_ROOT` because the internal
    specs carry patterns relative to it. Passing None used to mean "resolve
    against the CWD", which silently matched nothing and reported every local
    source as missing -- a confusing failure for what is a one-argument mistake.
    """
    specs = specs if specs is not None else all_specs()
    token = token or os.environ.get("HF_TOKEN")
    root = INTERNAL_ROOT if root is None else root
    out: list[CheckResult] = []
    for spec in specs:
        try:
            if spec.kind in (Kind.LOCAL_JSONL, Kind.LOCAL_AUDIO):
                res = check_local(spec, root, sample)
            else:
                res = check_hub(spec, token)
        except Exception as exc:
            # Same principle as in probe_fields: report, never abort. A check
            # that explodes (a weird mount, a parser bug) is one FAIL row.
            res = CheckResult(spec_name=spec.name, lang=spec.lang, kind=spec.kind.value)
            res.fail(f"check crashed: {type(exc).__name__}: {exc}")
        out.append(res)
        mark = "ok  " if res.ok else "FAIL"
        LOGGER.info("%s %-24s %s", mark, spec.name, "; ".join(res.problems) or "-")
    if probe:
        for i, spec in enumerate(specs):
            try:
                out[i] = probe_fields(spec, root, token)
            except Exception as exc:
                # probe_fields is defensive but not total; keep the check
                # result and mark it failed rather than losing the report.
                out[i].fail(f"probe crashed: {type(exc).__name__}: {exc}")
    return out


def format_report(results: list[CheckResult]) -> str:
    """Human-readable table, failures first."""
    lines = [
        f"{'status':6} {'spec':26} {'lang':5} {'kind':12} {'gated':6} {'license':14} notes",
        "-" * 130,
    ]
    for r in sorted(results, key=lambda r: (r.ok, r.lang, r.spec_name)):
        notes = "; ".join(r.problems) or (f"{r.files:,} file(s)" if r.files else "")
        lines.append(
            f"{'ok' if r.ok else 'FAIL':6} {r.spec_name:26} {r.lang:5} {r.kind:12} "
            f"{r.gated if r.gated is not None else '-'!s:6} "
            f"{(r.license or '-')[:14]:14} {notes[:70]}"
        )
    failed = [r for r in results if not r.ok]
    warned = [r for r in results if r.ok and r.problems]
    lines += [
        "-" * 130,
        f"{len(results)} checked   {len(results) - len(failed)} resolved   "
        f"{len(failed)} FAILED   {len(warned)} with warnings",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="preflight",
        description="Resolve registry entries before an ingest. Exit 1 if any fail.",
    )
    parser.add_argument("--lang", action="append", choices=["ar", "en", "zh", "hi", "ml"],
                        help="restrict to a language (repeatable)")
    parser.add_argument("--root", default=INTERNAL_ROOT,
                        help=f"root for relative local paths (default {INTERNAL_ROOT})")
    parser.add_argument("--probe-fields", action="store_true",
                        help="stream one row per source and verify column names")
    parser.add_argument("--sample", type=int, default=5_000,
                        help="rows sampled for duration coverage (default 5000)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    specs = tuple(
        s for lang in (args.lang or []) for s in specs_for(lang)
    ) or None
    results = run(specs, root=args.root, probe=args.probe_fields, sample=args.sample)

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2, ensure_ascii=False))
    else:
        print(format_report(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CheckResult", "check_hub", "check_local", "format_report", "main", "probe_fields", "run"]
