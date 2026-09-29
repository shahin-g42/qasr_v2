#!/usr/bin/env python3
"""Verify BOTH internal trees on disk against configs/corpus/internal_ingest.yaml.

The ingest contract is the contract: its ``train_manifest`` names every file
the corpus build may ingest, and its ``eval_manifest`` names every file that
must stay out of the pools and visible to the leak gate. The registry mirrors
that list in code (``internal_v76_<lang>`` / ``internal_sft_<lang>``) and a
unit test pins the registry against the YAML -- but no test can see the
cluster's actual tree, so a build could still ingest a file the config never
named (a stray download, a reprocess leftover) or silently skip one the config
requires, and nothing would say so until training.

This is the runnable half of that contract. It resolves the two trees the way
the corpus drivers will use them -- ``--internal-root`` / ``--sft-root``,
else ``QASR_INTERNAL_ROOT`` (or ``INTERNAL_ROOT``) / ``QASR_SFT_ROOT``, else
the config's own absolute paths -- and reports, per language and tree:

* config ``train_manifest`` files that are NOT ingestible: absent from the
  tree, caught by the spec's exclusion tokens, or not matched by its glob;
* on-disk files that WOULD be ingested but are NOT named in the config
  (unlisted data entering training silently);
* config ``eval_manifest`` files that are not separated: not excluded from
  the spec's ingest, or invisible to the ``eval_*.jsonl*`` glob that
  ``distribute.load_eval_exclusions`` and the stage-4 audit scan.

    python3 scripts/corpus/check_internal_sources.py           # human report
    python3 scripts/corpus/check_internal_sources.py --json    # machine report

Exit codes: 0 = trees match the config; 1 = findings (listed on stderr);
2 = cannot verify (the config or a tree is unreachable -- expected off the
cluster, where the trees are not mounted).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data_processing.datasets import registry  # noqa: E402
from data_processing.datasets.local import expand_paths, is_excluded  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "configs" / "corpus" / "internal_ingest.yaml"

#: What the leak gate globs: ``distribute.load_eval_exclusions`` and
#: ``bundle._eval_paths`` both scan ``<root>/<lang>/eval_*.jsonl*``. An eval
#: file whose name does not match this pattern is invisible to every path that
#: learns "held out" from the disk.
GATE_GLOB = "eval_*.jsonl*"

#: tree key -> registry spec name template, checked in this order.
_TREES: dict[str, str] = {
    "v76": "internal_v76_{lang}",
    "sft": "internal_sft_{lang}",
}

_ROOT_FLAGS: dict[str, str] = {"v76": "--internal-root", "sft": "--sft-root"}

#: Env fallbacks per tree, in precedence order: the registry's names first,
#: then the driver's (scripts/corpus/env.sh exports INTERNAL_ROOT /
#: QASR_SFT_ROOT).
_ENV_ROOTS: dict[str, tuple[str, ...]] = {
    "v76": ("QASR_INTERNAL_ROOT", "INTERNAL_ROOT"),
    "sft": ("QASR_SFT_ROOT",),
}


def _tree_of(path: Path) -> str | None:
    """Which internal tree a config path belongs to, by its directory name."""
    if "q3asr_sft_manifests" in path.parts:
        return "sft"
    if "training_manifests" in path.parts:
        return "v76"
    return None


def _routed(cfg: dict, key: str, tree: str) -> list[Path]:
    """Every path one config list (train_manifest / eval_manifest) files under a tree."""
    return [
        Path(f)
        for files in (cfg.get(key) or {}).values()
        for f in files
        if _tree_of(Path(f)) == tree
    ]


def _config_tree(cfg: dict, tree: str) -> Path:
    """The tree root the config itself implies: the parent of the ``<lang>/`` dirs."""
    parents = {
        p.parents[1]
        for p in _routed(cfg, "train_manifest", tree) + _routed(cfg, "eval_manifest", tree)
    }
    if not parents:
        raise ValueError(
            f"cannot locate the {tree} tree: the config names no file under it "
            f"and no {_ROOT_FLAGS[tree]} was given"
        )
    # A config spanning several directories is a finding (see audit); take the
    # first so the rest of the report stays deterministic.
    return sorted(parents)[0]


def _env_root(tree: str) -> Path | None:
    for name in _ENV_ROOTS[tree]:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


def _resolve_roots(
    cfg: dict, internal_root: Path | None, sft_root: Path | None
) -> dict[str, Path]:
    """Flag > env > the config's own paths, then require both trees to exist."""
    overrides = {"v76": internal_root, "sft": sft_root}
    roots: dict[str, Path] = {}
    for tree in _TREES:
        roots[tree] = overrides[tree] or _env_root(tree) or _config_tree(cfg, tree)
    for tree, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(
                f"{tree} tree not found at {root} -- mount it or pass {_ROOT_FLAGS[tree]}"
            )
    return roots


def _missing_reason(
    root: Path, lang: str, name: str, exclude: tuple[str, ...], patterns: tuple[str, ...]
) -> str:
    """Why a config-named train file is not ingestible, in the spec's own terms."""
    path = root / lang / name
    if is_excluded(path, exclude):
        return f"excluded by the spec's tokens {exclude!r}"
    if not path.is_file():
        return "not on disk in this tree"
    return f"on disk but not matched by the spec's pattern(s) {patterns!r}"


def audit(
    config_path: Path | str,
    *,
    internal_root: Path | str | None = None,
    sft_root: Path | str | None = None,
) -> dict:
    """Check both trees against the config; findings never raise.

    Raises OSError / ValueError only for "cannot verify" conditions: an
    unreadable config or a tree that is not mounted.
    """
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: not a YAML mapping")
    train: dict = cfg.get("train_manifest") or {}
    evals: dict = cfg.get("eval_manifest") or {}

    problems: list[str] = []
    # The plan is the registry's LANGUAGES: a config that disagrees either way
    # means some language is planned but unnamed, or named but unspec'd.
    for key, listed in (("train_manifest", train), ("eval_manifest", evals)):
        for lang in sorted(set(registry.LANGUAGES) - set(listed)):
            problems.append(
                f"{key} has no entry for {lang!r} (the registry plans every language)"
            )
        for lang in sorted(set(listed) - set(registry.LANGUAGES)):
            problems.append(
                f"{key} names unknown language {lang!r} (not in the registry's LANGUAGES)"
            )
        for lang, files in listed.items():
            for f in files:
                path = Path(f)
                if _tree_of(path) is None:
                    problems.append(
                        f"{key}[{lang}]: not under either internal tree "
                        f"(training_manifests/ or q3asr_sft_manifests/): {f}"
                    )
                elif path.parent.name != lang:
                    problems.append(
                        f"{key}[{lang}]: filed under {path.parent.name!r}, not {lang!r}: {f}"
                    )
    for tree in _TREES:
        parents = {
            p.parents[1]
            for p in _routed(cfg, "train_manifest", tree) + _routed(cfg, "eval_manifest", tree)
        }
        if len(parents) > 1:
            problems.append(
                f"config {tree} entries span multiple directories: "
                + ", ".join(sorted(str(p) for p in parents))
            )

    roots = _resolve_roots(
        cfg,
        Path(internal_root) if internal_root is not None else None,
        Path(sft_root) if sft_root is not None else None,
    )

    sources: list[dict] = []
    for lang in sorted(set(train) | set(evals) | set(registry.LANGUAGES)):
        for tree, template in _TREES.items():
            spec_name = template.format(lang=lang)
            try:
                spec = registry.by_name(spec_name)
            except KeyError:
                spec = None
                problems.append(
                    f"config names {lang}/{tree} files but the registry has no {spec_name!r} spec"
                )
            root = roots[tree]
            train_files = [Path(f) for f in train.get(lang, []) if _tree_of(Path(f)) == tree]
            eval_files = [Path(f) for f in evals.get(lang, []) if _tree_of(Path(f)) == tree]
            matched = (
                {p.name for p in expand_paths(replace(spec, local_root=str(root)), root)}
                if spec is not None
                else set()
            )
            listed = {p.name for p in train_files}

            missing = sorted(listed - matched)
            extra = sorted(matched - listed)
            reasons = {
                name: (
                    _missing_reason(root, lang, name, spec.exclude, spec.paths)
                    if spec is not None
                    else "no registry spec to ingest it"
                )
                for name in missing
            }

            eval_ingested: list[str] = []
            eval_gate_invisible: list[str] = []
            eval_missing: list[str] = []
            for path in eval_files:
                if not (root / lang / path.name).is_file():
                    eval_missing.append(path.name)
                    continue
                if spec is not None and not is_excluded(path, spec.exclude):
                    eval_ingested.append(path.name)
                if not fnmatch.fnmatch(path.name, GATE_GLOB):
                    eval_gate_invisible.append(path.name)

            src_problems: list[str] = []
            by_reason: dict[str, list[str]] = {}
            for name in missing:
                by_reason.setdefault(reasons[name], []).append(name)
            for reason, names in sorted(by_reason.items()):
                src_problems.append(
                    f"{lang}/{tree}: {len(names)} config train file(s) not ingestible -- "
                    f"{reason}: {', '.join(names)}"
                )
            if extra:
                src_problems.append(
                    f"{lang}/{tree}: {len(extra)} on-disk train file(s) not named in the "
                    f"config (they would be ingested unlisted): {', '.join(extra)}"
                )
            if eval_ingested:
                src_problems.append(
                    f"{lang}/{tree}: {len(eval_ingested)} config eval file(s) NOT excluded "
                    f"from the spec (they would be ingested): {', '.join(eval_ingested)}"
                )
            if eval_gate_invisible:
                src_problems.append(
                    f"{lang}/{tree}: {len(eval_gate_invisible)} config eval file(s) invisible "
                    f"to the {GATE_GLOB!r} leak gate: {', '.join(eval_gate_invisible)}"
                )
            if eval_missing:
                src_problems.append(
                    f"{lang}/{tree}: {len(eval_missing)} config eval file(s) missing on disk: "
                    f"{', '.join(eval_missing)}"
                )

            sources.append(
                {
                    "lang": lang,
                    "tree": tree,
                    "root": str(root),
                    "listed": len(listed),
                    "matched": len(matched),
                    "missing": missing,
                    "missing_reasons": reasons,
                    "extra": extra,
                    "eval_listed": len(eval_files),
                    "eval_missing": eval_missing,
                    "eval_ingested": eval_ingested,
                    "eval_gate_invisible": eval_gate_invisible,
                    "problems": src_problems,
                }
            )
            problems.extend(src_problems)

    totals = {
        "config_train_entries": sum(len(v) for v in train.values()),
        "config_eval_entries": sum(len(v) for v in evals.values()),
        "listed_train": sum(s["listed"] for s in sources),
        "matched_train": sum(s["matched"] for s in sources),
        "missing_train": sum(len(s["missing"]) for s in sources),
        "extra_train": sum(len(s["extra"]) for s in sources),
        "eval_listed": sum(s["eval_listed"] for s in sources),
        "eval_missing": sum(len(s["eval_missing"]) for s in sources),
        "eval_ingested": sum(len(s["eval_ingested"]) for s in sources),
        "eval_gate_invisible": sum(len(s["eval_gate_invisible"]) for s in sources),
    }
    return {
        "config": str(config_path),
        "roots": {tree: str(root) for tree, root in roots.items()},
        "sources": sources,
        "problems": problems,
        "totals": totals,
        "ok": not problems,
    }


def _print_human(report: dict) -> None:
    totals = report["totals"]
    print(f"config   : {report['config']}")
    print(f"v76 root : {report['roots']['v76']}")
    print(f"sft root : {report['roots']['sft']}")
    print()
    print(f"{'lang':<5} {'tree':<4} {'listed/matched':>14} {'eval':>5}  status")
    for src in report["sources"]:
        pair = f"{src['listed']}/{src['matched']}"
        status = "ok" if not src["problems"] else "PROBLEM"
        print(f"{src['lang']:<5} {src['tree']:<4} {pair:>14} {src['eval_listed']:>5}  {status}")
    print()
    print(
        f"train: {totals['listed_train']}/{totals['config_train_entries']} config entries "
        f"checked, {totals['matched_train']} ingestible"
    )
    print(
        f"eval : {totals['eval_listed']}/{totals['config_eval_entries']} config entries "
        f"checked; ingested {totals['eval_ingested']}, gate-invisible "
        f"{totals['eval_gate_invisible']}, missing {totals['eval_missing']}"
    )
    if report["ok"]:
        print("OK: every config train file is ingestible; every eval file is separated")
    else:
        print(f"PROBLEM: {len(report['problems'])} finding(s):", file=sys.stderr)
        for problem in report["problems"]:
            print(f"  - {problem}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Verify the internal trees on disk against "
        "configs/corpus/internal_ingest.yaml (train ingestible, eval separated)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="training config to verify against (default: %(default)s)",
    )
    parser.add_argument(
        "--internal-root",
        type=Path,
        default=None,
        help="v7.6 manifests tree (default: $QASR_INTERNAL_ROOT / $INTERNAL_ROOT, "
        "else the config's own absolute paths)",
    )
    parser.add_argument(
        "--sft-root",
        type=Path,
        default=None,
        help="q3asr SFT envelopes tree (default: $QASR_SFT_ROOT, else the config's "
        "own absolute paths)",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    try:
        report = audit(args.config, internal_root=args.internal_root, sft_root=args.sft_root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot verify: {exc}", file=sys.stderr)
        return 2

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
