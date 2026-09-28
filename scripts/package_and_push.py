"""Package a trained QASR checkpoint into a self-contained HuggingFace repo and push it.

Run this on the cluster where the trained weights live. It assembles a clean
release directory from a Phase-2 checkpoint (``output/full``) and an optional
Phase-3b EAGLE head (``output/eagle_v2/checkpoint-10000``), then uploads it to
the Hub.

What the release contains
-------------------------
* Model weights + config + generation config (copied from ``--model-dir``).
* Processor / tokenizer / feature-extractor artifacts (copied from ``--model-dir``).
* Bundled remote-code modules so ``trust_remote_code=True`` works with zero
  install: configuration.py, modeling.py, feature_extraction.py, processing.py,
  utils.py, eagle.py (copied verbatim from ``--src-dir``).
* ``auto_map`` entries injected into config.json / preprocessor_config.json /
  processor_config.json so the Auto* factories resolve the QASR classes.
* The EAGLE head at ``eagle/eagle_head.pt`` (for speculative decoding).
* A polished model card written to ``README.md``.

Usage
-----
    # Build the staging dir locally and inspect it (no upload):
    python scripts/package_and_push.py --no-upload --verify

    # Build and push (private) to the default repo:
    huggingface-cli login          # once, stores the token
    python scripts/package_and_push.py

    # Override anything:
    python scripts/package_and_push.py \
        --model-dir output/full \
        --eagle-dir output/eagle_v2/checkpoint-10000 \
        --repo-id audarai/Audar-ASR-V1-Pro \
        --private \
        --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

LOGGER = logging.getLogger("qasr.package")

DEFAULT_REPO_ID = "audarai/Audar-ASR-V1-Pro"
DEFAULT_MODEL_DIR = "output/full"
DEFAULT_EAGLE_DIR = "output/eagle_v2/checkpoint-10000"
DEFAULT_STAGING = "output/hf_release"
DEFAULT_CARD = "scripts/HF_MODEL_CARD.md"

# Rendered into the card for any metric that was never measured for THIS
# artifact. Deliberately visible: before this, the card template carried
# hard-coded numbers from one earlier eval run, so packaging a new head
# silently published the old head's measurements as if they were the new one's.
UNMEASURED = "_not measured_"

# evaluate.py mode name -> placeholder suffix used in the card. A mode absent
# from the report (e.g. the acceptance gate runs only offline+eagle) fills as
# UNMEASURED rather than retaining a stale value.
CARD_MODE_SUFFIX = {
    "offline": "_OFFLINE",
    "eagle": "_EAGLE",
    "streaming": "_STREAM",
    "streaming-eagle": "_STREAM_EAGLE",
}

# Placeholders that match this shape are checked for leftovers before the card
# is written, so a typo'd or renamed key can never ship as a literal "{{X}}".
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")

# Remote-code modules bundled so ``trust_remote_code=True`` can load the model
# with no package install. Order does not matter; the dynamic loader resolves
# the relative imports (modeling->configuration, processing->feature_extraction
# + utils, eagle->modeling) by fetching sibling files from the repo.
REMOTE_CODE_MODULES = (
    "configuration.py",
    "modeling.py",
    "feature_extraction.py",
    "processing.py",
    "utils.py",
    "eagle.py",
)

# auto_map for the model config (config.json).
MODEL_AUTO_MAP = {
    "AutoConfig": "configuration.QASRConfig",
    "AutoModel": "modeling.QASRModel",
    "AutoModelForCausalLM": "modeling.QASRForConditionalGeneration",
}

# auto_map for the processor / feature-extractor configs.
PROCESSOR_AUTO_MAP = {
    "AutoFeatureExtractor": "feature_extraction.QASRFeatureExtractor",
    "AutoProcessor": "processing.QASRProcessor",
}

# Training-only artifacts that must never ship in a release.
SKIP_FILE_PATTERNS = (
    "optimizer",
    "scheduler",
    "rng_state",
    "trainer_state",
    "training_args",
    "zero_to_fp32",
    "latest",
    "global_step",
    "eagle_head.pt",  # the head is copied separately into eagle/
)


def _should_skip(name: str) -> bool:
    return any(token in name for token in SKIP_FILE_PATTERNS)


def _copy_model_files(model_dir: Path, staging: Path) -> list[str]:
    """Copy top-level model + processor files (skipping training artifacts).

    Only files at the top level are copied so intermediate ``checkpoint-*``
    subdirectories are never pulled into the release.
    """
    copied: list[str] = []
    for entry in sorted(model_dir.iterdir()):
        if entry.is_dir():
            continue
        if _should_skip(entry.name):
            LOGGER.info("  skip  %s (training artifact)", entry.name)
            continue
        shutil.copy2(entry, staging / entry.name)
        copied.append(entry.name)
        LOGGER.info("  copy  %s", entry.name)
    if not any(name.endswith(".safetensors") for name in copied):
        raise FileNotFoundError(
            f"No .safetensors weights found in {model_dir}. Is this a QASR checkpoint?"
        )
    if "config.json" not in copied:
        raise FileNotFoundError(f"config.json missing in {model_dir}")
    return copied


def _inject_auto_map(path: Path, entries: dict[str, str]) -> None:
    """Merge ``entries`` into the ``auto_map`` of a JSON config file in place."""
    if not path.is_file():
        return
    with path.open() as handle:
        data = json.load(handle)
    auto_map = dict(data.get("auto_map", {}))
    auto_map.update(entries)
    data["auto_map"] = auto_map
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    LOGGER.info("  auto_map -> %s (%s)", path.name, ", ".join(entries))


def _bundle_remote_code(src_dir: Path, staging: Path) -> None:
    for module in REMOTE_CODE_MODULES:
        source = src_dir / module
        if not source.is_file():
            raise FileNotFoundError(f"Remote-code module not found: {source}")
        shutil.copy2(source, staging / module)
        LOGGER.info("  code  %s", module)


def _copy_eagle_head(eagle_dir: Path, staging: Path) -> bool:
    head = eagle_dir / "eagle_head.pt"
    if not head.is_file():
        LOGGER.warning("EAGLE head not found at %s; releasing base model only", head)
        return False
    target_dir = staging / "eagle"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(head, target_dir / "eagle_head.pt")
    LOGGER.info("  eagle eagle/eagle_head.pt (%.1f MB)", head.stat().st_size / 1e6)
    return True


def _fmt_pct(value: object, nd: int = 2) -> str:
    return f"{value * 100:.{nd}f}%" if isinstance(value, (int, float)) else UNMEASURED


def _fmt_num(value: object, nd: int = 3, suffix: str = "") -> str:
    return f"{value:.{nd}f}{suffix}" if isinstance(value, (int, float)) else UNMEASURED


def build_card_vars(report: dict | None) -> dict[str, str]:
    """Derive every measured metric in the card from one ``evaluate.py`` report.

    ``report`` is the JSON that ``evaluate.py --output`` writes: top-level
    ``modes[<mode>][summary]`` (wer/cer/rtf_mean/latency_mean_s/
    tokens_per_second_mean/samples), ``eagle_stats`` and ``comparison``.
    Pass ``None`` to get an all-UNMEASURED map.
    """
    report = report or {}
    modes = report.get("modes") or {}
    variables: dict[str, str] = {}

    samples = None
    for mode, suffix in CARD_MODE_SUFFIX.items():
        summary = (modes.get(mode) or {}).get("summary") or {}
        if samples is None and summary.get("samples"):
            samples = summary["samples"]
        variables[f"WER{suffix}"] = _fmt_pct(summary.get("wer"))
        variables[f"CER{suffix}"] = _fmt_pct(summary.get("cer"))
        variables[f"RTF{suffix}"] = _fmt_num(summary.get("rtf_mean"), 3)
        variables[f"LAT{suffix}"] = _fmt_num(summary.get("latency_mean_s"), 3, " s")
        tps = summary.get("tokens_per_second_mean")
        variables[f"TPS{suffix}"] = (
            f"{tps:.0f} tok/s" if isinstance(tps, (int, float)) else "—"
        )

    stats = report.get("eagle_stats") or {}
    comparison = report.get("comparison") or {}
    by_position = stats.get("acceptance_by_position") or []
    variables["EAGLE_ACCEPTANCE"] = _fmt_pct(stats.get("acceptance_rate"), 1)
    variables["EAGLE_TPF"] = _fmt_num(stats.get("tokens_per_target_forward"), 2)
    variables["EAGLE_MEAN_ACCEPTED"] = _fmt_num(stats.get("mean_accepted_per_round"), 2)
    variables["EAGLE_POS0"] = _fmt_pct(by_position[0] if by_position else None, 1)
    variables["EAGLE_BY_POSITION"] = (
        " · ".join(f"{p * 100:.0f}%" for p in by_position)
        if by_position and all(isinstance(p, (int, float)) for p in by_position)
        else UNMEASURED
    )
    variables["EAGLE_SPEEDUP"] = _fmt_num(comparison.get("eagle_speedup"), 2, "×")
    variables["EAGLE_EXACT_MATCH"] = str(
        comparison.get("exact_transcript_match") or UNMEASURED
    )
    variables["EAGLE_NUM_DRAFT"] = str(report.get("num_draft_tokens") or UNMEASURED)
    variables["EVAL_SAMPLES"] = str(samples) if samples else UNMEASURED
    variables["EVAL_LANGUAGE"] = str(report.get("language") or UNMEASURED)
    # Basenames only. The report records absolute cluster paths, and a card is
    # published -- those would leak the internal filesystem layout and usernames.
    for key, field in (("EVAL_MANIFEST", "manifest"), ("EVAL_MODEL", "model"),
                       ("EVAL_EAGLE", "eagle")):
        raw = report.get(field)
        variables[key] = Path(str(raw)).name if raw else UNMEASURED
    return variables


def _write_model_card(
    card_template: Path,
    staging: Path,
    repo_id: str,
    has_eagle: bool,
    card_vars: dict[str, str] | None = None,
) -> None:
    if not card_template.is_file():
        raise FileNotFoundError(f"Model card template not found: {card_template}")
    text = card_template.read_text(encoding="utf-8")
    text = text.replace("{{REPO_ID}}", repo_id)
    # Display name tracks the repo id so a --repo-id override never publishes a
    # card titled after a different model.
    text = text.replace("{{MODEL_NAME}}", repo_id.split("/")[-1])
    for name, value in (card_vars or {}).items():
        text = text.replace("{{" + name + "}}", value)
    # Anything still unsubstituted is a renamed or typo'd key. Blank it to the
    # visible marker instead of shipping a literal placeholder to the Hub.
    leftovers = sorted(set(PLACEHOLDER_RE.findall(text)))
    if leftovers:
        LOGGER.warning(
            "card placeholders unfilled (%d): %s", len(leftovers), ", ".join(leftovers)
        )
        for name in leftovers:
            text = text.replace("{{" + name + "}}", UNMEASURED)
    if not has_eagle:
        # The card documents EAGLE usage unconditionally; flag its absence
        # rather than shipping instructions for a head that is not in the repo.
        text = text.replace(
            "## Evaluation",
            "> **Note:** this release was packaged without the EAGLE draft head; "
            "the speculative-decoding sections below require training one first "
            "(`qasr-train-eagle`).\n\n## Evaluation",
            1,
        )
    (staging / "README.md").write_text(text, encoding="utf-8")
    LOGGER.info("  card  README.md (repo_id=%s)", repo_id)


def _write_gitattributes(staging: Path) -> None:
    lines = [
        "*.safetensors filter=lfs diff=lfs merge=lfs -text",
        "*.pt filter=lfs diff=lfs merge=lfs -text",
        "*.bin filter=lfs diff=lfs merge=lfs -text",
    ]
    (staging / ".gitattributes").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("  lfs   .gitattributes")


def _verify_release(staging: Path) -> None:
    """Reload config + processor (and the model shell) from the staged repo."""
    LOGGER.info("Verifying staged release re-loads locally ...")
    import qasr  # noqa: F401  (registers the Auto* classes)
    from qasr import QASRConfig, QASRProcessor

    config = QASRConfig.from_pretrained(staging)
    if config.audio_config.hidden_size != 1280:
        raise RuntimeError("Staged config did not round-trip (encoder hidden_size)")
    processor = QASRProcessor.from_pretrained(staging)
    if processor.feature_extractor.sampling_rate != 16000:
        raise RuntimeError("Staged processor sampling rate is not 16 kHz")

    head = staging / "eagle" / "eagle_head.pt"
    if head.is_file():
        from qasr.eagle import EagleHead

        EagleHead.from_pretrained(staging / "eagle")
    LOGGER.info("Verification passed (config + processor%s reload OK)",
                " + eagle head" if head.is_file() else "")


def build_release(
    *,
    model_dir: Path,
    eagle_dir: Path | None,
    src_dir: Path,
    staging: Path,
    card_template: Path,
    repo_id: str,
    card_vars: dict[str, str] | None = None,
) -> bool:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"--model-dir does not exist: {model_dir}")
    if staging.exists():
        LOGGER.info("Clearing existing staging dir %s", staging)
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Copying model + processor files from %s", model_dir)
    _copy_model_files(model_dir, staging)

    LOGGER.info("Bundling remote-code modules from %s", src_dir)
    _bundle_remote_code(src_dir, staging)

    LOGGER.info("Injecting auto_map for trust_remote_code")
    _inject_auto_map(staging / "config.json", MODEL_AUTO_MAP)
    _inject_auto_map(staging / "preprocessor_config.json", PROCESSOR_AUTO_MAP)
    _inject_auto_map(staging / "processor_config.json", PROCESSOR_AUTO_MAP)

    has_eagle = False
    if eagle_dir is not None:
        LOGGER.info("Copying EAGLE head from %s", eagle_dir)
        has_eagle = _copy_eagle_head(eagle_dir, staging)

    _write_model_card(card_template, staging, repo_id, has_eagle, card_vars)
    _write_gitattributes(staging)

    LOGGER.info("Release assembled at %s", staging)
    return has_eagle


def push_release(staging: Path, repo_id: str, private: bool, commit_message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    LOGGER.info("Creating repo %s (private=%s) if needed", repo_id, private)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    LOGGER.info("Uploading %s -> %s", staging, repo_id)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    LOGGER.info("Push complete: https://huggingface.co/%s", repo_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help=f"Trained QASR checkpoint (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--eagle-dir", default=DEFAULT_EAGLE_DIR, help=f"Dir containing eagle_head.pt (default: {DEFAULT_EAGLE_DIR}); pass 'none' to skip")
    parser.add_argument("--src-dir", default="src/qasr", help="Package source dir to bundle remote code from")
    parser.add_argument("--staging-dir", default=DEFAULT_STAGING, help=f"Where to assemble the release (default: {DEFAULT_STAGING})")
    parser.add_argument("--card", default=DEFAULT_CARD, help=f"Model card template (default: {DEFAULT_CARD})")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Target Hub repo id (default: {DEFAULT_REPO_ID})")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true", help="Create a private repo (default)")
    visibility.add_argument("--public", dest="private", action="store_false", help="Create a public repo")
    parser.set_defaults(private=True)
    parser.add_argument("--verify", action="store_true", help="Reload the staged release locally before pushing")
    parser.add_argument("--no-upload", action="store_true", help="Assemble the staging dir only; skip the Hub push")
    parser.add_argument("--commit-message", default=None, help="Commit message for the upload (default: derived from --repo-id)")
    parser.add_argument("--eval-json", default=None,
        help="evaluate.py report whose metrics fill the card. Without it every measured "
             "number renders as 'not measured' instead of inheriting the template's stale values.")
    parser.add_argument("--card-var", action="append", default=[], metavar="NAME=VALUE",
        help="Extra card placeholder, repeatable, e.g. "
             "--card-var TARGET_CHECKPOINT=checkpoint-150000")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    eagle_dir = None if str(args.eagle_dir).lower() == "none" else Path(args.eagle_dir)
    staging = Path(args.staging_dir)

    report = None
    if args.eval_json:
        report_path = Path(args.eval_json)
        if not report_path.is_file():
            raise FileNotFoundError(f"--eval-json not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        LOGGER.info("Card metrics sourced from %s", report_path)
    else:
        LOGGER.warning(
            "no --eval-json: the card's measured metrics will render as %r. Pass the "
            "report written by the acceptance gate to publish numbers for THIS artifact.",
            UNMEASURED,
        )
    card_vars = build_card_vars(report)
    for item in args.card_var:
        name, sep, value = item.partition("=")
        if not name or not sep:
            raise ValueError(f"--card-var expects NAME=VALUE, got {item!r}")
        card_vars[name.strip().upper()] = value

    has_eagle = build_release(
        model_dir=Path(args.model_dir),
        eagle_dir=eagle_dir,
        src_dir=Path(args.src_dir),
        staging=staging,
        card_template=Path(args.card),
        repo_id=args.repo_id,
        card_vars=card_vars,
    )

    if args.verify:
        _verify_release(staging)

    if args.no_upload:
        LOGGER.info("--no-upload set; staged release ready at %s (eagle=%s)", staging, has_eagle)
        return

    commit_message = args.commit_message or f"Upload {args.repo_id.split('/')[-1]} release"
    push_release(staging, args.repo_id, args.private, commit_message)


if __name__ == "__main__":
    main()
