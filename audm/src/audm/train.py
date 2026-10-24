"""Distributed training entrypoint; dataset/augmentation semantics come from QASR."""

import copy
import json
import logging
import os
from pathlib import Path

import torch
from qasr.augmentation import build_augmenter
from qasr.data import JsonlSpeechDataset, ResilientAudioDataset
from qasr.losses import make_smoothed_causal_lm_loss
from qasr.train import _build_code_switcher, _sample_subset, _StatsSubset, _training_arguments
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from .collator import AUDDataCollator
from .eval_callback import AUDWEREvalCallback
from .export import write_json
from .memory import CUDAMemoryCallback, write_memory_report
from .modeling import AUDForConditionalGeneration
from .processing import AUDProcessor
from .trainer import AUDTrainer
from .training_config import load_recipe

LOGGER = logging.getLogger("audm")


def dataset_parts(config, evaluation=False):
    specs = config.eval_manifest_specs if evaluation else config.train_manifest_specs
    kwargs = dict(
        min_duration_seconds=config.min_duration_seconds,
        max_duration_seconds=config.max_duration_seconds,
        audio_root=config.audio_root,
        validate_audio_paths=config.validate_audio_paths,
    )
    parts = [JsonlSpeechDataset(path, language=lang, **kwargs) for path, lang in specs]
    if not evaluation:
        parts += [
            JsonlSpeechDataset(path, language=lang, empty_target_ok=True, **kwargs)
            for path, lang in config.non_speech_manifest_specs
        ]
    return parts


def resilient_parts(parts, **options):
    # Substitute bad samples within the same source manifest. Wrapping the
    # concatenation could replace the last bad Malayalam row with Chinese and
    # silently violate the sampler's language quota.
    wrapped = [ResilientAudioDataset(part, **options) for part in parts]
    return (
        (wrapped[0] if len(wrapped) == 1 else torch.utils.data.ConcatDataset(wrapped))
        if wrapped
        else None
    )


def resume_contract(receipt):
    value = copy.deepcopy(receipt)
    # Compare resolved semantics; YAML formatting/paths and evaluation-only
    # changes must not require throwing away the optimizer and scheduler.
    for key in ("recipe", "data_recipe", "recipe_sha256", "data_recipe_sha256"):
        value.pop(key, None)
    for key in ("resume_from_checkpoint", "smoke_test", "output_dir", "model_name_or_path"):
        value["resolved"].pop(key, None)
    for key in ("eval_manifest", "per_device_eval_batch_size"):
        value["resolved"].pop(key, None)
    if "per_device_train_batch_size" in value["resolved"]:
        value["resolved"]["examples_per_rank_update"] = value["resolved"].pop(
            "per_device_train_batch_size"
        ) * value["resolved"].pop("gradient_accumulation_steps")
    # v5 added optional effects but left v4's waveform/SpecAugment behavior
    # unchanged. Compare those legacy policies across versions only when none
    # of the new effect keys is present; older numerical policies stay blocked.
    effects = value.get("robustness", {}).get("augmentation", {}).get("effects", {})
    if value.get("robustness_version") in (4, 5) and not {
        "silence",
        "pitch",
        "clipping",
        "dropout",
        "compression",
    }.intersection(effects):
        value["robustness_version"] = 4
    return value


def validate_resume_recipe(previous, current):
    if resume_contract(previous) != resume_contract(current):
        raise ValueError(
            "Resume recipe/data changed; start a new fine-tuning run with --model instead"
        )
    old = previous["resolved"].get("eval_manifest")
    new = current["resolved"].get("eval_manifest")
    if old != new and (
        not isinstance(old, dict)
        or not isinstance(new, dict)
        or any(old.get(language) != paths for language, paths in new.items())
    ):
        raise ValueError("Resume permits removing evaluation languages, not adding/relabeling data")


def validate_resume_snapshot(previous, current):
    # Resume contracts already require identical training manifests. Allow
    # evaluation removal only; every retained manifest must still match exactly.
    if any(previous.get(path) != metadata for path, metadata in current.items()):
        raise ValueError("Training/evaluation manifests changed since the original run")


def run(recipe, model=None, output=None, smoke=False, resume=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config, stage, receipt = load_recipe(recipe, model, output, smoke, resume)
    smoke = config.smoke_test
    set_seed(config.seed)
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    root = Path(config.output_dir) / "smoke-test" if config.smoke_test else Path(config.output_dir)
    checkpoint = config.resume_from_checkpoint
    if checkpoint == "auto":
        checkpoint = get_last_checkpoint(str(root)) if root.exists() else None
        if checkpoint is None:
            raise ValueError("--resume auto found no complete Trainer checkpoint")
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
        for name in ("trainer_state.json", "config.json"):
            if not Path(checkpoint, name).is_file():
                raise ValueError(f"Resume checkpoint is missing {name}: {checkpoint}")
        if not list(Path(checkpoint).glob("optimizer*")) and not list(
            Path(checkpoint).glob("global_step*")
        ):
            raise ValueError(
                "Resume needs optimizer state; use --model for weights-only fine-tuning"
            )
        previous_path = root / "training_recipe.json"
        if not previous_path.is_file():
            raise ValueError("Resume requires the original run's training_recipe.json")
        previous = json.loads(previous_path.read_text())
        validate_resume_recipe(previous, receipt)
    elif root.exists() and any(root.iterdir()):
        # A successful rehearsal lives in its own subdirectory; production may
        # create new files in the parent without overwriting that rehearsal.
        rehearsal_only = (
            not smoke
            and {p.name for p in root.iterdir()} == {"smoke-test"}
            and (root / "smoke-test/training_recipe.json").is_file()
        )
        if not rehearsal_only:
            raise FileExistsError(
                f"Run directory is not empty: {root}; use --resume or a new --output"
            )
    if config.deepspeed:
        ds = json.loads(Path(config.deepspeed).read_text())
        if ds.get("zero_optimization", {}).get("stage") != 2:
            raise ValueError(
                "This release supports DeepSpeed ZeRO-2; ZeRO-3 needs different generation/checkpoint handling"
            )
    args = _training_arguments(config, bool(config.eval_manifest_specs))
    if args.device.type == "cpu" and args.device.index is not None:
        # Accelerate's multi-CPU device may be cpu:0, which torch.load rejects
        # as map_location during optimizer resume. Normalize the cached device
        # and shared distributed state; CUDA placement is untouched.
        args.__dict__["_setup_devices"] = torch.device("cpu")
        args.distributed_state.device = torch.device("cpu")
    if config.expected_world_size is not None and args.world_size != config.expected_world_size:
        raise ValueError(f"Expected {config.expected_world_size} ranks; launched {args.world_size}")
    if torch.distributed.is_initialized():
        # Every rank must finish the output-directory checks before rank 0 creates it.
        torch.distributed.barrier()
    config.resume_from_checkpoint = checkpoint
    processor = AUDProcessor.from_pretrained(checkpoint or config.model_name_or_path)
    if processor.config.audio_type_version and config.non_speech_manifest_specs:
        raise ValueError("Typed checkpoints require the categorized robustness.non_speech pool")
    if config.max_duration_seconds > processor.config.max_audio_seconds:
        raise ValueError("Dataset maximum duration exceeds the exported audio contract")
    model = AUDForConditionalGeneration.from_pretrained(
        checkpoint or config.model_name_or_path,
        local_files_only=True,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation=config.attn_implementation or "sdpa",
    )
    model.configure_training(stage)
    # Qwen's nested causal model owns the loss, unlike QASR's outer lm_head.
    if config.label_smoothing:
        model.aud_decoder.loss_function = make_smoothed_causal_lm_loss(config.label_smoothing)
    train_parts, eval_parts = dataset_parts(config), dataset_parts(config, evaluation=True)
    snapshot = {
        str(part.manifest_path): {
            "size": part.manifest_path.stat().st_size,
            "mtime_ns": part.manifest_path.stat().st_mtime_ns,
            "records": len(part),
            "hours": part.stats.total_kept_hours,
        }
        for part in train_parts + eval_parts
    }
    if checkpoint:
        snapshot_path = root / "data_snapshot.json"
        if not snapshot_path.is_file():
            raise ValueError("Resume requires the original run's data_snapshot.json")
        validate_resume_snapshot(json.loads(snapshot_path.read_text()), snapshot)
    if args.should_save:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "training_recipe.json", receipt)
        write_json(root / "data_snapshot.json", snapshot)
    mixer = _build_code_switcher(config, processor, args, train_parts)
    if config.smoke_test:
        count = max(2, -(-config.smoke_test_train_samples // len(train_parts)))
        train_parts = [
            _StatsSubset(p, _sample_subset(p, min(count, len(p)), config.data_seed + i).indices)
            for i, p in enumerate(train_parts)
        ]
        eval_parts = [_sample_subset(p, min(2, len(p)), config.data_seed + 1) for p in eval_parts]
    resilient = dict(
        sampling_rate=16000,
        min_audio_seconds=config.min_duration_seconds,
        max_audio_seconds=config.max_duration_seconds,
        tokenizer=processor.tokenizer,
        max_target_length=config.max_target_length,
        truncate_long_transcripts=config.truncate_long_transcripts,
    )
    train_dataset = resilient_parts(train_parts, **resilient)
    eval_dataset = resilient_parts(eval_parts, **resilient)
    robust = config.aud_robustness
    if robust.get("augmentation"):
        from .robustness import RobustAugmenter

        augmenter = RobustAugmenter(robust["augmentation"], config.augmentation)
    else:
        augmenter = build_augmenter(config.augmentation, sampling_rate=16000)
    collator_options = dict(
        processor=processor,
        language=config.language,
        min_audio_seconds=config.min_duration_seconds,
        max_audio_seconds=config.max_duration_seconds,
        max_target_length=config.max_target_length,
        enforce_augmented_duration=True,
    )
    collator = AUDDataCollator(
        **collator_options,
        augmenter=augmenter,
        code_switcher=mixer,
        code_switch_probability=config.code_switch_probability,
    )
    if robust.get("non_speech"):
        from .robustness import NonSpeechPool

        collator.non_speech_pool = NonSpeechPool(
            robust["non_speech"],
            config.min_duration_seconds,
            config.max_duration_seconds,
            require_audio_type=bool(processor.config.audio_type_version),
        )
    clean_collator = AUDDataCollator(**collator_options)
    trainer = AUDTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        prediction_processor=processor,
        prediction_collator=clean_collator,
        eval_collator=clean_collator,
        eval_log_samples=min(config.eval_log_samples, 2) if smoke else config.eval_log_samples,
        eval_generation_max_new_tokens=min(config.eval_generation_max_new_tokens, 16)
        if smoke
        else config.eval_generation_max_new_tokens,
        balanced_parts=train_parts,
        balanced_sampling=config.balanced_sampling,
        sampling_strategy=config.sampling_strategy,
        sampling_temperature=config.sampling_temperature,
        language_batch_counts=config.language_batch_counts,
        allow_partial_epoch=smoke or stage in ("adapter", "decoder"),
        llm_lr_factor=config.llm_lr_factor,
        embed_lr_factor=config.embed_lr_factor,
    )
    trainer.aud_receipt = receipt
    trainer.aud_label_smoothing = config.label_smoothing
    trainer.add_callback(CUDAMemoryCallback())
    trainer.aud_source = str(Path(checkpoint or config.model_name_or_path).resolve())
    if eval_parts and config.eval_wer_samples_per_language:
        callback = AUDWEREvalCallback(
            batch_size=config.per_device_eval_batch_size,
            processor=processor,
            eval_manifest_specs=config.eval_manifest_specs,
            samples_per_language=min(2, config.eval_wer_samples_per_language)
            if smoke
            else config.eval_wer_samples_per_language,
            seed=config.data_seed,
            max_new_tokens=16 if smoke else config.eval_generation_max_new_tokens,
            global_language_budget=config.eval_wer_global_language_budget,
            dataset_kwargs=dict(
                min_duration_seconds=config.min_duration_seconds,
                max_duration_seconds=config.max_duration_seconds,
                audio_root=config.audio_root,
            ),
        ).bind_trainer(trainer)
        trainer.add_callback(callback)
    LOGGER.info(
        "AUD %s stage=%s ranks=%d global_batch=%d trainable=%s",
        model.config.variant,
        stage,
        args.world_size,
        args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps,
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    LOGGER.info(
        "AUD loss uses supervised-token chunks of 128; label_smoothing=%s", config.label_smoothing
    )
    LOGGER.info(
        "AUD evaluation batch_size=%d for both loss and WER/CER generation",
        config.per_device_eval_batch_size,
    )
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
    except torch.OutOfMemoryError:
        try:
            write_memory_report(args, root, "oom", trainer.state.global_step)
        except Exception:
            LOGGER.exception("Could not write CUDA memory diagnostics")
        raise
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    if eval_dataset is not None:
        trainer.save_metrics("eval", trainer.evaluate())
    trainer.save_model()
    trainer.save_state()


def preflight(recipe, model, output, scan=False):
    config, stage, receipt = load_recipe(recipe, model, output)
    processor = AUDProcessor.from_pretrained(model)
    paths = (
        config.train_manifest_paths
        + config.eval_manifest_paths
        + [p for p, _ in config.non_speech_manifest_specs]
    )
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing manifests: {missing}")
    if config.code_switch_index and not Path(config.code_switch_index).is_file():
        raise FileNotFoundError(config.code_switch_index)
    if config.code_switch_index:
        from data_processing.code_switch.common import load_plan
        from data_processing.code_switch.index import check_frozen
        from qasr.code_switch import OnlineCodeSwitch

        mixer = OnlineCodeSwitch(
            config.code_switch_index, processor.tokenizer, allow_pilot=config.smoke_test
        )
        check_frozen(mixer.index["root"], load_plan(mixer.index["root"]))
        if (
            mixer.cfg["sample_rate"] != 16000
            or mixer.cfg["max_seconds"] > config.max_duration_seconds
            or mixer.cfg["max_target_tokens"] > config.max_target_length
        ):
            raise ValueError("Code-switch index limits differ from the training recipe")
        indexed = {s["path"] for s in mixer.index["sources"]}
        if not {str(Path(p).resolve()) for p in config.train_manifest_paths}.issubset(indexed):
            raise ValueError("Training manifests are absent from the code-switch index")
    if processor.config.audio_type_version and config.non_speech_manifest_specs:
        raise ValueError("Typed checkpoints require the categorized robustness.non_speech pool")
    if config.max_duration_seconds > processor.config.max_audio_seconds:
        raise ValueError("Dataset duration exceeds model limit")
    noise = (config.augmentation or {}).get("noise_injection", {})
    policy = config.aud_robustness.get("augmentation")
    uses_noise = not policy or policy.get("effects", {}).get("noise", {}).get("p", 0)
    if uses_noise and noise.get("enabled") and not Path(noise["noise_dir"]).is_dir():
        raise FileNotFoundError(noise["noise_dir"])
    robustness_scan = {}
    if config.aud_robustness:
        from .robustness import NonSpeechPool, RobustAugmenter

        if config.aud_robustness.get("augmentation"):
            augmenter = RobustAugmenter(config.aud_robustness["augmentation"], config.augmentation)
            if scan:
                robustness_scan["augmentation"] = augmenter.scan_assets()
        if config.aud_robustness.get("non_speech"):
            pool = NonSpeechPool(
                config.aud_robustness["non_speech"],
                config.min_duration_seconds,
                config.max_duration_seconds,
                require_audio_type=bool(processor.config.audio_type_version),
            )
            if scan:
                robustness_scan["non_speech"] = pool.scan_audio()
    if processor.config.audio_type_version and stage == "adapter":
        raise ValueError("Audio-type training requires decoder or full stage")
    report = {
        "variant": processor.config.variant,
        "stage": stage,
        "manifests": len(paths),
        "expected_world_size": config.expected_world_size,
        "global_batch": (config.expected_world_size or 1)
        * config.per_device_train_batch_size
        * config.gradient_accumulation_steps,
        "recipe": receipt,
        "scan": scan,
        "robustness_scan": robustness_scan,
    }
    if scan:
        parts = dataset_parts(config)
        from qasr.sampling import WeightedStratifiedLanguageSampler

        if config.sampling_strategy == "weighted_stratified":
            sampler = WeightedStratifiedLanguageSampler(
                parts,
                batch_size=receipt["quota_window"],
                seed=config.data_seed,
                num_replicas=config.expected_world_size or 1,
                rank=0,
                language_batch_counts=config.language_batch_counts,
            )
            import math

            updates = math.ceil(
                len(sampler)
                / config.per_device_train_batch_size
                / config.gradient_accumulation_steps
            )
            report["updates_per_coverage_epoch"] = updates
            report["language_repeat_factors"] = sampler.repeat_factors
        report["records"] = sum(map(len, parts))
        report["hours"] = sum(p.stats.total_kept_hours for p in parts)
    return report
