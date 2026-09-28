"""Per-language WER/CER evaluation callback for QASR training runs.

Loss curves do not tell us whether ml/hi are catching up or whether the decoder
is forgetting. ``WEREvalCallback`` decodes a fixed small sample per language
from the configured eval manifests at every Trainer evaluation step and logs
``eval_wer_<lang>``/``eval_cer_<lang>``/``eval_wer_macro`` so the step-budget
stop criterion (WER flattening in the WSD stable region) is observable live.

Only rank 0 runs the decode pass; the metrics are broadcast through the
normal Trainer logging pipeline (wandb/console).
"""

from __future__ import annotations

import logging
import random
import zlib
from collections.abc import Sequence
from typing import Any

import torch
from transformers import TrainerCallback

from .audio import load_mono_audio
from .data import JsonlSpeechDataset
from .evaluate import error_counts

LOGGER = logging.getLogger("qasr.eval_callback")


class WEREvalCallback(TrainerCallback):
    """Greedy-decode a fixed per-language sample subset at each evaluation.

    Args:
        processor: The ``QASRProcessor`` used for training (also handles
            prompt construction and decoding).
        eval_manifest_specs: ``(manifest_path, language)`` pairs, the same
            format produced by ``TrainConfig.eval_manifest_specs``.
        samples_per_language: Decoded samples per language per evaluation.
        seed: Fixed sampling seed — the same utterances are evaluated every
            time so curves are comparable across steps.
        dataset_kwargs: Extra ``JsonlSpeechDataset`` constructor kwargs
            (duration filters, audio root, ...).
        max_new_tokens: Greedy decode budget per utterance.
    """

    def __init__(
        self,
        *,
        processor: Any,
        eval_manifest_specs: Sequence[tuple[str, str]],
        samples_per_language: int = 100,
        seed: int = 42,
        dataset_kwargs: dict[str, Any] | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        if samples_per_language < 0:
            raise ValueError("samples_per_language must be non-negative")
        self.processor = processor
        self.eval_manifest_specs = list(eval_manifest_specs)
        self.samples_per_language = int(samples_per_language)
        self.seed = int(seed)
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.max_new_tokens = int(max_new_tokens)
        # Indexing 100M-line manifests is expensive; build each part lazily
        # once (rank 0 only) and reuse it for every evaluation.
        self._datasets: dict[tuple[str, str], JsonlSpeechDataset] = {}
        # transformers' CallbackHandler passes a fixed kwarg set that never
        # includes the trainer, so it must be bound explicitly after
        # construction (see bind_trainer). Relying on an on_evaluate kwarg
        # silently disables the callback.
        self._trainer: Any | None = None

    def bind_trainer(self, trainer: Any) -> WEREvalCallback:
        """Attach the Trainer whose model/log pipeline the decode pass uses."""
        self._trainer = trainer
        return self

    def _dataset_for(self, path: str, language: str) -> JsonlSpeechDataset:
        key = (path, language)
        dataset = self._datasets.get(key)
        if dataset is None:
            dataset = JsonlSpeechDataset(path, language=language, **self.dataset_kwargs)
            self._datasets[key] = dataset
        return dataset

    def _sample_indices(self, dataset: JsonlSpeechDataset, language: str) -> list[int]:
        count = min(self.samples_per_language, len(dataset))
        # Deterministic per language so every evaluation decodes the same
        # utterances regardless of world size or epoch. random.Random rejects
        # tuple seeds, and hash() is PYTHONHASHSEED-randomized, so mix the
        # language in with crc32.
        rng = random.Random(self.seed * 1_000_003 + zlib.crc32(language.encode("utf-8")))
        return rng.sample(range(len(dataset)), count)

    def _transcribe(self, trainer: Any, feature: dict[str, Any]) -> str:
        waveform = load_mono_audio(
            feature["audio_path"], int(self.processor.feature_extractor.sampling_rate)
        )
        batch = dict(
            self.processor.apply_transcription_request(
                audio=[waveform],
                language=[feature.get("language")],
                return_tensors="pt",
            )
        )
        inputs = trainer._prepare_inputs(batch)
        model_dtype = next(trainer.model.parameters()).dtype
        for name, value in inputs.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[name] = value.to(dtype=model_dtype)
        prompt_length = inputs["input_ids"].shape[1]
        output_ids = trainer.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        return self.processor.decode(
            output_ids[0, prompt_length:],
            return_format="transcription_only",
        ).strip()

    def _evaluate_language(
        self, trainer: Any, language: str, manifests: list[tuple[str, str]]
    ) -> dict[str, float] | None:
        totals = {"word_errors": 0, "word_count": 0, "char_errors": 0, "char_count": 0}
        decoded = 0
        for path, manifest_language in manifests:
            dataset = self._dataset_for(path, manifest_language)
            for index in self._sample_indices(dataset, language):
                feature = dataset[index]
                try:
                    hypothesis = self._transcribe(trainer, feature)
                except Exception as exc:  # keep training alive on one bad file
                    LOGGER.warning(
                        "WER callback skipped %s: %s", feature.get("audio_path"), exc
                    )
                    continue
                counts = error_counts(feature["text"], hypothesis)
                for key, value in counts.items():
                    totals[key] += value
                decoded += 1
        if decoded == 0 or totals["word_count"] == 0:
            return None
        return {
            "wer": totals["word_errors"] / totals["word_count"],
            "cer": totals["char_errors"] / max(totals["char_count"], 1),
            "samples": decoded,
        }

    def on_evaluate(self, args, state, control, model=None, trainer=None, **kwargs) -> None:
        trainer = trainer if trainer is not None else self._trainer
        if trainer is None:
            LOGGER.warning(
                "WER callback registered but no trainer bound; call "
                "bind_trainer(trainer) after construction — skipping decode pass"
            )
            return
        if not state.is_world_process_zero or self.samples_per_language == 0:
            return
        # Group manifests by language (several manifests may share a language).
        by_language: dict[str, list[tuple[str, str]]] = {}
        for path, language in self.eval_manifest_specs:
            by_language.setdefault(language, []).append((path, language))

        was_training = trainer.model.training
        trainer.model.eval()
        metrics: dict[str, float] = {}
        wer_values: list[float] = []
        try:
            with torch.no_grad():
                for language in sorted(by_language):
                    result = self._evaluate_language(trainer, language, by_language[language])
                    if result is None:
                        LOGGER.warning("WER callback: no decodable samples for %s", language)
                        continue
                    metrics[f"eval_wer_{language}"] = round(result["wer"], 4)
                    metrics[f"eval_cer_{language}"] = round(result["cer"], 4)
                    wer_values.append(result["wer"])
                    LOGGER.info(
                        "WER callback [%s]: WER=%.2f%% CER=%.2f%% (%d samples)",
                        language,
                        100 * result["wer"],
                        100 * result["cer"],
                        result["samples"],
                    )
        finally:
            if was_training:
                trainer.model.train()
        if wer_values:
            metrics["eval_wer_macro"] = round(sum(wer_values) / len(wer_values), 4)
        if metrics:
            trainer.log(metrics)


__all__ = ["WEREvalCallback"]
