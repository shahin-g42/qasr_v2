"""Batched AUD generation with the existing fixed QASR evaluation sample sets."""

import logging

import torch
from qasr.audio import load_mono_audio
from qasr.data import CombinedSpeechDataset
from qasr.eval_callback import WEREvalCallback
from qasr.evaluate import error_counts

LOGGER = logging.getLogger("audm.eval_callback")


class AUDWEREvalCallback(WEREvalCallback):
    def __init__(self, *, batch_size=1, **kwargs):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("WER batch_size must be a positive integer")
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def _transcribe_batch(self, trainer, features, waves):
        batch = dict(
            self.processor.apply_transcription_request(
                audio=waves, language=[row.get("language") for row in features], return_tensors="pt"
            )
        )
        inputs = trainer._prepare_inputs(batch)
        dtype = next(trainer.model.parameters()).dtype
        for name, value in inputs.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[name] = value.to(dtype=dtype)
        width = inputs["input_ids"].shape[1]
        output = trainer.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            synced_gpus=False,
        )
        if len(output) != len(features):
            raise RuntimeError("WER generation returned the wrong number of sequences")
        return [
            self.processor.decode(row[width:], return_format="transcription_only").strip()
            for row in output
        ]

    def _evaluate_language(self, trainer, language, manifests):
        totals = {"word_errors": 0, "word_count": 0, "char_errors": 0, "char_count": 0}
        decoded = 0
        datasets = [self._dataset_for(path, lang) for path, lang in manifests]
        if self.global_language_budget and len(datasets) > 1:
            datasets = [CombinedSpeechDataset(datasets)]
        pending_features, pending_waves = [], []

        def flush():
            nonlocal decoded
            if not pending_features:
                return
            # Generation/OOM failures must be visible. Silently dropping a whole
            # batch could make a broken checkpoint appear more accurate.
            hypotheses = self._transcribe_batch(trainer, pending_features, pending_waves)
            for feature, hypothesis in zip(pending_features, hypotheses, strict=True):
                for key, value in error_counts(feature["text"], hypothesis).items():
                    totals[key] += value
                decoded += 1
            pending_features.clear()
            pending_waves.clear()

        for dataset in datasets:
            for index in self._sample_indices(dataset, language):
                try:
                    feature = dataset[index]
                    wave = load_mono_audio(feature["audio_path"], 16000)
                    # Validate each recording before batching so one invalid file
                    # cannot discard the other 39 evaluation examples.
                    wave = self.processor._waves([wave], 16000)[0]
                except (OSError, ValueError, RuntimeError) as exc:
                    LOGGER.warning("WER callback skipped %s[%s]: %s", language, index, exc)
                    continue
                pending_features.append(feature)
                pending_waves.append(wave)
                if len(pending_features) == self.batch_size:
                    flush()
        flush()
        if decoded == 0 or totals["word_count"] == 0:
            return None
        return dict(
            wer=totals["word_errors"] / totals["word_count"],
            cer=totals["char_errors"] / max(totals["char_count"], 1),
            samples=decoded,
        )
