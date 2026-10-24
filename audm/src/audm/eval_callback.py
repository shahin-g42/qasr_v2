"""Batched AUD generation with the existing fixed QASR evaluation sample sets."""

import logging

import torch
from qasr.audio import load_mono_audio
from qasr.data import CombinedSpeechDataset
from qasr.eval_callback import WEREvalCallback
from qasr.evaluate import error_counts

LOGGER = logging.getLogger("audm.eval_callback")


class AUDWEREvalCallback(WEREvalCallback):
    def __init__(self, *, batch_size=1, distributed=False, index_cache=None, **kwargs):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("WER batch_size must be a positive integer")
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.distributed = distributed
        self.index_cache = index_cache
        self.decode_rank, self.decode_world = 0, 1

    def _dataset_for(self, path, language):
        if not self.index_cache:
            return super()._dataset_for(path, language)
        from .indexed_data import IndexedSpeechDataset

        key = (path, language)
        if key not in self._datasets:
            self._datasets[key] = IndexedSpeechDataset(
                path, language=language, cache_directory=self.index_cache, **self.dataset_kwargs
            )
        return self._datasets[key]

    def on_evaluate(self, args, state, control, model=None, trainer=None, **kwargs):
        if not self.distributed or not torch.distributed.is_initialized():
            return super().on_evaluate(args, state, control, model=model, trainer=trainer, **kwargs)
        trainer = trainer or self._trainer
        if trainer is None or not self.samples_per_language:
            return
        self.decode_rank = torch.distributed.get_rank()
        self.decode_world = torch.distributed.get_world_size()
        languages = {}
        for path, language in self.eval_manifest_specs:
            languages.setdefault(language, []).append((path, language))
        was_training = trainer.model.training
        trainer.model.eval()
        metrics, wers = {}, []
        try:
            with torch.no_grad():
                for language in sorted(languages):
                    result = self._evaluate_language(trainer, language, languages[language])
                    if result:
                        metrics[f"eval_wer_{language}"] = round(result["wer"], 4)
                        metrics[f"eval_cer_{language}"] = round(result["cer"], 4)
                        wers.append(result["wer"])
                        if self.decode_rank == 0:
                            LOGGER.info("Distributed WER [%s]: %s", language, result)
        finally:
            if was_training:
                trainer.model.train()
        if wers:
            metrics["eval_wer_macro"] = round(sum(wers) / len(wers), 4)
        if metrics and self.decode_rank == 0:
            trainer.log(metrics)

    def _evaluate_language(self, trainer, language, manifests):
        error = None
        try:
            totals = self._local_totals(trainer, language, manifests)
        except Exception as exc:
            if self.decode_world == 1:
                raise
            error = exc
            totals = [0] * 5
        if self.decode_world > 1:
            device = trainer.args.device if torch.distributed.get_backend() == "nccl" else "cpu"
            counts = torch.tensor(
                [int(error is not None), *totals], dtype=torch.long, device=device
            )
            torch.distributed.all_reduce(counts)
            failures, *totals = counts.tolist()
            if failures:
                raise RuntimeError(
                    f"Distributed WER failed on {failures} rank(s): {error}"
                ) from error
        word_errors, words, char_errors, chars, decoded = totals
        if not decoded or not words:
            return None
        return dict(wer=word_errors / words, cer=char_errors / max(1, chars), samples=decoded)

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

    def _local_totals(self, trainer, language, manifests):
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
            for index in self._sample_indices(dataset, language)[
                self.decode_rank :: self.decode_world
            ]:
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
        return [
            totals["word_errors"],
            totals["word_count"],
            totals["char_errors"],
            totals["char_count"],
            decoded,
        ]
