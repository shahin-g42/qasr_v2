from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
from transformers.audio_utils import AudioInput, make_list_of_audio_chat_template
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_asr.processing_qwen3_asr import (
    LANGUAGE_CODE_TO_NAME,
    Qwen3ASRProcessor,
)

from .feature_extraction import QASRFeatureExtractor
from .utils import get_subsampling_output_lengths


LOGGER = logging.getLogger("qasr.processing")


LANGUAGE_NAMES = {**LANGUAGE_CODE_TO_NAME, "ml": "Malayalam"}


def _audio_content_item(audio_item: Any) -> dict[str, Any]:
    if isinstance(audio_item, str):
        return {"type": "audio", "path": audio_item}
    return {"type": "audio", "audio": audio_item}


def _normalize_rows(value: str | Sequence[str], size: int, name: str) -> list[str]:
    rows = [value] * size if isinstance(value, str) else list(value)
    if len(rows) != size:
        raise ValueError(f"Got {len(rows)} {name} value(s) for {size} audio sample(s)")
    return rows


def resolve_qasr_language(language: str) -> str:
    resolved = LANGUAGE_NAMES.get(language.lower())
    if resolved is not None:
        return resolved
    for name in LANGUAGE_NAMES.values():
        if name.lower() == language.lower():
            return name
    raise ValueError(f"Unsupported QASR language: {language!r}")


class QASRProcessor(Qwen3ASRProcessor):
    """Qwen3 chat/tokenizer processing with Cohere encoder audio lengths."""

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        timestamp_segment_time: float = 80,
        subsampling_factor: int = 8,
        subsampling_conv_kernel_size: int = 3,
        subsampling_conv_stride: int = 2,
    ) -> None:
        super().__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            timestamp_segment_time=timestamp_segment_time,
        )
        self.subsampling_factor = int(subsampling_factor)
        self.subsampling_conv_kernel_size = int(subsampling_conv_kernel_size)
        self.subsampling_conv_stride = int(subsampling_conv_stride)
        if feature_extractor is not None and not isinstance(feature_extractor, QASRFeatureExtractor):
            LOGGER.warning(
                "QASRProcessor loaded with %s instead of QASRFeatureExtractor. "
                "Audio processing will adapt automatically, but mel features may "
                "differ from training. Re-save the processor with "
                "QASRFeatureExtractor for best results.",
                type(feature_extractor).__name__,
            )

    def _process_audio(self, audio: AudioInput, **kwargs):
        """Process audio, handling both QASR and CohereAsr feature extractor outputs.

        QASRFeatureExtractor returns ``attention_mask``; the native
        CohereAsrFeatureExtractor returns ``length`` instead.  This override
        normalises both formats so downstream code always gets
        ``input_features_mask``.

        Chunking is disabled (chunk_long_audio=False) because the QASR model
        processes each audio sample as a single sequence.  Energy-based
        splitting is only useful for offline long-form transcription with
        explicit stitching logic, which is not used in this pipeline.
        """
        n_window = kwargs.get("n_window", 50)
        # Disable feature-extractor chunking: training audio is pre-validated
        # by the collator and may exceed the 30 s fast-path threshold after
        # speed perturbation.  The encoder handles variable lengths natively.
        kwargs.setdefault("chunk_long_audio", False)
        audio_inputs = self.feature_extractor(audio, **kwargs)

        if "attention_mask" in audio_inputs:
            audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")
        elif "length" in audio_inputs:
            # CohereAsrFeatureExtractor returns per-sample valid time lengths;
            # both extractors produce time-major (batch, time, num_mel_bins),
            # so the time dim is shape[1].
            lengths = audio_inputs.pop("length")
            if not torch.is_tensor(lengths):
                lengths = torch.as_tensor(lengths)
            max_time = audio_inputs["input_features"].shape[1]
            audio_inputs["input_features_mask"] = (
                torch.arange(max_time, device=lengths.device)[None, :] < lengths[:, None]
            ).to(torch.int64)
        else:
            raise KeyError(
                f"Feature extractor {type(self.feature_extractor).__name__} returned neither "
                f"'attention_mask' nor 'length'. Got keys: {list(audio_inputs.keys())}"
            )

        audio_lengths = self._get_audio_token_length(
            audio_inputs["input_features_mask"].sum(-1), n_window
        )
        audio_inputs["num_audio_tokens"] = audio_lengths
        audio_replacements = [
            self.replace_audio_token(audio_inputs, idx) for idx in range(len(audio))
        ]
        return audio_inputs, audio_replacements

    def __call__(
        self,
        text,
        audio,
        output_labels: bool | None = False,
        **kwargs: Any,
    ) -> BatchFeature:
        """Apply Qwen processing and create causal labels from token IDs.

        Transformers 5.14 exposes the native ``output_labels`` switch, but its
        Qwen3-ASR implementation returns modality type IDs as labels. QASR keeps
        the same public contract while constructing the intended causal labels.
        Prompt masking is added by :meth:`prepare_training_batch`, where the
        prompt boundary is known.
        """
        if output_labels:
            kwargs["return_mm_token_type_ids"] = True
        model_inputs = super().__call__(
            text=text,
            audio=audio,
            output_labels=False,
            **kwargs,
        )
        if output_labels:
            mm_token_type_ids = model_inputs.pop("mm_token_type_ids")
            labels = model_inputs["input_ids"].clone()
            labels[mm_token_type_ids != 0] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
        return BatchFeature(data=model_inputs, tensor_type="pt")

    def _get_audio_token_length(self, audio_lengths, n_window=50):
        del n_window
        if not torch.is_tensor(audio_lengths):
            audio_lengths = torch.as_tensor(audio_lengths)
        lengths = get_subsampling_output_lengths(
            audio_lengths,
            subsampling_factor=self.subsampling_factor,
            kernel_size=self.subsampling_conv_kernel_size,
            stride=self.subsampling_conv_stride,
        )
        return lengths.cpu().numpy()

    def apply_transcription_request(
        self,
        audio: AudioInput | list[AudioInput],
        language: str | list[str] | None = None,
        **kwargs: Any,
    ):
        audio_items = make_list_of_audio_chat_template(audio)
        if language is None:
            languages: list[str | None] = [None] * len(audio_items)
        else:
            language_rows = _normalize_rows(language, len(audio_items), "language")
            languages = [resolve_qasr_language(item) for item in language_rows]

        conversations = []
        for language_name, audio_item in zip(languages, audio_items):
            messages = []
            if language_name is not None:
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": language_name}]}
                )
            messages.append(
                {"role": "user", "content": [_audio_content_item(audio_item)]}
            )
            conversations.append(messages)
        return_tensors = kwargs.pop("return_tensors", None)
        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors=return_tensors,
            processor_kwargs=kwargs,
        )

    def prepare_training_batch(
        self,
        audio: AudioInput | list[AudioInput],
        text: str | Sequence[str],
        language: str | Sequence[str],
        **kwargs: Any,
    ):
        audio_items = make_list_of_audio_chat_template(audio)
        texts = _normalize_rows(text, len(audio_items), "text")
        language_rows = _normalize_rows(language, len(audio_items), "language")
        conversations = []
        language_names = []
        for audio_item, transcript, language_code in zip(audio_items, texts, language_rows):
            language_name = resolve_qasr_language(language_code)
            language_names.append(language_name)
            conversations.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": language_name}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": transcript},
                            _audio_content_item(audio_item),
                        ],
                    },
                ]
            )

        # The native Qwen fine-tuning sequence is the prompt-only chat text
        # followed by the model's language marker, transcript, and ChatML EOS.
        prefix_texts = self.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        eos = self.tokenizer.eos_token or ""
        targets = [
            f"language {language_name}<asr_text>{transcript}{eos}"
            for language_name, transcript in zip(language_names, texts)
        ]
        full_texts = [prefix + target for prefix, target in zip(prefix_texts, targets)]
        inputs = self(
            text=full_texts,
            audio=audio_items,
            output_labels=True,
            **kwargs,
        )

        # Mask the full prompt. Re-expand its single audio placeholder using
        # the same count that the processor inserted into the full sequence.
        for row, prefix_text in enumerate(prefix_texts):
            audio_token_count = int(inputs["input_ids"][row].eq(self.audio_token_id).sum())
            expanded_prefix = prefix_text.replace(
                self.audio_token,
                self.audio_token * audio_token_count,
            )
            prefix_ids = self.tokenizer(
                expanded_prefix,
                add_special_tokens=False,
            )["input_ids"]
            valid_positions = inputs["attention_mask"][row].to(dtype=torch.bool).nonzero().flatten()
            if len(prefix_ids) > len(valid_positions):
                raise RuntimeError("Qwen prompt is longer than the prepared full sequence")
            actual_prefix = inputs["input_ids"][row, valid_positions[: len(prefix_ids)]]
            expected_prefix = torch.as_tensor(
                prefix_ids,
                dtype=actual_prefix.dtype,
                device=actual_prefix.device,
            )
            if not torch.equal(actual_prefix, expected_prefix):
                raise RuntimeError("Could not locate the Qwen prompt boundary in training inputs")
            inputs["labels"][row, valid_positions[: len(prefix_ids)]] = -100
        return inputs


__all__ = ["LANGUAGE_NAMES", "QASRProcessor", "resolve_qasr_language"]
