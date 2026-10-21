"""Variant-specific audio frontends, with trained text prompts kept intact."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    Gemma4AudioFeatureExtractor,
    PreTrainedTokenizerFast,
    WhisperFeatureExtractor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

from .chat_processing import AUDChatProcessor, resolve_language
from .configuration import AUDConfig
from .feature_extraction import AUDCohereFeatureExtractor


class AUDTurboChatProcessor(AUDChatProcessor):
    def __init__(self, feature_extractor, tokenizer, chat_template, n_window):
        Qwen3ASRProcessor.__init__(self, feature_extractor, tokenizer, chat_template=chat_template)
        self.n_window = n_window

    def _process_audio(self, audio, **kwargs):
        for key in ("n_window", "chunk_long_audio", "max_length"):
            kwargs.pop(key, None)
        kwargs.update(
            truncation=False, padding="longest", return_attention_mask=True, return_tensors="pt"
        )
        result = self.feature_extractor(audio, **kwargs)
        result["input_features_mask"] = result.pop("attention_mask")
        result["num_audio_tokens"] = self._get_audio_token_length(
            result["input_features_mask"].sum(-1)
        )
        return result, [self.replace_audio_token(result, idx) for idx in range(len(audio))]

    def _get_audio_token_length(self, lengths, n_window=None):
        return (
            _get_feat_extract_output_lengths(torch.as_tensor(lengths), self.n_window).cpu().numpy()
        )


class AUDRawFeatureExtractor:
    sampling_rate = 16000
    model_input_names = ["input_features", "input_features_mask"]

    def __init__(self, frame_samples=80, max_audio_clip_s=35.0):
        self.frame_samples = int(frame_samples)
        self.max_audio_clip_s = max_audio_clip_s

    def __call__(self, waves, **kwargs):
        lengths = torch.tensor([len(w) for w in waves])
        width = int(
            (lengths.max() + self.frame_samples - 1) // self.frame_samples * self.frame_samples
        )
        features = torch.zeros(len(waves), width)
        for i, wave in enumerate(waves):
            features[i, : len(wave)] = torch.as_tensor(wave)
        return BatchFeature(
            {
                "input_features": features,
                "input_features_mask": torch.arange(width)[None] < lengths[:, None],
            }
        )

    def save_pretrained(self, directory):
        Path(directory, "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "feature_extractor_type": "AUDRawFeatureExtractor",
                    "sampling_rate": 16000,
                    "frame_samples": self.frame_samples,
                    "max_audio_clip_s": self.max_audio_clip_s,
                },
                indent=2,
            )
            + "\n"
        )


class AUDProcessor:
    model_input_names = ["input_ids", "attention_mask", "input_features", "input_features_mask"]

    def __init__(self, config, tokenizer, feature_extractor, chat_template=None):
        self.config, self.tokenizer, self.feature_extractor = config, tokenizer, feature_extractor
        self.audio_token_id = config.audio_token_id
        self.chat_template = chat_template
        self.chat = None
        if config.variant == "audpro":
            a = config.audio_config
            self.chat = AUDChatProcessor(
                feature_extractor,
                tokenizer,
                chat_template,
                subsampling_factor=a.subsampling_factor,
                subsampling_conv_kernel_size=a.subsampling_conv_kernel_size,
                subsampling_conv_stride=a.subsampling_conv_stride,
                projector_pool_stride=config.projector_pool_stride,
            )
        elif config.variant == "audturbo":
            self.chat = AUDTurboChatProcessor(
                feature_extractor, tokenizer, chat_template, config.audio_config.n_window
            )
        if self.chat is not None and self.chat.audio_token_id != config.audio_token_id:
            raise ValueError("Tokenizer audio token ID differs from model config")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The trained tokenizer must have a pad token")
        if max(self.tokenizer.get_vocab().values()) >= config.text_config.vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds the trained decoder embeddings")
        if not 0 <= self.audio_token_id < config.text_config.vocab_size:
            raise ValueError("Audio token ID is outside the decoder vocabulary")
        if self.chat is not None and not chat_template:
            raise ValueError("The trained QASR chat template is required")

    @classmethod
    def from_pretrained(cls, directory, **kwargs):
        directory = Path(directory)
        config = AUDConfig.from_pretrained(directory, local_files_only=True)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True)
        data = json.loads((directory / "preprocessor_config.json").read_text())
        if data.get("sampling_rate", 16000) != 16000:
            raise ValueError("AUD currently requires 16 kHz audio")
        if config.variant == "audpro":
            frontend = AUDCohereFeatureExtractor.from_dict(data)
        elif config.variant == "audturbo":
            frontend = WhisperFeatureExtractor.from_dict(data)
            if frontend.feature_size != config.audio_config.num_mel_bins:
                raise ValueError("Captioner mel bins differ from encoder configuration")
        elif config.flash_encoder == "gemma4":
            frontend = Gemma4AudioFeatureExtractor.from_dict(data)
            if frontend.feature_size != config.audio_config.subsampling_conv_channels[0]:
                raise ValueError("Gemma frontend feature size differs from encoder")
        else:
            frontend = AUDRawFeatureExtractor(data["frame_samples"], config.max_audio_seconds)
        template = directory / "chat_template.jinja"
        return cls(
            config, tokenizer, frontend, template.read_text() if template.is_file() else None
        )

    def save_pretrained(self, directory, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(directory)
        self.feature_extractor.save_pretrained(directory)
        if self.chat_template:
            (directory / "chat_template.jinja").write_text(self.chat_template)
        (directory / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": "AUDProcessor",
                    "variant": self.config.variant,
                },
                indent=2,
            )
            + "\n"
        )

    def _waves(self, audio, sampling_rate):
        if sampling_rate != 16000:
            raise ValueError("Resample to 16 kHz before calling AUDProcessor")
        if isinstance(audio, (np.ndarray, torch.Tensor)) and audio.ndim == 1:
            audio = [audio]
        waves = [np.asarray(w, dtype=np.float32) for w in audio]
        if not waves:
            raise ValueError("Empty audio batch")
        for wave in waves:
            if wave.ndim != 1 or len(wave) < 160 or not np.isfinite(wave).all():
                raise ValueError("Expected finite mono waveforms with at least 160 samples")
            if len(wave) > round(16000 * self.config.max_audio_seconds):
                raise ValueError("Audio exceeds max_audio_seconds; segment it before processing")
        return waves

    def audio_token_lengths(self, mask):
        lengths = mask.long().sum(-1)
        if self.chat is not None:
            return torch.as_tensor(self.chat._get_audio_token_length(lengths), device=mask.device)
        if self.config.flash_encoder == "gemma4":
            lengths = (lengths + 3) // 4  # two stride-2 convolutions, mask[:, ::2]
            return (lengths + self.config.flash_audio_stride - 1) // self.config.flash_audio_stride
        frames = lengths // self.feature_extractor.frame_samples
        return ((frames + 1) // 2 + 1) // 2

    def _flash_batch(self, waves, texts=None, languages=None):
        result = self.feature_extractor(
            waves,
            sampling_rate=16000,
            padding=True,
            truncation=False,
            max_length=None,
            return_tensors="pt",
        )
        lengths = self.audio_token_lengths(result["input_features_mask"])
        rows, label_rows = [], []
        if texts is not None and (
            languages is None or len(texts) != len(waves) or len(languages) != len(waves)
        ):
            raise ValueError("Training requires one transcript and language per waveform")
        for i, length in enumerate(lengths.tolist()):
            prefix = (
                self.config.prompt_prefix
                + [self.audio_token_id] * length
                + self.config.prompt_suffix
            )
            target = []
            if texts is not None:
                target = self.tokenizer.encode(
                    f"language {resolve_language(languages[i])}<asr_text>{texts[i]}",
                    add_special_tokens=False,
                ) + [self.config.eos_token_id]
            rows.append(prefix + target)
            label_rows.append([-100] * len(prefix) + target)
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, row in enumerate(rows):
            start = 0 if texts is not None else width - len(row)
            ids[i, start : start + len(row)] = torch.tensor(row)
            attention[i, start : start + len(row)] = 1
            labels[i, start : start + len(row)] = torch.tensor(label_rows[i])
        result.update(input_ids=ids, attention_mask=attention)
        if texts is not None:
            result["labels"] = labels
        return result

    def apply_transcription_request(self, audio, language=None, sampling_rate=16000, **kwargs):
        waves = self._waves(audio, sampling_rate)
        if self.chat is None:
            return self._flash_batch(waves)
        self.tokenizer.padding_side = "left"
        kwargs.update(return_tensors="pt", padding=True, sampling_rate=16000)
        return self.chat.apply_transcription_request(waves, language=language, **kwargs)

    def prepare_training_batch(self, audio, text, language, sampling_rate=16000, **kwargs):
        waves = self._waves(audio, sampling_rate)
        texts = [text] * len(waves) if isinstance(text, str) else list(text)
        languages = [language] * len(waves) if isinstance(language, str) else list(language)
        if self.chat is None:
            return self._flash_batch(waves, texts, languages)
        self.tokenizer.padding_side = "right"
        kwargs.update(return_tensors="pt", padding=True, sampling_rate=16000)
        return self.chat.prepare_training_batch(waves, texts, languages, **kwargs)

    def decode(self, ids, return_format="transcription_only", **kwargs):
        if self.chat is not None:
            return self.chat.decode(ids, return_format=return_format, **kwargs)
        raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        return (
            raw.split("<asr_text>", 1)[-1]
            .split("<|im_end|>", 1)[0]
            .split("<|endoftext|>", 1)[0]
            .strip()
        )
