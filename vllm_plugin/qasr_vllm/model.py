"""vLLM model for QASR: HF Parakeet encoder + QASR projector + vLLM Qwen3 decoder.

Written against vLLM v0.26.0 (the image the cluster's corrector nodes run) and
modelled on the in-tree ``qwen2_audio.py`` / ``qwen3_asr.py`` /
``parakeet.py`` of that release. The encoder stays the transformers module the
model was trained with (as vLLM's own ``parakeet.py`` does), so audio
embeddings are bit-for-bit the training ones; only the 1.7B decoder runs on
vLLM's paged-attention kernels, which is where generation time goes.

Prompt contract (identical to training, see ``qasr.processing``):

    <|im_start|>system\\n{Language}<|im_end|>\\n
    <|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\\n
    <|im_start|>assistant\\n

and the model answers ``language {Language}<asr_text>{transcript}``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import MultiModalDataDict, PromptType, TokensPrompt
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config

from .audio import QASRAudioTower, QASRMelExtractor, subsampling_output_lengths
from .config import QASRVllmConfig

AUDIO_TOKEN = "<|audio_pad|>"
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
ASR_TEXT_TAG = "<asr_text>"
SAMPLE_RATE = 16_000
# Training drops clips over 35 s (data.py duration filter); the transcription
# endpoint chunks longer uploads at this boundary.
MAX_AUDIO_CLIP_S = 35
LANGUAGES = {"ar": "Arabic", "en": "English", "zh": "Chinese", "hi": "Hindi", "ml": "Malayalam"}


def build_prompt(language: str | None) -> str:
    """The exact training prompt; ``language`` is an ISO code, a name, or None."""
    name = LANGUAGES.get(language, language) if language else ""
    return (
        f"<|im_start|>system\n{name}<|im_end|>\n"
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def split_output(text: str) -> tuple[str | None, str]:
    """``"language Arabic<asr_text>..."`` -> ``("Arabic", "...")``."""
    if ASR_TEXT_TAG not in text:
        return None, text
    head, body = text.rsplit(ASR_TEXT_TAG, 1)
    head = head.strip()
    return (head[len("language "):].strip() if head.startswith("language ") else None), body


class QASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> QASRVllmConfig:
        return self.ctx.get_hf_config(QASRVllmConfig)

    def get_hf_processor(self, **kwargs: object) -> QASRProcessingInfo:
        # The checkpoint's processor_config.json names QASRProcessor, which
        # AutoProcessor cannot resolve here; everything it did lives below.
        return self

    @cached_property
    def mel(self) -> QASRMelExtractor:
        return QASRMelExtractor(feature_size=int(self.get_hf_config().audio_config.num_mel_bins))

    def num_audio_tokens(self, num_frames: int | torch.Tensor) -> torch.Tensor:
        audio = self.get_hf_config().audio_config
        return subsampling_output_lengths(
            torch.as_tensor(num_frames).reshape(-1),
            subsampling_factor=int(audio.subsampling_factor),
            kernel_size=int(audio.subsampling_conv_kernel_size),
            stride=int(audio.subsampling_conv_stride),
        )

    @property
    def audio_token_id(self) -> int:
        return int(self.get_hf_config().audio_token_id)

    def get_data_parser(self) -> MultiModalDataParser:
        # vLLM decodes files with soundfile at the native rate and averages
        # channels, as training does (qasr.audio.load_mono_audio). Resampling
        # must also match: training uses scipy resample_poly(up, down), which
        # is exactly vLLM's "scipy" method; the default "pyav" is not.
        return MultiModalDataParser(
            target_sr=SAMPLE_RATE,
            target_channels=1,
            audio_resample_method="scipy",
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        frames = self.mel.num_frames(MAX_AUDIO_CLIP_S * SAMPLE_RATE)
        return {"audio": int(self.num_audio_tokens(frames)[0])}


class QASRDummyInputsBuilder(BaseDummyInputsBuilder[QASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        return {
            "audio": self._get_dummy_audios(
                length=MAX_AUDIO_CLIP_S * SAMPLE_RATE,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


def _field_config(hf_inputs: Mapping[str, torch.Tensor]) -> Mapping[str, MultiModalFieldConfig]:
    return {
        "input_features": MultiModalFieldConfig.batched("audio"),
        "feature_lengths": MultiModalFieldConfig.batched("audio"),
    }


class QASRMultiModalProcessor(BaseMultiModalProcessor[QASRProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        ids = self.info.get_tokenizer().encode(prompt, add_special_tokens=False)
        audios = list(mm_data.get("audios") or mm_data.get("audio") or [])
        if not audios:
            return BatchFeature({"input_ids": [ids]}, tensor_type="pt")

        features, frames = zip(*(self.info.mel(a) for a in audios), strict=False)
        tokens = self.info.num_audio_tokens(torch.tensor(frames)).tolist()
        audio_id = self.info.audio_token_id
        if ids.count(audio_id) != len(audios):
            raise ValueError(
                f"prompt has {ids.count(audio_id)} {AUDIO_TOKEN} placeholder(s) for {len(audios)} audio item(s)"
            )
        # Expand each single placeholder to that clip's encoder length, as the
        # training processor does; vLLM then locates the runs via _get_prompt_updates.
        expanded: list[int] = []
        k = 0
        for t in ids:
            if t == audio_id:
                expanded.extend([audio_id] * tokens[k])
                k += 1
            else:
                expanded.append(t)

        padded = torch.zeros(len(features), max(frames), features[0].shape[-1])
        for i, f in enumerate(features):
            padded[i, : f.shape[0]] = f
        return BatchFeature(
            {
                "input_ids": torch.tensor([expanded]),
                "input_features": padded,
                "feature_lengths": torch.tensor(frames, dtype=torch.long),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audio_id = self.info.audio_token_id
        lengths = out_mm_kwargs.get_data().get("feature_lengths")
        tokens = [] if lengths is None else self.info.num_audio_tokens(lengths).tolist()

        def replacement(item_idx: int) -> list[int]:
            if tokens[item_idx] <= 0:
                raise ValueError("audio clip is too short to produce any encoder frame")
            return [audio_id] * tokens[item_idx]

        return [PromptReplacement(modality="audio", target=[audio_id], replacement=replacement)]


@MULTIMODAL_REGISTRY.register_processor(
    QASRMultiModalProcessor,
    info=QASRProcessingInfo,
    dummy_inputs=QASRDummyInputsBuilder,
)
class QASRForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsTranscription):
    supported_languages = LANGUAGES

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.audio_tower.": "audio_tower.encoder.",
            "model.multi_modal_projector.": "audio_tower.projector.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: QASRVllmConfig = vllm_config.model_config.hf_config
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = QASRAudioTower(config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen3ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        features = kwargs.pop("input_features", None)
        lengths = kwargs.pop("feature_lengths", None)
        if features is None:
            return []
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.reshape(-1).tolist()
        items = [features[i].reshape(-1, features[i].shape[-1])[: int(n)] for i, n in enumerate(lengths)]
        return tuple(self.audio_tower(items))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip = ["model.ctc_head."]  # training-only auxiliary head
        if self.config.tie_word_embeddings:
            skip.append("lm_head.")
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # ---- /v1/audio/transcriptions -------------------------------------------------

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig, task_type: str) -> SpeechToTextConfig:
        return SpeechToTextConfig(max_audio_clip_s=MAX_AUDIO_CLIP_S, sample_rate=SAMPLE_RATE)

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError("QASR only supports task_type='transcribe'")
        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        return TokensPrompt(
            prompt_token_ids=tokenizer.encode(build_prompt(stt_params.language), add_special_tokens=False),
            multi_modal_data={"audio": stt_params.audio},
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return split_output(text)[1]
