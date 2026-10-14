"""Self-contained speech checkpoints; inference never requires the upstream weights."""

import os
from pathlib import Path

import torch

from .asr import ASRModel
from .backbone import component_from_config
from .config import ModelConfig
from .text import SpeechTokenizer
from .tts import TTSModel


def build_model(config, component, tokenizer, speakers, languages, decoder=None):
    if config.task == "asr":
        if config.asr_decoder == "qwen":
            from .qwen_asr import QwenASRModel, load_qwen_decoder

            return QwenASRModel(component, decoder or load_qwen_decoder(config), config, tokenizer)
        return ASRModel(component, config, len(tokenizer))
    return TTSModel(component, config, len(tokenizer), len(speakers), len(languages))


def save_checkpoint(
    path,
    model,
    config,
    tokenizer,
    speakers,
    languages,
    assets,
    step,
    optimizer=None,
    run_metadata=None,
    training_state=None,
):
    path = Path(path)
    payload = {
        "format_version": 1,
        "config": config.to_dict(),
        "step": step,
        "model": model.state_dict(),
        "tokenizer": tokenizer.serialized,
        "speakers": speakers,
        "languages": languages,
        "assets": assets,
        "backbone_config": model.backbone.component.config.to_dict() if model.backbone else None,
        "run_metadata": run_metadata or {},
        "training_state": training_state or {},
        "tokenizer_kind": config.model.asr_decoder if config.model.task == "asr" else "ctc",
        "decoder_config": model.decoder.config.to_dict() if hasattr(model, "decoder") else None,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu", model_override=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config = model_override or ModelConfig(**payload["config"]["model"])
    decoder = None
    if payload.get("tokenizer_kind") == "qwen":
        from .qwen_asr import QwenTokenizer, decoder_from_config

        tokenizer = QwenTokenizer(payload["tokenizer"])
        decoder = decoder_from_config(payload["decoder_config"])
    else:
        tokenizer = SpeechTokenizer(payload["tokenizer"])
    component = (
        component_from_config(payload["backbone_config"], config.task)
        if payload["backbone_config"] is not None
        else None
    )
    model = build_model(
        config, component, tokenizer, payload["speakers"], payload["languages"], decoder
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), tokenizer, payload
