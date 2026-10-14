"""Load only the requested tower from the pinned official safetensors checkpoint."""

import json
from pathlib import Path

import torch
from torch import nn


def resolve_file(model_id, filename, revision, local_files_only=False):
    if Path(model_id).is_dir():
        return str(Path(model_id) / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename, revision=revision, local_files_only=local_files_only)


def component_from_config(config, task):
    from transformers import (
        EmbeddingGemma2TextConfig,
        EmbeddingGemma2TextModel,
        Gemma4AudioConfig,
        Gemma4AudioModel,
    )

    if task == "asr":
        return Gemma4AudioModel(Gemma4AudioConfig(**config))
    return EmbeddingGemma2TextModel(EmbeddingGemma2TextConfig(**config))


def load_component(model_config):
    from safetensors import safe_open

    resolve = lambda name: resolve_file(  # noqa: E731
        model_config.model_id, name, model_config.revision, model_config.local_files_only
    )
    config = json.loads(Path(resolve("config.json")).read_text())
    if config.get("model_type") != "embedding_gemma2":
        raise ValueError("expected an EmbeddingGemma 2 checkpoint")
    task = model_config.task
    component = component_from_config(
        config["audio_config" if task == "asr" else "text_config"], task
    )
    prefix = "audio_tower." if task == "asr" else "language_model."
    # The pinned upstream release has one safetensors file. No remote Python is executed.
    with safe_open(resolve("model.safetensors"), framework="pt", device="cpu") as weights:
        state = {
            key.removeprefix(prefix): weights.get_tensor(key)
            for key in weights.keys()
            if key.startswith(prefix)
        }
    if not state:
        raise ValueError(f"no {prefix} weights found; unsupported checkpoint layout")
    component.load_state_dict(state, strict=True)
    return component


class Backbone(nn.Module):
    def __init__(self, component, train_last=0):
        super().__init__()
        if train_last > len(component.layers):
            raise ValueError("train_backbone_layers exceeds the number of backbone layers")
        self.component = component
        self.train_last = train_last
        component.requires_grad_(False)
        if train_last:
            for layer in component.layers[-train_last:]:
                layer.requires_grad_(True)
            for name in ("output_proj", "norm", "embedding_projection"):
                if module := getattr(component, name, None):
                    module.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.component.eval()
        if mode and self.train_last:
            for layer in self.component.layers[-self.train_last :]:
                layer.train()
        return self

    def forward(self, **inputs):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_last > 0):
            return self.component(**inputs, return_dict=True)
