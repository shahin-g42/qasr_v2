"""GQASR public model names and tensor namespace.

The computational primitives derive from the upstream architectures identified
in provenance.json. Local subclasses change identifiers, not their mathematics.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Gemma4AudioConfig, Gemma4AudioModel, Qwen3Config, Qwen3ForCausalLM
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioLayer


class GQASRAcousticConfig(Gemma4AudioConfig):
    model_type = "gqasr_acoustic"


class GQASRDecoderConfig(Qwen3Config):
    model_type = "gqasr_decoder"


def public_config(config, model_type):
    data = dict(config)
    for key in ("_name_or_path", "architectures", "auto_map", "transformers_version"):
        data.pop(key, None)
    data["model_type"] = model_type
    return data


class GQASRAcousticLayer(Gemma4AudioLayer):
    gradient_checkpointing = False

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
        return super().forward(*args, **kwargs)


def name_core_modules(module):
    """Give this instance's architecture-specific classes local public identifiers.

    No upstream class is mutated. Subclasses retain isinstance compatibility with
    HF output hooks. Checkpoints serialize tensors/configs, never Python objects.
    """
    for child in module.modules():
        cls = type(child)
        if cls.__name__.startswith(("Gemma4", "Qwen3")):
            name = (
                cls.__name__.replace("Gemma4Audio", "GQASRAcoustic")
                .replace("Gemma4", "GQASR")
                .replace("Qwen3", "GQASRText")
            )
            if name not in globals():
                globals()[name] = type(name, (cls,), {"__module__": __name__})
            child.__class__ = globals()[name]


@dataclass
class GQASRConfig:
    acoustic_config: dict
    decoder_config: dict
    prompt_prefix: list[int]
    prompt_suffix: list[int]
    eos_id: int
    bridge_width: int = 384
    audio_stride: int = 2
    model_type: str = "gqasr"
    architectures: tuple = ("GQASRForSpeechRecognition",)
    format_version: int = 1

    def __post_init__(self):
        if self.model_type != "gqasr" or self.format_version != 1:
            raise ValueError("unsupported GQASR model format")
        if self.audio_stride not in (1, 2) or self.bridge_width < 1:
            raise ValueError("invalid bridge width or temporal stride")
        self.acoustic_config = public_config(self.acoustic_config, "gqasr_acoustic")
        self.decoder_config = public_config(self.decoder_config, "gqasr_decoder")


class GQASRAcousticStack(nn.Module):
    def __init__(self, component):
        super().__init__()
        self.core = component
        self.train_last = 0

    def train(self, mode=True):
        super().train(mode)
        if self.train_last == len(self.core.layers):
            self.core.train(mode)
        else:
            self.core.eval()
            if mode and self.train_last:
                for layer in self.core.layers[-self.train_last :]:
                    layer.train()
        return self

    def forward(self, **inputs):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_last > 0):
            return self.core(**inputs, return_dict=True)


class GQASRForSpeechRecognition(nn.Module):
    def __init__(self, config, acoustic=None, decoder=None):
        super().__init__()
        self.config = config
        acoustic = acoustic or Gemma4AudioModel(GQASRAcousticConfig(**config.acoustic_config))
        decoder = decoder or Qwen3ForCausalLM(GQASRDecoderConfig(**config.decoder_config))
        name_core_modules(acoustic)
        name_core_modules(decoder)
        self.gqasr_acoustic = GQASRAcousticStack(acoustic)
        self.gqasr_decoder = decoder
        self.gqasr_bridge = nn.Sequential(
            nn.LayerNorm(acoustic.config.output_proj_dims),
            nn.Linear(acoustic.config.output_proj_dims, config.bridge_width),
            nn.GELU(),
            nn.Linear(config.bridge_width, decoder.config.hidden_size),
        )
        self.register_buffer(
            "gqasr_prompt_prefix", torch.tensor(config.prompt_prefix, dtype=torch.long)
        )
        self.register_buffer(
            "gqasr_prompt_suffix", torch.tensor(config.prompt_suffix, dtype=torch.long)
        )
        self.eos_id = config.eos_id
        self.label_smoothing = 0.0
        self.configure_training()

    def configure_training(
        self,
        acoustic_layers=0,
        decoder_layers=0,
        train_embeddings=False,
        gradient_checkpointing=False,
        label_smoothing=0.0,
    ):
        audio, decoder = self.gqasr_acoustic.core, self.gqasr_decoder
        if not 0 <= acoustic_layers <= len(audio.layers) or not 0 <= decoder_layers <= len(
            decoder.model.layers
        ):
            raise ValueError("trainable layer count exceeds architecture")
        if not 0 <= label_smoothing < 1:
            raise ValueError("invalid label smoothing")
        self.label_smoothing = label_smoothing
        audio.requires_grad_(False)
        decoder.requires_grad_(False)
        self.gqasr_acoustic.train_last = acoustic_layers
        if acoustic_layers == len(audio.layers):
            audio.requires_grad_(True)
        elif acoustic_layers:
            for layer in audio.layers[-acoustic_layers:]:
                layer.requires_grad_(True)
            audio.output_proj.requires_grad_(True)
        if decoder_layers:
            for layer in decoder.model.layers[-decoder_layers:]:
                layer.requires_grad_(True)
            decoder.model.norm.requires_grad_(True)
        if train_embeddings:
            decoder.get_input_embeddings().requires_grad_(True)
            decoder.get_output_embeddings().requires_grad_(True)
        for layer in audio.layers:
            layer.gradient_checkpointing = bool(gradient_checkpointing and acoustic_layers)
        if gradient_checkpointing and decoder_layers:
            decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            decoder.gradient_checkpointing_disable()
        self.train(self.training)

    def save_pretrained(self, directory):
        from safetensors.torch import save_model

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        save_model(
            self,
            str(directory / "model.safetensors"),
            metadata={"format": "pt", "model_type": "gqasr"},
        )

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        from safetensors.torch import load_model

        directory = Path(directory)
        config = GQASRConfig(**json.loads((directory / "config.json").read_text()))
        model = cls(config)
        load_model(model, str(directory / "model.safetensors"), strict=True)
        return model.to(device)

    def audio_embeddings(self, features, mask):
        output = self.gqasr_acoustic(input_features=features, attention_mask=mask)
        x, valid = self.gqasr_bridge(output.last_hidden_state), output.attention_mask.bool()
        if torch.any(valid.sum(-1) == 0):
            raise ValueError("audio produced no valid frames")
        x = x.masked_fill(~valid[..., None], 0)
        if self.config.audio_stride == 2:
            padding = x.shape[1] % 2
            x = F.pad(x, (0, 0, 0, padding))
            valid = F.pad(valid, (0, padding))
            count = valid.reshape(valid.shape[0], -1, 2).sum(-1)
            x = x.reshape(x.shape[0], -1, 2, x.shape[-1]).sum(2) / count.clamp_min(1)[..., None]
            valid = count > 0
        return x, valid

    def fuse(self, audio, valid, targets=None, target_mask=None, left_pad=False):
        embed = self.gqasr_decoder.get_input_embeddings()
        prefix, suffix = embed(self.gqasr_prompt_prefix), embed(self.gqasr_prompt_suffix)
        sequences, label_rows = [], []
        for i in range(audio.shape[0]):
            prompt = torch.cat([prefix, audio[i, valid[i]], suffix]).to(audio.dtype)
            if targets is None:
                sequences.append(prompt)
                continue
            ids = targets[i, target_mask[i]]
            sequences.append(torch.cat([prompt, embed(ids).to(audio.dtype)]))
            label_rows.append(torch.cat([ids.new_full((len(prompt),), -100), ids]))
        lengths = torch.tensor([len(s) for s in sequences], device=audio.device)
        if int(lengths.max()) > self.gqasr_decoder.config.max_position_embeddings:
            raise ValueError("audio and text exceed GQASR context; segment the input")
        width = int(lengths.max())
        padded = torch.stack(
            [
                F.pad(s, (0, 0, width - len(s), 0) if left_pad else (0, 0, 0, width - len(s)))
                for s in sequences
            ]
        )
        mask = length_mask(lengths, width)
        if left_pad:
            mask = mask.flip(1)
        labels = (
            None
            if targets is None
            else torch.stack([F.pad(row, (0, width - len(row)), value=-100) for row in label_rows])
        )
        return padded, mask, labels

    def forward(self, input_features, input_features_mask, targets, target_mask):
        audio, valid = self.audio_embeddings(input_features, input_features_mask)
        embeds, mask, labels = self.fuse(audio, valid, targets, target_mask)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.gqasr_decoder.model(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        # Compute vocabulary logits only at supervised positions, avoiding audio-prefix logits.
        selected = labels[:, 1:] != -100
        if not selected.any():
            raise ValueError("ASR batch has no supervised target tokens")
        logits = self.gqasr_decoder.lm_head(hidden[:, :-1][selected])
        return {
            "loss": F.cross_entropy(
                logits.float(), labels[:, 1:][selected], label_smoothing=self.label_smoothing
            )
        }

    @torch.no_grad()
    def generate_transcripts(self, input_features, input_features_mask, max_new_tokens=256):
        audio, valid = self.audio_embeddings(input_features, input_features_mask)
        embeds, mask, _ = self.fuse(audio, valid, left_pad=True)
        return self.gqasr_decoder.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.eos_id,
            pad_token_id=self.eos_id,
        )


def length_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]
