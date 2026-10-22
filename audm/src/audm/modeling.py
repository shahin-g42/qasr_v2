"""AUD tensor namespace over pretrained primitives. See NOTICE.md for attribution."""

from typing import ClassVar

import torch
from torch import nn
from transformers import Gemma4AudioModel, PreTrainedModel, Qwen3ForCausalLM
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    MoonshineStreamingCausalConv1d,
    MoonshineStreamingEncoder,
)
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoder
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from .configuration import AUDConfig
from .lengths import get_projector_pool_output_lengths


class AUDMoonshineCausalConv1d(MoonshineStreamingCausalConv1d):
    def forward(self, x, mask=None):
        # Upstream's any-overlap mask marks padded right tails as valid. That
        # changes the token count and look-ahead context when batching lengths.
        # Preserve the convolution; retain only outputs on the valid time grid.
        x = nn.Conv1d.forward(self, nn.functional.pad(x, (self.left_pad, 0)))
        if mask is not None:
            lengths = (mask.long().sum(-1) + self.stride[0] - 1) // self.stride[0]
            mask = torch.arange(x.shape[-1], device=x.device)[None] < lengths[:, None]
            x = x * mask[:, None]
        return x, mask


def name_aud_layers(module):
    """Instance-local subclasses: preserve kernels, hooks and isinstance contracts."""
    prefixes = {
        "Qwen3OmniMoe": "AUDOmni",
        "MoonshineStreaming": "AUDMoonshine",
        "Gemma4Audio": "AUDGemma4Audio",
        "Parakeet": "AUDCohere",
        "Qwen3": "AUDText",
    }
    for child in module.modules():
        cls = type(child)
        for prefix, replacement in prefixes.items():
            if cls.__name__.startswith(prefix):
                name = replacement + cls.__name__[len(prefix) :]
                if name not in globals():
                    globals()[name] = type(name, (cls,), {"__module__": __name__})
                child.__class__ = globals()[name]
                break


class AUDProProjector(nn.Module):
    """Exact QASR bridge, including trained deep residual blocks."""

    def __init__(self, config):
        super().__init__()
        size, depth = config.audio_config.hidden_size, config.projector_depth
        self.input_layernorm = nn.LayerNorm(size) if depth > 1 else None
        self.linear_1 = nn.Linear(size, size)
        self.act = ACT2FN[config.projector_hidden_act]
        self.hidden_layers = (
            nn.ModuleList(
                nn.Sequential(nn.LayerNorm(size), nn.Linear(size, size)) for _ in range(depth - 1)
            )
            if depth > 1
            else None
        )
        self.linear_2 = nn.Linear(size, config.text_config.hidden_size)

    def forward(self, hidden):
        hidden = self.input_layernorm(hidden) if self.input_layernorm is not None else hidden
        hidden = self.act(self.linear_1(hidden))
        for block in self.hidden_layers or []:
            hidden = hidden + self.act(block(hidden))
        return self.linear_2(hidden)


class AUDAdapter(nn.Module):
    def __init__(self, input_size, output_size, width, residual=False):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.down = nn.Linear(input_size, width)
        self.up = nn.Linear(width, output_size)
        self.residual = residual and input_size == output_size
        if self.residual:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        projected = self.up(nn.functional.gelu(self.down(self.norm(hidden))))
        return hidden + projected if self.residual else projected


class AUDPoolingConv(nn.Module):
    def __init__(self, size, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(size, size, 3, stride=stride, padding=1)

    def forward(self, hidden, lengths):
        hidden = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        lengths = get_projector_pool_output_lengths(lengths, pool_stride=self.stride)
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
        return hidden, mask


class AUDForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = AUDConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules: ClassVar[list[str]] = [
        "AUDCohereEncoderBlock",
        "AUDOmniAudioEncoderLayer",
        "AUDMoonshineEncoderLayer",
        "AUDGemma4AudioLayer",
        "AUDTextDecoderLayer",
    ]
    _tied_weights_keys = {"aud_decoder.lm_head.weight": "aud_decoder.model.embed_tokens.weight"}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        report_requested = kwargs.pop("output_loading_info", False)
        model, report = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        if any(
            report.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"Incomplete AUD checkpoint: {report}")
        if model.aud_ctc_head is not None:
            model.aud_ctc_head.requires_grad_(False)
        return (model, report) if report_requested else model

    def __init__(self, config):
        super().__init__(config)
        constructors = {
            "audpro": ParakeetEncoder,
            "audturbo": Qwen3OmniMoeAudioEncoder,
            "audflash": Gemma4AudioModel
            if config.flash_encoder == "gemma4"
            else MoonshineStreamingEncoder,
        }
        self.aud_encoder = constructors[config.variant](config.audio_config)
        self.aud_decoder = Qwen3ForCausalLM(config.text_config)
        self.aud_pool = None
        if config.variant == "audpro":
            self.aud_projector = AUDProProjector(config)
            if config.projector_pool_stride > 1:
                self.aud_pool = AUDPoolingConv(
                    config.audio_config.hidden_size, config.projector_pool_stride
                )
        else:
            size = (
                config.audio_config.output_dim
                if config.variant == "audturbo"
                else config.audio_config.output_proj_dims
                if config.flash_encoder == "gemma4"
                else config.audio_config.hidden_size
            )
            self.aud_projector = AUDAdapter(
                size,
                config.text_config.hidden_size,
                config.adapter_width,
                residual=config.variant == "audturbo",
            )
        self.aud_ctc_head = (
            nn.Linear(config.audio_config.hidden_size, config.ctc_vocab_size, bias=False)
            if config.ctc_vocab_size
            else None
        )
        if self.aud_ctc_head is not None:
            self.aud_ctc_head.requires_grad_(False)  # retained scaffold; no implicit CTC objective
        name_aud_layers(self)
        # HF also registers tied-weight and output-hook metadata in post_init.
        # Nested pretrained modules carry _is_hf_initialized, so their math is untouched.
        self.post_init()
        if isinstance(self.aud_projector, AUDAdapter) and self.aud_projector.residual:
            nn.init.zeros_(self.aud_projector.up.weight)
            nn.init.zeros_(self.aud_projector.up.bias)

    def get_input_embeddings(self):
        return self.aud_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.aud_decoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.aud_decoder.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.aud_decoder.set_output_embeddings(value)

    def configure_training(self, stage="full"):
        if stage not in ("adapter", "full", "decoder"):
            raise ValueError("stage must be adapter, full or decoder")
        if self.config.audio_type_version and stage == "adapter":
            raise ValueError(
                "Audio-type training requires an unfrozen decoder: use decoder or full stage"
            )
        self.aud_encoder.requires_grad_(stage == "full")
        self.aud_decoder.requires_grad_(stage in ("full", "decoder"))
        self.aud_projector.requires_grad_(True)
        if self.aud_pool is not None:
            self.aud_pool.requires_grad_(True)
        if self.aud_ctc_head is not None:
            self.aud_ctc_head.requires_grad_(False)
        self.config.use_cache = False
        self.aud_decoder.config.use_cache = False

    def train(self, mode=True):
        super().train(mode)
        # A frozen encoder needs no backward graph. The decoder does: adapter
        # gradients pass through it, and eval() would disable its checkpointing.
        if not any(p.requires_grad for p in self.aud_encoder.parameters()):
            self.aud_encoder.eval()
        # Match QASR: preserve pretrained Conformer running statistics.
        for module in self.aud_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def encode_audio(self, input_features, input_features_mask):
        if input_features_mask is None or input_features_mask.ndim != 2:
            raise ValueError("AUD requires a 2D mask with valid audio lengths")
        mask = input_features_mask.bool()
        lengths = mask.sum(-1)
        if (lengths <= 0).any() or not torch.equal(
            mask, torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        ):
            raise ValueError("Audio masks must describe nonempty, right-padded sequences")
        dtype = next(self.aud_encoder.parameters()).dtype
        features = input_features.to(dtype=dtype)
        if self.config.variant == "audturbo":
            if (
                features.ndim != 3
                or features.shape[0] != mask.shape[0]
                or features.shape[2] != mask.shape[1]
            ):
                raise ValueError("AudTurbo expects [batch, mel, frames] features")
            packed = features.transpose(1, 2)[mask].transpose(0, 1)
            hidden = self.aud_encoder(input_features=packed, feature_lens=lengths).last_hidden_state
            lengths = _get_feat_extract_output_lengths(lengths, self.config.audio_config.n_window)
        else:
            if self.config.variant == "audflash" and self.config.flash_encoder == "moonshine":
                if features.shape != mask.shape:
                    raise ValueError("AudFlash expects [batch, samples] raw waveforms")
                output = self.aud_encoder(input_values=features, attention_mask=mask)
            else:
                if features.shape[:2] != mask.shape:
                    raise ValueError("Mel encoders expect [batch, frames, mel] features")
                output = self.aud_encoder(input_features=features, attention_mask=mask)
            hidden, mask = output.last_hidden_state, output.attention_mask.bool()
            if self.config.variant == "audflash" and self.config.flash_encoder == "gemma4":
                # Preserve GQASR: project first, then average valid adjacent frames.
                hidden = self.aud_projector(hidden).masked_fill(~mask[..., None], 0)
                if self.config.flash_audio_stride == 2:
                    padding = hidden.shape[1] % 2
                    hidden = nn.functional.pad(hidden, (0, 0, 0, padding))
                    mask = nn.functional.pad(mask, (0, padding))
                    count = mask.reshape(mask.shape[0], -1, 2).sum(-1)
                    hidden = hidden.reshape(hidden.shape[0], -1, 2, hidden.shape[-1]).sum(2)
                    hidden = hidden / count.clamp_min(1)[..., None]
                    mask = count > 0
                return hidden[mask], mask.sum(-1)
            if self.aud_pool is not None:
                hidden, mask = self.aud_pool(hidden, mask.sum(-1))
            lengths = mask.sum(-1)
            hidden = hidden[mask]
        return self.aud_projector(hidden), lengths

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        input_features_mask=None,
        inputs_embeds=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if input_features is not None:
            if input_ids is None:
                raise ValueError("input_ids are required to locate audio placeholders")
            audio, lengths = self.encode_audio(input_features, input_features_mask)
            placeholders = input_ids.eq(self.config.audio_token_id)
            counts = placeholders.sum(-1)
            if counts.shape != lengths.shape or not torch.equal(counts.to(lengths.device), lengths):
                raise ValueError(
                    f"Per-row audio placeholder mismatch: {counts.tolist()} vs {lengths.tolist()}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                placeholders.unsqueeze(-1).expand_as(inputs_embeds), audio.to(inputs_embeds)
            )
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            position_ids = position_ids[:, -inputs_embeds.shape[1] :]
        return self.aud_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        **kwargs,
    ):
        features = kwargs.pop("input_features", None)
        mask = kwargs.pop("input_features_mask", None)
        result = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if is_first_iteration or not kwargs.get("use_cache", True):
            result.update(input_features=features, input_features_mask=mask)
        return result

    def save_pretrained(self, save_directory, *args, **kwargs):
        kwargs.setdefault("max_shard_size", "100GB")
        return super().save_pretrained(save_directory, *args, **kwargs)
