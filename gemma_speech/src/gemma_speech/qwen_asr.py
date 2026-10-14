"""EmbeddingGemma audio encoder → trained bridge → Qwen3-ASR's pretrained decoder.

The Qwen speech checkpoint uses ordinary 1-D positions on all three MRoPE axes;
therefore its text weights are compatible with native HF Qwen3's default 1-D RoPE.
The Qwen audio tower is never instantiated or retained here.
"""

import base64
import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone, resolve_file
from .layers import length_mask

LANGUAGES = {
    "ar": "Arabic",
    "en": "English",
    "hi": "Hindi",
    "zh": "Chinese",
    "ml": "Malayalam",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}


class QwenTokenizer:
    def __init__(self, serialized):
        from transformers import AutoTokenizer

        self.serialized = serialized
        data = json.loads(serialized)
        with tempfile.TemporaryDirectory() as directory:
            for name, encoded in data["files"].items():
                if Path(name).name != name:
                    raise ValueError("invalid tokenizer filename")
                (Path(directory) / name).write_bytes(base64.b64decode(encoded))
            self.hf = AutoTokenizer.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False
            )
        self.prefix = data["prefix"]
        self.suffix = data["suffix"]
        self.eos_id = data["eos_id"]

    @classmethod
    def from_pretrained(cls, config):
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(
            config.qwen_model_id,
            revision=config.qwen_revision,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        path = resolve_file(
            config.qwen_model_id,
            "chat_template.json",
            config.qwen_revision,
            config.local_files_only,
        )
        hf.chat_template = json.loads(Path(path).read_text())["chat_template"]
        prompt = hf.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        parts = prompt.split("<|audio_pad|>")
        if len(parts) != 2:
            raise ValueError("expected exactly one Qwen audio placeholder")
        eos = hf.encode("<|im_end|>", add_special_tokens=False)
        if len(eos) != 1:
            raise ValueError("Qwen end-of-turn must be a single special token")
        with tempfile.TemporaryDirectory() as directory:
            hf.save_pretrained(directory)
            data = {
                "files": {
                    p.name: base64.b64encode(p.read_bytes()).decode()
                    for p in Path(directory).iterdir()
                },
                "prefix": hf.encode(parts[0], add_special_tokens=False),
                "suffix": hf.encode(parts[1], add_special_tokens=False),
                "eos_id": eos[0],
            }
        return cls(json.dumps(data).encode())

    def __len__(self):
        return len(self.hf)

    def encode(self, text):
        return self.hf.encode(text, add_special_tokens=False)

    def target(self, text, language):
        # Malayalam remains a new ASR language to learn; the original checkpoint does not list it.
        name = LANGUAGES.get(language, language)
        return self.encode(f"language {name}<asr_text>{text}") + [self.eos_id]

    def decode(self, ids):
        raw = self.hf.decode(ids, skip_special_tokens=False)
        text = raw.split("<asr_text>", 1)[-1]
        return text.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def decoder_from_config(config):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = dict(config)
    old_rope = config.pop("rope_scaling", None) or {}
    if old_rope and old_rope.get("rope_type", "default") != "default":
        raise ValueError("only the inspected default Qwen3-ASR RoPE is supported")
    config.setdefault(
        "rope_parameters", {"rope_type": "default", "rope_theta": config.pop("rope_theta", 1000000)}
    )
    return Qwen3ForCausalLM(Qwen3Config(**config))


def load_qwen_decoder(config):
    from safetensors import safe_open

    path = resolve_file(
        config.qwen_model_id, "config.json", config.qwen_revision, config.local_files_only
    )
    root = json.loads(Path(path).read_text())
    if "thinker_config" not in root:
        raise ValueError("expected the original Qwen3-ASR checkpoint with thinker_config")
    text_config = root["thinker_config"]["text_config"]
    path = resolve_file(
        config.qwen_model_id, "model.safetensors", config.qwen_revision, config.local_files_only
    )
    with safe_open(path, framework="pt", device="cpu") as source:
        state = {
            "model." + k.removeprefix("thinker.model."): source.get_tensor(k)
            for k in source.keys()
            if k.startswith("thinker.model.")
        }
        if "thinker.lm_head.weight" in source.keys():
            state["lm_head.weight"] = source.get_tensor("thinker.lm_head.weight")
        elif text_config.get("tie_word_embeddings") and "model.embed_tokens.weight" in state:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
    # Do not let nested tie_word_embeddings overwrite a distinct trained LM head.
    # The outer Qwen speech config can have different tying behavior from text_config.
    text_config["tie_word_embeddings"] = torch.equal(
        state["lm_head.weight"], state["model.embed_tokens.weight"]
    )
    decoder = decoder_from_config(text_config)
    decoder.load_state_dict(state, strict=True)
    return decoder


class QwenASRModel(nn.Module):
    def __init__(self, component, decoder, config, tokenizer):
        super().__init__()
        self.config = config
        self.backbone = Backbone(component, config.train_backbone_layers)
        self.decoder = decoder
        if config.qwen_train_layers > len(decoder.model.layers):
            raise ValueError("qwen_train_layers exceeds decoder depth")
        decoder.requires_grad_(False)
        if config.qwen_train_layers:
            for layer in decoder.model.layers[-config.qwen_train_layers :]:
                layer.requires_grad_(True)
            decoder.model.norm.requires_grad_(True)
        self.bridge = nn.Sequential(
            nn.LayerNorm(component.config.output_proj_dims),
            nn.Linear(component.config.output_proj_dims, config.width),
            nn.GELU(),
            nn.Linear(config.width, decoder.config.hidden_size),
        )
        self.register_buffer("prefix", torch.tensor(tokenizer.prefix, dtype=torch.long))
        self.register_buffer("suffix", torch.tensor(tokenizer.suffix, dtype=torch.long))
        self.eos_id = tokenizer.eos_id

    def audio_embeddings(self, features, mask):
        output = self.backbone(input_features=features, attention_mask=mask)
        x, valid = self.bridge(output.last_hidden_state), output.attention_mask.bool()
        if torch.any(valid.sum(-1) == 0):
            raise ValueError("audio produced no valid frames")
        x = x.masked_fill(~valid[..., None], 0)
        if self.config.qwen_audio_stride == 2:
            padding = x.shape[1] % 2
            x = F.pad(x, (0, 0, 0, padding))
            valid = F.pad(valid, (0, padding))
            count = valid.reshape(valid.shape[0], -1, 2).sum(-1)
            x = x.reshape(x.shape[0], -1, 2, x.shape[-1]).sum(2) / count.clamp_min(1)[..., None]
            valid = count > 0
        return x, valid

    def fuse(self, audio, valid, targets=None, target_mask=None, left_pad=False):
        embed = self.decoder.get_input_embeddings()
        prefix, suffix = embed(self.prefix), embed(self.suffix)
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
        if int(lengths.max()) > self.decoder.config.max_position_embeddings:
            raise ValueError("audio and text exceed Qwen context; segment the input")
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
        hidden = self.decoder.model(
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
        logits = self.decoder.lm_head(hidden[:, :-1][selected])
        return {"loss": F.cross_entropy(logits.float(), labels[:, 1:][selected])}

    @torch.no_grad()
    def generate_transcripts(self, input_features, input_features_mask, max_new_tokens=256):
        audio, valid = self.audio_embeddings(input_features, input_features_mask)
        embeds, mask, _ = self.fuse(audio, valid, left_pad=True)
        return self.decoder.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.eos_id,
            pad_token_id=self.eos_id,
        )
