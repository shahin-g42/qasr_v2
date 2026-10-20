"""Selective, strict export. Never instantiate an upstream multimodal/text model."""

import hashlib
import json
import shutil
import tempfile
from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from transformers import GenerationConfig, PreTrainedTokenizerFast, WhisperFeatureExtractor

from .configuration import AUDConfig
from .feature_extraction import AUDCohereFeatureExtractor
from .modeling import AUDAdapter, AUDForConditionalGeneration
from .processing import AUDProcessor, AUDRawFeatureExtractor

CAPTIONER = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
MOONSHINE = "moonshine-ai/moonshine-streaming-medium"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WeightSource:
    """Index-first shard selection and safe tensor loading, pinned to an immutable Hub revision."""

    def __init__(self, source, revision="main"):
        self.source = str(source)
        self.local = Path(source).is_dir()
        self.revision = None
        self.stack = ExitStack()
        self.handles, self.paths, self.aliases = {}, {}, {}
        if self.local:
            self.root = Path(source).resolve()
            self.files = {
                p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()
            }
        else:
            if str(source).startswith(("/", ".")):
                raise FileNotFoundError(source)
            info = HfApi().model_info(source, revision=revision)
            self.revision = info.sha
            self.files = {s.rfilename for s in info.siblings}
        self.config = self.json("config.json")
        index = "model.safetensors.index.json"
        if index in self.files:
            self.weight_map = self.json(index)["weight_map"]
        elif "model.safetensors" in self.files:
            handle = self._open("model.safetensors")
            self.weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
        else:
            raise ValueError(
                f"{source}: expected model.safetensors or its shard index; pickle loading is disabled"
            )

    def file(self, name):
        if name not in self.files:
            raise FileNotFoundError(f"{self.source}: missing {name}")
        path = (
            self.root / name
            if self.local
            else Path(hf_hub_download(self.source, name, revision=self.revision))
        )
        self.paths[name] = path
        return path

    def json(self, name):
        return json.loads(self.file(name).read_text())

    def _open(self, name):
        if name not in self.handles:
            handle = self.stack.enter_context(
                safe_open(self.file(name), framework="pt", device="cpu")
            )
            self.handles[name] = handle
            # safetensors.save_model records removed shared tensor aliases in metadata.
            self.aliases.update(
                {k: v for k, v in (handle.metadata() or {}).items() if v in handle.keys()}
            )
        return self.handles[name]

    def tensor(self, key):
        if key in self.weight_map:
            return self._open(self.weight_map[key]).get_tensor(key)
        if key in self.aliases:
            return self.tensor(self.aliases[key])
        raise KeyError(f"{self.source}: missing pretrained tensor {key}")

    def choose_prefix(self, choices):
        found = [p for p in choices if any(k.startswith(p) for k in self.weight_map)]
        if len(found) != 1:
            raise ValueError(
                f"{self.source}: expected exactly one audio encoder prefix among {choices}; got {found}"
            )
        return found[0]

    def provenance(self):
        return {
            "source": self.source,
            "revision": self.revision,
            "files_read": {
                k: {"sha256": sha256(p), "bytes": p.stat().st_size}
                for k, p in sorted(self.paths.items())
            },
        }

    def close(self):
        self.stack.close()


def _template(source, tokenizer):
    if "chat_template.jinja" in source.files:
        return source.file("chat_template.jinja").read_text()
    if "chat_template.json" in source.files:
        return source.json("chat_template.json")["chat_template"]
    if "processor_config.json" in source.files:
        value = source.json("processor_config.json").get("chat_template")
        if value:
            return value
    return tokenizer.chat_template


def _frontend_config(source):
    if "preprocessor_config.json" in source.files:
        return source.json("preprocessor_config.json")
    value = source.json("processor_config.json").get("feature_extractor")
    if not isinstance(value, dict):
        raise ValueError(f"{source.source}: missing audio feature extractor configuration")
    return value


def build_export(
    variant, text_source, encoder_source=None, dtype="bfloat16", adapter_width=384, seed=42
):
    """Return a fully materialized CPU model plus explicit tensor provenance."""
    torch.manual_seed(seed)
    source = text_source.config
    if variant in ("audpro", "audturbo"):
        if source.get("model_type") != "qasr":
            raise ValueError(
                "--qasr-checkpoint must be a trained local QASR checkpoint (model_type=qasr)"
            )
        text_config = dict(source["text_config"])
        text_config["tie_word_embeddings"] = source.get(
            "tie_word_embeddings", text_config.get("tie_word_embeddings", True)
        )
        common = {
            "audio_token_id": source["audio_token_id"],
            "pad_token_id": source.get("pad_token_id"),
            "eos_token_id": source.get("eos_token_id"),
        }
    else:
        if source.get("model_type") != "gqasr":
            raise ValueError("--gqasr-checkpoint must be a trained local GQASR checkpoint")
        text_config = source["decoder_config"]
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            text_source.root / "tokenizer", local_files_only=True
        )
        audio_token = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        if audio_token is None or tokenizer.convert_ids_to_tokens(audio_token) != "<|audio_pad|>":
            raise ValueError("GQASR tokenizer does not contain <|audio_pad|>")
        common = {
            "audio_token_id": audio_token,
            "prompt_prefix": source["prompt_prefix"],
            "prompt_suffix": source["prompt_suffix"],
            "eos_token_id": source["eos_id"],
            "pad_token_id": tokenizer.pad_token_id,
        }
    prefix = None
    if variant == "audpro":
        audio_config = source["audio_config"]
        common.update(
            {
                key: source.get(key, default)
                for key, default in (
                    ("projector_depth", 1),
                    ("projector_hidden_act", "gelu"),
                    ("projector_pool_stride", 1),
                )
            }
        )
        if "model.ctc_head.weight" in text_source.weight_map:
            common["ctc_vocab_size"] = text_source.tensor("model.ctc_head.weight").shape[0]
    elif variant == "audturbo":
        audio_config = encoder_source.config["thinker_config"]["audio_config"]
        prefix = encoder_source.choose_prefix(
            ("thinker.audio_tower.", "model.thinker.audio_tower.")
        )
    else:
        audio_config = encoder_source.config["encoder_config"]
        prefix = encoder_source.choose_prefix(("model.encoder.", "encoder."))
    config = AUDConfig(
        variant=variant,
        audio_config=audio_config,
        text_config=text_config,
        requires_alignment=variant != "audpro",
        adapter_width=adapter_width,
        **common,
    )
    embed_key = (
        "gqasr_decoder.model.embed_tokens.weight"
        if variant == "audflash"
        else "model.language_model.embed_tokens.weight"
    )
    head_key = "gqasr_decoder.lm_head.weight" if variant == "audflash" else "lm_head.weight"
    if (
        config.text_config.tie_word_embeddings
        and embed_key in text_source.weight_map
        and head_key in text_source.weight_map
        and not torch.equal(text_source.tensor(embed_key), text_source.tensor(head_key))
    ):
        raise ValueError("Source claims tied decoder weights but original tensors differ")
    if variant == "audflash":
        for name in ("prompt_prefix", "prompt_suffix"):
            key = "gqasr_" + name
            if key in text_source.weight_map and text_source.tensor(key).tolist() != source[name]:
                raise ValueError(f"GQASR {name} buffer differs from config")
    # Meta parameters avoid constructing discarded random multi-billion parameter tensors.
    # Nonpersistent buffers (RoPE, sinusoidal positions) stay on CPU and remain valid.
    with init_empty_weights(include_buffers=False):
        model = AUDForConditionalGeneration(config)
    new_state = {}
    if variant != "audpro":
        size = audio_config["output_dim"] if variant == "audturbo" else audio_config["hidden_size"]
        adapter = AUDAdapter(
            size, config.text_config.hidden_size, adapter_width, residual=variant == "audturbo"
        )
        new_state = {"aud_projector." + k: v for k, v in adapter.state_dict().items()}
    replacements = {
        "aud_encoder.": "model.audio_tower.",
        "aud_projector.": "model.multi_modal_projector.",
        "aud_pool.": "model.pooling_conv.",
        "aud_ctc_head.": "model.ctc_head.",
        "aud_decoder.model.": "model.language_model.",
        "aud_decoder.lm_head.": "lm_head.",
    }
    mapping, state, used_text, used_audio = {}, {}, set(), set()
    tied = config.text_config.tie_word_embeddings
    for target, expected in model.state_dict().items():
        if target in new_state:
            value, origin = new_state[target], {"initialization": "new_adapter", "seed": seed}
        else:
            owner = text_source
            if target.startswith("aud_encoder.") and encoder_source is not None:
                key, owner = prefix + target.removeprefix("aud_encoder."), encoder_source
            elif variant == "audflash":
                key = "gqasr_decoder." + target.removeprefix("aud_decoder.")
            else:
                matches = [p for p in replacements if target.startswith(p)]
                if len(matches) != 1:
                    raise ValueError(f"Unmapped target tensor: {target}")
                key = replacements[matches[0]] + target[len(matches[0]) :]
            if tied and target.startswith("aud_decoder."):
                embed = (
                    "gqasr_decoder.model.embed_tokens.weight"
                    if variant == "audflash"
                    else "model.language_model.embed_tokens.weight"
                )
                head = "gqasr_decoder.lm_head.weight" if variant == "audflash" else "lm_head.weight"
                if key in (embed, head) and key not in owner.weight_map:
                    key = head if key == embed else embed
            value = owner.tensor(key)
            (used_text if owner is text_source else used_audio).add(key)
            origin = {"source": owner.source, "key": key}
        if value.shape != expected.shape:
            raise ValueError(
                f"Shape mismatch for {target}: {tuple(value.shape)} != {tuple(expected.shape)}"
            )
        state[target] = value.to(DTYPES[dtype]) if value.is_floating_point() else value
        mapping[target] = origin
    if tied and not torch.equal(
        state["aud_decoder.lm_head.weight"], state["aud_decoder.model.embed_tokens.weight"]
    ):
        raise ValueError(
            "Source claims tied decoder weights but embedding and output tensors differ"
        )
    if variant == "audpro":
        unused = set(text_source.weight_map) - used_text
        if unused:
            raise ValueError(f"AudPro would discard checkpoint tensors: {sorted(unused)[:20]}")
    else:
        prefixes = (
            ("gqasr_decoder.",) if variant == "audflash" else ("model.language_model.", "lm_head.")
        )
        unused = {k for k in text_source.weight_map if k.startswith(prefixes)} - used_text
        if unused:
            raise ValueError(f"Unmapped trained decoder tensors: {sorted(unused)[:20]}")
    if encoder_source is not None:
        unused = {k for k in encoder_source.weight_map if k.startswith(prefix)} - used_audio
        if unused:
            raise ValueError(f"Unmapped encoder tensors: {sorted(unused)[:20]}")
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Export still contains meta tensors")
    model.eval()
    return model, mapping


def export_model(
    variant,
    output,
    *,
    qasr_checkpoint=None,
    gqasr_checkpoint=None,
    encoder_source=None,
    encoder_revision="main",
    dtype="bfloat16",
    adapter_width=384,
    seed=42,
):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new export directory")
    checkpoint = gqasr_checkpoint if variant == "audflash" else qasr_checkpoint
    if not checkpoint or not Path(checkpoint).is_dir():
        raise ValueError("Provide an existing trained local QASR/GQASR checkpoint directory")
    sources = []
    staging = None
    try:
        text = WeightSource(checkpoint)
        sources.append(text)
        audio = None
        if variant != "audpro":
            audio = WeightSource(
                encoder_source or (CAPTIONER if variant == "audturbo" else MOONSHINE),
                encoder_revision,
            )
            sources.append(audio)
        model, mapping = build_export(variant, text, audio, dtype, adapter_width, seed)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            text.root / "tokenizer" if variant == "audflash" else text.root, local_files_only=True
        )
        template = _template(text, tokenizer) if variant != "audflash" else None
        if variant == "audpro":
            frontend = AUDCohereFeatureExtractor.from_dict(_frontend_config(text))
            model.config.max_audio_seconds = frontend.max_audio_clip_s
        elif variant == "audturbo":
            data = _frontend_config(audio)
            if (
                data.get("feature_size") != model.config.audio_config.num_mel_bins
                or data.get("sampling_rate") != 16000
            ):
                raise ValueError("Captioner audio frontend does not match its encoder")
            frontend = WhisperFeatureExtractor.from_dict(data)
        else:
            a = model.config.audio_config
            frontend = AUDRawFeatureExtractor(round(a.frame_ms * a.sample_rate / 1000))
        processor = AUDProcessor(model.config, tokenizer, frontend, template)
        if "generation_config.json" in text.files:
            model.generation_config = GenerationConfig(**text.json("generation_config.json"))
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = model.config.eos_token_id
        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        model.save_pretrained(staging)
        processor.save_pretrained(staging)
        if not (staging / "model.safetensors").is_file():
            raise RuntimeError("One-piece export did not produce model.safetensors")
        for i, src in enumerate(sources):
            for name in sorted(src.files):
                if Path(name).name.lower().startswith(("license", "notice")) or name == "README.md":
                    target = staging / "licenses" / str(i) / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src.file(name), target)
        (staging / "NOTICE.md").write_text(files("audm").joinpath("NOTICE.md").read_text())
        write_json(
            staging / "provenance.json",
            {
                "format_version": 1,
                "variant": variant,
                "dtype": dtype,
                "seed": seed,
                "requires_alignment": model.config.requires_alignment,
                "sources": [src.provenance() for src in sources],
                "tensor_mapping": mapping,
                "parameter_counts": {
                    name: sum(p.numel() for p in module.parameters())
                    for name, module in model.named_children()
                },
                "scope": "Weights and assets only. No optimizer or scheduler state imported.",
                "runtime_modifications": (
                    ["Moonshine valid-time convolution padding mask"]
                    if variant == "audflash"
                    else []
                ),
            },
        )
        (staging / "README.md").write_text(
            f"# {variant}\n\nLoad with audm.AUDForConditionalGeneration and audm.AUDProcessor.\n\n"
            + (
                "New encoder-to-decoder adapter: alignment training is required; this is not a validated ASR model.\n"
                if model.config.requires_alignment
                else "QASR repack with trained components preserved.\n"
            )
            + "\nSee provenance.json for source revisions, tensor mappings and parameter counts; source licenses apply.\n"
        )
        write_json(
            staging / "checksums.json",
            {
                p.relative_to(staging).as_posix(): sha256(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        )
        staging.rename(output)
        staging = None
        return {
            "output": str(output),
            "variant": variant,
            "requires_alignment": model.config.requires_alignment,
        }
    finally:
        for source in sources:
            source.close()
        if staging is not None:
            shutil.rmtree(staging)
