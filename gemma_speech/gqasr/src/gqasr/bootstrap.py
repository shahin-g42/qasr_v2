"""One-time upstream conversion to the standalone GQASR artifact format."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    Gemma4AudioConfig,
    Gemma4AudioFeatureExtractor,
    Gemma4AudioModel,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from .modeling import GQASRConfig, GQASRForSpeechRecognition

SOURCES = {
    "acoustic": {
        "repository": "google/embeddinggemma-2",
        "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "license": "Apache-2.0",
        "terms": "https://ai.google.dev/gemma/docs/gemma_4_license",
    },
    "decoder": {
        "repository": "Qwen/Qwen3-ASR-0.6B",
        "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "license": "Apache-2.0",
        "terms": "https://www.apache.org/licenses/LICENSE-2.0",
    },
}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def export_base(acoustic_path, decoder_path, output, seed=42, bridge_width=384):
    acoustic_path, decoder_path, output = map(
        lambda p: Path(p).resolve(), (acoustic_path, decoder_path, output)
    )
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(seed)
    aroot = json.loads((acoustic_path / "config.json").read_text())
    droot = json.loads((decoder_path / "config.json").read_text())
    if aroot.get("model_type") != "embedding_gemma2" or "thinker_config" not in droot:
        raise ValueError("unexpected upstream checkpoint format")
    acoustic = Gemma4AudioModel(Gemma4AudioConfig(**aroot["audio_config"]))
    with safe_open(acoustic_path / "model.safetensors", framework="pt") as source:
        acoustic.load_state_dict(
            {
                k.removeprefix("audio_tower."): source.get_tensor(k)
                for k in source.keys()
                if k.startswith("audio_tower.")
            },
            strict=True,
        )
    dconfig = dict(droot["thinker_config"]["text_config"])
    old_rope = dconfig.pop("rope_scaling", None)
    if old_rope and old_rope.get("rope_type", "default") != "default":
        raise ValueError("unsupported decoder rotary configuration")
    dconfig.setdefault(
        "rope_parameters", {"rope_type": "default", "rope_theta": dconfig.pop("rope_theta", 1e6)}
    )
    with safe_open(decoder_path / "model.safetensors", framework="pt") as source:
        state = {
            "model." + k.removeprefix("thinker.model."): source.get_tensor(k)
            for k in source.keys()
            if k.startswith("thinker.model.")
        }
        state["lm_head.weight"] = source.get_tensor("thinker.lm_head.weight")
    dconfig["tie_word_embeddings"] = torch.equal(
        state["lm_head.weight"], state["model.embed_tokens.weight"]
    )
    decoder = Qwen3ForCausalLM(Qwen3Config(**dconfig))
    decoder.load_state_dict(state, strict=True)
    del state
    tokenizer = AutoTokenizer.from_pretrained(
        decoder_path, local_files_only=True, trust_remote_code=False
    )
    tokenizer.chat_template = json.loads((decoder_path / "chat_template.json").read_text())[
        "chat_template"
    ]
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    parts = prompt.split("<|audio_pad|>")
    eos = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(parts) != 2 or len(eos) != 1:
        raise ValueError("unexpected upstream prompt contract")
    config = GQASRConfig(
        acoustic.config.to_dict(),
        decoder.config.to_dict(),
        tokenizer.encode(parts[0], add_special_tokens=False),
        tokenizer.encode(parts[1], add_special_tokens=False),
        eos[0],
        bridge_width,
    )
    model = GQASRForSpeechRecognition(config, acoustic, decoder).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gqasr-export-", dir=output.parent))
    try:
        artifact = staging / "model"
        model.save_pretrained(artifact)
        tokenizer.save_pretrained(artifact / "tokenizer")
        tpath = artifact / "tokenizer/tokenizer_config.json"
        tconfig = json.loads(tpath.read_text())
        tconfig["tokenizer_class"] = "PreTrainedTokenizerFast"
        for k in ("name_or_path", "_name_or_path", "auto_map"):
            tconfig.pop(k, None)
        tpath.write_text(json.dumps(tconfig, indent=2, ensure_ascii=False))
        frontend = Gemma4AudioFeatureExtractor.from_pretrained(acoustic_path, local_files_only=True)
        fconfig = json.loads(frontend.to_json_string())
        fconfig["feature_extractor_type"] = "GQASRFeatureExtractor"
        (artifact / "preprocessor_config.json").write_text(json.dumps(fconfig, indent=2))
        provenance = {
            "model_name": "gqasr-base",
            "derived_model": True,
            "sources": SOURCES,
            "source_weights_sha256": {
                "acoustic": digest(acoustic_path / "model.safetensors"),
                "decoder": digest(decoder_path / "model.safetensors"),
            },
            "initialization": {
                "acoustic": "pretrained",
                "decoder": "pretrained",
                "bridge": "random; requires warm-up training",
                "seed": seed,
            },
            "parameters": sum(p.numel() for p in model.parameters()),
            "bridge_parameters": sum(p.numel() for p in model.gqasr_bridge.parameters()),
            "torch": str(torch.__version__),
            "transformers": __import__("transformers").__version__,
        }
        (artifact / "provenance.json").write_text(json.dumps(provenance, indent=2))
        licenses = Path(__file__).resolve().parent / "licenses"
        shutil.copytree(licenses, artifact / "licenses")
        (artifact / "MODEL_CARD.md").write_text(
            "# GQASR Base\n\nDerived multilingual ASR initialization. The acoustic encoder and text decoder are pretrained; the alignment bridge is newly initialized. This is not a validated speech recognizer before adaptation. See provenance.json and licenses/ for upstream attribution and applicable terms. Renaming does not change the architecture or remove those terms.\n"
        )
        checksums = {
            str(p.relative_to(artifact)): digest(p) for p in artifact.rglob("*") if p.is_file()
        }
        (artifact / "checksums.json").write_text(json.dumps(checksums, indent=2))
        os.rename(artifact, output)
    finally:
        shutil.rmtree(staging)
    return {
        "output": str(output),
        "parameters": provenance["parameters"],
        "bridge_parameters": provenance["bridge_parameters"],
        "weights_sha256": checksums["model.safetensors"],
    }


def copy_assets(source, destination):
    source, destination = Path(source), Path(destination)
    for name in ("tokenizer", "licenses"):
        shutil.copytree(source / name, destination / name)
    for name in ("preprocessor_config.json", "provenance.json", "MODEL_CARD.md"):
        shutil.copyfile(source / name, destination / name)
