"""Cluster-side base conversion; never train or overwrite an existing artifact."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from gqasr.bootstrap import SOURCES, digest
from safetensors import safe_open
from transformers import Gemma4AudioConfig, Gemma4AudioFeatureExtractor, Gemma4AudioModel

from .base import GQTTSBase
from .speaker import QWEN_TTS_REVISION, SpeakerAdapter, load_qwen_tts, weights_digest


def export_base(acoustic_path, tts_path, output, seed=42):
    acoustic_path, tts_path, output = (Path(p).resolve() for p in (acoustic_path, tts_path, output))
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    cfg = json.loads((acoustic_path / "config.json").read_text())
    if cfg.get("model_type") != "embedding_gemma2":
        raise ValueError("unexpected acoustic source checkpoint")
    acoustic = Gemma4AudioModel(Gemma4AudioConfig(**cfg["audio_config"]))
    with safe_open(acoustic_path / "model.safetensors", framework="pt") as source:
        acoustic.load_state_dict(
            {
                k.removeprefix("audio_tower."): source.get_tensor(k)
                for k in source.keys()
                if k.startswith("audio_tower.")
            },
            strict=True,
        )
    api = load_qwen_tts(tts_path)
    codec = api.model.speech_tokenizer
    dimensions = dict(
        input_dim=acoustic.config.output_proj_dims,
        width=512,
        output_dim=api.model.config.speaker_encoder_config.enc_dim,
    )
    torch.manual_seed(seed)
    adapter = SpeakerAdapter(**dimensions)
    config = dict(
        model_type="gqtts",
        format_version=1,
        architectures=["GQTTSBase"],
        acoustic_config=acoustic.config.to_dict(),
        generator_config=api.model.config.to_dict(),
        codec_config=codec.model.config.to_dict(),
        adapter_dimensions=dimensions,
        generate_defaults=api.generate_defaults,
    )
    model = GQTTSBase(config, acoustic, api.model, codec.model, adapter).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gqtts-export-", dir=output.parent))
    try:
        artifact = staging / "artifact"
        model.save_pretrained(artifact)
        api.processor.save_pretrained(artifact / "processor")
        codec.feature_extractor.save_pretrained(artifact / "codec_frontend")
        Gemma4AudioFeatureExtractor.from_pretrained(
            acoustic_path, local_files_only=True
        ).save_pretrained(artifact / "acoustic_frontend")
        provenance = dict(
            model_name="gqtts-base",
            derived_model=True,
            sources={
                "acoustic": SOURCES["acoustic"],
                "tts": {
                    "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    "revision": QWEN_TTS_REVISION,
                    "license": "Apache-2.0",
                },
            },
            source_weights_sha256={
                "acoustic": weights_digest(acoustic_path),
                "tts": weights_digest(tts_path),
                "codec": weights_digest(tts_path / "speech_tokenizer"),
            },
            initialization={
                "acoustic": "pretrained",
                "talker_and_code_predictor": "pretrained",
                "codec": "pretrained; retained",
                "speaker_adapter": "random; must be trained",
                "original_speaker_encoder": "excluded; retain upstream source separately for teacher caching",
                "seed": seed,
            },
            parameters=sum(p.numel() for p in model.parameters()),
            adapter_parameters=sum(p.numel() for p in adapter.parameters()),
            torch=str(torch.__version__),
            transformers=__import__("transformers").__version__,
        )
        (artifact / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        from gqasr import bootstrap as asr_bootstrap

        (artifact / "licenses").mkdir()
        shutil.copyfile(
            Path(asr_bootstrap.__file__).parent / "licenses/Apache-2.0.txt",
            artifact / "licenses/Apache-2.0.txt",
        )
        (artifact / "licenses/NOTICE.md").write_text(
            "# GQTTS derivation notice\n\n"
            "Derived from Google's EmbeddingGemma 2 audio encoder and Alibaba Qwen's "
            "Qwen3-TTS-12Hz-0.6B-Base talker/code predictor and speech codec, using "
            "Hugging Face Transformers primitives. These sources use Apache-2.0. "
            "Upstream copyrights and applicable notices are retained. Changes include "
            "replacement of the reference speaker encoder with an acoustic encoder and "
            "new speaker adapter, local GQTTS class/tensor identifiers and a standalone "
            "artifact loader. Source revisions and weight hashes are in provenance.json.\n"
        )
        from gemma_speech._vendor import qwen_tts

        shutil.copyfile(
            Path(qwen_tts.__file__).parent / "NOTICE.md", artifact / "licenses/TTS_PORT_NOTICE.md"
        )
        (artifact / "MODEL_CARD.md").write_text(
            "# GQTTS Base\n\nDerived initialization with pretrained acoustic encoder, talker/code predictor and retained codec. The replacement speaker adapter is random and requires training. No voice-cloning quality or Arabic pronunciation is established by export. Original speaker encoder is excluded. See provenance.json and licenses/ for attribution.\n"
        )
        checksums = {
            str(p.relative_to(artifact)): digest(p) for p in artifact.rglob("*") if p.is_file()
        }
        (artifact / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
        os.rename(artifact, output)
    finally:
        shutil.rmtree(staging)
    return dict(
        output=str(output),
        parameters=provenance["parameters"],
        adapter_parameters=provenance["adapter_parameters"],
        weights_sha256=checksums["model.safetensors"],
    )
