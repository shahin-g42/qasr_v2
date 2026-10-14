"""Frozen Gemma audio frames -> learned Qwen speaker embedding, in one runtime."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from gemma_speech.assets import load_assets, processors_from_assets
from gemma_speech.backbone import load_component
from gemma_speech.config import ModelConfig
from gemma_speech.data import load_audio, read_manifest

QWEN_TTS_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"


def weights_digest(directory):
    """Fingerprint local weights, so a checkpoint name alone cannot imply provenance."""
    paths = sorted(Path(directory).glob("*.safetensors"))
    if not paths:
        raise ValueError(f"no safetensors weights in {directory}")
    result = {}
    for path in paths:
        with path.open("rb") as f:
            result[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    return result


class SpeakerAdapter(nn.Module):
    def __init__(self, input_dim=1536, width=512, output_dim=1024):
        super().__init__()
        self.dimensions = dict(input_dim=input_dim, width=width, output_dim=output_dim)
        self.attention = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, width),
            nn.GELU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, frames, mask):
        if frames.ndim != 3 or mask.shape != frames.shape[:2] or not mask.any(-1).all():
            raise ValueError("speaker pooling requires at least one valid audio frame per item")
        frames = frames.masked_fill(~mask[..., None], 0)
        scores = self.attention(frames).squeeze(-1).float().masked_fill(~mask, -torch.inf)
        weights = scores.softmax(-1)[..., None]
        mean = (weights * frames.float()).sum(1)
        variance = (weights * (frames.float() - mean[:, None]).square()).sum(1)
        return self.project(torch.cat([mean, variance.clamp_min(1e-6).sqrt()], -1))


def distillation_loss(prediction, teacher):
    # Preserve the teacher's scale as well as direction: Qwen consumes unnormalized vectors.
    cosine = 1 - F.cosine_similarity(prediction.float(), teacher.float(), dim=-1)
    relative_mse = (prediction.float() - teacher.float()).square().mean(
        -1
    ) / teacher.float().square().mean(-1).clamp_min(1e-6)
    return (cosine + relative_mse).mean(), cosine.mean()


def load_qwen_tts(path, device="cpu", dtype=torch.float32):
    from gemma_speech._vendor.qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError("Qwen TTS requires a local prefetched model directory")
    model = Qwen3TTSModel.from_pretrained(
        str(path), dtype=dtype, attn_implementation="sdpa", local_files_only=True
    )
    if (
        model.model.tokenizer_type != "qwen3_tts_tokenizer_12hz"
        or model.model.tts_model_type != "base"
    ):
        raise ValueError("only Qwen 12Hz Base checkpoints are supported")
    model.model.to(device).eval()
    model.device = torch.device(device)
    # The tokenizer wrapper is not an nn.Module child of the talker.
    model.model.speech_tokenizer.model.to(device).eval()
    model.model.speech_tokenizer.device = torch.device(device)
    return model


def audio_frames(component, extractor, wave, device):
    features = extractor(
        [wave],
        sampling_rate=16000,
        padding=True,
        truncation=False,
        max_length=None,
        return_tensors="pt",
    )
    result = component(
        input_features=features["input_features"].to(device),
        attention_mask=features["input_features_mask"].to(device).bool(),
    )
    return result.last_hidden_state[0, result.attention_mask[0].bool()].float()


@torch.no_grad()
def cache_speaker(
    manifest,
    output,
    gemma,
    qwen,
    device="cpu",
    max_seconds=15,
    shard=0,
    shards=1,
    include_codes=False,
):
    """Cache frames/teacher vectors (and optional codec targets) in the SAME environment.

    Launch disjoint --shard/--shards processes to parallelize; each writes its own manifest.
    """
    if not 0 <= shard < shards:
        raise ValueError("invalid shard")
    rows = read_manifest(manifest, "tts")[shard::shards]
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / f"shard-{shard:05d}.jsonl"
    if index_path.exists():
        raise FileExistsError(index_path)
    config = ModelConfig(model_id=str(Path(gemma).resolve()), local_files_only=True)
    component = load_component(config).to(device).eval()
    extractor, _ = processors_from_assets(load_assets(config))
    teacher = load_qwen_tts(qwen, device)
    records = []
    for row in rows:
        filename = hashlib.sha256(row.id.encode()).hexdigest() + ".pt"
        destination = directory / filename
        if destination.exists():
            raise FileExistsError(destination)
        audio16 = load_audio(row.audio, 16000, max_seconds)
        audio24 = load_audio(row.audio, 24000, max_seconds)
        frames = audio_frames(component, extractor, audio16, device)
        target = teacher.model.extract_speaker_embedding(audio24, sr=24000).float().cpu()
        payload = {"frames": frames.cpu(), "teacher": target}
        if include_codes:
            codes = teacher.model.speech_tokenizer.encode(audio24, sr=24000).audio_codes[0]
            payload["codes"] = codes.cpu().long()
        torch.save(payload, destination)
        records.append(
            {
                "id": row.id,
                "audio": str(row.audio),
                "text": row.tts_text,
                "speaker": row.speaker,
                "language": row.language,
                "file": filename,
                "audio_sha256": hashlib.sha256(Path(row.audio).read_bytes()).hexdigest(),
            }
        )
    metadata = {
        "gemma_config": component.config.to_dict(),
        "gemma_revision": config.revision,
        "qwen_revision": QWEN_TTS_REVISION,
        "qwen_path": str(Path(qwen).resolve()),
        "gemma_path": config.model_id,
        "shards": shards,
        "include_codes": include_codes,
        "gemma_weights": weights_digest(gemma),
        "qwen_weights": weights_digest(qwen),
    }
    (directory / f"metadata-{shard:05d}.json").write_text(json.dumps(metadata, indent=2))
    with index_path.open("x") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"manifest": str(index_path), "rows": len(records)}


def read_cache(directory):
    directory = Path(directory).resolve()
    files = sorted(directory.glob("shard-*.jsonl"))
    if not files:
        raise ValueError(f"no cache shards in {directory}")
    metadata = json.loads((directory / "metadata-00000.json").read_text())
    if len(files) != metadata["shards"]:
        raise ValueError("incomplete feature cache; wait for all shards")
    records = []
    for index, filename in enumerate(files):
        if filename.name != f"shard-{index:05d}.jsonl":
            raise ValueError("noncontiguous cache shards")
        if json.loads((directory / f"metadata-{index:05d}.json").read_text()) != metadata:
            raise ValueError("mixed model identities in cache")
        for line in filename.read_text().splitlines():
            row = json.loads(line)
            row["file"] = str((directory / row["file"]).resolve())
            records.append(row)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("empty cache or duplicate utterance IDs")
    return records, metadata


def collate_cached(rows):
    data = [torch.load(row["file"], map_location="cpu", weights_only=True) for row in rows]
    lengths = torch.tensor([len(d["frames"]) for d in data])
    frames = pad_sequence([d["frames"] for d in data], batch_first=True)
    mask = torch.arange(frames.shape[1])[None] < lengths[:, None]
    targets = torch.stack([d["teacher"] for d in data])
    return frames, mask, targets


@torch.no_grad()
def speaker_vector(checkpoint, audio, gemma, device="cpu", max_seconds=15):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "gemma-qwen-speaker-v1":
        raise ValueError("invalid speaker adapter checkpoint")
    if saved["metadata"].get("gemma_weights") != weights_digest(gemma):
        raise ValueError("Gemma weights differ from those used to train this speaker adapter")
    adapter = SpeakerAdapter(**saved["dimensions"]).to(device).eval()
    adapter.load_state_dict(saved["adapter"], strict=True)
    cfg = ModelConfig(model_id=str(Path(gemma).resolve()), local_files_only=True)
    component = load_component(cfg).to(device).eval()
    # Compare architectural fields, ignoring where the local files are stored.
    before = dict(saved["metadata"]["gemma_config"])
    after = component.config.to_dict()
    for k in ("_name_or_path", "transformers_version", "dtype"):
        before.pop(k, None)
        after.pop(k, None)
    if before != after:
        raise ValueError("Gemma audio configuration differs from the adapter's training backbone")
    extractor, _ = processors_from_assets(load_assets(cfg))
    frames = audio_frames(component, extractor, load_audio(audio, 16000, max_seconds), device)
    return adapter(frames[None], torch.ones(1, len(frames), device=device, dtype=torch.bool))[0]


@torch.no_grad()
def synthesize_qwen(
    checkpoint,
    audio,
    gemma,
    qwen,
    text,
    language,
    output,
    device="cpu",
    seed=42,
    talker_checkpoint=None,
):
    import soundfile as sf

    from gemma_speech._vendor.qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(seed)
    vector = speaker_vector(checkpoint, audio, gemma, device)
    model = load_qwen_tts(qwen, device)
    if talker_checkpoint:
        saved = torch.load(talker_checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "gemma-qwen-talker-v1":
            raise ValueError("invalid adapted talker checkpoint")
        model.model.talker.load_state_dict(saved["talker"], strict=True)
    prompt = VoiceClonePromptItem(
        ref_code=None,
        ref_spk_embedding=vector,
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )
    waves, sr = model.generate_voice_clone(
        text=text, language=language, voice_clone_prompt=[prompt], non_streaming_mode=True
    )
    if not __import__("numpy").isfinite(waves[0]).all():
        raise FloatingPointError("nonfinite Qwen audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, waves[0], sr)
    return {"output": str(output.resolve()), "sample_rate": sr, "seconds": len(waves[0]) / sr}
