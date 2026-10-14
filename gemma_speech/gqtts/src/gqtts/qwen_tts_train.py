"""Qwen 12Hz talker adaptation with frozen Gemma speaker conditioning.

Train non-streaming text-to-code prediction in Auto language mode. Arabic/Hindi/
Malayalam are learned from paired examples; no fictitious language token is added.
The speech codec is retained. Both autoregressive losses are explicitly aligned
once, avoiding a second implicit label shift in a generic LM loss helper.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from gemma_speech.data import WeightedDistributedSampler, sampling_weights
from gemma_speech.distributed import Runtime
from gemma_speech.train import learning_rate_factor

from .speaker import SpeakerAdapter, load_qwen_tts, read_cache, weights_digest


def identity_collate(rows):
    return rows


class QwenTTSFineTune(nn.Module):
    def __init__(self, api, adapter, max_tokens=256):
        super().__init__()
        self.talker = api.model.talker
        self.adapter = adapter.requires_grad_(False)
        self.tokenizer = api.processor.tokenizer
        self.config = api.model.config
        self.max_tokens = max_tokens

    def pack(self, rows):
        talker, cfg = self.talker, self.config
        device = next(talker.parameters()).device

        def codec(ids):
            return talker.get_input_embeddings()(torch.as_tensor(ids, device=device))

        def text(ids):
            return talker.text_projection(
                talker.get_text_embeddings()(torch.as_tensor(ids, device=device))
            )

        text_bos, text_eos, text_pad = text(
            [cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]
        )
        role_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        # Match the upstream non_streaming_mode=True, x-vector, Auto-language prompt.
        tag_ids = [
            cfg.talker_config.codec_nothink_id,
            cfg.talker_config.codec_think_bos_id,
            cfg.talker_config.codec_think_eos_id,
        ]
        sequences, labels, bounds, all_codes = [], [], [], []
        for row in rows:
            cached = torch.load(row["file"], map_location="cpu", weights_only=True)
            codes = cached["codes"].to(device)
            if (
                codes.ndim != 2
                or codes.shape[1] != 16
                or not len(codes)
                or codes.min() < 0
                or codes.max() >= 2048
            ):
                raise ValueError("expected nonempty 16-codebook Qwen 12Hz targets in [0, 2048)")
            ids = self.tokenizer.encode(row["text"], add_special_tokens=False)
            if not ids or len(ids) > self.max_tokens:
                raise ValueError("empty or overlong TTS text; segment the utterance")
            frames = cached["frames"].to(device)
            with torch.no_grad():
                speaker = self.adapter(
                    frames[None], torch.ones(1, len(frames), device=device, dtype=torch.bool)
                )[0]
            prompt = torch.cat(
                [
                    text(role_ids),
                    codec(tag_ids) + text_pad,
                    (speaker + text_pad)[None],
                    (codec(cfg.talker_config.codec_pad_id) + text_bos)[None],
                    text(ids) + codec(cfg.talker_config.codec_pad_id),
                    (text_eos + codec(cfg.talker_config.codec_pad_id))[None],
                    (text_pad + codec(cfg.talker_config.codec_bos_id))[None],
                ]
            )
            encoded = codec(codes[:, 0])
            for i in range(1, 16):
                encoded = encoded + talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i])
            encoded = encoded + text_pad
            end = (text_pad + codec(cfg.talker_config.codec_eos_token_id))[None]
            sequences.append(torch.cat([prompt, encoded, end]))
            labels.append(
                torch.cat(
                    [
                        codes.new_full((len(prompt),), -100),
                        codes[:, 0],
                        codes.new_tensor([cfg.talker_config.codec_eos_token_id]),
                    ]
                )
            )
            bounds.append((len(prompt), len(codes)))
            all_codes.append(codes)
        lengths = torch.tensor([len(s) for s in sequences], device=device)
        if int(lengths.max()) > cfg.talker_config.max_position_embeddings:
            raise ValueError("TTS input exceeds the context window")
        embeddings = pad_sequence(sequences, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        mask = torch.arange(embeddings.shape[1], device=device)[None] < lengths[:, None]
        return embeddings, mask, labels, bounds, all_codes

    def forward(self, rows):
        embeddings, mask, labels, bounds, codes = self.pack(rows)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)[None].expand(3, -1, -1)
        hidden = self.talker.model(
            inputs_embeds=embeddings,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        selected = labels[:, 1:] != -100
        first_logits = self.talker.codec_head(hidden[:, :-1][selected])
        main_loss = F.cross_entropy(first_logits.float(), labels[:, 1:][selected])
        previous = torch.cat(
            [hidden[i, start - 1 : start + count - 1] for i, (start, count) in enumerate(bounds)]
        )
        target = torch.cat(codes)
        inputs = [previous[:, None], self.talker.get_input_embeddings()(target[:, :1])]
        inputs += [
            self.talker.code_predictor.get_input_embeddings()[i - 1](target[:, i : i + 1])
            for i in range(1, 15)
        ]
        logits = self.talker.code_predictor.forward_finetune(
            inputs_embeds=torch.cat(inputs, dim=1), use_cache=False, return_dict=True
        ).logits
        residual_loss = F.cross_entropy(logits.float().flatten(0, 1), target[:, 1:].flatten())
        return main_loss + 0.3 * residual_loss


def train_qwen_tts(config, resume=None):
    path = Path(config).resolve()
    cfg = json.loads(path.read_text())
    defaults = dict(
        batch_size=2,
        max_steps=20000,
        learning_rate=2e-5,
        precision="bf16",
        workers=4,
        seed=42,
        eval_every=250,
        max_text_tokens=256,
        language_weights={},
    )
    required = {"train_cache", "validation_cache", "output_dir", "qwen", "speaker_checkpoint"}
    if cfg.keys() - (required | defaults.keys()) or not required <= cfg.keys():
        raise ValueError("missing or unknown Qwen TTS config keys")
    cfg = defaults | cfg
    for key in required:
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if (
        cfg["precision"] not in {"bf16", "fp32"}
        or min(cfg[k] for k in ("batch_size", "max_steps", "learning_rate", "eval_every")) <= 0
    ):
        raise ValueError("invalid Qwen training settings")
    rt = Runtime(cfg["precision"])
    try:
        _train(cfg, rt, resume)
    finally:
        rt.close()


def _train(cfg, rt, resume):
    torch.manual_seed(cfg["seed"] + rt.rank)
    rows, metadata = read_cache(cfg["train_cache"])
    validation, val_metadata = read_cache(cfg["validation_cache"])
    if metadata != val_metadata or not metadata["include_codes"]:
        raise ValueError("matching train/validation caches with codec targets are required")
    for key in ("id", "audio_sha256"):
        if {r[key] for r in rows} & {r[key] for r in validation}:
            raise ValueError(f"TTS split leakage in {key}")
    if len({r["language"] for r in rows}) < 2:
        raise ValueError("multilingual TTS training requires at least two languages")
    saved_adapter = torch.load(cfg["speaker_checkpoint"], map_location="cpu", weights_only=True)
    if saved_adapter["metadata"]["gemma_config"] != metadata["gemma_config"] or saved_adapter[
        "metadata"
    ].get("gemma_weights") != metadata.get("gemma_weights"):
        raise ValueError("speaker adapter and feature cache use different Gemma configurations")
    if weights_digest(cfg["qwen"]) != metadata["qwen_weights"]:
        raise ValueError("Qwen weights differ from the feature/codec cache")
    adapter = SpeakerAdapter(**saved_adapter["dimensions"])
    adapter.load_state_dict(saved_adapter["adapter"], strict=True)
    api = load_qwen_tts(cfg["qwen"])
    model = QwenTTSFineTune(api, adapter, cfg["max_text_tokens"]).to(rt.device)
    del api  # Codec/speaker teacher are not needed during cached training.
    optimizer = torch.optim.AdamW(model.talker.parameters(), lr=cfg["learning_rate"])
    step, epoch, offset = 0, 0, 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)

        def comparable(c):
            return {k: v for k, v in c.items() if k not in {"output_dir", "workers"}}

        if comparable(saved["config"]) != comparable(cfg) or saved["world"] != rt.world:
            raise ValueError("resume requires identical config/world size")
        model.talker.load_state_dict(saved["talker"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, offset = saved["step"], saved["epoch"], saved["offset"]
        torch.set_rng_state(saved["rng"][rt.rank]["cpu"])
        if rt.device.type == "cuda":
            torch.cuda.set_rng_state(saved["rng"][rt.rank]["cuda"], rt.device)
    output = Path(cfg["output_dir"])
    if rt.main:
        output.mkdir(parents=True, exist_ok=False)
    rt.barrier()
    wrapped = (
        DistributedDataParallel(
            model, device_ids=[rt.local_rank] if rt.device.type == "cuda" else None
        )
        if rt.world > 1
        else model
    )
    weights = sampling_weights(
        [SimpleNamespace(language=r["language"]) for r in rows], cfg["language_weights"]
    )
    sampler = WeightedDistributedSampler(
        weights, len(rows), cfg["batch_size"], 1, rt.rank, rt.world, cfg["seed"]
    )
    loader = DataLoader(
        rows,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        collate_fn=identity_collate,
        num_workers=cfg["workers"],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    model.train()
    while step < cfg["max_steps"]:
        sampler.epoch, sampler.offset = epoch, offset * cfg["batch_size"]
        for index, batch in enumerate(loader, offset):
            optimizer.zero_grad(set_to_none=True)
            with rt.autocast():
                loss = wrapped(batch)
            if not rt.all_finite(loss):
                raise FloatingPointError("nonfinite TTS loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.talker.parameters(), 1.0)
            if not rt.all_finite(norm):
                raise FloatingPointError("nonfinite TTS gradients")
            optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * learning_rate_factor(
                step, min(250, cfg["max_steps"] // 10), cfg["max_steps"]
            )
            optimizer.step()
            step, offset = step + 1, index + 1
            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                model.eval()
                totals = [0.0, 0]
                local = validation[rt.rank :: rt.world]
                with torch.no_grad():
                    for start in range(0, len(local), cfg["batch_size"]):
                        selected = local[start : start + cfg["batch_size"]]
                        with rt.autocast():
                            value = model(selected)
                        totals[0] += float(value) * len(selected)
                        totals[1] += len(selected)
                totals = rt.gather(totals)
                rng = {"cpu": torch.get_rng_state()}
                if rt.device.type == "cuda":
                    rng["cuda"] = torch.cuda.get_rng_state(rt.device)
                rngs = rt.gather(rng)
                if rt.main:
                    report = {
                        "step": step,
                        "validation_teacher_forced_loss": sum(t[0] for t in totals)
                        / sum(t[1] for t in totals),
                    }
                    print(json.dumps(report), flush=True)
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(report) + "\n")
                    torch.save(
                        {
                            "format": "gemma-qwen-talker-v1",
                            "talker": model.talker.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": cfg,
                            "metadata": metadata,
                            "step": step,
                            "epoch": epoch,
                            "offset": offset,
                            "world": rt.world,
                            "rng": rngs,
                        },
                        output / "last.tmp",
                    )
                    os.replace(output / "last.tmp", output / "last.pt")
                model.train()
            if step >= cfg["max_steps"]:
                break
        epoch, offset = epoch + 1, 0
