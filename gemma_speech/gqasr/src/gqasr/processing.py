"""Local-only GQASR tokenizer and acoustic frontend."""

import json
from pathlib import Path

import torch
from transformers import Gemma4AudioFeatureExtractor, PreTrainedTokenizerFast

LANGUAGES = {"ar": "Arabic", "en": "English", "hi": "Hindi", "ml": "Malayalam", "zh": "Chinese"}


class GQASRFeatureExtractor(Gemma4AudioFeatureExtractor):
    pass


class GQASRProcessor:
    def __init__(self, directory):
        directory = Path(directory)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            directory / "tokenizer", local_files_only=True
        )
        frontend = json.loads((directory / "preprocessor_config.json").read_text())
        self.frontend = GQASRFeatureExtractor.from_dict(frontend)
        self.config = json.loads((directory / "config.json").read_text())
        if self.frontend.sampling_rate != 16000:
            raise ValueError("GQASR requires 16 kHz frontend")

    def target(self, text, language):
        if language not in LANGUAGES:
            raise ValueError(f"unknown language: {language}")
        return self.tokenizer.encode(
            f"language {LANGUAGES[language]}<asr_text>{text}", add_special_tokens=False
        ) + [self.config["eos_id"]]

    def decode(self, ids):
        raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        return (
            raw.split("<asr_text>", 1)[-1]
            .split("<|im_end|>", 1)[0]
            .split("<|endoftext|>", 1)[0]
            .strip()
        )

    def batch(self, waves, texts=None, languages=None, max_target_length=512):
        features = self.frontend(
            waves,
            sampling_rate=16000,
            padding=True,
            truncation=False,
            max_length=None,
            return_tensors="pt",
        )
        batch = {
            "input_features": features["input_features"],
            "input_features_mask": features["input_features_mask"].bool(),
        }
        if texts is not None:
            targets = [self.target(t, lang) for t, lang in zip(texts, languages, strict=True)]
            if max(map(len, targets)) > max_target_length:
                raise ValueError(
                    "target exceeds max_target_length; filter/segment during preflight"
                )
            lengths = torch.tensor([len(t) for t in targets])
            tokens = torch.zeros(len(targets), int(lengths.max()), dtype=torch.long)
            for i, target in enumerate(targets):
                tokens[i, : len(target)] = torch.tensor(target)
            batch.update(
                targets=tokens, target_mask=torch.arange(tokens.shape[1])[None] < lengths[:, None]
            )
        return batch
