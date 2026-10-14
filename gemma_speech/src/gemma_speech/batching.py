import torch
from torch.nn.utils.rnn import pad_sequence

from .audio import LogMel
from .data import load_audio, pad_sequences


class Collator:
    def __init__(
        self,
        config,
        tokenizer,
        feature_extractor=None,
        hf_tokenizer=None,
        speakers=None,
        languages=None,
    ):
        self.config, self.tokenizer = config, tokenizer
        self.feature_extractor, self.hf_tokenizer = feature_extractor, hf_tokenizer
        self.speakers, self.languages = speakers or {}, languages or {}
        self.mel = LogMel() if config.model.task == "tts" else None

    def __call__(self, rows):
        cfg = self.config
        is_asr = cfg.model.task == "asr"
        texts = [r.text if is_asr else r.tts_text for r in rows]
        sequences = (
            [self.tokenizer.target(r.text, r.language) for r in rows]
            if is_asr and cfg.model.asr_decoder == "qwen"
            else [self.tokenizer.encode(t) for t in texts]
        )
        if any(len(s) > cfg.max_text_tokens for s in sequences):
            raise ValueError("transcript exceeds token limit; segment it instead of truncating")
        tokens, mask = pad_sequences(sequences)
        waves = [
            load_audio(r.audio, 16000 if is_asr else 24000, cfg.max_audio_seconds) for r in rows
        ]
        if is_asr:
            features = self.feature_extractor(
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
                "targets": tokens,
                "target_mask": mask,
            }
        else:
            mels = [self.mel(torch.from_numpy(w)) for w in waves]
            batch = {
                "tokens": tokens,
                "token_mask": mask,
                "mel": pad_sequence(mels, batch_first=True),
                "mel_lengths": torch.tensor([m.shape[0] for m in mels]),
                "speaker_ids": torch.tensor([self.speakers[r.speaker] for r in rows]),
                "language_ids": torch.tensor([self.languages[r.language] for r in rows]),
            }
            if self.hf_tokenizer is not None:
                encoded = self.hf_tokenizer(
                    texts, padding=True, truncation=False, return_tensors="pt"
                )
                if encoded["input_ids"].shape[1] > cfg.max_text_tokens:
                    raise ValueError(
                        "Gemma text exceeds token limit; segment it instead of truncating"
                    )
                batch.update(
                    input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"].bool()
                )
        return batch, rows


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
