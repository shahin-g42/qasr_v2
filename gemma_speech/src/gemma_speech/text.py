"""Multilingual text handling; training never strips Arabic diacritics."""

import io
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


class SpeechTokenizer:
    """Independent SentencePiece vocabulary. ID 0 is padding / CTC blank."""

    def __init__(self, model: str | Path | bytes):
        import sentencepiece as spm

        self.serialized = model if isinstance(model, bytes) else Path(model).read_bytes()
        self.sp = spm.SentencePieceProcessor(model_proto=self.serialized)
        if self.sp.pad_id() != 0 or self.sp.unk_id() != 1:
            raise ValueError("tokenizer requires pad/blank=0, unk=1")

    def __len__(self):
        return self.sp.vocab_size()

    def encode(self, text):
        ids = self.sp.encode(normalize_text(text), out_type=int)
        if 0 in ids or 1 in ids:
            raise ValueError("blank or unknown target token; train with byte fallback")
        return ids

    def decode(self, ids):
        return self.sp.decode([int(i) for i in ids if i != 0])


def train_tokenizer(texts, output, vocab_size=8192):
    import sentencepiece as spm

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=(normalize_text(t) for t in texts if normalize_text(t)),
        model_writer=model,
        model_type="unigram",
        vocab_size=vocab_size,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        byte_fallback=True,
        normalization_rule_name="identity",
        character_coverage=1.0,
        hard_vocab_limit=False,
        shuffle_input_sentence=False,
        num_threads=1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        f.write(model.getvalue())
    return SpeechTokenizer(model.getvalue())
