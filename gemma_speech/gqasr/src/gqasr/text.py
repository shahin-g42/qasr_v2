"""Preserve language distinctions and Arabic diacritics in training text."""

import unicodedata


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())
