"""Pinned positional answer vocabulary, without model-runtime imports."""

from __future__ import annotations

import itertools
import string
from collections.abc import Sequence
from typing import Any

AUTOJEV_MODEL = "denis-pplx/autojev-27b"
AUTOJEV_REVISION = "6f5b557e037f5edb25c7dc92dbc6553e5a19c015"
AUTOJEV_SOURCE_REVISION = "ee63c1515980491a742f0bd0685c8dc5ca1f00c3"
BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MAX_OPTIONS = 255


def discover_codes(tokenizer: Any, count: int = MAX_OPTIONS) -> tuple[list[str], list[int]]:
    """Match upstream selection order, then verify the exact non-thinking prefix.

    Unstable/duplicate selected codes are errors, not reasons to skip ahead and
    silently change the positional readout mapping.
    """
    if type(count) is not int or not 1 <= count <= MAX_OPTIONS:
        raise ValueError("count must be an integer in 1..255")
    candidates = list(string.ascii_uppercase) + [
        "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    codes, token_ids = [], []
    for code in candidates:
        encoded = tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) == 1:
            codes.append(code)
            token_ids.append(encoded[0])
        if len(codes) == count:
            break
    if len(codes) != count or any(type(i) is not int or i < 0 for i in token_ids):
        raise ValueError(f"Tokenizer must provide {count} valid single-token answer codes")
    if len(set(token_ids)) != count:
        raise ValueError("Answer token IDs must be distinct")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Choose an option."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Tokenizer must provide a nonempty non-thinking chat prefix")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for code, token_id in zip(codes, token_ids, strict=True):
        if tokenizer.encode(prefix + code, add_special_tokens=False) != prefix_ids + [token_id]:
            raise ValueError(f"Answer code {code!r} is unstable after the chat prefix")
    return codes, token_ids


def verify_codes(
    tokenizer: Any, codes: Sequence[str], token_ids: Sequence[int],
) -> tuple[list[str], list[int]]:
    """Require all 255 saved positions to equal rediscovery, including ID types."""
    actual = discover_codes(tokenizer)
    if (
        isinstance(codes, (str, bytes)) or isinstance(token_ids, (str, bytes))
        or any(type(i) is not int for i in token_ids)
        or list(codes) != actual[0] or list(token_ids) != actual[1]
    ):
        raise ValueError("Checkpoint answer vocabulary differs from its tokenizer")
    return actual
