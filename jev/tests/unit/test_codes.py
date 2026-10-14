from __future__ import annotations

import itertools
import os
import string
import subprocess
import sys
from pathlib import Path

import pytest

from jev.codes import discover_codes, verify_codes


class FakeTokenizer:
    padding_side = "right"
    prefix = "chat-prefix\n"

    def __init__(self):
        candidates = list(string.ascii_uppercase) + [
            "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
        ]
        self.vocab = {code: index + 1 for index, code in enumerate(candidates)}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == self.prefix:
            return [900]
        if text.startswith(self.prefix):
            return [900, *self.encode(text[len(self.prefix):], add_special_tokens=False)]
        return [self.vocab[text]] if text in self.vocab else [800, 801]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        }
        assert messages == [{"role": "user", "content": "Choose an option."}]
        return self.prefix


def test_order_and_saved_mapping():
    tokenizer = FakeTokenizer()
    codes, ids = discover_codes(tokenizer)
    assert codes[:28] == [*string.ascii_uppercase, "AA", "AB"]
    assert len(codes) == len(set(ids)) == 255
    assert verify_codes(tokenizer, codes, ids) == (codes, ids)
    with pytest.raises(ValueError, match="differs"):
        verify_codes(tokenizer, codes[::-1], ids)
    with pytest.raises(ValueError, match="differs"):
        verify_codes(tokenizer, codes, [True, *ids[1:]])


def test_skip_multitoken_but_never_skip_unstable_selected_codes():
    tokenizer = FakeTokenizer()
    del tokenizer.vocab["AA"]
    assert discover_codes(tokenizer)[0][26] == "AB"
    original = tokenizer.encode
    tokenizer.encode = lambda text, **kw: [999] if text == tokenizer.prefix + "A" else original(
        text, **kw
    )
    with pytest.raises(ValueError, match="unstable"):
        discover_codes(tokenizer)


def test_insufficient_or_duplicate_tokens():
    tokenizer = FakeTokenizer()
    tokenizer.vocab["B"] = tokenizer.vocab["A"]
    with pytest.raises(ValueError, match="distinct"):
        discover_codes(tokenizer)
    tokenizer.vocab = {"A": 1}
    with pytest.raises(ValueError, match="255"):
        discover_codes(tokenizer)


@pytest.mark.parametrize("count", [0, 256, True, 1.5])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        discover_codes(FakeTokenizer(), count)


def test_namespace_and_processing_import_do_not_load_runtime():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(root / "packages/core/src"), str(root / "packages/contracts/src"),
    ]))
    subprocess.run(
        [sys.executable, "-c", "import jev, jev.codes; from jev import PreparedBatch; "
         "import sys; assert 'torch' not in sys.modules; "
         "assert 'transformers' not in sys.modules"],
        env=env, check=True,
    )
