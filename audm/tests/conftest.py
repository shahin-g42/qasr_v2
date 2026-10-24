import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

torch.set_num_threads(1)

CHAT_TEMPLATE = """{%- set ns = namespace(system_text='') -%}
{%- for m in messages -%}{%- if m.role == 'system' -%}
{%- for c in m.content -%}{%- if c.type == 'text' -%}{%- set ns.system_text = ns.system_text + c.text -%}{%- endif -%}{%- endfor -%}
{%- endif -%}{%- endfor -%}
{%- set audio_tokens = '<|audio_start|><|audio_pad|><|audio_end|>' -%}
{{- '<|im_start|>system\n' + ns.system_text + '<|im_end|>\n' -}}
{{- '<|im_start|>user\n' + audio_tokens + '<|im_end|>\n' -}}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"""


def toy_tokenizer():
    special_tokens = [
        "<unk>",
        "<pad>",
        "<|im_start|>",
        "<|im_end|>",
        "<|audio_start|>",
        "<|audio_pad|>",
        "<|audio_end|>",
        "<asr_text>",
    ]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        [
            "system user assistant language Arabic English Chinese Hindi Malayalam",
            "hello world a longer transcript",
        ],
        trainers.BpeTrainer(vocab_size=128, special_tokens=special_tokens),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=special_tokens[2:],
        audio_token="<|audio_pad|>",
        audio_bos_token="<|audio_start|>",
        audio_eos_token="<|audio_end|>",
    )


@pytest.fixture
def tokenizer():
    return toy_tokenizer()


@pytest.fixture(autouse=True)
def cpu_backend_on_non_cuda_hosts(monkeypatch):
    # These are CPU/CUDA contract tests. Transformers can select MPS separately
    # from Accelerate's CPU flag, producing mixed devices on macOS; CTC also
    # lacks an MPS kernel. Keep this test-only backend choice explicit.
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
