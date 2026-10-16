import torch
from gqasr.language_replay import append_language


def test_language_prefill_appends_without_changing_audio_or_padding():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "language Chinese<asr_text>"
            assert not add_special_tokens
            return [2, 3]
    class Decoder:
        def get_input_embeddings(self):
            return torch.nn.Embedding(5, 4)
    embeds = torch.randn(1, 7, 4)
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    result, result_mask = append_language(embeds, mask, Decoder(), Tokenizer())
    assert result.shape == (1, 9, 4)
    assert torch.equal(result[:, :7], embeds)
    assert torch.equal(result_mask[:, :7], mask)
    assert result_mask[:, 7:].all()
