import tempfile
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from qasr import QASRFeatureExtractor, QASRProcessor
from qasr.processing import resolve_qasr_language

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


def toy_processor() -> QASRProcessor:
    extractor = QASRFeatureExtractor(
        feature_size=128,
        sampling_rate=16_000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        dither=1e-5,
        max_audio_clip_s=35.0,
        overlap_chunk_second=5.0,
        min_energy_window_samples=1_600,
        return_attention_mask=True,
    )
    return QASRProcessor(
        feature_extractor=extractor,
        tokenizer=toy_tokenizer(),
        chat_template=CHAT_TEMPLATE,
    )


class LanguageMappingTest(unittest.TestCase):
    def test_supported_and_experimental_languages(self) -> None:
        self.assertEqual(resolve_qasr_language("ar"), "Arabic")
        self.assertEqual(resolve_qasr_language("en"), "English")
        self.assertEqual(resolve_qasr_language("zh"), "Chinese")
        self.assertEqual(resolve_qasr_language("hi"), "Hindi")
        self.assertEqual(resolve_qasr_language("ml"), "Malayalam")
        self.assertEqual(resolve_qasr_language("malayalam"), "Malayalam")

    def test_unknown_language_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported QASR language"):
            resolve_qasr_language("xx")


class ProcessorContractTest(unittest.TestCase):
    def test_feature_extraction_is_deterministic_and_time_major(self) -> None:
        processor = toy_processor()
        audio = np.linspace(-0.1, 0.1, 3_200, dtype=np.float32)

        first = processor.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        second = processor.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")

        self.assertEqual(first.input_features.shape[-1], 128)
        self.assertNotIn("audio_chunk_index", first)
        self.assertTrue(torch.equal(first.input_features, second.input_features))

    def test_placeholder_counts_and_causal_label_masking(self) -> None:
        processor = toy_processor()
        audios = [
            np.zeros(1_600, dtype=np.float32),
            np.zeros(3_200, dtype=np.float32),
        ]

        batch = processor.prepare_training_batch(
            audio=audios,
            text=["hello", "a longer transcript"],
            language=["en", "ml"],
            sampling_rate=16_000,
            padding=True,
            return_tensors="pt",
        )

        expected = processor._get_audio_token_length(batch.input_features_mask.sum(-1))
        actual = batch.input_ids.eq(processor.audio_token_id).sum(-1).cpu().numpy()
        self.assertEqual(expected.tolist(), actual.tolist())
        self.assertTrue(torch.all(batch.labels[batch.input_ids == processor.audio_token_id] == -100))
        self.assertTrue(torch.all(batch.labels[batch.attention_mask == 0] == -100))
        for row, language in enumerate(("English", "Malayalam")):
            target = processor.tokenizer.decode(batch.labels[row][batch.labels[row] != -100])
            normalized = target.replace("Ġ", "").replace(" ", "")
            self.assertIn(f"language{language}<asr_text>", normalized)

    def test_prompt_only_request_has_no_labels(self) -> None:
        processor = toy_processor()

        batch = processor.apply_transcription_request(
            audio=np.zeros(1_600, dtype=np.float32),
            language="ar",
            sampling_rate=16_000,
            return_tensors="pt",
        )

        self.assertNotIn("labels", batch)
        self.assertGreater(batch.input_ids.eq(processor.audio_token_id).sum().item(), 0)

    def test_empty_transcript_produces_tag_then_eos_targets(self) -> None:
        # Non-speech / silence records train with text="": the target must be
        # exactly `language <Lang><asr_text><eos>` with nothing between the
        # tag and EOS, and labels must exist (the row still supervises).
        processor = toy_processor()

        batch = processor.prepare_training_batch(
            audio=[np.zeros(3_200, dtype=np.float32)],
            text=[""],
            language=["ar"],
            sampling_rate=16_000,
            padding=True,
            return_tensors="pt",
        )

        labels = batch.labels[0]
        target_ids = labels[labels != -100]
        self.assertGreater(len(target_ids), 0)
        target = processor.tokenizer.decode(target_ids)
        normalized = target.replace("Ġ", "").replace(" ", "")
        self.assertIn("languageArabic<asr_text>", normalized)
        # Nothing follows the tag except the end-of-sequence turn marker.
        tail = normalized.split("<asr_text>", 1)[1]
        self.assertEqual(tail, "<|im_end|>")

    def test_processor_save_reload_round_trip(self) -> None:
        processor = toy_processor()
        processor.projector_pool_stride = 2
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            reloaded = QASRProcessor.from_pretrained(directory)

        self.assertIsInstance(reloaded.feature_extractor, QASRFeatureExtractor)
        self.assertEqual(reloaded.subsampling_factor, 8)
        # The Tier B pool stride must survive save/reload or a reloaded
        # processor would silently insert the wrong placeholder count.
        self.assertEqual(reloaded.projector_pool_stride, 2)
        self.assertEqual(reloaded.audio_token_id, processor.audio_token_id)
        self.assertEqual(reloaded.chat_template, processor.chat_template)


if __name__ == "__main__":
    unittest.main()
