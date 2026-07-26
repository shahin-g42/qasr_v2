from __future__ import annotations

import argparse

import torch
from transformers.audio_utils import load_audio

from qasr import QASRForConditionalGeneration, QASRProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one file with QASR.")
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--eagle",
        default=None,
        help="Path to trained EAGLE head directory for speculative decoding",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=5,
        help="Number of draft tokens for speculative decoding",
    )
    args = parser.parse_args()

    processor = QASRProcessor.from_pretrained(args.model)
    audio = load_audio(args.audio, sampling_rate=processor.feature_extractor.sampling_rate)
    inputs = processor.apply_transcription_request(
        audio=audio,
        language=args.language,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
    )

    if args.eagle:
        # Speculative decoding with EAGLE-2
        from qasr.eagle import EagleSpeculativeDecoder

        decoder = EagleSpeculativeDecoder.from_pretrained(
            model_path=args.model,
            eagle_path=args.eagle,
            dtype=torch.bfloat16,
            device="auto",
        )
        inputs = inputs.to(decoder.device, decoder.dtype)
        output_ids = decoder.generate(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs["input_features_mask"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
        )
    else:
        # Standard autoregressive decoding
        # The full model is ~8 GB in bf16 and fits comfortably on one GPU;
        # sharding with device_map="auto" only adds cross-device hops.
        model = QASRForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        ).eval()
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    print(processor.decode(generated_ids[0], return_format="transcription_only"))


if __name__ == "__main__":
    main()
