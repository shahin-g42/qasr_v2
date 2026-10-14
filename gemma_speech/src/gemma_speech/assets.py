import json
import tempfile
from pathlib import Path


def load_assets(config):
    from transformers import AutoTokenizer, Gemma4AudioFeatureExtractor

    options = {"revision": config.revision, "local_files_only": config.local_files_only}
    if config.task == "asr":
        extractor = Gemma4AudioFeatureExtractor.from_pretrained(config.model_id, **options)
        if extractor.sampling_rate != 16000:
            raise ValueError("expected the official 16 kHz audio frontend")
        return {"feature_extractor": json.loads(extractor.to_json_string())}
    if not config.use_semantic_text:
        return {}
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=False, **options)
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.save_pretrained(directory)
        return {"hf_tokenizer_files": {p.name: p.read_bytes() for p in Path(directory).iterdir()}}


def processors_from_assets(assets):
    from transformers import AutoTokenizer, Gemma4AudioFeatureExtractor

    extractor, tokenizer = None, None
    if "feature_extractor" in assets:
        extractor = Gemma4AudioFeatureExtractor.from_dict(assets["feature_extractor"])
    if "hf_tokenizer_files" in assets:
        with tempfile.TemporaryDirectory() as directory:
            for name, data in assets["hf_tokenizer_files"].items():
                if Path(name).name != name:
                    raise ValueError("invalid tokenizer asset filename")
                (Path(directory) / name).write_bytes(data)
            tokenizer = AutoTokenizer.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False
            )
    return extractor, tokenizer
