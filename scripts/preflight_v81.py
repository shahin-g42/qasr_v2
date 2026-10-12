"""Validate the online code-switch index and compute the configured world-size coverage budget."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from preflight_v80 import check_inputs

from qasr.code_switch import OnlineCodeSwitch
from qasr.config import TrainConfig
from qasr.data import JsonlSpeechDataset
from qasr.processing import QASRProcessor
from qasr.sampling import WeightedStratifiedLanguageSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v8.1/02_full_10node.yaml')
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--resume-from-checkpoint')
    args = parser.parse_args()
    print(f'preflight: loading config {args.config}', flush=True)
    cfg = TrainConfig.from_yaml(args.config)
    print('preflight: checking checkpoint files and manifest paths', flush=True)
    check_inputs(cfg, resume=args.resume_from_checkpoint, smoke=args.smoke_test)
    print('preflight: loading processor and tokenizer', flush=True)
    processor = QASRProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    model_config = json.loads((Path(cfg.model_name_or_path) / 'config.json').read_text())
    if getattr(processor, 'projector_pool_stride', 1) != model_config.get('projector_pool_stride', 1):
        raise ValueError('Processor/model pooling stride mismatch')
    if cfg.max_duration_seconds > processor.feature_extractor.max_audio_clip_s:
        raise ValueError('Configured duration exceeds the processor limit')
    print('preflight: opening assembled code-switch index', flush=True)
    mixer = OnlineCodeSwitch(cfg.code_switch_index, processor.tokenizer, allow_pilot=args.smoke_test)
    from data_processing.code_switch.common import load_plan
    from data_processing.code_switch.index import check_frozen
    print('preflight: checking frozen manifest metadata and tokenizer hashes', flush=True)
    check_frozen(mixer.index['root'], load_plan(mixer.index['root']))
    print(f"Per-example code-switch trigger probability: {cfg.code_switch_probability}; pools: {mixer.index['pools']}")
    if args.scan:
        print('preflight: scanning full training manifests for coverage counts', flush=True)
        parts = [JsonlSpeechDataset(path, language=lang, audio_root=cfg.audio_root,
                    min_duration_seconds=cfg.min_duration_seconds, max_duration_seconds=cfg.max_duration_seconds,
                    validate_audio_paths=False) for path, lang in cfg.train_manifest_specs]
        originals = Counter()
        for part in parts:
            originals[part.language] += len(part)
        sampler = WeightedStratifiedLanguageSampler(parts, batch_size=cfg.per_device_train_batch_size,
                    seed=cfg.seed, language_batch_counts=cfg.language_batch_counts, num_replicas=cfg.expected_world_size)
        steps = math.ceil(sampler.num_batches / cfg.gradient_accumulation_steps)
        print(f'Original eligible counts: {dict(originals)}; full coverage: {steps:,} updates')
        if not args.smoke_test and steps != cfg.max_steps:
            raise ValueError(f'Coverage budget differs: configured max_steps={cfg.max_steps}, computed={steps}')
        print('Run code-switch verify --audio separately for a full source/index integrity check.')
    print('v8.1 preflight passed; GPU memory/throughput still require a cluster rehearsal.')


if __name__ == '__main__':
    main()
