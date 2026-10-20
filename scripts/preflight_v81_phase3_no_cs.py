"""Stage 3 validation without any code-switch index dependency."""
import argparse
import json
from pathlib import Path

from preflight_v80 import check_inputs
from preflight_v81_phase3 import check_stage2_export, scan_coverage

from qasr.config import TrainConfig
from qasr.processing import QASRProcessor


def check_no_code_switch(cfg):
    if cfg.code_switch_index is not None or cfg.code_switch_probability != 0:
        raise ValueError('This launcher requires code_switch_index: null and code_switch_probability: 0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v8.1/03_hq_8node_b20_no_cs.yaml')
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--resume-from-checkpoint')
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg.validate()
    check_no_code_switch(cfg)
    print('preflight: code-switching disabled; no partner index required', flush=True)
    check_stage2_export(cfg.model_name_or_path, cfg.output_dir)
    print('preflight: checking checkpoint shards, manifests, DeepSpeed and MUSAN', flush=True)
    check_inputs(cfg, resume=args.resume_from_checkpoint, smoke=args.smoke_test)
    print('preflight: loading processor/tokenizer', flush=True)
    processor = QASRProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    model_config = json.loads((Path(cfg.model_name_or_path) / 'config.json').read_text())
    if getattr(processor, 'projector_pool_stride', 1) != model_config.get('projector_pool_stride', 1):
        raise ValueError('Processor/model pooling stride mismatch')
    if cfg.max_duration_seconds > processor.feature_extractor.max_audio_clip_s:
        raise ValueError('Configured duration exceeds processor limit')
    if args.scan:
        scan_coverage(cfg, exact=False)
    print('Stage 3 preflight passed. GPU memory still requires a cluster smoke test.', flush=True)


if __name__ == '__main__':
    main()
