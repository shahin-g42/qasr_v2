"""Audit real training batches on all logical ranks, without model training."""
import argparse
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Collator, ManifestIndex, StratifiedUpdateSampler
from .processing import LANGUAGES, GQASRProcessor


class AuditedCollator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processor = GQASRProcessor(cfg['base_model'])
        self.collator = Collator(self.processor, cfg, training=True)

    def __call__(self, inputs):
        import gqasr.data as data_module
        loaded = []
        mixtures = []
        loader = data_module.load_wave
        def load(path, *args):
            loaded.append(str(path))
            return loader(path, *args)
        switcher = self.collator.switcher
        original = switcher.augment if switcher else None
        original_record = switcher._record if switcher else None
        def record(pair, language, index):
            value = original_record(pair, language, index)
            if value['language'] != language:
                raise ValueError(f'Code-switch pool {language} returned {value["language"]}')
            return value
        def mix(feature, *args, **kwargs):
            result, audit = original(feature, *args, **kwargs)
            if audit['applied']:
                segments = result['segments']
                if result['text'] != ' '.join(s['text'] for s in segments):
                    raise ValueError('Code-switch transcript/segment order mismatch')
                if switcher.cfg['language_policy'] == 'first' and result['language'] != feature['language']:
                    raise ValueError('Code-switch first-language policy violated')
                mixtures.append(dict(anchor=feature['language'], assigned=result['language'],
                                     languages=[s['language'] for s in segments],
                                     texts=[s['text'] for s in segments], text=result['text']))
            return result, audit
        data_module.load_wave = load
        if switcher:
            switcher.augment = mix
            switcher._record = record
        try:
            batch, rows = self.collator(inputs)
        finally:
            data_module.load_wave = loader
            if switcher:
                switcher.augment = original
                switcher._record = original_record
        if loaded != [r['audio'] for r in inputs]:
            raise ValueError('Source waveform loading order differs from sampler rows')
        traces = []
        for i, (before, after) in enumerate(zip(inputs, rows, strict=True)):
            if before['id'] != after['id']:
                raise ValueError('Collator reordered examples')
            if before['id'].split(':')[1] != before['language']:
                raise ValueError('Index key/language mismatch')
            tokens = batch['targets'][i][batch['target_mask'][i]].tolist()
            expected = self.processor.target(after['text'], after['language'])
            if tokens != expected:
                raise ValueError('Padded target row does not match transcript/language')
            header = self.processor.tokenizer.decode(tokens, skip_special_tokens=False).split('<asr_text>')[0].strip()
            if header != 'language ' + LANGUAGES[after['language']]:
                raise ValueError(f'Wrong language header: {header}')
            traces.append(dict(id=before['id'], audio=before['audio'], anchor=before['language'],
                               assigned=after['language'], header=header, text=after['text'],
                               augmentation=after.get('augmentation_audit', {})))
        return traces, mixtures


def audit_rank(task):
    cfg, rank, start, updates, workers, output = task
    torch.set_num_threads(1)
    dataset = ManifestIndex(cfg['index_path'], 'train')
    sampler = StratifiedUpdateSampler(dataset.counts, cfg['language_batch_counts'], cfg['batch_size'],
                                      start+updates, rank, cfg['expected_world_size'], cfg['data_seed'],
                                      start, augmentation_seeds=True)
    expected = iter(StratifiedUpdateSampler(dataset.counts, cfg['language_batch_counts'], cfg['batch_size'],
                                            start+updates, rank, cfg['expected_world_size'], cfg['data_seed'],
                                            start, augmentation_seeds=True))
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=AuditedCollator(cfg), num_workers=workers,
                        multiprocessing_context='spawn' if workers else None,
                        generator=torch.Generator().manual_seed(cfg['data_seed']))
    counts, assigned, chinese_context = Counter(), Counter(), Counter()
    accumulation = cfg['accumulation_steps']
    update_counts = Counter()
    with Path(output, f'rank-{rank:03d}-start-{start}.jsonl').open('x') as stream:
        for batch_number, (rows, mixes) in enumerate(loader):
            keys = next(expected)
            if [r['id'] for r in rows] != [f'train:{k[0]}:{k[1]}' for k in keys]:
                raise ValueError('DataLoader reordered sampler batches')
            counts.update(r['anchor'] for r in rows)
            assigned.update(r['assigned'] for r in rows)
            update_counts.update(r['anchor'] for r in rows)
            if (batch_number+1) % accumulation == 0:
                if dict(update_counts) != cfg['language_batch_counts']:
                    raise ValueError(f'Rank {rank}: wrong per-update quotas: {update_counts}')
                update_counts.clear()
            for mix in mixes:
                if 'zh' in mix['languages']:
                    chinese_context[mix['assigned']] += 1
            stream.write(json.dumps(dict(rank=rank, microbatch=batch_number, rows=rows, mixtures=mixes), ensure_ascii=False)+'\n')
    return dict(rank=rank, start=start, anchors=dict(counts), assigned=dict(assigned),
                mixed_examples_containing_chinese_by_header=dict(chinese_context))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parallel-ranks', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--starts', nargs='+', type=int, default=[0, 9376])
    args = parser.parse_args()
    if min(args.parallel_ranks, args.updates) < 1 or args.workers < 0 or min(args.starts) < 0:
        parser.error('Invalid worker/update counts')
    cfg = json.loads(Path(args.run_config).read_text())
    if cfg.get('streaming_datasets'):
        raise ValueError('This audit targets the indexed-data training run; streaming requires a separate trace')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(cfg, indent=2))
    tasks = [(cfg, rank, start, args.updates, args.workers, str(output))
             for start in args.starts for rank in range(cfg['expected_world_size'])]
    results = []
    with ProcessPoolExecutor(args.parallel_ranks, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(audit_rank, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            print(json.dumps(dict(event='batch_audit_progress', done=len(results), total=len(tasks))), flush=True)
    (output/'summary.json').write_text(json.dumps(dict(passed=True, results=results,
        scope='Real indexed audio/augmentation/DataLoader/targets; logical rank replay, no NCCL or model forward. Sampled update windows, not every historical batch.'), indent=2))
    print(f'COMPLETE: {output / "summary.json"}')


if __name__ == '__main__':
    main()
