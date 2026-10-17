"""Trace Chinese clips alone and in real multilingual training batches."""
import argparse
import copy
import json
from pathlib import Path

import torch

from .comparison_audit import dump, save_rows
from .data import Collator, ManifestIndex, StratifiedUpdateSampler
from .language_replay import teacher_losses
from .modeling import GQASRForSpeechRecognition
from .processing import GQASRProcessor
from .qwen_baseline import select_samples


def difference(a, b):
    if a.shape != b.shape:
        return dict(shape_equal=False, solo_shape=list(a.shape), batch_shape=list(b.shape))
    a, b = a.float(), b.float()
    delta = b-a
    return dict(shape_equal=True, exact=bool(torch.equal(a,b)),
                max_absolute=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                relative_l2=float(delta.norm()/a.norm().clamp_min(1e-12)))


def configuration(cfg, mode):
    c = copy.deepcopy(cfg)
    c['online_code_switching'] = mode in ('code_switch', 'all')
    if mode in ('solo', 'batch_clean', 'code_switch'):
        c['augmentation'] = None
    elif mode == 'waveform':
        c['augmentation'].pop('spec_augment', None)
    elif mode == 'specaugment':
        c['augmentation'] = {'spec_augment': c['augmentation']['spec_augment']}
    return c


def run(args):
    device = args.device
    prior = Path(args.comparison)
    cfg = json.loads((prior/'training-config.json').read_text())
    checkpoint = prior/'snapshots/current'
    jobs = json.loads((prior/'jobs.json').read_text())
    command = jobs[0]['command']
    samples = select_samples(cfg['index_path'], command[command.index('--predictions')+1])[:args.samples]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output/'samples.jsonl', samples)
    dump(output/'settings.json', vars(args))
    processor = GQASRProcessor(checkpoint)
    model = GQASRForSpeechRecognition.from_pretrained(checkpoint, device).eval()
    dataset = ManifestIndex(cfg['index_path'], 'train')
    sampler = iter(StratifiedUpdateSampler(dataset.counts, cfg['language_batch_counts'], cfg['batch_size'],
                                          args.start+len(samples), 0, cfg['expected_world_size'], cfg['data_seed'],
                                          args.start, augmentation_seeds=True))
    records = []
    for number, sample in enumerate(samples):
        source_rows = [dataset[k] for k in next(sampler)]
        position = next(i for i,r in enumerate(source_rows) if r['language']=='zh')
        source_rows[position] = dict(id=sample['id'], audio=sample['audio'], text=sample['reference'],
                                    duration=sample['duration'], language='zh', dialect='unspecified',
                                    augmentation_seed=source_rows[position]['augmentation_seed'])
        baseline = None
        for mode in ('solo', 'batch_clean', 'waveform', 'specaugment', 'code_switch', 'all'):
            cc = configuration(cfg, mode)
            selected = [source_rows[position]] if mode == 'solo' else source_rows
            index = 0 if mode == 'solo' else position
            batch, rows = Collator(processor, cc, training=True)(selected)
            for i, row in enumerate(rows):
                tokens = batch['targets'][i][batch['target_mask'][i]].tolist()
                if tokens != processor.target(row['text'], row['language']):
                    raise ValueError('Target/transcript mismatch at collation')
            feature = batch['input_features'][index][batch['input_features_mask'][index]].clone()
            batch = {k:v.to(device) for k,v in batch.items()}
            capture = {}
            def hook(module, inputs, result, capture=capture, index=index):
                capture['encoder'] = result.last_hidden_state[index][result.attention_mask[index].bool()].detach().cpu()
            handle = model.gqasr_acoustic.register_forward_hook(hook)
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                    audio, valid = model.audio_embeddings(batch['input_features'], batch['input_features_mask'])
                    # Verify model fusion for every multilingual row, including padding and shift.
                    _, _, labels = model.fuse(audio, valid, batch['targets'], batch['target_mask'])
                    for i in range(len(rows)):
                        if not torch.equal(labels[i][labels[i]!=-100], batch['targets'][i][batch['target_mask'][i]]):
                            raise ValueError('Target association changed in model fusion')
                    bridge = audio[index][valid[index]].detach().cpu()
                    row = rows[index]
                    if row['language'] != 'zh':
                        raise ValueError('Chinese anchor changed language')
                    nll = teacher_losses(model, processor, audio[index:index+1], valid[index:index+1],
                                         [dict(reference=row['text'])])[0]
                    embeds, mask, _ = model.fuse(audio[index:index+1], valid[index:index+1], left_pad=True)
                    ids = model.gqasr_decoder.generate(inputs_embeds=embeds, attention_mask=mask,
                        max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=model.eos_id, pad_token_id=model.eos_id)
                    raw = processor.tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
            finally:
                handle.remove()
            tensors = dict(frontend=feature, encoder=capture['encoder'], bridge=bridge)
            if baseline is None:
                baseline = tensors
            record = dict(id=sample['id'], mode=mode, batch_size=len(rows), batch_position=index,
                          original_reference=sample['reference'], supervised_text=row['text'],
                          assigned_language=row['language'], augmentation=row.get('augmentation_audit', {}),
                          differences_from_clean_solo={k:difference(baseline[k],v) for k,v in tensors.items()},
                          teacher_forced=nll, raw_output=raw, hypothesis=processor.decode(ids[0].tolist()))
            records.append(record)
            with (output/'trace.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            print(json.dumps(dict(event='batch_invariance', sample=number+1,total=len(samples),mode=mode,
                                  differences=record['differences_from_clean_solo'])), flush=True)
            del batch, audio, valid, embeds, labels, tensors, capture
        # Release per-sample baseline before processing the next clip.
        del baseline
    clean = [r for r in records if r['mode']=='batch_clean']
    dump(output/'summary.json', dict(clean_batch_comparisons=clean, examples=len(samples),
        scope='Eval-mode encoder/bridge on actual train-collated batches; supervised model fusion checked on all rows. Decoder generation/NLL scored on Chinese row individually; no backward/DDP. Augmented differences are expected, not automatically bugs.'))
    print(f'COMPLETE: {output / "summary.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--start', type=int, default=9376)
    parser.add_argument('--max-new-tokens', type=int, default=768)
    args = parser.parse_args()
    if args.samples < 1 or args.start < 0 or args.max_new_tokens < 1:
        parser.error('Positive samples and nonnegative start required')
    torch.set_num_threads(2)
    run(args)


if __name__ == '__main__':
    main()
