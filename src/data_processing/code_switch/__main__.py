"""Staged preparation of an offline index for online code-switch augmentation."""
import argparse
import json
import os
import socket
import time
from pathlib import Path

from .common import lock
from .index import assemble, plan, run, status, stop_run, tokenizer_from, verify


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run', 'stop', 'status', 'verify', 'assemble', 'peek', 'lock-test'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--run-root', required=True)
        if name in ('plan', 'verify', 'assemble'):
            cmd.add_argument('--jobs', type=positive, default=min(16, os.cpu_count() or 1))
        if name == 'plan':
            cmd.add_argument('--config', default='configs/code_switch/full_v1.yaml')
            cmd.add_argument('--samples', type=positive, help='pilot preview count; production training rejects pilots')
            cmd.add_argument('--limit', type=positive, help='pilot only: index first N rows per train manifest; all eval rows excluded')
        if name == 'run':
            cmd.add_argument('--procs', type=positive, default=8)
        if name == 'stop':
            cmd.add_argument('--all', action='store_true')
        if name == 'verify':
            cmd.add_argument('--reset', action='store_true')
            modes = cmd.add_mutually_exclusive_group()
            modes.add_argument('--audio', action='store_true', help='fully decode eligible audio again (default: headers)')
            modes.add_argument('--metadata-only', action='store_true', help='hash/validate manifests and index records without reopening source audio')
        if name == 'peek':
            cmd.add_argument('-n', type=positive, default=2, help='examples per pair')
            cmd.add_argument('--export-audio', action='store_true')
        if name == 'lock-test':
            cmd.add_argument('--hold', type=positive, default=1)
    args = parser.parse_args(argv)
    root = Path(args.run_root).resolve()
    try:
        if args.command == 'plan':
            plan(root, args.config, args.jobs, args.samples, args.limit)
        elif args.command == 'run':
            run(root, args.procs)
        elif args.command == 'stop':
            stop_run(root, args.all)
        elif args.command == 'status':
            print(json.dumps(status(root), indent=2))
        elif args.command == 'verify':
            bad = verify(root, args.jobs, args.reset, args.audio, args.metadata_only)
            print(json.dumps({'bad': bad, 'reset': args.reset}, indent=2))
            return int(bool(bad))
        elif args.command == 'assemble':
            assemble(root, args.jobs)
        elif args.command == 'lock-test':
            with lock(root / '_state/filesystem-test.lock'):
                print(f'{socket.gethostname()}: acquired lock for {args.hold}s; probe from another node now', flush=True)
                time.sleep(args.hold)
        elif args.command == 'peek':
            import soundfile as sf

            from qasr.code_switch import OnlineCodeSwitch
            mixer = OnlineCodeSwitch(root / 'assembled/index.json', tokenizer_from(root / '_state/tokenizer'), allow_pilot=True)
            for part in mixer.parts():
                for i in range(min(args.n, len(part))):
                    row = part[i]
                    waveform = row.pop('waveform')
                    if args.export_audio:
                        folder = root / '_preview'
                        folder.mkdir(exist_ok=True)
                        path = folder / f'{row["pair"]}-{part.start+i}.flac'
                        sf.write(path, waveform, mixer.cfg['sample_rate'])
                        row['preview_audio'] = str(path)
                    print(json.dumps(row, ensure_ascii=False))
    except BlockingIOError:
        parser.exit(1, 'Lock busy. Another owner is active; stop workers before verify/assemble/reset.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
