"""Launch feature-extraction workers with explicit GPU groups and checked failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def split_range(start, end, n):
    """Partition [start, end) without dropping, duplicating, or creating empty ranges."""
    if start < 0 or end <= start or n < 1:
        raise ValueError("Require 0 <= start < end and at least one worker")
    n = min(n, end - start)
    size, extra = divmod(end - start, n)
    ranges = []
    for i in range(n):
        stop = start + size + (i < extra)
        ranges.append((start, stop))
        start = stop
    return ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('outdir', 'data_path', 'model_path', 'ass_path'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--e', type=int, required=True, help='Exclusive end index after dataset shuffle (seed 42)')
    parser.add_argument('--gpu-groups', default='0;1;2;3', help='Semicolon-separated workers, comma-separated GPUs per worker; e.g. "0;1" or "0,1;2,3"')
    parser.add_argument('--family', choices=['qwen2', 'qwen3'], default='qwen3')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    groups = [[int(g) for g in group.split(',')] for group in args.gpu_groups.split(';')]
    flat = [g for group in groups for g in group]
    if any(g < 0 for g in flat) or len(flat) != len(set(flat)):
        parser.error('GPU groups must be nonnegative and disjoint')
    # Preserve the original feature-directory layout.
    outdir = Path(args.outdir) / f"sharegpt_{args.s}_{args.e - 1}_mufp16"
    commands = []
    for index, (start, end) in enumerate(split_range(args.s, args.e, len(groups))):
        commands.append([sys.executable, '-m', 'ge_data.ge_data_all_' + args.family,
                         '--start', str(start), '--end', str(end), '--index', str(index),
                         '--gpu_index', *map(str, groups[index]), '--outdir', str(outdir),
                         '--data_path', args.data_path, '--model_path', args.model_path,
                         '--ass_path', args.ass_path])
    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=len(commands)) as pool:
            futures = [pool.submit(subprocess.run, command, cwd=ROOT, check=True) for command in commands]
            for future in futures:
                future.result()


if __name__ == '__main__':
    main()
