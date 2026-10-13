"""Validate released measurements and their integrity without private inputs."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['run_id', 'variant', 'seq_len', 'batch_size', 'max_batches', 'dtype',
          'wall_seconds', 'tokens_per_second', 'peak_memory_gib', 'mean_loss',
          'perplexity', 'token_accuracy', 'status']
STATUSES = {'measured', 'allocator_failure', 'allocator_failure_after_prior_failure'}

def read(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)

def validate_measurements(path):
    fields, rows = read(path)
    assert fields == FIELDS, f'Unexpected columns: {path}'
    assert rows, f'Empty measurements: {path}'
    seen = set()
    for row in rows:
        identity = (row['run_id'], row['variant'], row['seq_len'])
        assert identity not in seen, f'Duplicate observation: {path}: {identity}'
        seen.add(identity)
        assert re.fullmatch(r'[GLP][0-9]{2}', row['run_id'])
        assert row['variant'] in {'baseline', *[f'variant-{x}' for x in 'ABCDE']}
        assert row['status'] in STATUSES
        n, b, limit = [int(row[k]) for k in ['seq_len', 'batch_size', 'max_batches']]
        assert min(n, b, limit) > 0
        assert row['dtype'] == 'bf16'
        for field in FIELDS[6:12]:
            if row[field]:
                assert math.isfinite(float(row[field])), f'Non-finite metric: {path}: {field}'
        if row['status'] == 'measured':
            duration, speed = float(row['wall_seconds']), float(row['tokens_per_second'])
            assert duration > 0 and speed > 0
            # Token count inferred from independent timing columns must be whole batches.
            count = duration * speed
            batches = count / (n * b)
            assert abs(batches - round(batches)) < 1e-6 and 1 <= round(batches) <= limit
            assert float(row['peak_memory_gib']) > 0
            assert float(row['mean_loss']) >= 0
            assert math.isclose(math.exp(float(row['mean_loss'])), float(row['perplexity']), rel_tol=1e-8)
            if row['token_accuracy']:
                assert 0 <= float(row['token_accuracy']) <= 1
        else:
            assert all(not row[k] for k in FIELDS[6:12]), f'Failure has metrics: {path}'
    return rows

def checksum_lines():
    files = sorted([*ROOT.glob('results/**/*.csv'), *ROOT.glob('results/**/*.json')])
    return [f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}' for path in files]

def validate_links():
    for path in ROOT.rglob('*.md'):
        if '.git' in path.parts: continue
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#')): continue
            assert (path.parent / target.split('#')[0]).exists(), f'Broken local link in {path}: {target}'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-checksums', action='store_true', help='Refresh after an intentional data change')
    args = parser.parse_args()
    rows = []
    for path in sorted(ROOT.glob('results/*/measurements.csv')):
        rows.extend(validate_measurements(path))
        metadata = json.loads(path.with_name('metadata.json').read_text())
        assert metadata['trained_replacement'] is False and metadata['warmup_passes'] == 0
    validate_measurements(ROOT / 'results/gemma-dolma/additional-probes.csv')
    _, coverage = read(ROOT / 'results/llama8b-pg19/coverage.csv')
    measured = {(r['variant'], r['seq_len']) for r in rows if r['run_id'] == 'P01'}
    scheduled = set()
    for row in coverage:
        key = (row['variant'], row['seq_len'])
        assert key not in scheduled
        scheduled.add(key)
        assert row['status'] in {'measured', 'quota_blocked', 'not_attempted_after_quota'}
        assert (row['status'] == 'measured') == (key in measured)
    assert measured.issubset(scheduled)
    sums = ROOT / 'results/SHA256SUMS'
    expected = '\n'.join(checksum_lines()) + '\n'
    if args.write_checksums: sums.write_text(expected)
    assert sums.read_text() == expected, 'Released file checksum mismatch'
    validate_links()
    print(f'Validated {len(rows)} primary rows, supplementary probes, coverage, checksums, and local links.')

if __name__ == '__main__': main()
