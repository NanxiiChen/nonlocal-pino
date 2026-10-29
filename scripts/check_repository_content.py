"""Check the Git index excludes generated research payloads and oversized files."""
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 1024 * 1024


def main():
    manifest = json.loads((REPO/'artifacts/local_files.json').read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unsupported local artifact manifest schema')
    extensions = set(manifest['generated_suffixes'])
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).split(b'\0')
    total = count = 0
    rejected = []
    for raw in paths:
        if not raw:
            continue
        relative = Path(raw.decode())
        path = REPO/relative
        size = path.stat().st_size
        if relative.parts[:2] == ('docs', 'tasks') and path.suffix.lower() in extensions:
            rejected.append(f'generated report output: {relative}')
        if size > MAX_TRACKED_BYTES:
            rejected.append(f'over 1 MiB: {relative} ({size} bytes)')
        total += size
        count += 1
    if rejected:
        raise ValueError('Keep payloads in local artifacts, not Git:\n'+'\n'.join(rejected))
    print(f'{count} indexed files, {total / 2**20:.2f} MiB; no generated report payload or oversized file')


if __name__ == '__main__':
    main()
