"""List or restore optional report artifacts without overwriting local changes."""
import argparse
import filecmp
import json
from pathlib import Path, PurePosixPath
import shutil

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true')
    action.add_argument('--restore', action='store_true')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--path', action='append', help='Exact repo-relative artifact path; repeatable')
    parser.add_argument('--manifest', type=Path, default=REPO/'artifacts/local_files.json')
    parser.add_argument('--destination-root', type=Path, default=REPO)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unsupported local artifact manifest schema')
    entries = {}
    for record in manifest['files']:
        path = PurePosixPath(record['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[:2] != ('docs', 'tasks'):
            raise ValueError(f'Invalid report artifact path: {path}')
        if str(path) in entries or not isinstance(record['bytes'], int) or record['bytes'] < 0:
            raise ValueError(f'Invalid/duplicate report artifact: {path}')
        entries[str(path)] = record
    selected = sorted(entries) if args.all else args.path
    missing = set(selected) - entries.keys()
    if missing:
        raise ValueError(f'Unknown artifact paths: {sorted(missing)}')
    restored = present = 0
    for name in selected:
        record = entries[name]
        if args.list:
            print(f"{record['bytes']:>10}  {name}")
            continue
        source = Path(manifest['archive_root'])/name
        if not source.is_file() or source.stat().st_size != record['bytes']:
            raise ValueError(f'Archive file missing or size mismatch: {source}')
        target = args.destination_root/name
        if target.exists():
            if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
                raise FileExistsError(f'Preserving different local artifact: {target}')
            present += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if not filecmp.cmp(source, target, shallow=False):
                raise IOError(f'Artifact copy differs: {target}')
            restored += 1
    if args.restore:
        print(f'{restored} restored, {present} already identical; no existing file overwritten')


if __name__ == '__main__':
    main()
