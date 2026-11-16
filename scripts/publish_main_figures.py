"""Copy approved, byte-identical publication PDF/PNG pairs from an explicit manifest."""
import argparse
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def publish(manifest):
    if manifest['schema'] != 'pino_main_publication_v2':
        raise ValueError('Require the explicit ranked main-figure publication manifest')
    examples = [record['example'] for record in manifest['figures']]
    if len(examples) != 5 or len(set(examples)) != 5:
        raise ValueError('Require one selected figure per principal example')
    sources = []
    for record in manifest['figures']:
        if (len(record['samples']) != 3 or len(set(record['samples'])) != 3
                or record['cohort_count'] < 3 or len(record['phase_errors']) != 3
                or record['phase_errors'] != sorted(record['phase_errors'])):
            raise ValueError('Require best/middle/worst samples and increasing phase errors')
        for kind in ('pdf', 'png'):
            source = Path(record['source'][kind])
            target = REPO/record[kind]
            if record[kind] != f"assets/pino-main/{record['example']}.{kind}":
                raise ValueError('Unexpected publication destination')
            if source.stat().st_size != record['bytes'][kind]:
                raise ValueError(f'Approved source changed: {source}')
            if source.stat().st_size >= manifest['maximum_file_bytes']:
                raise ValueError(f'File exceeds publication budget: {source}')
            sources.append((source, target))
    if sum(p.stat().st_size for p, _ in sources) >= manifest['maximum_collection_bytes']:
        raise ValueError('Publication figure collection exceeds budget')
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError(f'Publication copy differs: {target}')
    print(f'Published {len(sources)} byte-identical PDF/PNG files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    publish(json.loads(args.manifest.read_text()))
