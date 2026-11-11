"""Copy only the five approved, byte-identical publication PDF/PNG pairs."""
import argparse
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def publish(manifest):
    sources = []
    for record in manifest['figures']:
        spec = json.loads((REPO/record['specification']).read_text())
        for kind in ('pdf', 'png'):
            source = Path(spec['output'][kind])
            target = REPO/record[kind]
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
