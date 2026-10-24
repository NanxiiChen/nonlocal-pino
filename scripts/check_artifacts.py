"""Check registered external roots and inventory; never create or rewrite them."""
import json
from pathlib import Path


def main():
    index = json.loads((Path(__file__).resolve().parents[1]/'artifacts/index.yaml').read_text())
    if index['schema'] != 1:
        raise ValueError('unsupported artifact schema')
    count = 0
    for case in index['cases']:
        for key in ('data_root', 'run_root'):
            path = Path(case[key])
            if not path.is_dir():
                raise FileNotFoundError(f"{case['case']} missing {key}: {path}")
            count += 1
        for key in ('data_children', 'run_children'):
            for name in case[key]:
                if not Path(name).is_dir():
                    raise FileNotFoundError(f'missing registered artifact: {name}')
                count += 1
    print(f"{index['project']}: {len(index['cases'])} cases, {count} registered directories OK")


if __name__ == '__main__':
    main()
