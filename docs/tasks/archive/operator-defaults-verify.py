"""Verify published defaults against saved runs and preserved historical controls."""
import importlib
import json
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path/'.git').exists())
HERE = Path(__file__).resolve().parent
before = json.loads((HERE/'operator-defaults-before.json').read_text())
result = {}
for name in ('train_operator_online', 'train_operator_single', 'train_operator_causal', 'train_operator_local'):
    current = json.loads(importlib.import_module(f'examples.electropolish2d.configs.{name}').get_config().to_json())
    assert current == before[name], name
    result[name] = 'exact historical config preserved'
legacy = json.loads(importlib.import_module('examples.electropolish2d.configs.train_operator_legacy').get_config().to_json())
assert legacy == before['train_operator']
result['train_operator_legacy'] = 'exact original finite-input config preserved'

def flatten(value, prefix=''):
    items = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            items.update(flatten(item, name))
        else:
            items[name] = item
    return items

runs = {
    'electropolish2d':'20261028-electro-operator-online-global-adam-01',
    'icemelting3d':'20261029-melting-operator-online-b16-01',
}
for case, run in runs.items():
    path = Path('/root/autodl-tmp/tf-logs')/case/run/'config.json'
    current = flatten(json.loads(importlib.import_module(f'examples.{case}.configs.train_operator').get_config().to_json()))
    saved = flatten(json.loads(path.read_text()))
    differences = [[key, current[key] if key in current else '<absent>', saved[key] if key in saved else '<absent>']
                   for key in sorted(current.keys() | saved.keys()) if current.get(key, '<absent>') != saved.get(key, '<absent>')]
    assert differences == [['log.run_name', '', run], ['train.seed', '<absent>', 42]], differences
    result[case] = {'saved_config':str(path),'differences':differences,'strategy_exact_match':True}
(HERE/'operator-defaults-audit.json').write_text(json.dumps(result, indent=2)+'\n')
print('Two recommended strategies match saved runs; five historical presets preserved.')
