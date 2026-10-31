"""Create an immutable hardlinked dense-reference view for remapped bank IDs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np


def prepare(bank, dense, output):
    if not all(path.is_absolute() for path in (bank, dense, output)):
        raise ValueError('Explicit absolute bank/dense/output paths required')
    validation = json.loads((dense / 'validation.json').read_text())
    source = Path(validation['dataset'])
    before = json.loads((source / 'manifest.json').read_text())
    after = json.loads((bank / 'manifest.json').read_text())
    if not validation['status'] == before['status'] == after['status'] == 'passed':
        raise ValueError('Passed source dense and physical bank validation required')
    if validation['indices'] != before['test_indices'] or validation['snapshots'] != 61:
        raise ValueError('Expected the original passed61time validation cohort')
    source_by_draw = {row['draw_index']: row for row in before['conditions']}
    source_checks = {row['index']: row for row in validation['conditions']}
    if len(source_by_draw) != len(before['conditions']):
        raise ValueError('Source accepted-draw identities must be unique')
    mapping, conditions = [], []
    for index in after['test_indices']:
        target = after['conditions'][index]
        original = source_by_draw[target['draw_index']]
        old_index = original['index']
        if old_index not in validation['indices'] or not target['has_reference']:
            raise ValueError('Expanded validation draw is not in the original dense cohort')
        path = dense / f'native_{old_index:03d}.npz'
        with np.load(bank / target['file'], allow_pickle=False) as target_arrays, np.load(path, allow_pickle=False) as reference:
            for key in ('points', 'volumes', 'holes'):
                np.testing.assert_array_equal(target_arrays[key], reference[key])
            for key in ('phi', 'mu'):
                np.testing.assert_allclose(reference[key][::15], target_arrays[key], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(reference['time'][::15], target_arrays['time'])
        mapping.append(dict(draw_index=target['draw_index'], source_index=old_index,
                            index=index, source_file=str(path), file=f'native_{index:03d}.npz'))
        check = deepcopy(source_checks[old_index])
        check.update(index=index, source_index=old_index, draw_index=target['draw_index'])
        conditions.append(check)
    if len(mapping) != 8 or len({row['draw_index'] for row in mapping}) != 8:
        raise ValueError('Exactly eight distinct matching reserved geometries required')
    output.mkdir(parents=True, exist_ok=False)
    for row in mapping:
        os.link(row['source_file'], output / row['file'])
        left, right = Path(row['source_file']).stat(), (output / row['file']).stat()
        if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
            raise AssertionError('Dense view must be a verified hardlink')
    result = deepcopy(validation)
    result.update(dataset=str(bank), indices=after['test_indices'], conditions=conditions,
        source_validation=str(dense / 'validation.json'), source_dataset=str(source),
        mapping=mapping, new_pde_solves=0, new_fem_solves=0,
        purpose='Unmodified passed61time references, remapped by exact accepted geometry draw and physical arrays. '
                'Original refinement gates preserved; no new reference solve or training supervision.')
    (output / 'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', output=str(output), indices=after['test_indices'],
                         mapping=mapping, new_pde_solves=0, new_fem_solves=0), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'dense', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.bank, args.dense, args.output)
