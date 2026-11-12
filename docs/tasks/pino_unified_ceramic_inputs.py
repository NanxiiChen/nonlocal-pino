"""Require identical known inputs and fixed execution shapes across samplers."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from examples.particle_corrosion2d.online import pad_initial_bank
from examples.particle_corrosion2d.train import load_bank


def audit(banks, presets, output):
    loaded = []
    for path, preset in zip(banks, presets, strict=True):
        c = importlib.import_module('examples.particle_corrosion2d.configs.'+preset).get_config()
        data, metadata = load_bank(path, c)
        padded = pad_initial_bank(data, c)
        loaded.append((data, metadata, padded))
    base, metadata, padded = loaded[0]
    for data, manifest, current in loaded[1:]:
        if manifest['geometries'] != metadata['geometries'] or manifest['runtime_seed'] != metadata['runtime_seed']:
            raise ValueError('Initial geometry prior/draws differ')
        for key in metadata['fixed_keys']:
            np.testing.assert_array_equal(data[key], base[key])
        if {k: a.shape for k, a in current.items()} != {k: a.shape for k, a in padded.items()}:
            raise ValueError('Padded execution shapes differ')
        if not np.all(data['pde_weights'] > 0):
            raise ValueError('Every sampled core needs a positive loss weight')
        np.testing.assert_allclose(data['pde_weights'].sum(axis=-1), 256., rtol=0, atol=1e-4)
    output.write_text(json.dumps(dict(status='passed', banks=[str(p) for p in banks], presets=presets,
        fixed_inputs_byte_equal=True, padded_shapes_equal=True, geometries=len(metadata['geometries']),
        preparation_seed=metadata['runtime_seed']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks', nargs='+', type=Path, required=True)
    parser.add_argument('--presets', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.banks, args.presets, args.output)
