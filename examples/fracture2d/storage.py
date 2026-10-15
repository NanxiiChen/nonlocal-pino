"""Atomic text and physical-field checkpoints."""
import json
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_fields(path, **fields):
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)


def load_fields(path, *, nodes, dofs):
    """Read physical arrays by path and validate their actual shape/dtype/range."""
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in ('q', 'd', 'H')}
    for name, shape in [('q', (dofs,)), ('d', (nodes,)), ('H', (nodes,))]:
        value = fields[name]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f'invalid checkpoint {name} shape/dtype/values: {path}')
    if fields['H'].min() < 0:
        raise ValueError(f'negative checkpoint history: {path}')
    return fields
