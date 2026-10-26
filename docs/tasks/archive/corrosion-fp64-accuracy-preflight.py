"""GPU initialization/operator parity for the isolated FP64 accuracy control."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.corrosion2d.configs.train_base import get_config
from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.training_precision import cast_floating, audit_floating
from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.runtime import with_run_seed

OUT = Path(__file__).with_suffix('.json')
BASE = Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42')


def main():
    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU parity preflight requires GPU')
    c = with_run_seed(get_config(), 42)
    with jax.enable_x64(False):
        kn, kt = jax.random.split(jax.random.PRNGKey(42))
        net = r.build_model(c, kn)
        old = eqx.tree_deserialise_leaves(BASE/'model_initial.eqx', net)
        params = lambda n: jax.tree.leaves(eqx.filter(n, eqx.is_array))
        for a, b in zip(params(net), params(old), strict=True):
            np.testing.assert_array_equal(a, b)
        _, _, points = r.grid(c)
        op = build_D(jnp.asarray(points, dtype=jnp.float32), c.nl.delta,
                     method=c.nl.method, kernel=c.nl.kernel)
        ref = r.load_reference(c)
        for a, key in ((op._row_idx, 'graph_row'), (op._col_idx, 'graph_col'),
                       (-op._vals, 'graph_weight')):
            np.testing.assert_array_equal(a, ref[key])
        times = [r.sample_training_times(c, kt, i) for i in (0, 1, 99, 4999)]
        xic, ic = r.training_initial_data(c, points, points)
    with jax.enable_x64(True):
        promoted = cast_floating((net, times, xic, ic), np.dtype('float64'))
        audit = audit_floating(promoted, np.dtype('float64'))
        for a, b in zip(jax.tree.leaves(eqx.filter((net, times, xic, ic), eqx.is_array)),
                        jax.tree.leaves(eqx.filter(promoted, eqx.is_array)), strict=True):
            np.testing.assert_array_equal(np.asarray(a, dtype=np.float64), b)
    oldconfig = json.loads((BASE/'config.json').read_text())
    newconfig = c.to_dict()
    differences = []
    def compare(a, b, path=''):
        for key in sorted(set(a)|set(b)):
            p = path+'.'+key if path else key
            if key not in a or key not in b:
                differences.append({'field': p, 'old': a[key] if key in a else None,
                                    'new': b[key] if key in b else None})
            elif isinstance(a[key], dict):
                compare(a[key], b[key], p)
            elif a[key] != b[key]:
                differences.append({'field': p, 'old': a[key], 'new': b[key]})
    # Normalize tuples to JSON lists before comparing saved and runtime configs.
    compare(oldconfig, json.loads(json.dumps(newconfig)))
    allowed = {'log.run_name', 'experiment.ch_rms_after', 'train.compute_dtype',
               'experiment.lr_tail_start', 'experiment.lr_tail_end'}
    assert c.experiment.lr_tail_start == c.experiment.ch_rms_after == 0
    if any(v['field'] not in allowed for v in differences):
        raise ValueError(differences)
    row = np.asarray(op._row_idx)
    sums = np.bincount(row, weights=-np.asarray(op._vals, dtype=np.float64), minlength=len(points))
    defect = np.asarray(op._diag, dtype=np.float64)-sums
    record = {'status': 'passed', 'device': str(jax.devices()[0]),
              'initial_parameters_exactly_match_baseline': True,
              'graph_edges_weights_exactly_match_reference': True,
              'promoted_inputs_exact': True, 'dtype_audit': audit,
              'config_differences': differences,
              'constant_defect_max': float(np.max(np.abs(defect))),
              'constant_defect_relative_to_diagonal_max': float(np.max(np.abs(defect)/sums)),
              'scaled_AC_constant_defect_max': float(r.nondim(c)[3]*np.max(np.abs(defect))),
              'baseline': str(BASE)}
    OUT.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
