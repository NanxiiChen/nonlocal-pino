"""Late-time boundary-only steady Stokes sensitivity; not a transient NSCH ablation."""
import argparse
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.sparse.linalg import splu
from ml_collections import ConfigDict
from docs.tasks.channel_nonlocal_reference import Collocation
os.sched_setaffinity(0, range(36, 40))
import equinox as eqx
import jax
import jax.numpy as jnp
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network, encode, spatial_field_jets


def main(run, numerical, output):
    run, numerical, output = Path(run), Path(numerical), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU-only diagnostic required')
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    scheme = Collocation(c, (192, 80), 1, 'interior')
    with np.load(run/'evaluation_inputs.npz', allow_pickle=False) as z:
        data = {k: jnp.asarray(z[k]) for k in z.files}
    key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    net = eqx.tree_deserialise_leaves(run/'model_final.eqx', build_network(c, key))
    e = eqx.filter_jit(encode)(net, data)
    x = jnp.asarray(scheme.points[scheme.outlet], dtype=c.train.compute_dtype)
    value, dx, dy = [np.asarray(a, dtype=np.float64)[0] for a in eqx.filter_jit(
        lambda n, e, x: spatial_field_jets(n, e, x, jnp.asarray([6.], dtype=x.dtype), data['amplitudes'], c))(net, e, x)]
    g = np.column_stack((-value[:, 4] + 2*c.physical.viscosity*dx[:, 2],
                          c.physical.viscosity*(dy[:, 2] + dx[:, 3])))
    initial = scheme.initial(); n = scheme.n
    _, jac = scheme.residual(initial, initial, 6., 0., 1., True)
    block = jac[2*n:, 2*n:].tocsc()
    factor = splu(block)
    right = np.zeros((block.shape[0], 2))
    ids = np.flatnonzero(scheme.outlet)
    right[ids, 0] = -g[:, 0]/c.loss.momentum_scale
    right[n + ids, 0] = -g[:, 1]/c.loss.momentum_scale
    right[ids, 1] = -1/c.loss.momentum_scale
    delta = factor.solve(right)
    residual = float(np.max(np.abs(block@delta - right)))
    calibration_velocity = float(np.max(np.abs(delta[:2*n, 1])))
    calibration_pressure = float(np.max(np.abs(delta[2*n:, 1] - 1)))
    if max(residual, calibration_velocity, calibration_pressure) > 1e-8:
        raise ValueError(f'Stokes pressure-shift calibration failed: {residual}, {calibration_velocity}, {calibration_pressure}')
    velocity = np.column_stack((delta[:n, 0], delta[n:2*n, 0]))
    dx, dy = scheme.Dx@velocity, scheme.Dy@velocity
    strain = np.column_stack((dx[:, 0], dy[:, 1], (dy[:, 0] + dx[:, 1])/np.sqrt(2)))
    with np.load(run/'diagnostic_fields.npz', allow_pickle=False) as z:
        learned = z['fields'][-1]
    with np.load(numerical/'solution.npz', allow_pickle=False) as z:
        reference = z['nodal_fields'][-1]
    error = learned[:, 2:4] - reference[:, 2:4]
    ex, ey = scheme.Dx@error, scheme.Dy@error
    strain_error = np.column_stack((ex[:, 0], ey[:, 1], (ey[:, 0] + ex[:, 1])/np.sqrt(2)))
    rows = {}
    for name, mask in [('droplet', reference[:, 0] < 0), ('interface', np.abs(reference[:, 0]) < .9),
                       ('near_outlet', scheme.points[:, 0] > 3.8)]:
        w = scheme.measure*mask
        norm = lambda v: float(np.sqrt(np.sum(v*v*w[:, None])/w.sum()))
        rows[name] = dict(boundary_response_velocity_rms=norm(velocity),
                          boundary_response_over_learned_velocity_error=norm(velocity)/norm(error),
                          boundary_response_over_learned_strain_error=norm(strain)/norm(strain_error))
    report = dict(status='steady_boundary_response_and_pressure_shift_calibration_passed',
        sparse_residual_max=residual, constant_traction_calibration_velocity=calibration_velocity,
        constant_traction_calibration_pressure_gap=calibration_pressure, regions=rows,
        scope='Fine192x80 homogeneous steady Stokes response to negative actual learned6s AD outlet traction; zero correction on wall/inlet. Calibrated by unit pressure shift. This is a static linear sensitivity, not proof excluding transient/nonlinear coupled boundary effects.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--numerical', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.run, a.numerical, a.output)
