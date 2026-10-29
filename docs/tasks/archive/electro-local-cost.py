"""Same-core operator PDE value/parameter-gradient cost, not full training."""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from nonlocal_pinn.training.runtime import configure_gpu_policy
from nonlocal_pinn.operators import build_D
from examples.electropolish2d.operator_training import _build_network, physical_fields
from examples.corrosion2d.residual_experiment import grid, boundary_values, nondim
from examples.corrosion2d.residuals import graph_action


RUN = Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-global-adam-01')
OUT = Path(__file__).resolve().parent
c = ConfigDict(json.loads((RUN / 'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
net = eqx.tree_deserialise_leaves(RUN / 'model_final.eqx',
                                 _build_network(c.arch, c.operator.image_shape, jax.random.key(0)))
_, _, points = grid(c)
op = build_D(jnp.asarray(points, dtype=jnp.float32), c.nl.delta,
             method='sparse', kernel=c.nl.kernel)
ids, sub, core, _ = op.build_subgraph_pool(c.sample.K_core, 1)[0]
z, o, _ = boundary_values(c, points)
mask = core & jnp.asarray(~(z | o))[ids]
core_ids = np.asarray(ids)[np.asarray(mask)]
assert len(core_ids) > 0
xg = jnp.asarray(points[np.asarray(ids)], dtype=jnp.float32)
xl = jnp.asarray(points[core_ids], dtype=jnp.float32)
with np.load(Path(c.operator.dataset_path) / 'conditions.npz') as data:
    images = jnp.asarray(data['images'][:8])
times = jnp.array([.05, .25, .5, 1.], dtype=jnp.float32)
nch, n1, n2, n3, dc, cle = nondim(c)


def fields(n, encoded, x, ts):
    h = n.prepare_encoded(encoded, x, jnp.zeros((len(x), 2), dtype=x.dtype))
    return physical_fields(n.decode(h, ts), c)


def spatial_fields(n, encoded, x, ts):
    f = fields(n, encoded, x, ts)
    phi = f[..., 0]
    q = f[..., 1] - dc*(3*phi**2-2*phi**3) - cle
    return jnp.stack([phi, q], -1)


def laplacian(n, encoded, x, ts):
    def fn(v):
        return spatial_fields(n, encoded, v, ts)
    result = jnp.zeros((len(ts), len(x), 2), dtype=x.dtype)
    for axis in range(2):
        tangent = jnp.broadcast_to(jax.nn.one_hot(axis, 2, dtype=x.dtype), x.shape)
        first = lambda v: jax.jvp(fn, (v,), (tangent,))[1]
        result = result + jax.jvp(first, (x,), (tangent,))[1]
    return result


def one_loss(n, image, x, local):
    encoded = n.encode(image)  # outside coordinate AD, inside parameter AD
    f, ft = jax.jvp(lambda ts: fields(n, encoded, x, ts), (times,), (jnp.ones_like(times),))
    phi = f[..., 0]
    q = f[..., 1] - dc*(3*phi**2-2*phi**3) - cle
    if local:
        lap = laplacian(n, encoded, x, times)
        dphi, dq = -lap[..., 0], -lap[..., 1]  # graph D approximates minus Laplacian
        weights = jnp.ones((1, len(x)), dtype=phi.dtype)
    else:
        dphi = graph_action(phi.T, sub).T
        dq = graph_action(q.T, sub).T
        weights = mask[None, :].astype(phi.dtype)
    ac = ft[..., 0] + n3*dphi + n2*(4*phi**3-6*phi**2+2*phi) - n1*q*dc*(6*phi-6*phi**2)
    ch = ft[..., 1] + nch*dq
    denominator = len(times)*weights.sum()
    return (c.loss.ac_scale*(jnp.abs(ac)*weights).sum()
            + c.loss.ch_scale*(jnp.abs(ch)*weights).sum())/denominator


# Query rows are independent in cross-attention. Verify directional trace
# against full single-query AD; promoted precision isolates FP32 reduction noise.
with jax.enable_x64(True):
    promoted = jax.tree.map(lambda a: a.astype(jnp.float64) if eqx.is_inexact_array(a) else a, net)
    im = images[0].astype(jnp.float64)
    encoded = promoted.encode(im)
    xt = xl[:2].astype(jnp.float64)
    ts = times[:1].astype(jnp.float64)
    directional = laplacian(promoted, encoded, xt, ts)[0]
    expected = jax.vmap(lambda x: jnp.trace(jax.hessian(
        lambda p: spatial_fields(promoted, encoded, p[None], ts)[0, 0])(x), axis1=-2, axis2=-1))(xt)
    np.testing.assert_allclose(directional, expected, rtol=1e-8, atol=1e-8)
    validation_gap = float(jnp.max(jnp.abs(directional-expected)))

records = {}
for mode, x in (('nonlocal', xg), ('local_ad', xl)):
    local = mode == 'local_ad'
    value_grad = eqx.filter_jit(eqx.filter_value_and_grad(
        lambda n, ims, query: jax.vmap(lambda image: one_loss(n, image, query, local))(ims).mean()))
    start = time.perf_counter()
    result = value_grad(net, images, x)
    jax.block_until_ready(result)
    compile_seconds = time.perf_counter()-start
    assert np.isfinite(float(result[0]))
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(result[1]) if eqx.is_array(a))
    for _ in range(10):
        jax.block_until_ready(value_grad(net, images, x))
    elapsed = []
    for _ in range(50):
        start = time.perf_counter()
        jax.block_until_ready(value_grad(net, images, x))
        elapsed.append(time.perf_counter()-start)
    records[mode] = dict(mean_ms=1000*float(np.mean(elapsed)), median_ms=1000*float(np.median(elapsed)),
                          compile_and_first_seconds=compile_seconds, pde_loss=float(result[0]),
                          evaluated_spatial_queries=len(x), core_points=len(core_ids))
    print(mode, json.dumps(records[mode]), flush=True)
output = dict(records=records, directional_hessian_max_gap=validation_gap,
              batch=8, times=times.tolist(), checkpoint=str(RUN / 'model_final.eqx'),
              ratio_local_over_nonlocal=records['local_ad']['median_ms']/records['nonlocal']['median_ms'],
              device=str(jax.devices()[0]),
              scope='PDE-only scalar value + parameter gradient, 10warm/50timed; same free core. '
                    'Local omits halo, shares encoded images across coordinates and time. '
                    'No IC/BC, optimizer, gradient balancing, sampling, full training or accuracy comparison.')
(OUT / 'electro-local-cost.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
print('RESULT', json.dumps(output), flush=True)
