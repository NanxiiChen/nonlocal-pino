"""CPU full-CH check for the isolated Taylor-AD candidate, with no live edits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from docs.tasks.spinodal_taylor_ad_probe import compare, taylor_spatial_jets
from examples.confined_ch2d.operator_local_ad import initial_jets, raw, condition_parts
from examples.confined_ch2d.operator_training import encode, phase


def candidate_parts(net, data, times, c):
    if c.loss.initial_residual_fraction != 0.:
        raise ValueError('This qualification covers the explicit positive-time comparison only')
    encoded = encode(net, data)
    prepared = net.prepare_encoded(encoded, data['points'])
    count = c.operator.local_ad.query_chunk

    def one(t):
        phi, dt = jax.jvp(lambda tt: phase(net, prepared, tt, data, c), (t,), (jnp.ones_like(t),))
        _, g0, l0, b0, _ = initial_jets(data['points'], c)

        @eqx.filter_checkpoint
        def block(points):
            return taylor_spatial_jets(lambda q: raw(net, encoded, q, t, c), points, True)

        values = jax.lax.map(block, data['points'].reshape((-1, count, 2)))
        gf, lf, bf = tuple(v.reshape((len(data['points']),)+v.shape[2:]) for v in values)
        s = t/c.physical.terminal_time
        gradient, laplacian, biharmonic = g0+s*gf, l0+s*lf, b0+s*bf
        p = c.physical
        lap_mu = p.beta*((3*phi**2-1)*laplacian+6*phi*jnp.sum(gradient**2, axis=-1))-p.epsilon**2*biharmonic
        filtered = data['metric']@(p.terminal_time*(dt-p.mobility*lap_mu))
        volume = data['volumes']
        pde = jnp.dot(volume, filtered**2)/jnp.sum(volume)
        x, n, w = data['condition'][:, :2], data['condition'][:, 2:4], data['condition_weights']
        value0, gradient0, _, _, lap_gradient0 = initial_jets(x, c)
        at = lambda q: raw(net, encoded, q, t, c)
        fb, fn = jax.jvp(at, (x,), (n,))
        lap_fn = lambda q: taylor_spatial_jets(at, q, False)[1]
        lap_normal = jax.jvp(lap_fn, (x,), (n,))[1]
        change = net.decode(prepared, jnp.reshape(s, (1,)))[0]
        mean = jnp.dot(volume, change)/jnp.sum(volume)
        phi_b = value0+s*(fb-mean)
        phi_n = jnp.sum(gradient0*n, axis=-1)+s*fn
        lap_phi_n = jnp.sum(lap_gradient0*n, axis=-1)+s*lap_normal
        mu_n = p.beta*(3*phi_b**2-1)*phi_n-p.epsilon**2*lap_phi_n
        bc_phi = jnp.dot(w, (phi_n/c.operator.local_ad.phase_flux_scale)**2)/jnp.sum(w)
        bc_mu = jnp.dot(w, (mu_n/c.operator.local_ad.chemical_flux_scale)**2)/jnp.sum(w)
        return jnp.stack((pde, bc_phi, bc_mu))

    return jax.vmap(one)(times[1:]).mean(axis=0)


def qualify(precision, architecture):
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Only CPU qualification is authorized for this isolated candidate')
    jax.config.update('jax_enable_x64', True)
    from examples.confined_ch2d.operator_training import build_network
    from examples.confined_ch2d.operator_precision import compute_arrays, compute_dtype
    from tests.test_confined_ch_operator_local_ad import config, data, Polynomial
    c = config(); c.train.compute_dtype=precision
    if architecture == 'small':
        c.arch.width=4; c.arch.heads=1; c.arch.condition_depth=1
        c.arch.decoder_depth=1; c.arch.spatial_features=2; c.arch.time_features=2
    elif architecture != 'production':
        raise ValueError(f'Unknown architecture {architecture}')
    d=compute_arrays(data(c), c)
    times=jnp.array([0., .3, .7], dtype=compute_dtype(c))
    polynomial=Polynomial(jnp.asarray(.4, dtype=compute_dtype(c)))
    atol, rtol = (2e-7, 2e-8) if precision == 'float64' else (3e-5, 3e-4)
    manufactured=compare(candidate_parts(polynomial,d,times,c),condition_parts(polynomial,d,times,c),
                         atol=atol,rtol=rtol)
    net=build_network(c,jax.random.key(11)); outputs={}; timings={}
    for name, parts_fn in (('nested_jvp',condition_parts),('taylor',candidate_parts)):
        def objective(model):
            parts=parts_fn(model,d,times,c)
            return parts[0]+c.operator.local_ad.boundary_weight*jnp.sum(parts[1:]),parts
        run=eqx.filter_jit(eqx.filter_value_and_grad(objective,has_aux=True))
        start=time.perf_counter(); result=run(net);jax.block_until_ready(result)
        first=time.perf_counter()-start
        start=time.perf_counter(); again=run(net);jax.block_until_ready(again)
        warm=time.perf_counter()-start
        outputs[name]=result;timings[name]=dict(first_seconds=first,warm_seconds=warm)
        print(json.dumps(dict(method=name,timing=timings[name],parts=np.asarray(result[0][1]).tolist())),flush=True)
    check=compare(outputs['taylor'],outputs['nested_jvp'],atol=atol,rtol=rtol)
    for result in outputs.values():
        for leaf in jax.tree.leaves(result):
            if eqx.is_inexact_array(leaf) and leaf.dtype != compute_dtype(c):
                raise AssertionError(f'Unexpected result precision: {leaf.dtype}')
    return dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),backend='cpu',
        precision=precision,architecture=c.arch.to_dict(),jax=jax.__version__,
        tolerance=dict(atol=atol,rtol=rtol),polynomial=manufactured,full_loss_parts_and_gradients=check,
        timings=timings,parts={k:np.asarray(v[0][1]).tolist() for k,v in outputs.items()},
        scope='Four domain points, physical wall queries, two positive times; full CH and both Neumann objectives, parameter-dependent hard mass and all parameter-gradient leaves; architecture and precision explicit',
        limitation='Not a full-query production GPU resource qualification or a training-convergence result',
        live_training_modified=False,production_sources_modified=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precision',choices=('float32','float64'),required=True)
    parser.add_argument('--architecture',choices=('small','production'),required=True)
    args=parser.parse_args(); report=qualify(args.precision,args.architecture)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
