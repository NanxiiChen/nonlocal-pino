"""Paired fixed-own-H damage optimizer experiment; no state promotion."""
import argparse
import json
from pathlib import Path
import runpy
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.fracture2d.at2_cg import damage_solver
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.damage_optimization import (
    make_objectives, optimize_lbfgs, validate_energy_operator,
)
from examples.fracture2d.local_damage_trial import load_geometry, sha
from examples.fracture2d.split_pinn_gradients import make_padded_matvec
from examples.fracture2d.train import _build_network, _nonnegative_damage


def run(c):
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise RuntimeError('optimizer experiment requires serialized FP64 GPU execution')
    controls = c.optimization
    methods = list(controls.methods)
    if (not methods or len(set(methods)) != len(methods)
            or set(methods)-{'adam_residual', 'lbfgs_residual', 'lbfgs_energy'}
            or type(controls.adam_updates) is not int or controls.adam_updates < 1
            or type(controls.adam_check_every) is not int or controls.adam_check_every < 1):
        raise ValueError('invalid explicit optimizer comparison controls')
    source, root = Path(controls.source), Path(controls.output)
    before = {name: sha(source/name) for name in ('fields.npz', 'network.eqx', 'result.json')}
    if (before['fields.npz'] != controls.source_field_sha256
            or before['network.eqx'] != controls.source_network_sha256):
        raise ValueError('frozen neural source checksum differs')
    root.mkdir()
    source_identity = json.loads((source.parent/'identity.json').read_text())
    for group in ('arch', 'train', 'physical', 'geometry', 'nl'):
        if source_identity['config'][group] != c.to_dict()[group]:
            raise ValueError(f'frozen neural source {group} config differs')
    mechanics, operator, geometry = load_geometry(c)
    if geometry != source_identity['geometry']:
        raise ValueError('frozen neural source geometry differs')
    atomic_json(root/'identity.json', dict(config=c.to_dict(), source=before,
        geometry=geometry, runner_sha256=sha(__file__),
        optimizer_sha256=sha(Path(__file__).with_name('damage_optimization.py')),
        diagnostic_only=True, states_promoted=False))
    validation = validate_energy_operator(operator, mechanics.volumes,
        roundoff_factor=c.validation.roundoff_factor)
    with np.load(source/'fields.npz', allow_pickle=False) as f:
        H_np, saved_image = f['H_used'].copy(), f['damage_image'].copy()
    if (H_np.shape != (mechanics.n_material,) or H_np.dtype != np.float64
            or not np.all(np.isfinite(H_np)) or np.min(H_np) < 0 or np.max(H_np) <= 0):
        raise ValueError('invalid own-history frozen source')
    source_report = json.loads((source/'result.json').read_text())
    H, load = jnp.asarray(H_np), jnp.asarray(source_report['load'])
    template = jax.tree.map(lambda x: x.astype(jnp.float64),
        _build_network(c, jax.random.PRNGKey(c.train.seed+1), out_dim=1))
    optimizer = optax.adam(c.train.learning_rate)
    initial_state = optimizer.init(eqx.filter(template, eqx.is_inexact_array))
    initial_net, initial_state = eqx.tree_deserialise_leaves(source/'network.eqx', (template, initial_state))
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    points = jnp.asarray(mechanics.coordinates[:mechanics.n_material])
    matvec, _ = make_padded_matvec(operator)
    volumes = jnp.asarray(mechanics.volumes)

    def image(net, prescribed):
        raw = c.physical.E/gc0*prescribed**2*jax.vmap(net)(points)[:, 0]
        return _nonnegative_damage(raw, c.physical.damage_floor)

    residual_objective, energy_objective = make_objectives(image, matvec, volumes, gc0, gc1)

    def metrics(net, history, prescribed):
        d = image(net, prescribed)
        parts = (gc0*d, gc1*matvec(d), 2*history*(d-1))
        scale = sum(jnp.max(abs(p)) for p in parts)
        return dict(AT2_relative=jnp.max(abs(sum(parts)))/scale,
                    damage_min=jnp.min(d), damage_max=jnp.max(d))

    evaluate, predict = eqx.filter_jit(metrics), eqx.filter_jit(image)
    reproduced = np.asarray(predict(initial_net, load))
    np.testing.assert_allclose(reproduced, saved_image, rtol=1e-12, atol=1e-13)
    # Independent CPU derivative check on the actual mesh, not the AD objective.
    D = operator.to_scipy()
    rho = (gc0+2*H_np)*reproduced+gc1*(D@reproduced)-2*H_np
    direction = rho/np.max(abs(rho))
    def host_energy(d):
        return .5*np.sum(mechanics.volumes*d*((gc0+2*H_np)*d+gc1*(D@d)))-2*np.sum(mechanics.volumes*H_np*d)
    step = 1e-6
    finite_difference = (host_energy(reproduced+step*direction)-host_energy(reproduced-step*direction))/(2*step)
    analytic = float(np.sum(mechanics.volumes*rho*direction))
    relative = abs(finite_difference-analytic)/abs(analytic)
    if relative > 1e-6:
        raise RuntimeError(f'fixed-H energy derivative fails CPU finite difference: {relative}')
    validation.update(energy_derivative_relative=float(relative), initial= {
        k: float(v) for k, v in evaluate(initial_net, H, load).items()})
    atomic_json(root/'validation.json', validation)
    print(json.dumps(dict(event='validated', **validation)), flush=True)

    @eqx.filter_jit
    def adam_step(net, state):
        value, gradient = eqx.filter_value_and_grad(residual_objective)(net, H, load)
        updates, state = optimizer.update(gradient, state, eqx.filter(net, eqx.is_inexact_array))
        return eqx.apply_updates(net, updates), state, value

    reports = []
    for method in methods:
        net, state = initial_net, initial_state
        folder = root/method
        folder.mkdir()
        def progress(row):
            print(json.dumps(dict(event='optimize', method=method, **row)), flush=True)
        if method == 'adam_residual':
            rows, start = [], time.perf_counter()
            status = 'iteration_budget_exhausted'
            for iteration in range(1, controls.adam_updates+1):
                net, state, loss = adam_step(net, state)
                if not np.isfinite(float(loss)):
                    raise FloatingPointError('Adam damage objective is nonfinite')
                if iteration % controls.adam_check_every == 0 or iteration == controls.adam_updates:
                    values = {k: float(v) for k, v in evaluate(net, H, load).items()}
                    ready = (values['AT2_relative'] <= controls.lbfgs.residual_limit
                        and values['damage_min'] >= -controls.lbfgs.range_tolerance
                        and values['damage_max'] <= 1+controls.lbfgs.range_tolerance)
                    row = dict(iteration=iteration, seconds=time.perf_counter()-start, ready=ready, **values)
                    rows.append(row)
                    progress(row)
                    if ready:
                        status = 'residual_reached'
                        break
            info = dict(status=status, iterations=iteration, evaluations=iteration,
                        seconds=time.perf_counter()-start, rows=rows)
        else:
            objective = residual_objective if method == 'lbfgs_residual' else energy_objective
            net, info = optimize_lbfgs(net, objective, metrics, H, load,
                controls.lbfgs.to_dict(), progress=progress)
        damage = np.asarray(predict(net, load))
        np.savez_compressed(folder/'fields.npz', damage=damage, H_used=H_np)
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        report = dict(method=method, **info, **{k: float(v) for k, v in evaluate(net, H, load).items()},
            diagnostic_only=True, states_promoted=False, source=before)
        atomic_json(folder/'result.json', report)
        reports.append(report)
    # Oracle is evaluated after training; its values never enter an optimizer.
    solve = damage_solver(operator, mechanics.volumes, gc0, gc1,
        tol=c.validation.cg_rtol, atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter)
    oracle = np.asarray(solve(H, jnp.zeros_like(H)))
    np.savez_compressed(root/'oracle.npz', damage=oracle, H_used=H_np)
    for report in reports:
        folder = root/report['method']
        with np.load(folder/'fields.npz', allow_pickle=False) as f:
            d = f['damage']
        host_parts = (gc0*d, gc1*(D@d), 2*H_np*(d-1))
        host_residual = float(np.max(abs(sum(host_parts)))/sum(np.max(abs(x)) for x in host_parts))
        if abs(host_residual-report['AT2_relative']) > 1e-10:
            raise RuntimeError('optimizer candidate CPU/GPU residual reports differ')
        report.update(damage_error_inf=float(np.max(abs(d-oracle))),
            damage_error_relative_l2=float(np.linalg.norm(d-oracle)/np.linalg.norm(oracle)),
            host_AT2_relative=host_residual)
        atomic_json(folder/'result.json', report)
    after = {name: sha(source/name) for name in before}
    if after != before:
        raise RuntimeError('frozen neural source was modified')
    atomic_json(root/'result.json', dict(complete=True, source_unchanged=True,
        diagnostic_only=True, states_promoted=False, validation=validation, cases=reports))
    print(json.dumps(dict(event='complete', results=[{k: r[k] for k in
        ('method', 'status', 'AT2_relative', 'damage_error_inf', 'iterations', 'seconds')} for r in reports])), flush=True)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    config = runpy.run_path(arguments.config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(config)
