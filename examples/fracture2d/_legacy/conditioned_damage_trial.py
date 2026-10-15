"""Stage1 frozen-own-history neural training; direct oracle only afterward."""
import argparse
import json
from pathlib import Path
import runpy
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.at2_cg import damage_solver
from examples.fracture2d._legacy.conditioned_damage import make_features, make_problem, _build_network
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.damage_optimization import optimize_lbfgs
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec


def run(c):
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise RuntimeError('conditioned damage trial requires serialized x64 GPU execution')
    if (not c.conditioned.methods or set(c.conditioned.methods)-{'residual_mse', 'residual_l4'}
            or len(set(c.conditioned.methods)) != len(c.conditioned.methods)
            or not np.isfinite(c.conditioned.loss_floor) or c.conditioned.loss_floor <= 0):
        raise ValueError('invalid explicit conditioned objective controls')
    start = time.perf_counter()
    source = Path(c.optimization.source)
    original = {key: sha(source/key) for key in ('fields.npz', 'network.eqx', 'result.json')}
    if (original['fields.npz'] != c.optimization.source_field_sha256
            or original['network.eqx'] != c.optimization.source_network_sha256):
        raise ValueError('own-neural frozen history source SHA differs')
    source_identity = json.loads((source.parent/'identity.json').read_text())
    for key in ('physical', 'geometry', 'nl'):
        if source_identity['config'][key] != c.to_dict()[key]:
            raise ValueError(f'conditioned own-history source {key} differs')
    mechanics, operator, geometry = load_geometry(c)
    if geometry != source_identity['geometry']:
        raise ValueError('own-history geometry differs')
    root = Path(c.conditioned.output_dir)
    root.mkdir()
    atomic_json(root/'identity.json', dict(schema='conditioned-neural-damage-fixedH-v1',
        config=c.to_dict(), geometry=geometry, source=original,
        runner_sha256=sha(__file__), ansatz_sha256=sha(Path(__file__).with_name('conditioned_damage.py')),
        optimizer_sha256=sha(Path(__file__).with_name('damage_optimization.py')),
        direct_targets=False, damage_correction=False, physical_state_promoted=False))
    with np.load(source/'fields.npz', allow_pickle=False) as f:
        H_np = f['H_used'].copy()
    if H_np.shape != (mechanics.n_material,) or not np.isfinite(H_np).all() or np.min(H_np) < 0 or np.max(H_np) <= 0:
        raise ValueError('invalid own frozen H')
    load = jnp.asarray(json.loads((source/'result.json').read_text())['load'])
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    features, info = make_features(operator, mechanics.coordinates[:mechanics.n_material],
        gc0=gc0, length=c.physical.l0, tip=list(c.conditioned.tip),
        powers=list(c.conditioned.powers), roundoff_factor=c.validation.roundoff_factor)
    H = jnp.asarray(H_np)
    matvec, _ = make_padded_matvec(operator)
    net = _build_network(c, info['input_dimension'])
    image, metrics, objectives = make_problem(features, matvec, c)
    evaluate, predict = eqx.filter_jit(metrics), eqx.filter_jit(image)
    np.savez_compressed(root/'features.npz', features=np.asarray(features(H)), H_used=H_np)
    info['parameter_count'] = sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_inexact_array)))
    atomic_json(root/'feature_info.json', info)
    print(json.dumps(dict(event='conditioned_features', **info)), flush=True)
    reports = []
    # Objective stages explicitly inherit neural weights, not a direct field.
    for method in c.conditioned.methods:
        folder = root/method
        folder.mkdir()
        def progress(row):
            print(json.dumps(dict(method=method, **row)), flush=True)
        net, optimization = optimize_lbfgs(net, objectives[method], metrics, H, load,
            c.optimization.lbfgs.to_dict(), progress=progress)
        d = np.asarray(predict(net, H, load))
        np.savez_compressed(folder/'fields.npz', damage=d, H_used=H_np)
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        report = dict(method=method, optimization=optimization,
                      **{k: float(v) for k, v in evaluate(net, H, load).items()},
                      network_sha256=sha(folder/'network.eqx'), fields_sha256=sha(folder/'fields.npz'))
        atomic_json(folder/'result.json', report)
        reports.append(report)
        if optimization['status'] == 'residual_reached':
            break
    # Direct solution is evaluated only after the neural training is finished.
    solve = damage_solver(operator, mechanics.volumes, gc0, gc1,
        tol=c.validation.cg_rtol, atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter)
    oracle = np.asarray(solve(H, jnp.zeros_like(H)))
    np.savez_compressed(root/'oracle.npz', damage=oracle, H_used=H_np)
    D = operator.to_scipy()
    for report in reports:
        folder = root/report['method']
        with np.load(folder/'fields.npz', allow_pickle=False) as f:
            d = f['damage']
        parts = (gc0*d, gc1*(D@d), 2*H_np*(d-1))
        host = float(np.max(abs(sum(parts)))/sum(np.max(abs(p)) for p in parts))
        if abs(host-report['AT2_relative']) > 1e-10:
            raise RuntimeError('conditioned damage CPU/GPU residual differs')
        report.update(host_AT2_relative=host, oracle_damage_inf=float(np.max(abs(d-oracle))),
                      oracle_damage_relative_l2=float(np.linalg.norm(d-oracle)/np.linalg.norm(oracle)))
        atomic_json(folder/'result.json', report)
    if original != {key: sha(source/key) for key in original}:
        raise RuntimeError('own-neural source changed during conditioned training')
    result = dict(complete=True, ready=reports[-1]['optimization']['status']=='residual_reached',
        seconds=time.perf_counter()-start, reports=reports, source_unchanged=True,
        direct_targets=False, damage_correction=False, physical_state_promoted=False)
    atomic_json(root/'result.json', result)
    print(json.dumps(dict(event='fixed_H_complete', ready=result['ready'], seconds=result['seconds'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    c = runpy.run_path(parser.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)
