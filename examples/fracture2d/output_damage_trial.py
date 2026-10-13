"""Matched own-neural warm-baseline output-head experiment; no physical commit."""
import argparse
import json
from pathlib import Path
import runpy
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags, load_npz

from examples.fracture2d.bounded_damage_output import (
    MODES, head, make_image, validate_graph,
)
from examples.fracture2d.compact_damage_features import Readout, pack_basis, residual_column_scale, residual_power_loss
from examples.fracture2d.compact_damage_loading import JointNetwork, history_feature_input
from examples.fracture2d.conditioned_damage import _build_network, make_features
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.damage_optimization import optimize_lbfgs
from examples.fracture2d.local_damage_trial import load_geometry, sha
from examples.fracture2d.split_pinn_gradients import make_padded_matvec
from examples.fracture2d.train import _nonnegative_damage


def setup(c):
    source = Path(c.head_trial.source)
    metadata = json.loads((source.parent/'identity.json').read_text())
    report = json.loads((source/'result.json').read_text())
    if (sha(source/'fields.npz') != c.head_trial.source_field_sha256
            or sha(source/'network.eqx') != c.head_trial.source_network_sha256
            or report['state_sha256'] != c.head_trial.source_field_sha256
            or report['network_sha256'] != c.head_trial.source_network_sha256
            or report['status'] != 'inner_rejected' or report['physical_state_promoted']
            or metadata['damage_correction'] or metadata['direct_targets']
            or not metadata['own_zero_initial_state']):
        raise ValueError('head study requires the pinned rejected own-neural diagnostic')
    for key in ('physical', 'geometry', 'nl', 'arch', 'conditioned'):
        if metadata['config'][key] != c.to_dict()[key]:
            raise ValueError(f'head-study source {key} differs')
    mechanics, operator, geometry = load_geometry(c)
    if geometry != metadata['geometry']:
        raise ValueError('head-study geometry differs')
    basis_path = Path(metadata['config']['loading']['source']).parent/'basis.npz'
    scales_path = basis_path.with_name('basis_geometry.npz')
    for path in (basis_path, scales_path):
        if sha(path) != metadata['seed'][str(path)]:
            raise ValueError('head-study frozen basis differs')
    matrix = load_npz(basis_path)
    with np.load(scales_path, allow_pickle=False) as f:
        old_scaling = f['column_scale'].copy()
    old_phi, _ = pack_basis(matrix, old_scaling)
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    features, info = make_features(operator, mechanics.coordinates[:mechanics.n_material],
        gc0=gc0, length=c.physical.l0, tip=list(c.conditioned.tip),
        powers=list(c.conditioned.powers), roundoff_factor=c.validation.roundoff_factor)
    template = JointNetwork(_build_network(c, info['input_dimension']),
                            Readout(jnp.zeros(matrix.shape[1], jnp.float64)))
    frozen = eqx.tree_deserialise_leaves(source/'network.eqx', template)
    reference_load = metadata['reference_load']
    scaling_mode = metadata['config']['loading']['history_feature_scaling']

    def known_features(H, load):
        return features(history_feature_input(H, load, reference_load, scaling_mode))

    def frozen_image(H, load):
        raw = (c.physical.E/gc0*load**2*jax.vmap(frozen.trunk)(known_features(H, load))[:, 0]
               +(load/reference_load)**2*old_phi(frozen.readout))
        return _nonnegative_damage(raw, c.physical.damage_floor)

    frozen_image = jax.jit(frozen_image)
    with np.load(source/'fields.npz', allow_pickle=False) as f:
        H_np = f['H_used'].copy()
        expected = f['damage_image'].copy()
    H, load = jnp.asarray(H_np), jnp.asarray(report['load'])
    np.testing.assert_allclose(frozen_image(H, load), expected, rtol=1e-11, atol=1e-13)
    if np.min(expected) < 0 or np.max(expected) >= 1:
        raise ValueError('bounded warm baseline must lie in[0,1)')
    D = operator.to_scipy()
    diagonal, graph = validate_graph(D, H_np, gc0=gc0, gc1=gc1,
                                    roundoff_factor=c.validation.roundoff_factor)
    trunk = eqx.tree_at(lambda n: (n.output.weight, n.output.bias), frozen.trunk,
        (jnp.zeros_like(frozen.trunk.output.weight), jnp.zeros_like(frozen.trunk.output.bias)))
    net = JointNetwork(trunk, Readout(jnp.zeros(matrix.shape[1], jnp.float64)))
    files = [source/'fields.npz', source/'network.eqx', source/'result.json',
             source.parent/'identity.json', basis_path, scales_path]
    hashes = {str(path): sha(path) for path in files}
    return dict(mechanics=mechanics, operator=operator, geometry=geometry, D=D,
        diagonal=diagonal, graph=graph, H_np=H_np, H=H, load=load,
        reference_load=reference_load, features=known_features, frozen_image=frozen_image,
        net=net, matrix=matrix, source_hashes=hashes)


def make_problem(c, s, mode, column_scale, trunk_gain):
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    phi, _ = pack_basis(s['matrix'], column_scale)
    diagonal = jnp.asarray(s['diagonal'])
    matvec, _ = make_padded_matvec(s['operator'])
    matvec = jax.jit(matvec)

    def increment(net, H, load):
        return (trunk_gain*c.physical.E/gc0*load**2
                *jax.vmap(net.trunk)(s['features'](H, load))[:, 0]
                +(load/s['reference_load'])**2*phi(net.readout))

    image, anchors = make_image(s['frozen_image'], increment, diagonal, mode=mode,
        gc0=gc0, gc1=gc1, floor=c.physical.damage_floor,
        initialization_epsilon=c.head_trial.initialization_epsilon)

    def residual(net, H, load):
        d = image(net, H, load)
        return (gc0+2*H)*d+gc1*matvec(d)-2*H

    def metrics(net, H, load):
        d = image(net, H, load)
        parts = (gc0*d, gc1*matvec(d), 2*H*(d-1))
        return dict(AT2_relative=jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts),
                    damage_min=jnp.min(d), damage_max=jnp.max(d))

    def peak(power):
        def objective(net, H, load):
            return residual_power_loss(residual(net, H, load), H, power=power,
                                       floor=c.conditioned.loss_floor)
        return objective
    return image, metrics, dict(residual_l32=peak(32), residual_l16=peak(16)), anchors


def normalization(c, s, mode):
    _, _, _, anchors = make_problem(c, s, mode, np.ones(s['matrix'].shape[1]), 1.)
    z, initial = anchors(s['H'], s['load'])
    derivative = jax.vmap(jax.grad(lambda x: head(x, mode, floor=c.physical.damage_floor)))(z)
    if mode.startswith('lifted_'):
        C = c.physical.Gc*c.physical.l0*jnp.asarray(s['diagonal'])
        derivative = derivative*C/(c.physical.Gc/c.physical.l0+2*s['H']+C)
    tangent = np.asarray(derivative)*(float(s['load'])/s['reference_load'])**2
    if not np.isfinite(tangent).all() or np.min(abs(tangent)) <= 0:
        raise ValueError('initial output tangent is nonfinite/zero; no silent normalization floor')
    gain = float(1/np.sqrt(np.mean(np.asarray(derivative)**2)))
    columns = residual_column_scale(s['D'], diags(tangent)@s['matrix'], s['H_np'],
        gc0=c.physical.Gc/c.physical.l0, gc1=c.physical.Gc*c.physical.l0)
    return columns, gain, np.asarray(initial)


def run(c):
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise RuntimeError('output-head comparison requires serialized x64 GPU execution')
    modes = list(c.head_trial.modes)
    if (not modes or len(set(modes)) != len(modes) or set(modes)-set(MODES)
            or list(c.head_trial.methods) != ['residual_l32', 'residual_l16']):
        raise ValueError('explicit unique heads and L32/L16 stages required')
    root = Path(c.head_trial.output_dir)
    root.mkdir()
    started = time.perf_counter()
    s = setup(c)
    identity = dict(schema='own-neural-output-fixedH-v1', config=c.to_dict(),
        geometry=s['geometry'], source=s['source_hashes'], graph=s['graph'],
        trainable_parameters=sum(x.size for x in jax.tree.leaves(eqx.filter(s['net'], eqx.is_inexact_array))),
        frozen_own_neural_baseline=True, direct_targets=False, damage_correction=False,
        physical_state_promoted=False, initialization='diagonal/H decoder of affine bounded own-neural image',
        code={name: sha(Path(__file__).with_name(name)) for name in
              ('output_damage_trial.py', 'bounded_damage_output.py', 'damage_optimization.py',
               'conditioned_damage.py', 'compact_damage_features.py')})
    atomic_json(root/'identity.json', identity)
    reports, common_initial = [], None
    for mode in modes:
        folder = root/mode
        folder.mkdir()
        columns, gain, initial = normalization(c, s, mode)
        if common_initial is None:
            common_initial = initial
        np.testing.assert_allclose(initial, common_initial, rtol=1e-13, atol=1e-14)
        np.savez_compressed(folder/'parameterization.npz', column_scale=columns, trunk_gain=gain)
        image, metrics, objectives, _ = make_problem(c, s, mode, columns, gain)
        predict, measure = eqx.filter_jit(image), eqx.filter_jit(metrics)
        net = s['net']
        np.testing.assert_allclose(predict(net, s['H'], s['load']), common_initial, rtol=1e-10, atol=1e-14)
        stages = []
        for method in c.head_trial.methods:
            def progress(row):
                print(json.dumps(dict(mode=mode, method=method, **row)), flush=True)
            net, optimization = optimize_lbfgs(net, objectives[method], metrics,
                s['H'], s['load'], c.head_trial.lbfgs.to_dict(), progress=progress)
            stages.append(dict(method=method, **optimization))
            if optimization['status'] == 'residual_reached':
                break
        damage = np.asarray(predict(net, s['H'], s['load']))
        parts = (c.physical.Gc/c.physical.l0*damage,
                 c.physical.Gc*c.physical.l0*(s['D']@damage), 2*s['H_np']*(damage-1))
        host = float(np.max(abs(sum(parts)))/sum(np.max(abs(p)) for p in parts))
        values = {k: float(v) for k, v in measure(net, s['H'], s['load']).items()}
        if abs(host-values['AT2_relative']) > 1e-10:
            raise RuntimeError('output-head CPU/GPU residual differs')
        np.savez_compressed(folder/'fields.npz', damage=damage, H_used=s['H_np'], initial_damage=common_initial)
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        result = dict(mode=mode, ready=optimization['status']=='residual_reached',
            stages=stages, host_AT2_relative=host, **values, trunk_gain=gain,
            network_sha256=sha(folder/'network.eqx'), fields_sha256=sha(folder/'fields.npz'),
            parameterization_sha256=sha(folder/'parameterization.npz'))
        atomic_json(folder/'result.json', result)
        reports.append(result)
        atomic_json(root/'result.json', dict(complete=len(reports)==len(modes), reports=reports,
            seconds=time.perf_counter()-started, physical_state_promoted=False))
        print(json.dumps(dict(event='head_complete', mode=mode, ready=result['ready'], AT2_relative=host)), flush=True)
    if s['source_hashes'] != {path: sha(path) for path in s['source_hashes']}:
        raise RuntimeError('head-study source changed')
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    c = runpy.run_path(p.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)
