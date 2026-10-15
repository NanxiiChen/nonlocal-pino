"""Fixed-own-history no-reference head test and certified-parent parameter transfer."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.local_damage_trial import sha
from examples.fracture2d._legacy.output_damage_safe_continue import safe_setup, verify_parent
from examples.fracture2d._legacy.output_damage_trial import make_problem as reference_problem
from examples.fracture2d._legacy.plain_sigmoid_damage import (
    make_problem, recenter_bias, make_transfer_objective, latent_basis_scaling,
)


def run(c):
    spec = c.continuation.no_reference
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or spec.kind not in ('pending_diagnostic', 'accepted_parent')
            or c.continuation.initializer != 'cap_frozen_predictor'
            or spec.transfer_policy != 'own_neural_logit_fit'
            or spec.parameterization not in ('inherited_reference_scales', 'latent_basis_rms')
            or spec.lbfgs.to_dict() != c.continuation.lbfgs.to_dict()
            or list(spec.methods) != list(c.continuation.methods)):
        raise ValueError('explicit GPU x64 same-policy own-neural no-reference trial required')
    _, physical_parent, protected = verify_parent(c)
    s = safe_setup(c)
    protected.update(s['source_hashes'])
    source = Path(spec.source)
    if spec.kind == 'pending_diagnostic':
        manifest = json.loads((source/'pending.json').read_text())
        if (manifest['completed_sweep'] != spec.completed_sweep or not manifest['has_image']
                or manifest['parent_sha256'] != physical_parent['state_sha256']
                or manifest['fields_sha256'] != spec.field_sha256
                or manifest['network_sha256'] != spec.network_sha256):
            raise ValueError('no-reference diagnostic checkpoint differs')
        fields_path, network_path = source/manifest['fields_file'], source/manifest['network_file']
        protected[str(source/'pending.json')] = sha(source/'pending.json')
        for filename in ('identity.json', 'audit.json', 'initializer_diagnostic.json'):
            path = source.parent/filename
            protected[str(path)] = sha(path)
        if not json.loads((source.parent/'audit.json').read_text())['pass_replay']:
            raise ValueError('no-reference source lacks independent diagnostic replay')
        load_value = manifest['load']
        with np.load(fields_path, allow_pickle=False) as f:
            H_transfer, H_final, teacher = [f[k].copy() for k in ('H_used', 'H', 'damage_image')]
    else:
        if source != Path(c.continuation.parent):
            raise ValueError('no-reference parent-transfer source must be physical accepted parent')
        fields_path, network_path = source/'fields.npz', source/'network.eqx'
        load_value = physical_parent['load']
        with np.load(fields_path, allow_pickle=False) as f:
            H_transfer = f['H'].copy()
            H_final, teacher = H_transfer.copy(), f['raw_current_image'].copy()
            np.testing.assert_array_equal(f['H'], np.maximum(f['H_parent'], s['mechanics'].driver(f['q'])))
    if sha(fields_path) != spec.field_sha256 or sha(network_path) != spec.network_sha256:
        raise ValueError('no-reference trial field/network pins differ')
    for path in (fields_path, network_path):
        protected[str(path)] = sha(path)
    if (H_final.shape != (s['mechanics'].n_material,) or H_transfer.shape != H_final.shape
            or H_final.dtype != np.float64 or H_transfer.dtype != np.float64
            or not np.isfinite(H_final).all() or not np.isfinite(H_transfer).all()
            or min(H_final.min(), H_transfer.min()) < 0 or H_final.max() <= 0):
        raise ValueError('no-reference trial history must be finite nonnegative FP64 material field')
    root = Path(spec.output_dir)
    root.mkdir()
    fixed = Path(c.head_loading.fixed_H_root)/'sigmoid'
    with np.load(fixed/'parameterization.npz', allow_pickle=False) as f:
        columns, gain = f['column_scale'].copy(), float(f['trunk_gain'])
    protected[str(fixed/'parameterization.npz')] = sha(fixed/'parameterization.npz')
    net = eqx.tree_deserialise_leaves(network_path, s['net'])
    load = jnp.asarray(load_value)
    H0, H1 = jnp.asarray(H_transfer), jnp.asarray(H_final)
    old_image, old_metrics, _, _ = reference_problem(c, s, 'sigmoid', columns, gain)
    np.testing.assert_allclose(eqx.filter_jit(old_image)(net, H0, load), teacher, rtol=1e-10, atol=1e-12)
    teacher_metrics = {k: float(v) for k, v in eqx.filter_jit(old_metrics)(net, H0, load).items()}
    if teacher_metrics['AT2_relative'] > spec.lbfgs.residual_limit:
        raise ValueError('no-reference transfer teacher has not solved its own fixed-H equation')
    if spec.parameterization == 'latent_basis_rms':
        previous_image, _, _, previous_latent = make_problem(c, s, columns, gain)
        converted, new_columns = latent_basis_scaling(net, s['matrix'], columns)
        converted_image, _, _, converted_latent = make_problem(c, s, new_columns, gain)
        for history in (H0, H1):
            np.testing.assert_allclose(eqx.filter_jit(previous_latent)(net, history, load),
                eqx.filter_jit(converted_latent)(converted, history, load), rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(eqx.filter_jit(previous_image)(net, history, load),
                eqx.filter_jit(converted_image)(converted, history, load), rtol=1e-11, atol=1e-12)
        net, columns = converted, new_columns
    image, metrics, objectives, latent = make_problem(c, s, columns, gain)
    predict, measure, raw = eqx.filter_jit(image), eqx.filter_jit(metrics), eqx.filter_jit(latent)
    net, target, shift = recenter_bias(net, raw, H0, load, jnp.asarray(teacher),
        trunk_factor=gain*c.physical.E/(c.physical.Gc/c.physical.l0)*load_value**2)
    initial = np.asarray(predict(net, H0, load))
    code_names = ['plain_sigmoid_damage.py', 'plain_sigmoid_trial.py',
        'cached_damage_optimization.py', 'output_damage_trial.py', 'bounded_damage_output.py',
        'conditioned_damage.py', 'compact_damage_features.py', 'compact_damage_loading.py',
        'local_damage_trial.py', 'local_mechanics.py', 'split_pinn_gradients.py', 'train.py']
    code = {name: sha(Path(__file__).with_name(name)) for name in code_names}
    identity = dict(schema='plain-sigmoid-fixed-ownH-v1', config=c.to_dict(), source=protected,
        code=code, geometry=s['geometry'], physical_parent=physical_parent,
        own_history=True, direct_targets=False, damage_correction=False,
        frozen_reference_in_output=False, physical_state_promoted=False,
        trainable_parameters=sum(a.size for a in jax.tree.leaves(eqx.filter(net, eqx.is_inexact_array))),
        teacher_metrics=teacher_metrics, bias_shift=shift, load=load_value)
    atomic_json(root/'identity.json', identity)
    (root/'code_snapshot').mkdir()
    for name in code_names:
        shutil.copyfile(Path(__file__).with_name(name), root/'code_snapshot'/name)
    started, stages = time.perf_counter(), []
    transfer = make_transfer_objective(raw, target)
    fit, _, _ = make_cached_lbfgs(net, transfer, metrics)
    def progress(row):
        print(json.dumps(dict(event='transfer', kind=spec.kind, **row)), flush=True)
    net, report = fit(net, H0, load, spec.lbfgs.to_dict(), progress=progress)
    stages.append(dict(method='own_neural_logit_transfer', **report))
    transferred = np.asarray(predict(net, H0, load))
    eqx.tree_serialise_leaves(root/'transfer_network.eqx', net)
    atomic_json(root/'transfer.json', dict(stages=stages, metrics={k:float(v) for k,v in measure(net,H0,load).items()}))
    for method in spec.methods:
        fit, _, _ = make_cached_lbfgs(net, objectives[method], metrics)
        def progress(row):
            print(json.dumps(dict(event='physical', kind=spec.kind, method=method, **row)), flush=True)
        net, report = fit(net, H1, load, spec.lbfgs.to_dict(), progress=progress)
        stages.append(dict(method=method, **report))
        if report['status'] == 'residual_reached':
            break
    damage = np.asarray(predict(net, H1, load))
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    parts = (gc0*damage, gc1*(s['D']@damage), 2*H_final*(damage-1))
    host = float(np.max(abs(sum(parts)))/sum(np.max(abs(p)) for p in parts))
    measured = {k:float(v) for k,v in measure(net,H1,load).items()}
    np.testing.assert_allclose(host, measured['AT2_relative'], rtol=1e-10, atol=1e-12)
    np.savez_compressed(root/'fields.npz', damage=damage, H_used=H_final,
        H_transfer=H_transfer, teacher_damage=teacher, initial_damage=initial, transferred_damage=transferred)
    np.savez_compressed(root/'parameterization.npz', column_scale=columns, trunk_gain=gain)
    eqx.tree_serialise_leaves(root/'network.eqx', net)
    result = dict(ready=report['status']=='residual_reached', kind=spec.kind, stages=stages,
        **measured, host_AT2_relative=host, seconds=time.perf_counter()-started,
        physical_state_promoted=False, frozen_reference_in_output=False,
        network_sha256=sha(root/'network.eqx'), fields_sha256=sha(root/'fields.npz'),
        transfer_network_sha256=sha(root/'transfer_network.eqx'),
        parameterization_sha256=sha(root/'parameterization.npz'))
    atomic_json(root/'result.json', result)
    if protected != {path:sha(path) for path in protected} or code != {
            name:sha(Path(__file__).with_name(name)) for name in code_names}:
        raise RuntimeError('no-reference trial protected source/code changed')
    print(json.dumps(dict(event='complete', ready=result['ready'], AT2_relative=host)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    c = runpy.run_path(parser.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)
