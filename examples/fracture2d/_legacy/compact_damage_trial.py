"""Fixed-own-H compact neural readout training, with direct oracle afterward."""
import argparse
import json
from pathlib import Path
import runpy
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz, save_npz

from examples.fracture2d._legacy.conditioned_damage import make_features, _build_network
from examples.fracture2d._legacy.compact_damage_features import Readout, build_basis, pack_basis, residual_column_scale, residual_power_loss
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.damage_optimization import optimize_lbfgs
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec
from examples.fracture2d._legacy.train import _nonnegative_damage
from examples.fracture2d._legacy.at2_cg import damage_solver


def run(c):
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise RuntimeError('compact neural damage requires serialized x64 GPU execution')
    begin = time.perf_counter()
    mechanics, operator, geometry = load_geometry(c)
    source = Path(c.compact.source)
    metadata = json.loads((source.parent/'identity.json').read_text())
    report = json.loads((source/'result.json').read_text())
    if metadata['geometry'] != geometry or report['network_sha256'] != sha(source/'network.eqx'):
        raise ValueError('compact neural source geometry/weights differ')
    for key in ('physical', 'geometry', 'nl', 'arch', 'conditioned'):
        if metadata['config'][key] != c.to_dict()[key]:
            raise ValueError(f'compact neural source {key} differs')
    before = {key: sha(source/key) for key in ('network.eqx', 'fields.npz', 'result.json')}
    root = Path(c.compact.output_dir)
    root.mkdir()
    with np.load(source/'fields.npz', allow_pickle=False) as f:
        H_np, previous_image = f['H_used'].copy(), f['damage'].copy()
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    load = jnp.asarray(json.loads((Path(c.optimization.source)/'result.json').read_text())['load'])
    H = jnp.asarray(H_np)
    features, info = make_features(operator, mechanics.coordinates[:mechanics.n_material],
        gc0=gc0, length=c.physical.l0, tip=list(c.conditioned.tip), powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    template = _build_network(c, info['input_dimension'])
    trunk = eqx.tree_deserialise_leaves(source/'network.eqx', template)
    base_raw = c.physical.E/gc0*load**2*eqx.filter_jit(jax.vmap(trunk))(features(H))[:, 0]
    np.testing.assert_allclose(_nonnegative_damage(base_raw, c.physical.damage_floor), previous_image,
                               rtol=1e-12, atol=1e-13)
    if c.compact.baseline_mode == 'positive_conditioned':
        # Reparameterize the frozen baseline before adding trainable readouts.
        # Former negative raw outputs had almost zero link derivative; their
        # positive images start close to zero with a usable link derivative.
        base_raw = _nonnegative_damage(base_raw, c.physical.damage_floor)
    elif c.compact.baseline_mode != 'raw_conditioned':
        raise ValueError('compact baseline_mode must be explicit and recognized')
    D = operator.to_scipy()
    residual_image = previous_image
    selection_before = None
    if c.compact.residual_source is not None:
        selection = Path(c.compact.residual_source)
        selection_identity = json.loads((selection.parent/'identity.json').read_text())
        if (selection_identity['geometry'] != geometry
                or selection_identity['direct_targets'] or selection_identity['damage_correction']):
            raise ValueError('adaptive selection must use the same uncorrected neural geometry')
        for key in ('physical', 'geometry', 'nl'):
            if selection_identity['config'][key] != c.to_dict()[key]:
                raise ValueError(f'adaptive neural selection {key} differs')
        selection_before = {key: sha(selection/key) for key in ('fields.npz', 'readout.eqx', 'result.json')}
        with np.load(selection/'fields.npz', allow_pickle=False) as f:
            np.testing.assert_array_equal(f['H_used'], H_np)
            residual_image = f['damage'].copy()
    residual = (gc0+2*H_np)*residual_image+gc1*(D@residual_image)-2*H_np
    inherited, inherited_before = None, None
    if c.compact.inherited_source is not None:
        parent = Path(c.compact.inherited_source)
        parent_identity = json.loads((parent.parent/'identity.json').read_text())
        parent_report = json.loads((parent/'result.json').read_text())
        if (parent_identity['geometry'] != geometry or parent_identity['source'] != before
                or parent_identity['direct_targets'] or parent_identity['damage_correction']):
            raise ValueError('nested features require the same uncorrected neural trunk')
        if parent_report['readout_sha256'] != sha(parent/'readout.eqx') or parent_report['fields_sha256'] != sha(parent/'fields.npz'):
            raise ValueError('nested neural parent checkpoint differs')
        with np.load(parent/'fields.npz', allow_pickle=False) as f:
            np.testing.assert_array_equal(f['H_used'], H_np)
            np.testing.assert_allclose(f['base_raw'], np.asarray(base_raw), rtol=0, atol=1e-14)
        with np.load(parent.parent/'basis_geometry.npz', allow_pickle=False) as f:
            inherited = (f['selected_indices'].copy(), f['radii'].copy())
            old_scale = f['column_scale'].copy()
        inherited_before = {str(path): sha(path) for path in
            [parent/'fields.npz', parent/'readout.eqx', parent/'result.json',
             parent.parent/'identity.json', parent.parent/'basis.npz', parent.parent/'basis_geometry.npz']}
    matrix, centers, radii, selected = build_basis(mechanics.coordinates[:mechanics.n_material], residual, c.compact, inherited=inherited)
    scaling = residual_column_scale(D, matrix, H_np, gc0=gc0, gc1=gc1)
    phi, width = pack_basis(matrix, scaling)
    net = Readout(jnp.zeros(matrix.shape[1], jnp.float64))
    if inherited is not None:
        old_matrix = load_npz(parent.parent/'basis.npz')
        difference = matrix[:, :old_matrix.shape[1]]-old_matrix
        if difference.nnz and np.max(abs(difference.data)) > 1e-14:
            raise ValueError('nested neural basis must preserve its parent columns')
        old_net = eqx.tree_deserialise_leaves(parent/'readout.eqx', Readout(jnp.zeros(len(old_scale), jnp.float64)))
        net = Readout(net.coefficients.at[:len(old_scale)].set(old_net.coefficients*old_scale/scaling[:len(old_scale)]))
        with np.load(parent/'fields.npz', allow_pickle=False) as f:
            np.testing.assert_allclose(_nonnegative_damage(base_raw+phi(net), c.physical.damage_floor), f['damage'], rtol=1e-11, atol=1e-13)
    matvec, _ = make_padded_matvec(operator)
    save_npz(root/'basis.npz', matrix)
    np.savez_compressed(root/'basis_geometry.npz', centers=centers, radii=radii,
                        selected_indices=selected, column_scale=scaling)
    atomic_json(root/'identity.json', dict(schema='compact-neural-damage-fixedH-v1', config=c.to_dict(),
        geometry=geometry, source=before, residual_selection_source=selection_before,
        inherited_neural_source=inherited_before,
        runner_sha256=sha(__file__), optimizer_sha256=sha(Path(__file__).with_name('damage_optimization.py')),
        basis_code_sha256=sha(Path(__file__).with_name('compact_damage_features.py')),
        frozen_conditioned_trunk=True, trainable_readouts=matrix.shape[1], basis_row_width=width,
        direct_targets=False, damage_correction=False, physical_state_promoted=False))
    print(json.dumps(dict(event='compact_basis', readouts=matrix.shape[1], row_width=width)), flush=True)

    def image(n, history, prescribed):
        return _nonnegative_damage(base_raw+phi(n), c.physical.damage_floor)
    def metrics(n, history, prescribed):
        d = image(n, history, prescribed)
        parts = (gc0*d, gc1*matvec(d), 2*history*(d-1))
        return dict(AT2_relative=jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts),
                    damage_min=jnp.min(d), damage_max=jnp.max(d))
    def mse(n, history, prescribed):
        d = image(n, history, prescribed)
        rho = (gc0+2*history)*d+gc1*matvec(d)-2*history
        return jnp.mean((rho/(2*jnp.sqrt(jnp.mean(history**2))))**2)
    def peak_objective(power):
        def objective(n, history, prescribed):
            d = image(n, history, prescribed)
            rho = (gc0+2*history)*d+gc1*matvec(d)-2*history
            return residual_power_loss(rho, history, power=power, floor=c.conditioned.loss_floor)
        return objective
    objectives = dict(residual_mse=mse, residual_l4=peak_objective(4), residual_l16=peak_objective(16), residual_l32=peak_objective(32))
    if (not c.compact.methods or set(c.compact.methods)-objectives.keys()
            or len(set(c.compact.methods)) != len(c.compact.methods)):
        raise ValueError('compact objective stages must be explicit and unique')
    reports = []
    for name in c.compact.methods:
        objective = objectives[name]
        folder = root/name
        folder.mkdir()
        def progress(row):
            print(json.dumps(dict(method=name, **row)), flush=True)
        net, optimization = optimize_lbfgs(net, objective, metrics, H, load,
            c.optimization.lbfgs.to_dict(), progress=progress)
        d = np.asarray(eqx.filter_jit(image)(net, H, load))
        np.savez_compressed(folder/'fields.npz', damage=d, H_used=H_np, base_raw=np.asarray(base_raw))
        eqx.tree_serialise_leaves(folder/'readout.eqx', net)
        r = dict(method=name, optimization=optimization,
                 **{k: float(v) for k, v in eqx.filter_jit(metrics)(net, H, load).items()},
                 readout_sha256=sha(folder/'readout.eqx'), fields_sha256=sha(folder/'fields.npz'))
        atomic_json(folder/'result.json', r)
        reports.append(r)
        if optimization['status'] == 'residual_reached':
            break
    oracle = np.asarray(damage_solver(operator, mechanics.volumes, gc0, gc1,
        tol=c.validation.cg_rtol, atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter)(H, jnp.zeros_like(H)))
    np.savez_compressed(root/'oracle.npz', damage=oracle, H_used=H_np)
    for r in reports:
        folder = root/r['method']
        with np.load(folder/'fields.npz', allow_pickle=False) as f:
            d = f['damage']
        parts = (gc0*d, gc1*(D@d), 2*H_np*(d-1))
        host = float(np.max(abs(sum(parts)))/sum(np.max(abs(p)) for p in parts))
        if abs(host-r['AT2_relative']) > 1e-10:
            raise RuntimeError('compact neural CPU/GPU residual differs')
        r.update(host_AT2_relative=host, oracle_damage_inf=float(np.max(abs(d-oracle))),
                 oracle_damage_relative_l2=float(np.linalg.norm(d-oracle)/np.linalg.norm(oracle)))
        atomic_json(folder/'result.json', r)
    if before != {key: sha(source/key) for key in before}:
        raise RuntimeError('frozen conditioned trunk changed')
    if selection_before is not None and selection_before != {key: sha(selection/key) for key in selection_before}:
        raise RuntimeError('neural residual selection source changed')
    if inherited_before is not None and inherited_before != {path: sha(path) for path in inherited_before}:
        raise RuntimeError('nested neural parent changed')
    result = dict(complete=True, ready=reports[-1]['optimization']['status']=='residual_reached',
                  reports=reports, seconds=time.perf_counter()-begin, source_unchanged=True,
                  damage_correction=False, direct_targets=False, physical_state_promoted=False)
    atomic_json(root/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    c = runpy.run_path(parser.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)
