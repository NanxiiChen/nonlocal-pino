"""Same-H detached-weight pilot; full original residual gates, no loading."""
import argparse
import json
from pathlib import Path
import runpy
import shutil

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz

from examples.fracture2d.asymmetric_damage_range import make_asymmetric_lbfgs
from examples.fracture2d.compact_damage_loading import history_feature_input
from examples.fracture2d.conditioned_damage import make_features
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.damage_residual_weighting import (
    make_weights, validate_weights, make_weighted_objective, build_core_halo, core_matvec,
)
from examples.fracture2d.local_damage_trial import load_geometry, sha
from examples.fracture2d.plain_damage_head import fresh_network, make_problem
from examples.fracture2d.range_damage_optimization import damage_diagnostics, make_augmented_objective
from examples.fracture2d.split_pinn_gradients import make_padded_matvec


def run(c):
    w = c.weighting
    if (not jax.config.jax_enable_x64 or jax.devices()[0].platform != 'gpu'
            or list(w.modes) != ['uniform', 'static', 'damage'] or w.power != 16
            or type(w.maxiter) is not int or w.maxiter < 1
            or type(w.core_count) is not int or w.core_count < 1
            or type(w.sample_seed) is not int or not np.isfinite(w.penalty) or w.penalty <= 0
            or w.load != .0053 or c.range.method != 'residual_l16'):
        raise ValueError('pilot requires explicit CUDA/x64, three modes, same load and L16 controls')
    snapshot, production = Path(w.snapshot_dir), Path(c.loading.output_dir)
    identity = json.loads((production/'identity.json').read_text())
    expected = c.to_dict()
    del expected['weighting']
    if identity['config'] != expected:
        raise ValueError('pilot physical/network/optimizer source configuration differs')
    for name, digest in identity['code'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'protected production code changed: {name}')
    for name, digest in [('current_fields.npz', w.fields_sha256), ('current_network.eqx', w.network_sha256)]:
        if sha(snapshot/name) != digest:
            raise ValueError(f'pilot checkpoint checksum differs: {name}')
    source_report = json.loads((snapshot/'error_report.json').read_text())
    if not source_report['pass_replay'] or source_report['current_outer'] != 153:
        raise ValueError('pilot requires independently replayed physical153')
    mechanics, operator, geometry = load_geometry(c)
    if geometry != identity['geometry'] or sha(c.cold.basis_path) != c.cold.basis_sha256:
        raise ValueError('pilot geometry/basis identity differs')
    matrix = load_npz(c.cold.basis_path)
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    points = mechanics.coordinates[:mechanics.n_material]
    features, info = make_features(operator, points, gc0=gc0, length=c.physical.l0,
        tip=list(c.conditioned.tip), powers=list(c.conditioned.powers), roundoff_factor=c.validation.roundoff_factor)
    def known_features(H, load):
        return features(history_feature_input(H, load, c.cold.reference_load, c.cold.history_feature_scaling))
    with np.load(production/'parameterization.npz', allow_pickle=False) as f:
        scales, gain = f['column_scale'].copy(), float(f['trunk_gain'])
    with np.load(snapshot/'current_fields.npz', allow_pickle=False) as f:
        H, saved_H, seed_damage, saved_image = [f[k].copy() for k in ('H', 'H_used', 'd', 'damage_image')]
    if (H.shape != (mechanics.n_material,) or H.dtype != np.float64
            or seed_damage.shape != H.shape or seed_damage.dtype != np.float64
            or not np.isfinite(points).all() or not np.isfinite(seed_damage).all()
            or not np.isfinite(H).all() or H.min() < 0 or H.max() <= 0):
        raise ValueError('pilot requires finite ordered physical seed and positive own history')
    template = fresh_network(c, info['input_dimension'], matrix, head='identity')
    initial = eqx.tree_deserialise_leaves(snapshot/'current_network.eqx', template)
    s = dict(matrix=matrix, operator=operator, reference_load=c.cold.reference_load, features=known_features)
    image, _, objectives, _ = make_problem(c, s, scales, gain, head='identity')
    predict = eqx.filter_jit(image)
    load, H = jnp.asarray(w.load, jnp.float64), jax.lax.stop_gradient(jnp.asarray(H))
    np.testing.assert_allclose(predict(initial, jnp.asarray(saved_H), load), saved_image, rtol=1e-10, atol=1e-12)
    matvec, packing = make_padded_matvec(operator)
    matvec = jax.jit(matvec)
    matrix_D = operator.to_scipy()
    row_width = packing['row_width']+1  # Include the original diagonal.
    volumes = jnp.asarray(mechanics.volumes)
    def metrics(net, context, load):
        return damage_diagnostics(image(net, context[0], load), context[0], matvec,
            gc0=gc0, gc1=gc1, volumes=volumes, thresholds=tuple(c.range.activity_thresholds))
    objective = make_weighted_objective(image, matvec, gc0=gc0, gc1=gc1, power=w.power, floor=c.conditioned.loss_floor)
    fit, value_grad, evaluate = make_asymmetric_lbfgs(initial, objective, metrics)
    zeros = jnp.zeros_like(H)
    penalty = jnp.asarray(w.penalty, jnp.float64)
    root = Path(w.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    names = ['weighted_damage_trial.py', 'damage_residual_weighting.py']
    atomic_json(root/'identity.json', dict(config=c.to_dict(), source_fields_sha256=w.fields_sha256,
        source_network_sha256=w.network_sha256, source_outer=153, load=w.load,
        physical_state_promoted=False, direct_training_targets=False, full_field_acceptance=True,
        weights_frozen_per_fit=True, weights_detached=True, production_unchanged=True,
        code={n:sha(Path(__file__).with_name(n)) for n in names}))
    for name in names: shutil.copyfile(Path(__file__).with_name(name), root/name)
    controls = c.range.lbfgs.to_dict()
    controls['maxiter'] = w.maxiter
    reports = []
    from jax.flatten_util import ravel_pytree
    flat, _ = ravel_pytree(eqx.filter(initial, eqx.is_inexact_array))
    original = make_augmented_objective(image, objectives[c.range.method], gc0=gc0)
    original_value = float(eqx.filter_jit(original)(initial, (H, zeros, zeros, penalty), load))
    for mode in w.modes:
        weights = make_weights(jnp.asarray(points), jnp.asarray(seed_damage), mode=mode,
            boost=w.boost, damage_power=w.damage_power, start=w.segment_start, end=w.segment_end, width=w.band_width)
        validate_weights(weights, H)
        context = (H, zeros, zeros, penalty, weights)
        # Warm both JIT paths so compilation is excluded from the fit timing.
        value, gradient = value_grad(flat, context, load)
        jax.block_until_ready((value, gradient, evaluate(flat, context, load)))
        if mode == 'uniform': np.testing.assert_allclose(float(value), original_value, rtol=1e-12, atol=1e-14)
        folder = root/mode
        folder.mkdir()
        np.savez_compressed(folder/'weights.npz', weights=np.asarray(weights), seed_damage=seed_damage)
        def progress(row):
            print(json.dumps(dict(mode=mode, **row)), flush=True)
        candidate, optimization = fit(initial, context, load, controls, progress=progress)
        damage = np.asarray(predict(candidate, H, load))
        stats = {k:float(v) for k,v in eqx.filter_jit(metrics)(candidate, context, load).items()}
        np.savez_compressed(folder/'fields.npz', d=damage, H=np.asarray(H))
        eqx.tree_serialise_leaves(folder/'network.eqx', candidate)
        report = dict(mode=mode, optimization=optimization, metrics=stats,
            weight_min=float(weights.min()), weight_max=float(weights.max()),
            ready=optimization['status']=='residual_reached', physical_state_promoted=False,
            gpu_time_shared_with_production=True, absolute_speedup_claim=False)
        atomic_json(folder/'result.json', report)
        reports.append(report)
        # Count exact support costs; core+halo is diagnosed, not used for fitting.
        probabilities = np.asarray(weights)/float(weights.sum())
        core = np.random.default_rng(w.sample_seed).choice(len(H), size=w.core_count, replace=True, p=probabilities)
        packed = build_core_halo(matrix_D, core, node_capacity=len(H), row_width=row_width)
        sampled = core_matvec(jnp.asarray(damage[packed['global_ids']])*packed['node_mask'], packed['columns'], packed['values'])
        np.testing.assert_allclose(sampled, (matrix_D@damage)[core], rtol=1e-11, atol=1e-9)
        atomic_json(folder/'core_halo_diagnostic.json', dict(core_draws=w.core_count,
            unique_core=int(len(np.unique(core))), halo_nodes=int(packed['node_mask'].sum()),
            full_nodes=len(H), retained_nonzero_coefficients=int(np.count_nonzero(packed['values'])),
            full_nonzero_coefficients=matrix_D.nnz, original_rows_equal=True, sampled_training=False,
            with_replacement=True, importance_factor='1/(N*p_i)', no_halo_truncation=True))
        atomic_json(root/'result.json', dict(complete=len(reports)==len(w.modes), reports=reports,
            load=w.load, same_initial_network=True, same_H=True, no_loading=True, physical_state_promoted=False))
        print(json.dumps(dict(event='case_complete', mode=mode, iterations=optimization['iterations'],
            seconds=optimization['seconds'], AT2_relative=stats['AT2_relative'], ready=report['ready'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    run(runpy.run_path(args.config)['get_config']())
