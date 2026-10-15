"""CPU native-network/CSR/P1/state/sampling replay for stagger controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import equinox as eqx
import jax
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import coo_matrix, load_npz
from scipy.special import expit

from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.low_order_damage import make_optimizer, optimizer_count
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.staggered_damage import case_configs


def read_fields(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def native_image(path, H, c, D, B, xy, scales, gain, g0):
    """NumPy Swish/sigmoid and host sparse features, no JAX network call."""
    offset = (xy-np.asarray(c.conditioned.tip))/c.physical.l0
    geom = np.column_stack((2*xy[:, 0]-1, 2*xy[:, 1]-1,
        np.tanh(offset[:, 0]), np.tanh(offset[:, 1]),
        np.exp(-.5*np.sum(offset**2, axis=1)), np.exp(-.5*offset[:, 1]**2)))
    off = -D.copy(); off.setdiag(0.); off.eliminate_zeros()
    current, values = H.copy(), []
    for k in range(max(c.conditioned.powers)+1):
        if k in c.conditioned.powers:
            ratio = 2*current/g0
            values.extend((np.log1p(ratio), ratio/(1+ratio)))
        if k < max(c.conditioned.powers):
            current = (off@current)/D.diagonal()
    x = np.column_stack((geom, *values))
    net = eqx.tree_deserialise_leaves(path, fresh_network(c, x.shape[1], B, head='sigmoid'))
    for layer in net.trunk.layers:
        z = x@np.asarray(layer.weight).T+np.asarray(layer.bias)
        x = z*expit(z)
    latent = gain*(x@np.asarray(net.trunk.output.weight).T+np.asarray(net.trunk.output.bias))[:, 0]
    return expit(latent+B@(scales*np.asarray(net.readout.coefficients)))


def residual(d, H, D, g0, g1):
    parts = [g0*d, g1*(D@d), 2*H*(d-1)]
    return float(abs(sum(parts)).max()/sum(abs(p).max() for p in parts))


def main(root, *, watch):
    if jax.devices()[0].platform != 'cpu':
        raise ValueError('independent stagger replay requires CPU')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    while not (root/'preflight.json').exists():
        if (root/'exit.json').exists(): raise RuntimeError('trial exits before GPU preflight')
        if not watch: raise RuntimeError('preflight not yet available')
        time.sleep(5)
    identity = json.loads((root/'identity.json').read_text())
    a, c = ConfigDict(identity['config'], allow_dotted_keys=True), ConfigDict(identity['source_config'], allow_dotted_keys=True)
    for name, digest in a.pins.items():
        assert sha(Path(a.source_root)/name) == digest, name
    for name, digest in identity['code'].items():
        assert sha(Path('/root/nonlocal-KKS')/name) == digest, name
    m, op, geometry = load_geometry(c)
    assert geometry == identity['geometry']
    D, B = op.to_scipy(), load_npz(c.l1_loading.basis_path)
    scales = read_fields(Path(a.source_root)/'parameterization.npz')
    parent = read_fields(Path(a.source_root)/f'state_{a.parent_index:04d}'/'fields.npz')
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    # Independently assembled consistent P1 mass, rather than worker's norm formula.
    rows = np.broadcast_to(tri[:, :, None], (len(tri), 3, 3)).ravel()
    cols = np.broadcast_to(tri[:, None, :], (len(tri), 3, 3)).ravel()
    values = (area[:, None, None]*(np.ones((3, 3))+np.eye(3))/12).ravel()
    mass = coo_matrix((values, (rows, cols)), shape=(m.n_material, m.n_material)).tocsr()
    def rms(v): return float(np.sqrt(np.sum(v*(mass@v))/area.sum()))
    t = c.l1_loading
    optimizer, _ = make_optimizer(kind=t.optimizer, learning_rate=t.learning_rate,
        decay_steps=t.decay_steps, decay_rate=t.decay_rate, minimum_rate=t.minimum_rate,
        grad_clip=t.grad_clip, b1=t.b1, b2=t.b2, eps=t.eps, eps_root=t.eps_root,
        shampoo_beta=t.shampoo_beta, correct_bias=t.correct_bias,
        precondition_frequency=t.precondition_frequency, max_precond_dim=t.max_precond_dim,
        precondition_1d=t.precondition_1d, weight_decay=t.weight_decay)
    features = json.loads((Path(a.source_root)/'identity.json').read_text())['features']
    template = fresh_network(c, features['input_dimension'], B, head='sigmoid')
    memory = optimizer.init(eqx.filter(template, eqx.is_inexact_array))
    probabilities = np.load(Path(a.source_root)/'sampling_probabilities.npy')
    generators = {}; block_sets = {}; done = set(); audited = []
    for case in case_configs(a.cases):
        generators[case.name] = np.random.default_rng()
        generators[case.name].bit_generator.state = identity['rng_state']
        block_sets[case.name] = set()
    review = root/'independent_review'; review.mkdir(exist_ok=True)
    reference_root = Path('/root/autodl-tmp/tf-logs/fracture2d/local_damage_load_refined_20261011_direct')
    refs = json.loads((reference_root/'result.json').read_text())['curve']
    matches = [r for r in refs if abs(r['load']-a.load)<1e-14]; assert len(matches) == 1
    reference = matches[0]
    ref_path = reference_root/f"state_{reference['index']:04d}"/'fields.npz'
    ref = read_fields(ref_path)['d']
    while True:
        for case in case_configs(a.cases):
            journal = root/case.name/'sampling_blocks.jsonl'
            if journal.exists():
                for line in journal.read_text().splitlines(keepends=True):
                    if not line.endswith('\n'): break
                    block = json.loads(line); seen = block_sets[case.name]
                    if block['block'] in seen: continue
                    rng = generators[case.name]
                    assert block['block'] == len(seen)+1 and rng.bit_generator.state == block['rng_before']
                    cores = np.stack([rng.choice(m.n_material, size=t.sample_count,
                        replace=True, p=probabilities).astype(np.int32) for _ in range(t.pool_size)])
                    assert hashlib.sha256(cores.tobytes()).hexdigest() == block['cores_sha256']
                    assert rng.bit_generator.state == block['rng_after']
                    assert 0 < block['halo_min'] <= block['halo_max'] <= t.halo_capacity
                    seen.add(block['block'])
            folders = sorted((root/case.name).glob('outer_[0-9][0-9][0-9][0-9]'))
            folders += [root/case.name/'final']
            for folder in folders:
                rp = folder/'report.json'
                if not rp.exists() or str(rp) in done: continue
                r, f = json.loads(rp.read_text()), read_fields(folder/'fields.npz')
                final = folder.name == 'final'
                assert sha(folder/'fields.npz') == r['fields_sha256']
                for name, value in f.items(): assert value.dtype == np.float64 and np.isfinite(value).all(), name
                np.testing.assert_array_equal(f['H_parent'], parent['H'])
                np.testing.assert_array_equal(f['d_parent'], parent['d'])
                np.testing.assert_allclose(f['d'], f['d_previous']+case.omega*(f['damage_image']-f['d_previous']), rtol=1e-12, atol=1e-14)
                stage_q = f['q_stage'] if final else f['q']
                np.testing.assert_array_equal(f['H_used'], np.maximum(parent['H'], m.driver(stage_q)))
                np.testing.assert_array_equal(f['H'], np.maximum(parent['H'], m.driver(f['q'])))
                np.testing.assert_array_equal(f['q'][m.constrained], (a.load*m.load_direction)[m.constrained])
                du = (stage_q-f['q_previous']).reshape(-1, 2)[:m.n_material]
                np.testing.assert_allclose(rms(du)/a.load, r['displacement_rms_over_load'], rtol=1e-10, atol=1e-14)
                np.testing.assert_allclose(rms(f['d']-f['d_previous']), r['damage_rms'], rtol=1e-10, atol=1e-14)
                np.testing.assert_allclose(residual(f['damage_image'], f['H_used'], D, g0, g1), r['inner_AT2'], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(residual(f['d'], f['H_used'], D, g0, g1), r['stage_AT2'], rtol=1e-10, atol=1e-12)
                force = m.matrix(f['d'] if final else f['d_previous'])@f['q']
                norm = float(np.linalg.norm(force[m.free]))
                assert norm <= r['mechanics']['convergence_limit']*(1+1e-3)
                np.testing.assert_allclose(force@m.load_direction, r['reaction'], rtol=1e-9, atol=1e-6)
                assert r['mechanics_calls'] == r['sweep']+int(final)
                if case.solver == 'pinn':
                    for name in ['network','optimizer']: assert sha(folder/(name+'.eqx')) == r[name+'_sha256']
                    image = native_image(folder/'network.eqx', f['H_used'], c, D, B,
                        m.coordinates[:m.n_material], scales['column_scale'], float(scales['trunk_gain']), g0)
                    np.testing.assert_allclose(image, f['damage_image'], rtol=1e-9, atol=1e-11)
                    state = eqx.tree_deserialise_leaves(folder/'optimizer.eqx', memory)
                    assert optimizer_count(state, kind='adam') == r['optimizer_count'] == a.parent_count+r['sweep']*t.inner_updates
                    assert r['learning_rate'] == t.learning_rate
                    if final:
                        image = native_image(folder/'network.eqx', f['H'], c, D, B,
                            m.coordinates[:m.n_material], scales['column_scale'], float(scales['trunk_gain']), g0)
                        np.testing.assert_allclose(image, f['raw_current_image'], rtol=1e-9, atol=1e-11)
                output = dict(case=case.name, sweep=r['sweep'], final=final, pass_replay=True,
                    report_sha256=sha(rp), fields_sha256=r['fields_sha256'], reaction=r['reaction'],
                    inner_AT2=r['inner_AT2'], stage_AT2=r['stage_AT2'], host_force_norm=norm)
                if final:
                    np.testing.assert_allclose(residual(f['d'], f['H'], D, g0, g1), r['physical_AT2'], rtol=1e-10, atol=1e-12)
                    assert not r['load_advanced'] and not r['certified_path']
                    output.update(physical_AT2=r['physical_AT2'], field_stable=r['field_stable'],
                        original_residual_pass=r['original_residual_pass'],
                        matched_load_damage_L2=float(np.sqrt(m.volumes@(f['d']-ref)**2/(m.volumes@ref**2))),
                        reference_reaction=reference['reaction'], matching_load=a.load,
                        reference_fields_sha256=sha(ref_path))
                audited.append(output); done.add(str(rp))
                print(json.dumps(dict(event='stagger_CPU_replay', **output)), flush=True)
        finished = (root/'result.json').exists() and json.loads((root/'result.json').read_text())['complete']
        complete = finished and sum(row['final'] for row in audited) == len(a.cases)
        atomic_json(review/'audit.json', dict(pass_replay=True, complete=complete,
            rows=audited, sampled_blocks={name:len(seen) for name,seen in block_sets.items()},
            source_unchanged=True, no_load_advancement=True, auditor_sha256=sha(__file__)))
        if complete or not watch: return
        if (root/'exit.json').exists() and json.loads((root/'exit.json').read_text())['worker_exit_code'] != 0:
            raise RuntimeError('stagger worker failed; successful independent snapshots retained')
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--watch', action='store_true'); args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    main(Path(args.root), watch=args.watch)
