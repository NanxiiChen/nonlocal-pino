"""Independent CPU native output, full loss, history, P1 and host-force replay."""
import argparse
import json
import os
from pathlib import Path
import time

import equinox as eqx
import jax
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import coo_matrix, load_npz

from examples.fracture2d._legacy.bounded_damage_morphology_trial import morphology
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.full_damage_optimization import adam_from_config, choose_optimizer
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.low_order_damage import optimizer_count
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.staggered_damage_audit import native_image, read_fields, residual


def main(root, *, watch):
    if jax.devices()[0].platform != 'cpu': raise ValueError('all-node independent audit requires CPU')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    while not (root/'preflight.json').exists():
        if (root/'exit.json').exists(): raise RuntimeError('full-domain trial fails before preflight')
        if not watch: raise RuntimeError('full-domain preflight unavailable')
        time.sleep(5)
    identity = json.loads((root/'identity.json').read_text())
    a, c = ConfigDict(identity['config'], allow_dotted_keys=True), ConfigDict(identity['source_config'], allow_dotted_keys=True)
    for name, digest in a.pins.items(): assert sha(Path(a.source_root)/name) == digest, name
    for name, digest in identity['code'].items(): assert sha(Path('/root/nonlocal-KKS')/name) == digest, name
    m, op, geometry = load_geometry(c); assert geometry == identity['geometry']
    assert identity['sampling'] == 'all_nodes' and identity['full_node_count'] == m.n_material
    D, B = op.to_scipy(), load_npz(c.l1_loading.basis_path)
    source = Path(a.source_root)
    parent = read_fields(source/f'state_{a.parent_index:04d}'/'fields.npz')
    scale = read_fields(source/'parameterization.npz')
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    rows = np.broadcast_to(tri[:, :, None], (len(tri), 3, 3)).ravel()
    cols = np.broadcast_to(tri[:, None, :], (len(tri), 3, 3)).ravel()
    vals = (area[:, None, None]*(np.ones((3, 3))+np.eye(3))/12).ravel()
    mass = coo_matrix((vals, (rows, cols)), shape=(m.n_material, m.n_material)).tocsr()
    def rms(v): return float(np.sqrt(np.sum(v*(mass@v))/area.sum()))
    def full_loss(d, H):
        raw = (g0+2*H)*d+g1*(D@d)-2*H
        return float(np.mean(abs(raw))/(2*np.sqrt(np.mean(H**2))))
    features = json.loads((source/'identity.json').read_text())['features']
    template = fresh_network(c, features['input_dimension'], B, head='sigmoid')
    optimizer, _ = adam_from_config(c.l1_loading)
    memory = optimizer.init(eqx.filter(template, eqx.is_inexact_array))
    def image(folder, H):
        return native_image(folder/'network.eqx', H, c, D, B,
            m.coordinates[:m.n_material], scale['column_scale'], float(scale['trunk_gain']), g0)
    review = root/'independent_review'; review.mkdir(exist_ok=True)
    done = set(); audited = []; screens = []
    ref_root = Path('/root/autodl-tmp/tf-logs/fracture2d/staggered_damage_trial_20261013/direct_full')
    reference = read_fields(ref_root/'final/fields.npz')
    selection_checked = False
    while True:
        for kind in ['adam', 'lbfgs']:
            folder = root/'screen'/kind; rp = folder/'report.json'
            if not rp.exists() or str(rp) in done: continue
            r, f = json.loads(rp.read_text()), read_fields(folder/'fields.npz')
            assert sha(folder/'fields.npz') == r['fields_sha256']
            assert sha(folder/'network.eqx') == r['network_sha256']
            np.testing.assert_array_equal(f['H_parent'], parent['H'])
            np.testing.assert_array_equal(f['d_parent'], parent['d'])
            np.testing.assert_array_equal(f['H'], np.maximum(parent['H'], m.driver(f['q_mechanics'])))
            np.testing.assert_allclose(image(folder, f['H']), f['damage_image'], rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(full_loss(f['damage_image'], f['H']), r['full_loss'], rtol=1e-9, atol=1e-12)
            force = m.matrix(parent['d'])@f['q_mechanics']
            assert np.linalg.norm(force[m.free]) <= r['mechanics']['convergence_limit']*(1+1e-3)
            np.testing.assert_allclose(force@m.load_direction, r['reaction_at_parent_damage'], rtol=1e-9, atol=1e-6)
            assert r['physical_state_promoted'] is False
            if kind == 'adam':
                state = eqx.tree_deserialise_leaves(folder/'optimizer.eqx', memory)
                assert optimizer_count(state, kind='adam') == r['optimizer_count'] == a.parent_count+a.full_training.pilot_updates
            else: assert r['optimizer_count'] is None and 0 <= r['iterations'] <= a.full_training.lbfgs.maxiter
            screens.append(dict(optimizer=kind, full_loss=r['full_loss'], pass_replay=True,
                iterations=r['iterations'], seconds=r['seconds'], AT2_relative=r['AT2_relative'], report_sha256=sha(rp)))
            done.add(str(rp))
        selection_path = root/'selection.json'
        if selection_path.exists() and len(screens) == 2 and not selection_checked:
            selection = json.loads(selection_path.read_text())
            assert choose_optimizer(screens) == selection['selected']
            selection_checked = True
        folders = sorted((root/'pinn_full').glob('outer_[0-9][0-9][0-9][0-9]'))+[root/'pinn_full/final']
        for folder in folders:
            rp = folder/'report.json'
            if not rp.exists() or str(rp) in done: continue
            assert selection_checked
            r, f = json.loads(rp.read_text()), read_fields(folder/'fields.npz')
            final = folder.name == 'final'
            assert r['optimizer'] == selection['selected'] and r['full_node_count'] == m.n_material
            assert sha(folder/'fields.npz') == r['fields_sha256'] and sha(folder/'network.eqx') == r['network_sha256']
            for name, v in f.items(): assert v.dtype == np.float64 and np.isfinite(v).all(), name
            np.testing.assert_array_equal(f['H_parent'], parent['H'])
            np.testing.assert_array_equal(f['d_parent'], parent['d'])
            np.testing.assert_array_equal(f['d'], f['damage_image'])
            q_stage = f['q_stage'] if final else f['q']
            np.testing.assert_array_equal(f['H_used'], np.maximum(parent['H'], m.driver(q_stage)))
            np.testing.assert_array_equal(f['H'], np.maximum(parent['H'], m.driver(f['q'])))
            np.testing.assert_array_equal(f['q'][m.constrained], (a.load*m.load_direction)[m.constrained])
            np.testing.assert_allclose(image(folder, f['H_used']), f['damage_image'], rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(full_loss(f['d'], f['H_used']), r['full_loss'], rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(residual(f['d'], f['H_used'], D, g0, g1), r['inner_AT2'], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(rms((q_stage-f['q_previous']).reshape(-1, 2)[:m.n_material])/a.load,
                r['displacement_rms_over_load'], rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(rms(f['d']-f['d_previous']), r['damage_rms'], rtol=1e-10, atol=1e-14)
            force = m.matrix(f['d'] if final else f['d_previous'])@f['q']
            norm = float(np.linalg.norm(force[m.free]))
            assert norm <= r['mechanics']['convergence_limit']*(1+1e-3)
            np.testing.assert_allclose(force@m.load_direction, r['reaction'], rtol=1e-9, atol=1e-6)
            assert r['mechanics_calls'] == r['sweep']+int(final)
            if r['optimizer'] == 'adam':
                assert sha(folder/'optimizer.eqx') == r['optimizer_sha256']
                state = eqx.tree_deserialise_leaves(folder/'optimizer.eqx', memory)
                assert optimizer_count(state, kind='adam') == r['optimizer_count'] == a.parent_count+r['sweep']*a.full_training.inner_updates
                assert r['learning_rate'] == c.l1_loading.learning_rate
            else: assert r['optimizer_count'] is None and r['learning_rate'] is None
            journal = [json.loads((root/'pinn_full'/f'sweep_{i:04d}.json').read_text()) for i in range(1, r['sweep']+1)]
            assert sum(x['optimization']['iterations'] for x in journal) == r['training_updates']
            assert sum(x['optimization']['evaluations'] for x in journal) == r['function_evaluations']
            assert all(0 <= x['optimization']['iterations'] <= a.full_training.inner_updates for x in journal)
            output = dict(sweep=r['sweep'], final=final, pass_replay=True, reaction=r['reaction'],
                full_loss=r['full_loss'], inner_AT2=r['inner_AT2'], optimizer=r['optimizer'],
                report_sha256=sha(rp), fields_sha256=r['fields_sha256'], host_force_norm=norm,
                morphology=morphology(f['d'], m.coordinates[:m.n_material], m.volumes,
                    band_width=3*c.physical.l0, threshold=.01))
            if final:
                np.testing.assert_allclose(image(folder, f['H']), f['raw_current_image'], rtol=1e-9, atol=1e-11)
                np.testing.assert_allclose(residual(f['d'], f['H'], D, g0, g1), r['physical_AT2'], rtol=1e-10, atol=1e-12)
                output.update(physical_AT2=r['physical_AT2'], field_stable=r['field_stable'],
                    damage_L2_vs_common_parent_direct=float(np.sqrt(m.volumes@(f['d']-reference['d'])**2/(m.volumes@reference['d']**2))))
                assert not r['certified_path'] and not r['load_advanced']
            audited.append(output); done.add(str(rp))
            print(json.dumps(dict(event='full_domain_CPU_replay', **output)), flush=True)
        finished = (root/'result.json').exists() and json.loads((root/'result.json').read_text())['complete']
        complete = finished and any(x['final'] for x in audited) and selection_checked
        atomic_json(review/'audit.json', dict(complete=complete, pass_replay=True, rows=audited,
            screening=screens, selection_checked=selection_checked, no_random_sampling=True,
            auditor_sha256=sha(__file__)))
        if complete or not watch: return
        if (root/'exit.json').exists() and json.loads((root/'exit.json').read_text())['worker_exit_code'] != 0:
            raise RuntimeError('full-domain worker failed; prior audited snapshots retained')
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--watch', action='store_true'); args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    main(Path(args.root), watch=args.watch)
