"""CPU native/CSR/history/FEM replay of cold all-node loading endpoints."""
import argparse
import json
import os
from pathlib import Path
import time

import equinox as eqx
import jax
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import coo_matrix, diags, load_npz
from scipy.sparse.linalg import cg

from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.full_damage_loading import common_config, load_grid
from examples.fracture2d._legacy.full_damage_optimization import adam_from_config
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.low_order_damage import optimizer_count
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.staggered_damage_audit import native_image, read_fields, residual


def main(root, *, watch):
    if jax.devices()[0].platform != 'cpu':
        raise ValueError('cold loading independent audit must run on CPU')
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[4:8] if len(allowed) >= 8 else allowed)
    suite = root.parent
    while not (root/'preflight.json').exists():
        if (suite/'exit.json').exists(): raise RuntimeError('loading suite exits before preflight')
        if not watch: raise RuntimeError('cold loading preflight not available')
        time.sleep(5)
    identity = json.loads((root/'identity.json').read_text())
    c = ConfigDict(identity['config'], allow_dotted_keys=True); a, t = c.full_loading, c.l1_loading
    assert identity['sampling'] == 'all_nodes' and not identity['symmetry_constraint']
    assert identity['reference_training_targets'] is False and identity['residual_is_stopping_gate'] is False
    init = json.loads((suite/'common_initialization.json').read_text())
    assert init['common_config'] == common_config(c) and init == identity['common_initialization']
    assert sha(root/'initial_network.eqx') == init['network_sha256']
    assert sha(root/'initial_optimizer.eqx') == init['optimizer_sha256']
    for name, digest in identity['source'].items(): assert sha(name) == digest, name
    for name, digest in identity['code'].items(): assert sha(Path('/root/nonlocal-KKS')/name) == digest, name
    m, op, geometry = load_geometry(c); assert geometry == identity['geometry']
    D, B = op.to_scipy(), load_npz(t.basis_path)
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    scales = read_fields(root/'parameterization.npz')
    zero = read_fields(root/'state_0000/fields.npz')
    for x in zero.values(): np.testing.assert_array_equal(x, np.zeros_like(x))
    xy = m.coordinates[:m.n_material]
    def image(path, H):
        return native_image(path, H, c, D, B, xy, scales['column_scale'], float(scales['trunk_gain']), g0)
    np.testing.assert_allclose(image(root/'initial_network.eqx', zero['H']), c.head.initial_damage,
        rtol=1e-12, atol=1e-14)
    optimizer, _ = adam_from_config(t)
    template = fresh_network(c, identity['features']['input_dimension'], B, head='sigmoid')
    memory = optimizer.init(eqx.filter(template, eqx.is_inexact_array))
    initial = eqx.tree_deserialise_leaves(root/'initial_optimizer.eqx', memory)
    assert optimizer_count(initial, kind='adam') == 0
    tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
    rows = np.broadcast_to(tri[:, :, None], (len(tri), 3, 3)).ravel()
    cols = np.broadcast_to(tri[:, None, :], (len(tri), 3, 3)).ravel()
    mass = coo_matrix(((area[:, None, None]*(np.ones((3, 3))+np.eye(3))/12).ravel(),
        (rows, cols)), shape=(m.n_material, m.n_material)).tocsr()
    def rms(v): return float(np.sqrt(np.sum(v*(mass@v))/area.sum()))
    def mae(d, H): return float(abs((g0+2*H)*d+g1*(D@d)-2*H).mean()/(2*np.sqrt(np.mean(H**2))))
    refroot = Path(a.reference_root)
    refs = json.loads((refroot/'result.json').read_text())['curve']
    done = set(); outputs = []; grid = load_grid(a)
    review = root/'independent_review'; review.mkdir(exist_ok=True)
    while True:
        for index, load in enumerate(grid, 1):
            state = root/f'state_{index:04d}'
            folders = [state/'outer_0001', state]
            if abs(load-a.critical_load)<1e-14:
                folders = sorted(state.glob('outer_[0-9][0-9][0-9][0-9]'))+[state]
            for folder in folders:
                rp = folder/'report.json'
                if not rp.exists() or str(rp) in done: continue
                r, f = json.loads(rp.read_text()), read_fields(folder/'fields.npz')
                final = folder == state
                assert sha(folder/'fields.npz') == r['fields_sha256']
                assert sha(folder/'network.eqx') == r['network_sha256']
                assert sha(folder/'optimizer.eqx') == r['optimizer_sha256']
                parent_path = root/f'state_{index-1:04d}'/'fields.npz'
                assert sha(parent_path) == r['parent_sha256']
                parent = read_fields(parent_path)
                for key, val in f.items(): assert val.dtype == np.float64 and np.isfinite(val).all(), key
                np.testing.assert_array_equal(f['H_parent'], parent['H'])
                np.testing.assert_array_equal(f['d_parent'], parent['d'])
                np.testing.assert_array_equal(f['d'], f['damage_image'])
                assert f['d'].min()>=0 and f['d'].max()<=1 and len(f['d'])==m.n_material
                q_stage = f['q_stage'] if final else f['q']
                np.testing.assert_array_equal(f['H_used'], np.maximum(parent['H'], m.driver(q_stage)))
                np.testing.assert_array_equal(f['H'], np.maximum(parent['H'], m.driver(f['q'])))
                np.testing.assert_array_equal(f['q'][m.constrained], (load*m.load_direction)[m.constrained])
                np.testing.assert_allclose(image(folder/'network.eqx', f['H_used']), f['d'], rtol=1e-9, atol=1e-11)
                np.testing.assert_allclose(mae(f['d'], f['H_used']), r['full_loss'], rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(residual(f['d'], f['H_used'], D, g0, g1), r['inner_AT2'], rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(rms((q_stage-f['q_previous']).reshape(-1, 2)[:m.n_material])/load,
                    r['displacement_rms_over_load'], rtol=1e-9, atol=1e-14)
                np.testing.assert_allclose(rms(f['d']-f['d_previous']), r['damage_rms'], rtol=1e-9, atol=1e-14)
                force = m.matrix(f['d'] if final else f['d_previous'])@f['q']
                force_norm = float(np.linalg.norm(force[m.free]))
                assert force_norm<=r['mechanics']['convergence_limit']*(1+1e-3)
                np.testing.assert_allclose(force@m.load_direction, r['reaction'], rtol=1e-9, atol=1e-6)
                assert r['mechanics_calls']==r['sweep']+int(final)
                restored = eqx.tree_deserialise_leaves(folder/'optimizer.eqx', memory)
                assert optimizer_count(restored, kind='adam')==r['optimizer_count']
                assert r['optimizer_count']==r['parent_optimizer_count']+r['sweep']*a.inner_updates
                previous_count = 0 if index==1 else json.loads((root/f'state_{index-1:04d}'/'result.json').read_text())['optimizer_count']
                assert r['parent_optimizer_count']==previous_count
                out = dict(index=index, load=float(load), sweep=r['sweep'], final=final,
                    reaction=r['reaction'], inner_AT2=r['inner_AT2'], optimizer_count=r['optimizer_count'],
                    host_force_norm=force_norm, pass_replay=True, report_sha256=sha(rp))
                if final:
                    np.testing.assert_allclose(image(folder/'network.eqx', f['H']), f['raw_current_image'], rtol=1e-9, atol=1e-11)
                    np.testing.assert_allclose(residual(f['d'], f['H'], D, g0, g1), r['physical_AT2'], rtol=1e-9, atol=1e-12)
                    assert r['certified_path'] is False
                    match = [x for x in refs if abs(x['load']-load)<1e-14]
                    if len(match)!=1: raise ValueError(f'no unique matched-load reference at {load}')
                    refpath = refroot/f"state_{match[0]['index']:04d}"/'fields.npz'
                    ref = read_fields(refpath)['d']
                    out.update(physical_AT2=r['physical_AT2'], field_stable=r['field_stable'],
                        matched_reference_reaction=match[0]['reaction'], reference_fields_sha256=sha(refpath),
                        matched_load_damage_L2=float(np.sqrt(m.volumes@(f['d']-ref)**2/(m.volumes@ref**2))),
                        damage_RMSE=float(np.sqrt(m.volumes@(f['d']-ref)**2/m.volumes.sum())))
                    if any(abs(load-x)<1e-14 for x in a.oracle_loads):
                        A = diags(g0+2*f['H'])+g1*D
                        WA = diags(m.volumes)@A; rhs=m.volumes*2*f['H']
                        exact, status = cg(WA, rhs, rtol=1e-10, atol=0., maxiter=10000,
                            M=diags(1/WA.diagonal()))
                        if status!=0 or residual(exact, f['H'], D, g0, g1)>1e-7:
                            raise RuntimeError('evaluation-only same-H oracle failed')
                        out['same_H_damage_L2']=float(np.sqrt(m.volumes@(f['d']-exact)**2/(m.volumes@exact**2)))
                outputs.append(out);done.add(str(rp))
                print(json.dumps(dict(event='cold_full_CPU_replay', **out)), flush=True)
        finished = (root/'result.json').exists() and json.loads((root/'result.json').read_text())['complete']
        complete = finished and sum(x['final'] for x in outputs)==len(grid)
        atomic_json(review/'audit.json', dict(complete=complete, pass_replay=True, rows=outputs,
            common_initialization_checked=True, zero_checked=True, no_sampling=True,
            reference_is_evaluation_only=True, auditor_sha256=sha(__file__)))
        if complete or not watch: return
        if (suite/'exit.json').exists(): raise RuntimeError('suite exited with incomplete audit')
        time.sleep(10)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    main(Path(args.root),watch=args.watch)
