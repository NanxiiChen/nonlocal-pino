"""Restore an audited fine prefix and complete longer all-node PINN loading."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import diags, load_npz

from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.core_halo_damage import pack_rows
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.full_damage_loading import common_config, load_grid
from examples.fracture2d._legacy.full_damage_optimization import adam_from_config
from examples.fracture2d._legacy.l1_damage_loading import committed_history, residual_metrics
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.low_order_damage import make_loss_fn, make_step, optimizer_count
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.provisional_damage import mesh_adjacency, crack_extent
from examples.fracture2d._legacy.staggered_damage import mechanics_first_sweep, field_changes, field_stop
from examples.fracture2d._legacy.staggered_damage_audit import read_fields


def targets(a):
    values = [a.source_load, a.load_step, a.transition_load, a.end_load, a.learning_rate,
              a.damage_change_limit, a.displacement_change_limit]
    if (not np.isfinite(values).all() or min(values) <= 0
            or a.end_load <= a.source_load or a.sampling != 'all_nodes'
            or a.loss_norm != 'l1' or a.optimizer != 'adam' or a.omega != 1.
            or a.symmetry_constraint is not False or a.residual_is_stopping_gate is not False
            or a.advance_on_budget_exit is not True
            or any(type(v) is not int or v < 1 for v in
                (a.source_index, a.source_optimizer_count, a.inner_updates,
                 a.max_outer_sweeps, a.minimum_sweeps, a.consecutive_passes, a.cpu_threads))
            or a.minimum_sweeps > a.max_outer_sweeps
            or any(type(v) is not int or not 1 <= v <= a.max_outer_sweeps for v in a.checkpoints)):
        raise ValueError('invalid explicit best-current full loading controls')
    last = round(a.end_load/a.load_step)
    if (not np.isclose(a.source_index*a.load_step, a.source_load, rtol=0, atol=1e-14)
            or last <= a.source_index
            or not np.isclose(last*a.load_step, a.end_load, rtol=0, atol=1e-14)):
        raise ValueError('fork and endpoint must lie on the declared fine load grid')
    full_grid=load_grid(ConfigDict(dict(pre_step=a.load_step,post_step=a.load_step,
        transition_load=a.transition_load,end_load=a.end_load)))
    return full_grid[a.source_index:]


def check_restored_fields(f, report, a, *, material_nodes, displacement_dofs):
    """Fail before training if a source is not the declared preceding-load state."""
    if (report['index'] != a.source_index or abs(report['load']-a.source_load)>1e-14
            or report['optimizer_count'] != a.source_optimizer_count
            or report['q_d_synchronized'] is not True):
        raise ValueError('preceding-load source index/load/count/synchronization differs')
    shapes = dict(q=(displacement_dofs,), d=(material_nodes,), H=(material_nodes,),
                  H_used=(material_nodes,), damage_image=(material_nodes,))
    for key, shape in shapes.items():
        if f[key].shape != shape or f[key].dtype != np.float64 or not np.isfinite(f[key]).all():
            raise ValueError(f'invalid restored source field {key}')
    if min(f['d'])<0 or max(f['d'])>1 or min(f['H'])<0 or max(f['H'])<=0:
        raise ValueError('source damage/history outside physical ranges')
    np.testing.assert_array_equal(f['d'], f['damage_image'])


def run(a, config_path):
    grid = targets(a)
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise ValueError('best-current loading requires serialized float64 GPU execution')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    suite = Path(a.output_dir); root = suite/'fine'; source = Path(a.source_root)
    if (root/'identity.json').exists(): raise ValueError('refusing existing best-current loading identity')
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in a.source_pins.items():
        if sha(source/name)!=digest: raise ValueError(f'fine-prefix source changed: {name}')
    origin = json.loads((source/'identity.json').read_text())
    audit = json.loads((source/'independent_review/audit.json').read_text())
    if not audit['complete'] or not audit['pass_replay'] or not origin['own_zero']:
        raise ValueError('full native source audit and own-zero prefix required')
    repo = Path(__file__).resolve().parents[3]
    for name, digest in origin['code'].items():
        if sha(repo/name)!=digest: raise ValueError(f'original numerical module changed: {name}')
    c = ConfigDict(origin['config'], allow_dotted_keys=True); t = c.l1_loading
    if (t.loss_norm!='l1' or t.optimizer!='adam' or t.learning_rate!=a.learning_rate
            or t.decay_rate!=1. or t.minimum_rate!=a.learning_rate
            or t.output_head!='sigmoid' or c.stagger.omega!=1.
            or c.full_loading.pre_step!=a.load_step or c.full_loading.post_step!=a.load_step
            or c.full_loading.transition_load!=a.transition_load):
        raise ValueError('source and continuation model/loss/optimizer/load policy differ')
    m, op, geometry = load_geometry(c)
    if geometry!=origin['geometry'] or sha(t.basis_path)!=t.basis_sha256:
        raise ValueError('source geometry or compact basis differs')
    B, D = load_npz(t.basis_path), op.to_scipy()
    scale = read_fields(source/'parameterization.npz')
    if (B.shape[0]!=m.n_material or scale['column_scale'].shape!=(B.shape[1],)
            or not np.isfinite(scale['column_scale']).all() or min(scale['column_scale'])<=0
            or float(scale['trunk_gain'])!=t.trunk_gain):
        raise ValueError('restored basis ordering/scales differ')
    parent_path = source/f'state_{a.source_index:04d}'
    f = read_fields(parent_path/'fields.npz')
    parent_report = json.loads((parent_path/'result.json').read_text())
    check_restored_fields(f, parent_report, a, material_nodes=m.n_material,
                          displacement_dofs=len(m.load_direction))
    g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    features, info = make_features(op, m.coordinates[:m.n_material], gc0=g0,
        length=c.physical.l0, tip=list(c.conditioned.tip), powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    if info!=origin['features']:raise ValueError('restored feature definition differs')
    features = eqx.filter_jit(features)
    net = eqx.tree_deserialise_leaves(parent_path/'network.eqx',
        fresh_network(c, info['input_dimension'], B, head='sigmoid'))
    optimizer, schedule = adam_from_config(t)
    memory = eqx.tree_deserialise_leaves(parent_path/'optimizer.eqx',
        optimizer.init(eqx.filter(net, eqx.is_inexact_array)))
    if optimizer_count(memory, kind='adam')!=a.source_optimizer_count:
        raise ValueError('restored Adam count differs')
    bc, bv = pack_rows(B@diags(scale['column_scale']), width=int(np.max(np.diff(B.indptr))))
    dc, dv = pack_rows(D, width=int(np.max(np.diff(D.indptr))))
    bc, bv, dc, dv = map(jnp.asarray, (bc, bv, dc, dv))
    local = jnp.arange(m.n_material, dtype=jnp.int32)
    probabilities = jnp.full(m.n_material, 1/m.n_material, jnp.float64)
    def context(H):
        H = jax.lax.stop_gradient(H)
        return dict(H=H, core_H=H, features=jax.lax.stop_gradient(features(H)),
            basis_columns=bc, basis_values=bv, core_local=local,
            operator_columns=dc, operator_values=dv, p_core=probabilities)
    image, objective = make_loss_fn(gc0=g0, gc1=g1, gain=t.trunk_gain,
        population_size=m.n_material, norm='l1')
    image, value = eqx.filter_jit(image), eqx.filter_jit(objective)
    step = make_step(objective, optimizer)
    np.testing.assert_allclose(image(net, context(jnp.asarray(f['H_used']))), f['damage_image'],
        rtol=1e-10, atol=1e-12)
    solve = m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                         maxiter=c.validation.cg_maxiter)
    warm_q, _, mechanical = solve(jnp.asarray(f['d']), float(grid[0]), jnp.asarray(f['q']))
    first_H = committed_history(jnp.asarray(f['H']), jnp.asarray(m.driver(np.asarray(warm_q))))
    ctx = context(first_H)
    raw = (g0+2*np.asarray(first_H))*np.asarray(image(net, ctx))+g1*(D@np.asarray(image(net,ctx)))-2*np.asarray(first_H)
    expected = float(abs(raw).mean()/(2*np.sqrt(np.mean(np.asarray(first_H)**2))))
    np.testing.assert_allclose(value(net,ctx,jnp.asarray(grid[0])),expected,rtol=1e-10,atol=1e-12)
    jax.block_until_ready(step(net,memory,ctx,jnp.asarray(grid[0])))  # compilation only; discard update
    if optimizer_count(memory,kind='adam')!=a.source_optimizer_count:
        raise RuntimeError('preflight changed the restored Adam state')
    for name in ('parameterization.npz','initial_network.eqx','initial_optimizer.eqx'):
        shutil.copyfile(source/name,root/name)
    curve=[];prefix=[]
    old_curve=json.loads((source/'result.json').read_text())['curve']
    for index in range(a.source_index+1):
        old=source/f'state_{index:04d}';new=root/f'state_{index:04d}';new.mkdir()
        names=['fields.npz','result.json'] if index==0 else ['fields.npz','network.eqx','optimizer.eqx','report.json','result.json']
        record=dict(index=index,source=str(old),files={})
        for name in names:
            shutil.copyfile(old/name,new/name)
            digest=sha(old/name)
            if sha(new/name)!=digest:raise RuntimeError('copied prefix differs')
            record['files'][name]=digest
        prefix.append(record)
        if index:
            row=old_curve[index-1]
            if row['index']!=index or abs(row['load']-index*a.load_step)>1e-14 or sha(new/'fields.npz')!=row['fields_sha256']:
                raise ValueError('inherited prefix load/field ancestry differs')
            curve.append(dict(row,inherited_prefix=True,inherited_source=str(old)))
    # Effective policy enables the existing independent cold-loading auditor to
    # replay the complete inherited+new trajectory without altering its source.
    full=c.full_loading
    full.suite_root=str(suite);full.end_load=a.end_load;full.inner_updates=a.inner_updates
    full.prefix_outer_cap=a.max_outer_sweeps;full.critical_outer_cap=a.max_outer_sweeps
    full.damage_change_limit=a.damage_change_limit;full.displacement_change_limit=a.displacement_change_limit
    full.minimum_sweeps=a.minimum_sweeps;full.consecutive_passes=a.consecutive_passes
    full.checkpoint_sweeps=list(a.checkpoints);full.oracle_loads=list(a.oracle_loads)
    init=dict(common_config=common_config(c),network_sha256=sha(root/'initial_network.eqx'),
        optimizer_sha256=sha(root/'initial_optimizer.eqx'))
    atomic_json(suite/'common_initialization.json',init)
    code={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and (path.is_relative_to(repo/'examples') or path.is_relative_to(repo/'src')):
                code[str(path.relative_to(repo))]=sha(path)
    code[str(Path(config_path).resolve().relative_to(repo))]=sha(config_path)
    for name in ('examples/fracture2d/_legacy/full_damage_loading_audit.py',
                 'examples/fracture2d/_legacy/best_full_damage_loading_audit.py'):
        code[name]=sha(repo/name)
    for name in code:
        path=root/'code_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,path)
    protected=dict(origin['source'])
    protected.update({str(source/name):digest for name,digest in a.source_pins.items()})
    atomic_json(root/'identity.json',dict(schema='best-current-fine-prefix-loading-v1',
        config=c.to_dict(),driver_config=a.to_dict(),source_config=origin['config'],
        loads=load_grid(full).tolist(),
        geometry=geometry,features=info,code=code,source=protected,common_initialization=init,
        sampling='all_nodes',full_node_count=m.n_material,own_zero=True,fresh_network=False,
        reference_training_targets=False,symmetry_constraint=False,optimizer_persistent=True,
        residual_is_stopping_gate=False,provisional_load_advancement=True,
        source_index=a.source_index,source_optimizer_count=a.source_optimizer_count,
        preflight_mechanics_calls=1,inherited_prefix_count=len(curve)))
    atomic_json(root/'prefix_receipt.json',dict(source_root=str(source),rows=prefix,
        source_audit_sha256=a.source_pins['independent_review/audit.json']))
    atomic_json(root/'preflight.json',dict(pass_restore=True,source_image_parity=True,
        full_loss_CSR_parity=True,gpu_step_compiled=True,optimizer_count=a.source_optimizer_count,
        source_mechanics=mechanical,inherited_prefix=True))
    q,d,H=map(jnp.asarray,(f['q'],f['d'],f['H']))
    carry=(net,memory);count=a.source_optimizer_count;started=time.perf_counter()
    tri,area=m.triangles[:m.n_material_cells],m.areas[:m.n_material_cells]
    adjacency=mesh_adjacency(tri,population_size=m.n_material)

    def train(history,previous_damage,state):
        del previous_damage
        candidate,mem=state;ctx=context(history);tic=time.perf_counter()
        for _ in range(a.inner_updates):
            candidate,mem,loss,grad=step(candidate,mem,ctx,jnp.asarray(load))
        output=image(candidate,ctx);output.block_until_ready();m.factors(np.asarray(output))
        if not np.isfinite([float(loss),float(grad)]).all():raise FloatingPointError('nonfinite Adam loss/gradient')
        raw,relative=residual_metrics(np.asarray(output),np.asarray(history),D,gc0=g0,gc1=g1)
        mae=float(abs(raw).mean()/(2*np.sqrt(np.mean(np.asarray(history)**2))))
        return output,(candidate,mem),dict(full_loss=mae,AT2_relative=relative,
            iterations=a.inner_updates,seconds=time.perf_counter()-tic)

    def checkpoint(folder,fields,row,state):
        folder.mkdir(exist_ok=True);atomic_fields(folder/'fields.npz',**fields)
        eqx.tree_serialise_leaves(folder/'network.eqx',state[0])
        eqx.tree_serialise_leaves(folder/'optimizer.eqx',state[1])
        row.update(fields_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'),
            optimizer_sha256=sha(folder/'optimizer.eqx'))
        atomic_json(folder/'report.json',row)

    for index,target in enumerate(grid,a.source_index+1):
        load=float(target);parent_d,parent_H=d,H
        folder=root/f'state_{index:04d}';folder.mkdir()
        parent=root/f'state_{index-1:04d}'/'fields.npz';parent_count=count;consecutive=0
        load_started=time.perf_counter()
        for sweep in range(1,a.max_outer_sweeps+1):
            tic=time.perf_counter()
            fields,carry=mechanics_first_sweep(q,d,parent_H,load,carry,
                mechanics=solve,driver=m.driver,damage=train,omega=1.)
            q,d,H=(fields[k] for k in ('q','d','H'));count+=a.inner_updates
            if optimizer_count(carry[1],kind='adam')!=count:raise RuntimeError('Adam state/count reset')
            changes=field_changes(q,fields['q_previous'],d,fields['d_previous'],tri,area,load=load)
            stop=field_stop(changes,displacement_limit=a.displacement_change_limit,
                damage_limit=a.damage_change_limit,sweep=sweep,minimum_sweeps=a.minimum_sweeps,
                consecutive_passes=a.consecutive_passes,previous_consecutive=consecutive)
            consecutive=stop['consecutive']
            row=dict(index=index,load=load,sweep=sweep,outer_cap=a.max_outer_sweeps,**changes,
                field_stopping=stop,field_stable=stop['converged'],reaction=fields['reaction'],
                q_d_synchronized=False,mechanics_calls=sweep,mechanics=fields['mechanics'],
                damage_min=float(d.min()),damage_max=float(d.max()),inner_AT2=fields['inner']['AT2_relative'],
                full_loss=fields['inner']['full_loss'],training_updates=sweep*a.inner_updates,
                optimizer_count=count,parent_optimizer_count=parent_count,learning_rate=float(schedule(count-1)),
                parent_sha256=sha(parent),seconds=time.perf_counter()-load_started,
                sweep_seconds=time.perf_counter()-tic,training_seconds=fields['inner']['seconds'],
                full_node_count=m.n_material)
            saved=dict(q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),H_used=np.asarray(H),
                damage_image=np.asarray(fields['damage_image']),q_previous=np.asarray(fields['q_previous']),
                d_previous=np.asarray(fields['d_previous']),H_parent=np.asarray(parent_H),d_parent=np.asarray(parent_d))
            atomic_json(folder/f'sweep_{sweep:04d}.json',row)
            dest=folder/f'outer_{sweep:04d}' if sweep in a.checkpoints or stop['converged'] else folder/'pending'
            checkpoint(dest,saved,dict(row),carry)
            atomic_json(root/'result.json',dict(complete=False,current=row,curve=curve,
                optimizer_count=count,end_load=a.end_load,inherited_prefix_count=a.source_index))
            if sweep==1 or sweep%8==0:print(json.dumps(dict(event='best_full_outer',**row)),flush=True)
            if stop['converged']:break
        q_sync,reaction,mechanical=solve(d,load,q)
        H_sync=committed_history(parent_H,jnp.asarray(m.driver(np.asarray(q_sync))))
        force=m.matrix(np.asarray(d))@np.asarray(q_sync);norm=float(np.linalg.norm(force[m.free]))
        if norm>mechanical['convergence_limit']*(1+1e-3):raise RuntimeError('final host FEM residual fails')
        np.testing.assert_allclose(force@m.load_direction,reaction,rtol=1e-9,atol=1e-6)
        physical=residual_metrics(np.asarray(d),np.asarray(H_sync),D,gc0=g0,gc1=g1)[1]
        saved.update(q_stage=np.asarray(q),q=np.asarray(q_sync),H=np.asarray(H_sync),
            raw_current_image=np.asarray(image(carry[0],context(H_sync))))
        report=dict(row,reaction=float(reaction),q_d_synchronized=True,mechanics_calls=sweep+1,
            mechanics=mechanical,host_force_norm=norm,physical_AT2=physical,
            original_residual_pass=bool(physical<=c.reporting.coupled_AT2_relative_limit),
            certified_path=False,status='provisional_field_stable' if stop['converged'] else 'provisional_budget_exhausted',
            healing=float(jnp.maximum(parent_d-d,0).max()),history_decrease=float(jnp.maximum(parent_H-H_sync,0).max()),
            seconds=time.perf_counter()-load_started,
            crack=crack_extent(np.asarray(d),m.coordinates[:m.n_material],adjacency,
                threshold=.9,tip=[.5,.5],tip_radius=c.physical.l0,right_edge=1.,tolerance=1e-12))
        checkpoint(folder,saved,report,carry);atomic_json(folder/'result.json',report)
        curve.append(report);q,H=q_sync,H_sync
        atomic_json(root/'result.json',dict(complete=False,current=report,curve=curve,
            optimizer_count=count,end_load=a.end_load,inherited_prefix_count=a.source_index))
        print(json.dumps(dict(event='best_full_load_complete',**report)),flush=True)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError(f'protected source changed: {path}')
    for name,digest in code.items():
        if sha(repo/name)!=digest:raise RuntimeError(f'numerical source changed: {name}')
    atomic_json(root/'result.json',dict(complete=True,current=curve[-1],curve=curve,
        optimizer_count=count,end_load=a.end_load,inherited_prefix_count=a.source_index,
        seconds=time.perf_counter()-started,certified_path=False,own_zero=True,sampling='all_nodes',loss_norm='l1'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    run(runpy.run_path(args.config)['get_config'](),args.config)
