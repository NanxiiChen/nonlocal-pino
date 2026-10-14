"""Common-parent all-node L1/L16 optimizer controls at the first .0053 H."""
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

from examples.fracture2d.at2_cg import damage_solver
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.damage_norm import make_loss_fn,eligible_l16
from examples.fracture2d.full_damage_optimization import adam_from_config,make_lbfgs
from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.low_order_damage import make_step,optimizer_count
from examples.fracture2d.output_damage_continue import atomic_fields
from examples.fracture2d.staggered_damage_trial import prepare,read_fields
from examples.fracture2d.staggered_damage_audit import residual


def prepare_trial(a):
    t=a.norm_trial
    if (jax.devices()[0].platform!='gpu' or not jax.config.jax_enable_x64
            or t.adam_initialization!='fresh_identical_zero_moments' or t.sampling!='all_nodes'
            or t.oracle_evaluation_only is not True or t.physical_state_promoted is not False
            or type(t.updates) is not int or t.updates<1
            or t.selection!='L16_field_L2_and_max_residual_no_worse_than_L1_then_min_field_L2'):
        raise ValueError('norm comparison requires the declared all-node frozen-H GPU protocol')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    p=prepare(a);source=Path(t.screen_source);repo=Path(__file__).resolve().parents[2]
    for name,digest in t.screen_pins.items():
        if sha(source/name)!=digest:raise ValueError(f'norm history source changed: {name}')
    old=json.loads((source/'identity.json').read_text())
    for name,digest in old['code'].items():
        if sha(repo/name)!=digest:raise ValueError(f'protected full-domain source changed: {name}')
    if old['source_config']!=p['c'].to_dict() or old['geometry']!=p['identity']['geometry']:
        raise ValueError('norm screen model/geometry differs')
    audit=json.loads((source/'independent_review/audit.json').read_text())
    if not audit['complete'] or not audit['pass_replay']:raise ValueError('norm H source unaudited')
    frozen=read_fields(source/'screen/adam/fields.npz');f=p['fields']
    np.testing.assert_array_equal(frozen['H_parent'],f['H'])
    np.testing.assert_array_equal(frozen['d_parent'],f['d'])
    np.testing.assert_array_equal(frozen['H'],np.maximum(f['H'],p['m'].driver(frozen['q_mechanics'])))
    p['frozen']=frozen
    return p


def objectives(a,p):
    t=a.norm_trial;scales=read_fields(Path(a.source_root)/'parameterization.npz')
    optimizer,schedule=adam_from_config(p['c'].l1_loading)
    memory=optimizer.init(eqx.filter(p['net'],eqx.is_inexact_array))
    if optimizer_count(memory,kind='adam')!=0:raise RuntimeError('fresh Adam count must be zero')
    tools={}
    for norm in ['l1','l16']:
        image,loss=make_loss_fn(gc0=p['g0'],gc1=p['g1'],gain=float(scales['trunk_gain']),
            population_size=p['m'].n_material,norm=norm,power=t.power,floor=t.floor)
        tools[norm]=dict(image=eqx.filter_jit(image),value=eqx.filter_jit(loss),
            step=make_step(loss,optimizer),loss=loss)
    return tools,optimizer,schedule,memory


def metrics(p,d,H,oracle):
    raw=(p['g0']+2*H)*d+p['g1']*(p['D']@d)-2*H
    rho=raw/(2*np.sqrt(np.mean(H**2)));delta=d-oracle;v=p['m'].volumes
    return dict(uniform_L1=float(np.mean(abs(rho))),AT2_relative=residual(d,H,p['D'],p['g0'],p['g1']),
        absolute_residual_max=float(abs(raw).max()),same_H_damage_L2=float(np.sqrt(v@delta**2/(v@oracle**2))),
        damage_RMSE=float(np.sqrt(v@delta**2/v.sum())),damage_min=float(d.min()),damage_max=float(d.max()))


def protected(a,code):
    for source,pins in [(Path(a.source_root),a.pins),(Path(a.norm_trial.screen_source),a.norm_trial.screen_pins)]:
        for name,digest in pins.items():
            if sha(source/name)!=digest:raise RuntimeError(f'norm protected input changed: {name}')
    repo=Path(__file__).resolve().parents[2]
    for name,digest in code.items():
        if sha(repo/name)!=digest:raise RuntimeError(f'norm source changed: {name}')


def screen(a,config_path):
    p=prepare_trial(a);t=a.norm_trial;root=Path(a.output_dir)
    if (root/'identity.json').exists():raise ValueError('refusing existing norm screen')
    root.mkdir(parents=True,exist_ok=True)
    H=jnp.asarray(p['frozen']['H']);context=p['full'](H)
    tools,optimizer,schedule,initial_memory=objectives(a,p)
    net=p['net'];image=tools['l1']['image']
    np.testing.assert_allclose(image(net,p['full'](jnp.asarray(p['fields']['H_used']))),
        p['fields']['damage_image'],rtol=1e-10,atol=1e-12)
    cases=[ConfigDict(x) for x in t.cases]
    if {x.name for x in cases}!={'l1_adam','l16_adam','l16_lbfgs'}:raise ValueError('norm cases differ')
    fit,lbfgs_grad=make_lbfgs(net,tools['l16']['loss'],t.lbfgs)
    for norm in tools:jax.block_until_ready(tools[norm]['step'](net,initial_memory,context,jnp.asarray(a.load)))
    from jax.flatten_util import ravel_pytree
    flat,_=ravel_pytree(eqx.filter(net,eqx.is_inexact_array))
    jax.block_until_ready(lbfgs_grad(flat,context,jnp.asarray(a.load)))
    initial=np.asarray(image(net,context));p['m'].factors(initial)
    solve=damage_solver(p['op'],p['m'].volumes,p['g0'],p['g1'],tol=p['c'].validation.cg_rtol,
        atol=p['c'].validation.cg_atol,maxiter=p['c'].validation.cg_maxiter)
    oracle=np.asarray(solve(H,jnp.asarray(initial)));p['m'].factors(oracle)
    atomic_fields(root/'initial.npz',**dict(p['frozen'],H=np.asarray(H),damage_image=initial,oracle=oracle))
    repo=Path(__file__).resolve().parents[2];code={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and (path.is_relative_to(repo/'examples') or path.is_relative_to(repo/'src')):
                code[str(path.relative_to(repo))]=sha(path)
    code[str(Path(config_path).resolve().relative_to(repo))]=sha(config_path)
    for name in code:
        target=root/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
    atomic_json(root/'identity.json',dict(config=a.to_dict(),source_config=p['c'].to_dict(),geometry=p['identity']['geometry'],
        code=code,all_node_count=p['m'].n_material,initial_optimizer_count=0,parent_optimizer_count=a.parent_count,
        history_fixed=True,reference_training_targets=False,physical_state_promoted=False))
    initial_row=metrics(p,initial,np.asarray(H),oracle)
    atomic_json(root/'preflight.json',dict(pass_restore=True,initial=initial_row,initial_sha256=sha(root/'initial.npz')))
    rows=[]
    def save(case,candidate,memory,folder,optimization):
        d=np.asarray(image(candidate,context));p['m'].factors(d)
        folder.mkdir(parents=True,exist_ok=False)
        atomic_fields(folder/'fields.npz',damage_image=d,H=np.asarray(H))
        eqx.tree_serialise_leaves(folder/'network.eqx',candidate)
        row=dict(case=case.name,norm=case.norm,optimizer=case.optimizer,load=a.load,
            **metrics(p,d,np.asarray(H),oracle),objective=float(tools[case.norm]['value'](candidate,context,jnp.asarray(a.load))),
            **optimization,fields_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'),
            physical_state_promoted=False,history_fixed=True,optimizer_count=None,optimizer_sha256=None)
        if case.optimizer=='adam':
            eqx.tree_serialise_leaves(folder/'optimizer.eqx',memory)
            row.update(optimizer_count=optimizer_count(memory,kind='adam'),optimizer_sha256=sha(folder/'optimizer.eqx'))
        atomic_json(folder/'report.json',row)
        atomic_json(root/'result.json',dict(complete=False,phase='screen',current=row,cases=rows,initial=initial_row))
        print(json.dumps(dict(event='norm_screen',**row)),flush=True)
        return row
    for case in cases:
        candidate,memory=net,initial_memory;started=time.perf_counter()
        if case.optimizer=='adam':
            for update in range(1,t.updates+1):
                candidate,memory,value,gradient=tools[case.norm]['step'](candidate,memory,context,jnp.asarray(a.load))
                if update in t.checkpoints:
                    if not np.isfinite([float(value),float(gradient)]).all():raise FloatingPointError('nonfinite norm Adam fit')
                    if optimizer_count(memory,kind='adam')!=update:raise RuntimeError('norm Adam state reset')
                    row=save(case,candidate,memory,root/'screen'/case.name/f'update_{update:05d}',
                        dict(iterations=update,evaluations=update,seconds=time.perf_counter()-started,
                            optimizer_success=True,status='declared_budget',learning_rate=float(schedule(update-1))))
        elif case.optimizer=='lbfgs':
            candidate,optimization=fit(net,context,jnp.asarray(a.load))
            optimization.pop('optimizer');optimization.pop('loss')
            row=save(case,candidate,None,root/'screen'/case.name/'final',dict(optimization,learning_rate=None))
        else:raise ValueError('unknown norm optimizer')
        rows.append(row)
    selected=eligible_l16(rows)
    atomic_json(root/'selection.json',dict(selected=selected,criterion=t.selection,rows=rows,
        coupled_initial_state='original .0052 network/physical state, screening weights discarded'))
    protected(a,code)
    atomic_json(root/'result.json',dict(complete=True,phase='screen',cases=rows,initial=initial_row,
        selected=selected,physical_state_promoted=False,load_advanced=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    screen(runpy.run_path(args.config)['get_config'](),args.config)
