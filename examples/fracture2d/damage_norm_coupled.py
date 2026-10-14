"""L16/full/L-BFGS stagger for coupled crack-path morphology evaluation."""
import argparse,json,os,runpy,shutil,sys,time
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.damage_norm import make_loss_fn
from examples.fracture2d.damage_norm_trial import prepare_trial,protected
from examples.fracture2d.full_damage_optimization import make_lbfgs
from examples.fracture2d.l1_damage_loading import committed_history,residual_metrics
from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.output_damage_continue import atomic_fields
from examples.fracture2d.provisional_damage import mesh_adjacency,crack_extent
from examples.fracture2d.staggered_damage import case_configs,mechanics_first_sweep,field_changes,field_stop


def run(a,config_path):
    t=a.coupled_training;case,=case_configs(a.cases)
    if (t.norm!='l16' or t.optimizer!='lbfgs' or t.sampling!='all_nodes' or t.stop_early is not False
            or t.selection!='user_crack_morphology_priority' or case.omega!=1. or case.solver!='pinn'
            or t.network_initialization!='original_preceding_load_parent' or t.inner_updates!=t.lbfgs.maxiter):
        raise ValueError('coupled norm control needs explicit morphology-first unrelaxed L16 protocol')
    for name,digest in t.screen_pins.items():
        if sha(Path(t.screen_root)/name)!=digest:raise ValueError(f'completed screen changed: {name}')
    review=json.loads((Path(t.screen_root)/'independent_review/audit.json').read_text())
    if not review['complete'] or not review['pass_replay']:raise ValueError('screen independent replay incomplete')
    p=prepare_trial(a);m,c,D,f=(p[k] for k in ['m','c','D','fields'])
    gain=float(np.load(Path(a.source_root)/'parameterization.npz')['trunk_gain'])
    image,objective=make_loss_fn(gc0=p['g0'],gc1=p['g1'],gain=gain,population_size=m.n_material,
        norm=t.norm,power=t.power,floor=t.floor)
    image,value=eqx.filter_jit(image),eqx.filter_jit(objective)
    fit,gradient=make_lbfgs(p['net'],objective,t.lbfgs)
    solve=m.gpu_solver(tol=c.validation.cg_rtol,atol=c.validation.cg_atol,maxiter=c.validation.cg_maxiter)
    q0,_,_=solve(jnp.asarray(f['d']),a.load,jnp.asarray(f['q']))
    h0=committed_history(jnp.asarray(f['H']),jnp.asarray(m.driver(np.asarray(q0))))
    np.testing.assert_allclose(h0,p['frozen']['H'],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(image(p['net'],p['full'](jnp.asarray(f['H_used']))),f['damage_image'],rtol=1e-10,atol=1e-12)
    from jax.flatten_util import ravel_pytree
    flat,_=ravel_pytree(eqx.filter(p['net'],eqx.is_inexact_array))
    jax.block_until_ready(gradient(flat,p['full'](h0),jnp.asarray(a.load)))
    root=Path(a.output_dir)
    if (root/'identity.json').exists():raise ValueError('refusing existing coupled norm trial')
    root.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[2];code={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and (path.is_relative_to(repo/'examples') or path.is_relative_to(repo/'src')):code[str(path.relative_to(repo))]=sha(path)
    code[str(Path(config_path).resolve().relative_to(repo))]=sha(config_path)
    for name in code:
        dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,dest)
    atomic_json(root/'identity.json',dict(config=a.to_dict(),source_config=c.to_dict(),geometry=p['identity']['geometry'],
        code=code,model_unchanged=True,full_node_count=m.n_material,loss_norm=t.norm,
        reference_training_targets=False,morphology_primary=True,straightness_constraint=False,
        screen_states_used=False,preflight_mechanics_calls=1,history_floor_load=a.parent_load))
    atomic_json(root/'preflight.json',dict(pass_restore=True,parent_image_parity=True,first_H_parity=True,
        GPU_gradient_compiled=True,full_node_count=m.n_material))
    q,d,parentH=map(jnp.asarray,(f['q'],f['d'],f['H']));net=p['net'];counter=0;updates=evaluations=0
    tri,area=m.triangles[:m.n_material_cells],m.areas[:m.n_material_cells]
    adjacency=mesh_adjacency(tri,population_size=m.n_material);folder=root/case.name;folder.mkdir()
    def damage(H,old_d,previous):
        candidate,optimization=fit(previous,p['full'](H),jnp.asarray(a.load))
        d_new=image(candidate,p['full'](H));d_new.block_until_ready();m.factors(np.asarray(d_new))
        optimization.update(full_loss=float(value(candidate,p['full'](H),jnp.asarray(a.load))),
            AT2_relative=residual_metrics(np.asarray(d_new),np.asarray(H),D,gc0=p['g0'],gc1=p['g1'])[1])
        return d_new,candidate,optimization
    def checkpoint(destination,fields,row,network):
        destination.mkdir(parents=True,exist_ok=True);atomic_fields(destination/'fields.npz',**fields)
        eqx.tree_serialise_leaves(destination/'network.eqx',network)
        row.update(fields_sha256=sha(destination/'fields.npz'),network_sha256=sha(destination/'network.eqx'))
        atomic_json(destination/'report.json',row)
    started=time.perf_counter()
    for sweep in range(1,case.max_sweeps+1):
        fields,net=mechanics_first_sweep(q,d,parentH,a.load,net,mechanics=solve,driver=m.driver,damage=damage,omega=1.)
        q,d,H=(fields[k] for k in ['q','d','H'])
        changes=field_changes(q,fields['q_previous'],d,fields['d_previous'],tri,area,load=a.load)
        stopping=field_stop(changes,displacement_limit=a.displacement_change_limit,damage_limit=a.damage_change_limit,
            sweep=sweep,minimum_sweeps=a.minimum_sweeps,consecutive_passes=a.consecutive_passes,previous_consecutive=counter)
        counter=stopping['consecutive'];updates+=fields['inner']['iterations'];evaluations+=fields['inner']['evaluations']
        row=dict(case=case.name,solver='pinn',optimizer='lbfgs',loss_norm='l16',omega=1.,sweep=sweep,load=a.load,
            **changes,field_stopping=stopping,mechanics_calls=sweep,reaction=fields['reaction'],reaction_damage='d_previous',
            q_d_synchronized=False,inner_AT2=fields['inner']['AT2_relative'],stage_AT2=fields['inner']['AT2_relative'],
            full_loss=fields['inner']['full_loss'],optimization=fields['inner'],training_updates=updates,
            function_evaluations=evaluations,full_node_count=m.n_material,mechanics=fields['mechanics'],
            damage_max=float(d.max()),optimizer_count=None,learning_rate=None,seconds=time.perf_counter()-started)
        saved=dict(q=np.asarray(q),d=np.asarray(d),H=np.asarray(H),H_used=np.asarray(H),
            damage_image=np.asarray(fields['damage_image']),q_previous=np.asarray(fields['q_previous']),
            d_previous=np.asarray(fields['d_previous']),H_parent=f['H'],d_parent=f['d'])
        atomic_json(folder/f'sweep_{sweep:04d}.json',row)
        if sweep in a.checkpoints or sweep==case.max_sweeps:checkpoint(folder/f'outer_{sweep:04d}',saved,dict(row),net)
        atomic_json(root/'result.json',dict(complete=False,current=row,fixed_load=a.load,load_advanced=False))
        if sweep==1 or sweep%8==0:print(json.dumps(dict(event='coupled_L16',**row)),flush=True)
    qsync,reaction,mechanical=solve(d,a.load,q)
    Hsync=committed_history(parentH,jnp.asarray(m.driver(np.asarray(qsync))))
    force=m.matrix(np.asarray(d))@np.asarray(qsync);norm=float(np.linalg.norm(force[m.free]))
    if norm>mechanical['convergence_limit']*(1+1e-3):raise RuntimeError('coupled L16 final host mechanics fail')
    np.testing.assert_allclose(force@m.load_direction,reaction,rtol=1e-9,atol=1e-6)
    saved.update(q_stage=saved['q'],q=np.asarray(qsync),H=np.asarray(Hsync),raw_current_image=np.asarray(image(net,p['full'](Hsync))))
    physical=residual_metrics(np.asarray(d),np.asarray(Hsync),D,gc0=p['g0'],gc1=p['g1'])[1]
    final=dict(row,reaction=float(reaction),reaction_damage='d',q_d_synchronized=True,mechanics=mechanical,
        mechanics_calls=sweep+1,final_sync_calls=1,host_force_norm=norm,physical_AT2=physical,
        original_residual_pass=bool(physical<=c.reporting.coupled_AT2_relative_limit),field_stable=stopping['converged'],
        certified_path=False,load_advanced=False,status='equal_outer_budget_complete',seconds=time.perf_counter()-started,
        crack=crack_extent(np.asarray(d),m.coordinates[:m.n_material],adjacency,threshold=.9,
            tip=[.5,.5],tip_radius=c.physical.l0,right_edge=1.,tolerance=1e-12))
    checkpoint(folder/'final',saved,final,net);atomic_json(folder/'result.json',final)
    protected(a,code)
    for name,digest in t.screen_pins.items():
        if sha(Path(t.screen_root)/name)!=digest:raise RuntimeError('completed norm screen changed')
    atomic_json(root/'result.json',dict(complete=True,current=final,fixed_load=a.load,load_advanced=False))
    print(json.dumps(dict(event='coupled_L16_complete',**final)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    jax.config.update('jax_enable_x64',True);run(runpy.run_path(args.config)['get_config'](),args.config)
