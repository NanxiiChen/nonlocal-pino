"""Explicit new-run imports of complete channel model and optimizer state."""
from pathlib import Path
import json
import equinox as eqx
import jax
import numpy as np


def continuation_spec(c):
    if 'continuation' not in c.train:return None
    setting=c.train.continuation
    if set(setting)!={'parent_run','kind'} or setting.kind not in ('strict','refine','operator','time_refine'):
        raise ValueError('Continuation requires explicit parent_run and kind=strict, refine, operator or time_refine')
    if any(key in c for key in ('balance','stagger')) or 'online' in c.operator:
        raise ValueError('Channel continuation currently supports uniform finite-bank optimizer state only')
    parent=Path(setting.parent_run)
    if not parent.is_absolute() or not parent.is_dir():raise ValueError('Continuation parent_run must be an existing absolute directory')
    for name in ('config.json','summary.json','model_final.eqx','optimizer_final.eqx','prediction.npz','validation.json'):
        if not (parent/name).is_file():raise FileNotFoundError(parent/name)
    saved=json.loads((parent/'config.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    audit=json.loads((parent/'validation.json').read_text())
    if summary['status']!='completed' or audit['status']!='replay_and_hard_constraints_passed':
        raise ValueError('Continuation requires a complete independently replayed parent')
    start=summary['iterations']
    if summary['metrics']['step']!=start or saved['train']['iterations']!=start:
        raise ValueError('Parent completed-step metadata disagrees')
    if c.train.iterations<start+2:raise ValueError('Continuation requires at least two new updates')
    current=json.loads(c.to_json());original=json.loads(json.dumps(saved))
    before=[original['sample']['Nx'],original['sample']['Ny']]
    after=[current['sample']['Nx'],current['sample']['Ny']]
    operator_change=None
    time_change=None
    if setting.kind=='time_refine':
        if 'time_discretization' not in original or 'time_discretization' not in current:
            raise ValueError('Time refinement requires explicit BDF2 time discretization in both runs')
        old=original['time_discretization'];new=current['time_discretization']
        old_dt=old['dt'];new_dt=new['dt']
        if (old['kind']!='bdf2' or new['kind']!='bdf2'
                or isinstance(new_dt,bool) or not np.isfinite(new_dt) or new_dt<=0
                or new_dt>=old_dt or abs(old_dt/new_dt-round(old_dt/new_dt))>1e-10):
            raise ValueError('Time refinement requires a strictly smaller positive integer subdivision of BDF2 dt')
        time_change=dict(before=old.copy(),after=new.copy())
        del old['dt'];del new['dt']
    if setting.kind=='operator':
        old=original['nl'].copy();new=current['nl'].copy()
        if old['method']!='gmls' or new['method']!='gmls':raise ValueError('Operator refinement requires the same explicit GMLS method')
        if any(isinstance(new[k],bool) or not isinstance(new[k],int) for k in ('degree','neighbors')):
            raise ValueError('Operator degree and neighbors must be explicit integers')
        if new['degree']<=old['degree'] or new['neighbors']<old['neighbors'] or not np.isfinite(new['weight_exponent']) or new['weight_exponent']<=0:
            raise ValueError('Operator refinement requires higher polynomial degree, no fewer neighbors and positive finite weights')
        operator_change=dict(before=old,after=new)
        for key in ('degree','neighbors','weight_exponent'):del original['nl'][key];del current['nl'][key]
    if setting.kind=='refine':
        if before==after or any(a<b or a%b for a,b in zip(after,before,strict=True)):
            raise ValueError('Refinement requires integer spatial refinements and at least one larger axis')
        for key in ('Nx','Ny'):del original['sample'][key];del current['sample'][key]
    for config in (original,current):
        del config['log'];del config['train']['iterations']
        if 'continuation' in config['train']:del config['train']['continuation']
    if original!=current:raise ValueError('Continuation numerical configuration differs outside the explicit refinement')
    elapsed=summary['cumulative_wall_seconds'] if 'cumulative_wall_seconds' in summary else summary['wall_seconds']
    result=dict(parent_run=str(parent),kind=setting.kind,start_step=start,
        model_path=str(parent/'model_final.eqx'),optimizer_path=str(parent/'optimizer_final.eqx'),
        prediction_path=str(parent/'prediction.npz'),parent_config=saved,
        parent_cumulative_wall_seconds=elapsed,resolution_before=before,resolution_after=after)
    if operator_change is not None:result['operator_refinement']=operator_change
    if time_change is not None:result['time_refinement']=time_change
    return result


def restore_state(model,optimizer,spec):
    if spec is None:return model,optimizer.init(eqx.filter(model,eqx.is_array)),0
    restored=eqx.tree_deserialise_leaves(spec['model_path'],model)
    state=eqx.tree_deserialise_leaves(spec['optimizer_path'],optimizer.init(eqx.filter(model,eqx.is_array)))
    for leaf in jax.tree.leaves((restored,state)):
        if eqx.is_array(leaf) and not np.isfinite(np.asarray(leaf)).all():
            raise FloatingPointError('Imported model or complete optimizer state is nonfinite')
    return restored,state,spec['start_step']
