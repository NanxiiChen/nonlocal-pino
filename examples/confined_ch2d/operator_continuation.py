"""Exact state import, strict continuation defaults and declared LR reduction."""
import json
import math
import os
from pathlib import Path
import equinox as eqx
import jax
import numpy as np


def continuation_spec(c):
    """Validate all parent inputs before expensive bank/backend construction."""
    if 'continuation' not in c.train:
        return None
    if c.operator.action!='train':raise ValueError('Continuation is a train-only option')
    unknown=set(c.train.continuation)-{'parent_run','learning_rate_factor','checkpoint'}
    if unknown:raise ValueError(f'Unknown continuation options: {sorted(unknown)}')
    factor=None
    if 'learning_rate_factor' in c.train.continuation:
        factor=c.train.continuation.learning_rate_factor
        if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or not 0<factor<1:
            raise ValueError('Continuation learning_rate_factor must be finite and strictly between 0 and 1')
    parent=Path(c.train.continuation.parent_run)
    if not parent.is_absolute() or not parent.is_dir():
        raise ValueError('Continuation parent_run must be an existing absolute run directory')
    previous=json.loads((parent/'config.json').read_text())
    current=json.loads(c.to_json())
    checkpoint='checkpoint' in c.train.continuation
    online_change='online' in current['operator'] and 'online' not in previous['operator']
    if online_change and not checkpoint:
        raise ValueError('Enabling online inputs requires an explicit paired checkpoint')
    for key in set(current)|set(previous):
        if key=='operator' and online_change:
            same={name:value for name,value in current[key].items() if name!='online'}
            if same!=previous[key]:raise ValueError('Online continuation changes non-sampling operator settings')
            continue
        if key not in ('train','log') and (key not in current or key not in previous or current[key]!=previous[key]):
            raise ValueError(f'Continuation changes numerical configuration: {key}')
    for key in (set(current['train'])|set(previous['train']))-{'iterations','continuation'}:
        if factor is not None and key in ('lr_init','lr_min'):
            if not math.isclose(current['train'][key],factor*previous['train'][key],rel_tol=1e-13,abs_tol=0):
                raise ValueError(f'Continuation must scale train.{key} by the declared learning_rate_factor')
            continue
        if current['train'][key]!=previous['train'][key]:
            raise ValueError(f'Continuation changes optimizer/runtime setting: train.{key}')
    if checkpoint:
        marker=Path(c.train.continuation.checkpoint)
        if not marker.is_absolute() or marker.parent!=parent:
            raise ValueError('Checkpoint must be an explicit absolute marker inside parent_run')
        saved=json.loads(marker.read_text());first=saved['step']
        if isinstance(first,bool) or not isinstance(first,int) or not 0<first<c.train.iterations or first>previous['train']['iterations']:
            raise ValueError('Checkpoint step must precede the requested total budget')
        if saved['kind']!='intermediate_full_state' or saved['config']!='config.json':
            raise ValueError('Require a published paired intermediate checkpoint')
        if marker.name!=f'checkpoint_{first:06d}.json':raise ValueError('Checkpoint marker/step mismatch')
        model_name,state_name=saved['model'],saved['optimizer']
        if (model_name!=f'ckpt_{first:06d}.eqx' or state_name!=f'optimizer_{first:06d}.eqx'):
            raise ValueError('Checkpoint payload names must match its global step')
        rows=[json.loads(line) for line in (parent/'metrics.jsonl').read_text().splitlines(keepends=True)
              if line.endswith('\n')]
        matching=[row for row in rows if row['step']==first]
        if len(matching)!=1:raise ValueError('Checkpoint requires one complete parent evaluation at its step')
        previous_wall=matching[0]['elapsed_seconds']
        if 'continuation' in previous['train']:
            inherited=json.loads((parent/'continuation.json').read_text())
            previous_wall+=inherited['previous_cumulative_wall_seconds']
    else:
        summary=json.loads((parent/'summary.json').read_text())
        first=summary['final_metrics']['step']
        if first!=previous['train']['iterations'] or not isinstance(first,int) or not 0<first<c.train.iterations:
            raise ValueError('Require complete parent and a strictly larger total iteration budget')
        if summary['checkpoint']!='model_final.eqx':raise ValueError('Expected complete fixed-budget parent checkpoint')
        model_name,state_name='model_final.eqx','optimizer_final.eqx'
        previous_wall=summary['cumulative_wall_seconds'] if 'cumulative_wall_seconds' in summary else summary['wall_seconds']
    for name in (model_name,state_name):
        if not (parent/name).is_file():raise FileNotFoundError(parent/name)
    provenance=dict(parent_run=str(parent),model=str(parent/model_name),
        optimizer_state=str(parent/state_name),completed_steps=first,
        total_requested_steps=c.train.iterations,new_updates=c.train.iterations-first,
        previous_cumulative_wall_seconds=previous_wall,
        policy='New run, exact model/optimizer import, same numerical configuration, global RNG counters')
    if checkpoint:
        provenance.update(checkpoint=str(marker),parent_evaluation=matching[0],
            policy='New run, paired intermediate model/SOAP import, global RNG and schedule counters')
        if 'online_state' in saved:
            name=Path(saved['online_state'])
            if name.name!=str(name) or not (parent/name).is_file():raise ValueError('Invalid checkpoint online state path')
            provenance['online_state']=str(parent/name)
        elif 'online' in previous['operator']:
            raise ValueError('Online parent checkpoint must publish its rolling-pool state')
        if online_change:
            if c.operator.online.origin_step!=first:raise ValueError('Online origin must equal imported fixed-bank checkpoint step')
            provenance['sampling_change']='Asynchronous deterministic rolling geometry inputs; physical/numerical settings unchanged'
    if factor is not None:
        provenance['learning_rate_scaling']=dict(factor=factor,
            previous={key:previous['train'][key] for key in ('lr_init','lr_min')},
            current={key:current['train'][key] for key in ('lr_init','lr_min')},
            schedule_counter='Restored from parent; no schedule restart')
        provenance['policy']='New run, exact state import, only declared learning-rate scaling, global RNG counters'
    return provenance


def initial_state(model,optimizer,provenance):
    """Restore the complete numerical state or initialize an explicit fresh run."""
    state=optimizer.init(eqx.filter(model,eqx.is_array))
    if provenance is None:return model,state,0,None
    model=eqx.tree_deserialise_leaves(provenance['model'],model)
    state=eqx.tree_deserialise_leaves(provenance['optimizer_state'],state)
    if not all(np.isfinite(a).all() for tree in (model,state) for a in jax.tree.leaves(tree) if eqx.is_array(a)):
        raise FloatingPointError('Continuation model/optimizer contains nonfinite arrays')
    return model,state,provenance['completed_steps'],provenance


def save_checkpoint(run,model,state,step,*,online_state=None):
    """Publish a paired intermediate snapshot; marker is written last.

    These are intermediate backups, not a completed fixed-budget run. Static
    architecture/config and stateless global RNG counters remain in config.json.
    """
    if isinstance(step,bool) or not isinstance(step,int) or step<1:
        raise ValueError('Checkpoint requires a positive global integer step')
    run=Path(run)
    if not (run/'config.json').is_file():raise FileNotFoundError(run/'config.json')
    model_path=run/f'ckpt_{step:06d}.eqx'
    state_path=run/f'optimizer_{step:06d}.eqx'
    marker=run/f'checkpoint_{step:06d}.json'
    for target in (model_path,state_path,marker):
        if target.exists():raise FileExistsError(target)
    for target,value in ((model_path,model),(state_path,state)):
        temporary=target.with_name(target.name+'.partial')
        with temporary.open('xb') as stream:
            eqx.tree_serialise_leaves(stream,value)
        os.replace(temporary,target)
    record=dict(step=step,kind='intermediate_full_state',model=model_path.name,
        optimizer=state_path.name,config='config.json',
        policy='Paired numerical state; not a completed fixed-budget result')
    if online_state is not None:
        online=Path(online_state)
        if online.parent!=run or not online.is_file():raise ValueError('Online state must be published inside the checkpoint run')
        record['online_state']=online.name
    temporary=marker.with_name(marker.name+'.partial')
    with temporary.open('x') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,marker)
    return marker
