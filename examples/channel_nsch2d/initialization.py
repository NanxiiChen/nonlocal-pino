"""Explicit model-only physics refinement; optimizer/pool counters start fresh."""
from io import BytesIO
import json
from pathlib import Path

from examples.channel_nsch2d.problem import identity


def model_initialization_spec(c):
    if 'initialization'not in c.train:return None
    q=c.train.initialization
    if (set(q)!={'kind','source_run','optimizer'} or q.kind!='physics_model' or q.optimizer!='fresh'
            or 'chemical_constraint'not in c or any(k in c.train for k in ('continuation','refinement'))):
        raise ValueError('Model initialization requires explicit physics_model/source_run/fresh and projected-mu refinement')
    source=Path(q.source_run)
    if not source.is_absolute():raise ValueError('Physics source run must be an absolute path')
    for name in ('config.json','summary.json','validation.json','model_final.eqx'):
        if not (source/name).is_file():raise FileNotFoundError(source/name)
    saved=json.loads((source/'config.json').read_text());summary=json.loads((source/'summary.json').read_text())
    validated=json.loads((source/'validation.json').read_text())
    if (summary['status']!='completed' or summary['supervised_evolved_field_samples']!=0
            or validated['status']!='replay_and_hard_constraints_passed'
            or 'time_derivative'not in saved or 'chemical_constraint'in saved):
        raise ValueError('Source must be a completed audited original pure-physics JVP model')
    parent_identity={k:saved[k]for k in ('geometry','physical','initial')}
    parent_identity['formulation']='matched-nsch-open-traction-v1'
    if json.loads(json.dumps(identity(c)))!=parent_identity:
        raise ValueError('Physics model source identity differs')
    current=json.loads(c.to_json());del current['chemical_constraint'];del current['train']['initialization']
    current['train']['lr_init']=saved['train']['lr_init'];current['log']['run_name']=saved['log']['run_name']
    if current!=saved:raise ValueError('Model refinement may change only chemical constraint, model-only initialization, lr_init and run name')
    return dict(kind='physics_model',source_run=str(source),model_path=str(source/'model_final.eqx'),optimizer='fresh',
        source_physics_updates=summary['iterations'],new_requested_updates=c.train.iterations,
        source_model_supervised_evolved_samples=0,source_numerical_validation=validated['status'],
        pool_and_global_counters='Fresh stage; not an exact rolling-pool/optimizer continuation')


def load_initialized_model(template,spec):
    if spec is None:return template
    import equinox as eqx
    return eqx.tree_deserialise_leaves(spec['model_path'],template)


def audit_model_initialization(run,c):
    import equinox as eqx
    import jax
    from examples.channel_nsch2d.network import build_network
    from nonlocal_pinn.training.optimizers import make_optimizer
    run=Path(run);spec=model_initialization_spec(c)
    if spec is None:raise ValueError('No explicit model initialization to audit')
    recorded=json.loads((run/'initialization.json').read_text())
    if recorded!=spec:raise ValueError('Saved model initialization provenance differs')
    if (run/'model_initial.eqx').read_bytes()!=Path(spec['model_path']).read_bytes():
        raise ValueError('Initial model is not exactly the specified pure-physics parent')
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',build_network(c,key))
    expected=make_optimizer(c).init(eqx.filter(initial,eqx.is_array));buffer=BytesIO()
    eqx.tree_serialise_leaves(buffer,expected)
    if buffer.getvalue()!=(run/'optimizer_initial.eqx').read_bytes():raise ValueError('Refinement optimizer is not the complete fresh factory state')
    return dict(status='model_only_source_and_fresh_optimizer_passed',source_run=spec['source_run'],
        initial_model_exactly_parent_final=True,complete_optimizer_exactly_fresh=True,
        source_physics_updates=spec['source_physics_updates'],new_updates=c.train.iterations,
        rolling_pool_and_counters_resumed=False,source_and_new_training_evolved_labels=0,
        scope='Global-time physics-only model refinement, not a cold start or equally trained single-factor control')
