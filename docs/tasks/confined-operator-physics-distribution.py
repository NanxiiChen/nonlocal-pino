"""Frozen physics-objective distribution on the complete512/8geometry bank.

No optimizer updates or reference-field gradients. Uses production make_loss;
the24known field errors are attached only as diagnostics after the PDE calls.
"""
import argparse
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_problem import evaluation_indices
from examples.confined_ch2d.operator_training import build_network,load_bank,make_loss
from examples.confined_ch2d.operator_conditioning import initial_rhs_enabled


def diagnose(run,positive_times,check_times,output):
    if positive_times<2 or check_times<=positive_times:
        raise ValueError('Require positive_times>=2 and a strictly finer quadrature check')
    jax.config.update('jax_enable_x64',True)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    manifest,refs,bank=load_bank(c)
    del refs  # Keep only label-free operator inputs during the PDE evaluations.
    expected={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
    if initial_rhs_enabled(c):expected.add('domain_fields')
    assert set(bank)==expected
    _,labeled=evaluation_indices(manifest)
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,jax.random.key(0)))
    loss=make_loss(c)
    evaluate=eqx.filter_jit(lambda n,d,t:loss(n,jax.tree.map(lambda a:a[None],d),t))
    def times(count):
        return jnp.concatenate((jnp.zeros(1),c.physical.terminal_time*(jnp.arange(count)+.5)/count))
    coarse=times(positive_times);fine=times(check_times)
    values=[];checks=[]
    for index in range(len(manifest['conditions'])):
        item=jax.tree.map(lambda a:a[index],bank)
        value=float(evaluate(model,item,coarse))
        if not np.isfinite(value) or value<=0:raise FloatingPointError('Require positive finite diagnostic PDE objective')
        values.append(value)
        if index in labeled:
            refined=float(evaluate(model,item,fine))
            if not np.isfinite(refined) or refined<=0:raise FloatingPointError('Invalid refined diagnostic objective')
            checks.append(dict(index=index,relative_quadrature_difference=abs(value-refined)/refined))
    values=np.asarray(values);train=values[manifest['train_indices']];test=values[manifest['test_indices']]
    def statistics(a):
        return dict(mean=float(a.mean()),quantile_levels=[0.,.05,.25,.5,.75,.95,1.],
            quantiles=np.quantile(a,[0.,.05,.25,.5,.75,.95,1.]).tolist())
    # Read field labels only after all frozen physics evaluations finish.
    summary=json.loads((run/'summary.json').read_text())
    field_errors={r['index']:r['phi_pooled'] for r in summary['final_metrics']['conditions']}
    rows=[dict(index=i,split='train' if i in manifest['train_indices'] else 'development',
        pde_objective=float(value),phase_relative_L2=field_errors[i] if i in field_errors else None)
        for i,value in enumerate(values)]
    result=dict(run=str(run),optimizer_updates=0,reference_fields_in_physics_inputs=False,
        scope='512training inputs and8reused development inputs; frozen residual diagnostic, not an error bound or a new generalization test.',
        positive_midpoint_nodes=positive_times,check_midpoint_nodes=check_times,
        train=statistics(train),development=statistics(test),
        development_percentile_among_training=[dict(index=i,percentile=float(100*np.mean(train<=values[i]))) for i in manifest['test_indices']],
        quadrature_checks=checks,conditions=rows)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('train','development','development_percentile_among_training')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--positive-times',type=int,required=True)
    parser.add_argument('--check-times',type=int,required=True)
    args=parser.parse_args();diagnose(args.run,args.positive_times,args.check_times,args.output)
