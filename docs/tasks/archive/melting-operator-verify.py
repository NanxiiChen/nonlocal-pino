"""Independent saved-field/stream/reference checks and final GPU replay."""
import argparse
import json
from pathlib import Path
import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict
from examples.icemelting3d.ellipsoids import signed_distance_numpy,validate_parameters
from examples.icemelting3d.operator_data import build_network,load_dataset,physics
from examples.icemelting3d.operator_training import make_predictor
from nonlocal_pinn.training.runtime import configure_gpu_policy

parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
run=Path(args.run);c=ConfigDict(json.loads((run/'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy);jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
manifest,refs=load_dataset(c);summary=json.loads((run/'summary.json').read_text())
assert summary['iterations']==5000 and not summary['positive_time_training_labels']
with np.load(run/'sampling.npz') as stored:
    parameters,times=stored['parameters'],stored['times']
assert parameters.shape==(5000,16,10) and times.shape==(5000,8)
validate_parameters(parameters)
assert (parameters[...,:3]>=c.initial.center_min).all() and (parameters[...,:3]<=c.initial.center_max).all()
assert (parameters[...,3:6]>=c.initial.axis_min).all() and (parameters[...,3:6]<=c.initial.axis_max).all()
assert (times>=c.sample.t_min).all() and (times<1).all()
flat=parameters.reshape(-1,10);unique=len(np.unique(flat,axis=0));assert unique==len(flat)
for p in np.asarray(manifest['parameters']):assert not np.any(np.all(flat==p,axis=1))
for name in json.loads((run/'source/files.json').read_text()):
    assert (run/'source'/name).read_bytes()==Path(name).read_bytes(),name
net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,jax.random.key(0)))
predict=make_predictor(c);records=[];gaps=[]
for index,ref in enumerate(refs):
    initial=np.tanh(signed_distance_numpy(ref['points'],ref['parameters'],c.initial.reference_sdf_iterations)/physics(c)[3])
    np.testing.assert_allclose(ref['phi'][0],initial,rtol=0,atol=2e-14)
    error=np.linalg.norm(ref['phi']-ref['coarse'],axis=1)/np.linalg.norm(ref['phi'],axis=1)
    assert error.max()<c.operator.reference_tolerance
    with np.load(run/f'prediction_{index:03d}.npz') as stored:
        p,target=stored['prediction'],stored['target']
        np.testing.assert_array_equal(target,ref['phi'])
        np.testing.assert_array_equal(stored['points'],ref['points'])
        assert np.isfinite(p).all() and p.min()>=-1 and p.max()<=1
        phi=float(np.sqrt(np.sum((p-target)**2)/np.sum(target**2)))
        terminal=float(np.sqrt(np.sum((p[-1]-target[-1])**2)/np.sum(target[-1]**2)))
        np.testing.assert_allclose(p[0],target[0],rtol=0,atol=1e-5)
        # Replay two predeclared cases through saved config/checkpoint/current sources.
        if index in (0,manifest['test_indices'][0]):
            replay=predict(net,ref['parameters'],ref['points'],ref['time'])
            gap=float(abs(replay-p).max());assert gap<2e-5;gaps.append(gap)
        if index in manifest['test_indices']:
            item=next(x for x in summary['test']['conditions'] if x['index']==index)
            np.testing.assert_allclose([phi,terminal],[item['pooled_phi'],item['terminal_phi']],rtol=1e-12)
        records.append(dict(index=index,pooled_phi=phi,terminal_phi=terminal,
            sampled_time_refinement=float(error.max())))
selected=[r for r in records if r['index'] in manifest['test_indices']]
np.testing.assert_allclose([np.mean([r[k] for r in selected]) for k in ('pooled_phi','terminal_phi')],
    [summary['test'][k] for k in ('pooled_phi','terminal_phi')],rtol=1e-12)
result=dict(all_checks_pass=True,actual_unique_online_conditions=unique,held_out_count=len(selected),
    source_bytes_match=True,hard_IC_maximum_gate=1e-5,replay_maximum_gap=max(gaps),records=records,
    scope='8held-out shapes on24^3diagnostic nodes; native128^3references, not full-grid inference error/continuum or seed stability')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('PASS',json.dumps({k:v for k,v in result.items() if k!='records'}))
