"""Dense genuine checkpoint fields for the requested 3D visualization only."""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict

from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.icemelting3d.operator_data import build_network
from examples.icemelting3d.operator_training import make_predictor
from examples.icemelting3d.ellipsoids import signed_distance_numpy

OUT=Path(__file__).resolve().parent
RUN=Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01')
DATA=Path('/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1')
ROOT=Path('/root/autodl-tmp/data/icemelting3d/operator_render3d_v1')
ROOT.mkdir(exist_ok=False)
c=ConfigDict(json.loads((RUN/'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64',False)
jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
assert jax.default_backend()=='gpu'
checkpoint=RUN/'model_final.eqx'
checkpoint_bytes=checkpoint.read_bytes()
net=eqx.tree_deserialise_leaves(checkpoint,build_network(c,jax.random.key(0)))
assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(net) if eqx.is_array(a))
predict=make_predictor(c)
axis=np.linspace(0,1,96,dtype=np.float32)
points=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
times=np.linspace(0,1,61,dtype=np.float32)
manifest=json.loads((DATA/'manifest.json').read_text())
records=[]
for index in (5,9,2):
    assert index in manifest['test_indices']
    start=time.perf_counter()
    with np.load(DATA/f'condition_{index:03d}.npz') as source:
        parameters=source['parameters']
        probe_points=source['points']
        probe_times=source['time']
    np.testing.assert_array_equal(parameters,manifest['parameters'][index])
    with np.load(RUN/f'prediction_{index:03d}.npz') as saved:
        replay=predict(net,parameters,probe_points,probe_times)
        replay_gap=float(abs(replay-saved['prediction']).max())
    assert replay_gap<2e-5,(index,replay_gap)
    phi=predict(net,parameters,points,times)
    assert phi.shape==(61,96**3) and np.isfinite(phi).all()
    assert phi.min()>=-1 and phi.max()<=1
    ids=np.linspace(0,len(points)-1,1024,dtype=int)
    width=np.sqrt(2)*c.physical.eps_star
    expected=np.tanh(signed_distance_numpy(points[ids],parameters,56)/width)
    initial_gap=float(abs(phi[0,ids]-expected).max())
    assert initial_gap<5e-5,(index,initial_gap)
    small=predict(net,parameters,points[ids],times[[0,15,30,45,60]])
    chunk_gap=float(abs(small-phi[[0,15,30,45,60]][:,ids]).max())
    assert chunk_gap<2e-5,(index,chunk_gap)
    target=ROOT/f'case{index:03d}-phi.npz'
    np.savez_compressed(target,phi=phi.reshape(61,96,96,96),axis=axis,time=times,
                        physical_time=times*c.nondim.T_c,parameters=parameters)
    record=dict(index=index,path=str(target),seconds=time.perf_counter()-start,
        replay_gap=replay_gap,hard_initial_field_gap=initial_gap,chunk_gap=chunk_gap,
        phi_min=float(phi.min()),phi_max=float(phi.max()))
    records.append(record)
    print(json.dumps(record),flush=True)
assert checkpoint.read_bytes()==checkpoint_bytes
report=dict(checkpoint=str(checkpoint),config=str(RUN/'config.json'),dataset=str(DATA),
    input_indices=[5,9,2],shape=[61,96,96,96],axis_order='time,x,y,z',
    time_scale_seconds=c.nondim.T_c,backend=jax.default_backend(),matmul_precision=c.train.matmul_precision,
    fields=records,checkpoint_unchanged=True,
    scope='Actual dense neural-operator inference, no training or mesh/field smoothing. Render resolution is not a new accuracy/convergence study.')
(ROOT/'prediction-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'melting-render-prediction.json').write_text(json.dumps(report,indent=2)+'\n')
print('DENSE_PREDICTIONS_COMPLETE',flush=True)
