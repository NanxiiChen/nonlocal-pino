"""Production-dtype physical query replay check; no fitted weights are exported."""
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.network import encode
from examples.channel_nsch2d.projected_fields import make_query_fn,query_rule
from examples.channel_nsch2d.problem import walls


def check(model,c,control):
    control=Path(control);dtype=c.train.compute_dtype
    with np.load(control/'evaluation_inputs.npz',allow_pickle=False)as f:
        data={k:jnp.asarray(f[k],dtype=f[k].dtype if np.issubdtype(f[k].dtype,np.integer)else dtype)for k in f.files}
    with np.load(control/'prediction.npz',allow_pickle=False)as f:
        point=jnp.asarray(f['points'],dtype=dtype);times=jnp.asarray(f['time'],dtype=dtype)
        amplitudes=f['amplitudes']
    probe=point[::max(1,len(point)//24)][:24]
    roots,_=np.polynomial.legendre.leggauss(64);xx,ee=np.meshgrid(np.linspace(0,c.geometry.length,17),(roots+1)/2,indexing='ij')
    b,t=walls(xx,amplitudes,c,xp=np)
    flow=jnp.asarray(np.stack((xx,b+ee*(t-b)),axis=-1).reshape(-1,2),dtype=dtype)
    joined=jnp.concatenate((point,probe,flow,data['boundary'][:,:2],data['points']))
    host={k:np.asarray(v)for k,v in data.items()};physical=make_query_fn(c)
    rule=lambda x:{k:jnp.asarray(v)for k,v in query_rule(host,np.asarray(x),c).items()}
    @eqx.filter_jit
    def predict(net,d,x,ts,r):return physical(net,encode(net,d),d,x,ts,r)
    saved=np.asarray(predict(model,data,point,times,rule(point)))
    combined=np.asarray(predict(model,data,joined,times,rule(joined)))
    ids=np.array([1,6,12]);probe_times=times[ids];probe_rule=rule(probe)
    @eqx.filter_jit
    def jets(net,d):
        e=encode(net,d)
        return jax.jvp(lambda ts:physical(net,e,d,probe,ts,probe_rule),
            (probe_times,),(jnp.ones_like(probe_times),))
    value,rate=[np.asarray(v)for v in jets(model,data)]
    replay_gap=float(np.max(np.abs(combined[:,:len(point)]-saved)))
    jvp_gap=float(np.max(np.abs(combined[ids,len(point):len(point)+len(probe)]-value)))
    finite=all(np.isfinite(v).all()for v in (saved,combined,value,rate))
    channels=np.max(np.abs(combined[:,:len(point)]-saved),axis=(0,1)).tolist()
    passed=finite and max(replay_gap,jvp_gap)<=2e-5
    result=dict(status=('production_physical_query_primal_JVP_and_batch_replay_passed'if passed else
        'production_physical_query_primal_JVP_or_batch_replay_failed'),
        query_batch_max_gap=replay_gap,physical_time_JVP_primal_max_gap=jvp_gap,
        per_channel_query_gap=channels,
        per_channel_JVP_primal_gap=np.max(np.abs(combined[ids,len(point):len(point)+len(probe)]-value),axis=(0,1)).tolist(),
        all_fields_and_time_rates_finite=finite,
        unchanged_absolute_gate=2e-5,query_count=len(point),native_count=len(data['points']),
        scope='Actual production dtype/model/grid, alternate physical query batches and physical timeJVP; no evolved labels or exported fitted weights')
    if not passed:
        raise ValueError(f'Projected physical query gate failed: {result}')
    return result
