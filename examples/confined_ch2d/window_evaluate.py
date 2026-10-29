"""Direct-path replay of a complete window trajectory; no surrogate state labels."""
from pathlib import Path
import json
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.problem import load_reference,identity,initial_field
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.window_network import build_window_network,window_fields
from examples.confined_ch2d.window_training import window_bounds,field_metrics


def replay_trajectory(run,c,ref,checkpoint):
    manifest=json.loads(checkpoint.read_text());bounds=np.asarray(manifest['bounds'])
    if manifest['model']!='sequential_nonlocal_ch_pinn_v1' or manifest['status']!='complete':raise ValueError('Require a complete window trajectory manifest')
    if manifest['identity']!=json.loads(json.dumps(identity(c))):raise ValueError('Window physical identity mismatch')
    np.testing.assert_array_equal(bounds,window_bounds(c))
    if len(manifest['windows'])!=c.train.windows.count:raise ValueError('Wrong window checkpoint count')
    x=jnp.asarray(ref['points']);vol=jnp.asarray(ref['volumes']);D=RegionalLaplacian(ref['points'],ref['volumes'],c)
    original=initial_field(x,c,xp=jnp);mean=jnp.dot(vol,original)/vol.sum()
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));template=build_window_network(c,key)
    @eqx.filter_jit
    def endpoint(net,start,width,t):return window_fields(net,x,t,width,start,mean,vol,D,c)
    @eqx.filter_jit
    def derivative(net,start,width,t):return jax.jvp(lambda tt:window_fields(net,x,tt,width,start,mean,vol,D,c),(t,),(jnp.ones_like(t),))[1][:,0]
    @eqx.filter_jit
    def queries(net,start,width,times):return jax.vmap(lambda t:window_fields(net,x,t,width,start,mean,vol,D,c))(times)
    owner=np.minimum(np.searchsorted(bounds[1:],ref['time'],side='left'),len(bounds)-2)
    fields=np.full((len(ref['time']),len(x),2),np.nan);previous=None;previous_dt=None
    continuity=[];derivative_jump=[]
    for i,record in enumerate(manifest['windows']):
        if record['start']!=bounds[i] or record['end']!=bounds[i+1]:raise ValueError('Window bounds/order mismatch')
        with np.load(run/record['initial'],allow_pickle=False) as d:
            start=d['phi'].copy();time=float(d['time'])
        if start.shape!=(len(x),) or not np.isfinite(start).all() or time!=bounds[i]:raise ValueError('Invalid fixed window initial state')
        if i==0:np.testing.assert_allclose(start,original,atol=1e-12,rtol=1e-12)
        else:np.testing.assert_allclose(start,previous[:,0],atol=1e-8,rtol=1e-8)
        model=eqx.tree_deserialise_leaves(run/record['model'],template)
        width=jnp.asarray(bounds[i+1]-bounds[i],dtype=jnp.float64);initial=jnp.asarray(start)
        now=np.asarray(endpoint(model,initial,width,jnp.asarray(0.,dtype=jnp.float64)))
        now_dt=np.asarray(derivative(model,initial,width,jnp.asarray(0.,dtype=jnp.float64)))
        if previous is not None:
            continuity.append(np.max(abs(now-previous),axis=0).tolist())
            derivative_jump.append(float(np.sqrt(np.sum(ref['volumes']*(now_dt-previous_dt)**2)/ref['volumes'].sum())))
        mask=owner==i
        if np.any(mask):
            times=jnp.asarray(np.clip(ref['time']-bounds[i],0,float(width)))
            value=np.asarray(queries(model,initial,width,times));fields[mask]=value[mask]
        previous=np.asarray(endpoint(model,initial,width,width));previous_dt=np.asarray(derivative(model,initial,width,width))
    if not np.isfinite(fields).all():raise ValueError('Incomplete/nonfinite physical trajectory')
    report=dict(continuity_max_absolute=np.max(np.asarray(continuity),axis=0).tolist() if continuity else [0.,0.],
        derivative_jump_rms=derivative_jump,final_metrics=field_metrics(fields[...,0],fields[...,1],ref),
        mass_max_drift=float(np.max(abs(fields[...,0]@ref['volumes']/ref['volumes'].sum()-float(mean)))),
        wall_normal_scope='Native closed graph on fixed quadrature; pointwise boundary extensions of window initial states are not supplied')
    phi=jnp.asarray(fields[...,0]);Aphi=jax.vmap(lambda a:D@a)(phi)
    report['energy']=np.asarray(jnp.sum(vol*(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*phi*Aphi),axis=1)).tolist()
    report['energy_max_increase']=max(0.,float(np.max(np.diff(report['energy']))))
    report['chemical_constraint_max_absolute']=float(jnp.max(jnp.abs(jnp.asarray(fields[...,1])-c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*Aphi)))
    return fields,report


def evaluate_windows(run,checkpoint):
    jax.config.update('jax_enable_x64',True)
    if checkpoint.name!='trajectory.json':raise ValueError('Windowed runs require their complete trajectory.json checkpoint')
    c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
    fields,report=replay_trajectory(run,c,ref,checkpoint)
    replay={}
    with np.load(run/'prediction.npz',allow_pickle=False) as saved:
        for j,key in enumerate(('phi','mu')):
            np.testing.assert_allclose(fields[...,j],saved[key],atol=1e-8,rtol=1e-8)
            replay[key]=float(np.max(abs(fields[...,j]-saved[key])))
    report['checkpoint']=str(checkpoint);report['cpu_replay_max_absolute']=replay
    (run/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(run/'replayed_prediction.npz',phi=fields[...,0],mu=fields[...,1])
    from examples.confined_ch2d.evaluate import field_plot
    field_plot(ref,[ref['phi_local'],fields[...,0],abs(fields[...,0]-ref['phi_local'])],run/'phi_fields.png',['Common FEM reference','Window PINN','Absolute error'])
    print(json.dumps(report,indent=2),flush=True);return report
