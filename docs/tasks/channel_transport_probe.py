"""Check conservative transport against a frozen trained channel streamfunction."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(run,output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network,encode,potentials
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.transport import build_transport,conservative_advection
    from examples.channel_nsch2d.problem import wall_slopes
    run=Path(run);c=ConfigDict(json.loads((run/'config.json').read_text()))
    if json.loads((run/'summary.json').read_text())['status']!='completed':raise ValueError('Frozen complete model required')
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False) as f:stored={k:f[k] for k in f.files}
    raw,_=geometry_data(stored['amplitudes'],c);operator=build_transport(raw['points'],stored['amplitudes'],c)
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    predict=eqx.filter_jit(lambda n:potentials(n,encode(n,data),data['points'],
        jnp.asarray(stored['time'],dtype=c.train.compute_dtype),data['amplitudes'],c))
    value=np.asarray(predict(model),dtype=np.float64)
    gap=float(np.max(np.abs(value[...,[0,1,3]]-stored['fields'][...,[0,1,4]])))
    if gap>2e-5:raise ValueError(f'Frozen potential replay failed: {gap}')
    adv,qx,qe=conservative_advection(value[...,0],value[...,2],operator,xp=np)
    nx,ny=c.sample.Nx+1,c.sample.Ny+1
    eta=np.linspace(0,1,ny);x=raw['points'].reshape(nx,ny,2)[:,0,0]
    db,dt=wall_slopes(x,stored['amplitudes'],c,xp=np);slope=db[:,None]+eta[None,:]*(dt-db)[:,None]
    ux=qx/operator['transport_jacobian'][None,:,None];uy=qe+slope[None]*ux
    velocity=np.stack((ux,uy),axis=-1).reshape(len(value),-1,2)
    ad=stored['fields'][...,2:4]
    relative_velocity=float(np.linalg.norm(velocity[1:]-ad[1:])/np.linalg.norm(ad[1:]))
    phase=value[...,0].reshape(-1,nx,ny)
    boundary=np.sum(operator['transport_weight_eta'][None]*(phase[:,-1]*qx[:,-1]-phase[:,0]*qx[:,0]),axis=1)
    boundary+=np.sum(operator['transport_weight_x'][None]*(phase[:,:,-1]*qe[:,:,-1]-phase[:,:,0]*qe[:,:,0]),axis=1)
    integral=adv@operator['transport_measure'];balance=float(np.max(np.abs(integral-boundary)))
    if balance>1e-10 or relative_velocity>.01:
        raise ValueError(f'Frozen transport gate failed: balance={balance},relative velocity={relative_velocity}')
    report=dict(status='frozen_transport_probe_passed_not_a_solution_accuracy_test',run=str(run),
        physical_potential_replay_max_gap=gap,sbp_vs_AD_velocity_relative_L2=relative_velocity,
        phase_advection_integral=integral.tolist(),phase_boundary_flux=boundary.tolist(),
        maximum_integral_gap=balance,stencil_dtype='float64',network_dtype=c.train.compute_dtype,
        scope='Same frozen model/geometry; checks published operator identities and resolved velocity agreement, not CH/NS stability')
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.run,a.output)
