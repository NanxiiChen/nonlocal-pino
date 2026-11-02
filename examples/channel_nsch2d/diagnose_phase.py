"""Compare physical open-boundary phase balance with the GMLS strong residual."""
from pathlib import Path
import argparse
import json
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network,encode,fields,potentials
from examples.channel_nsch2d.problem import quadrature


def diagnose(run,advection_grid):
    run=Path(run);c=ConfigDict(json.loads((run/'config.json').read_text()))
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    if 'chemical_constraint'in c:
        from examples.channel_nsch2d.diagnose_projected_phase import diagnose as projected_diagnose
        return projected_diagnose(run,c,advection_grid)
    with np.load(run/'prediction.npz',allow_pickle=False) as f:
        selected=np.maximum(1,np.rint(np.linspace(0,len(f['time'])-1,5)).astype(int))
        amplitudes=f['amplitudes'];times=f['time'][selected]
    raw,_=geometry_data(amplitudes,c)
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype)
          for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    @eqx.filter_jit
    def phase_jets(net,d,x,t):
        e=encode(net,d)
        fn=lambda x,t:potentials(net,e,x,t,d['amplitudes'],c)[...,:2]
        v,dt=jax.jvp(lambda t:fn(x,t),(t,),(jnp.ones_like(t),))
        ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=x.dtype),x.shape)
        ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=x.dtype),x.shape)
        _,dx=jax.jvp(lambda x:fn(x,t),(x,),(ex,))
        _,dy=jax.jvp(lambda x:fn(x,t),(x,),(ey,))
        return v,dt,dx,dy
    predict=eqx.filter_jit(lambda net,d,x,t:fields(net,encode(net,d),x,t,d['amplitudes'],c))
    times_device=jnp.asarray(times,dtype=c.train.compute_dtype)
    print('Evaluating independent open-boundary phase balance',flush=True)
    value,dt,dx,dy=[np.asarray(v,dtype=np.float64) for v in phase_jets(model,data,data['points'],times_device)]
    physical=np.asarray(predict(model,data,data['points'],times_device),dtype=np.float64)
    gap=float(np.max(np.abs(physical[...,:2]-value)))
    if not np.isfinite(physical).all() or gap>2e-5:raise ValueError(f'Physical/potential primal gap: {gap}')
    b=data['boundary'];bv,_,bx,by=[np.asarray(v,dtype=np.float64) for v in phase_jets(model,data,b[:,:2],times_device)]
    boundary_fields=np.asarray(predict(model,data,b[:,:2],times_device),dtype=np.float64)
    volume=raw['volumes'];normal=raw['boundary'][:,2:4];weight=raw['boundary_weights']
    un=np.sum(boundary_fields[...,2:4]*normal[None],axis=-1)
    mu_n=bx[...,1]*normal[None,:,0]+by[...,1]*normal[None,:,1]
    storage=dt[...,0]@volume
    advective_flux=(bv[...,0]*un)@weight
    chemical_flux=mu_n@weight
    weak_balance=storage+advective_flux-c.physical.mobility*chemical_flux
    advective_volume=(physical[...,2]*dx[...,0]+physical[...,3]*dy[...,0])@volume
    lap=np.sum(raw['laplacian'][None]*(value[:,raw['neighbors'],1]-value[:,:,None,1]),axis=-1)
    lap_volume=lap@volume
    strong_balance=storage+advective_volume-c.physical.mobility*lap_volume
    advection_gap=advective_volume-advective_flux
    gauss_gap=lap_volume-chemical_flux
    np.testing.assert_allclose(strong_balance,weak_balance+advection_gap-c.physical.mobility*gauss_gap,atol=1e-12)
    report=dict(status='diagnostic_completed_not_an_accuracy_pass',times=times.tolist(),
        diagnostic_time_derivative='Continuous time JVP, including when the training loss used discrete time',
        physical_potential_primal_gap=gap,phase_storage_rate=storage.tolist(),
        boundary_advective_phase_flux=advective_flux.tolist(),boundary_mu_normal_integral=chemical_flux.tolist(),
        weak_phase_balance_residual=weak_balance.tolist(),integrated_gmls_strong_residual=strong_balance.tolist(),
        advection_quadrature_gap=advection_gap.tolist(),gmls_laplacian_gauss_gap=gauss_gap.tolist(),
        mobility_scaled_laplacian_gauss_gap=(c.physical.mobility*gauss_gap).tolist(),
        scope='Same trained quadrature; includes every open boundary. No extra mass loss or reference label used in training.')
    if 'phase_transport' in c and c.phase_transport.kind=='sbp4_mapped_curl':
        from examples.channel_nsch2d.transport import conservative_advection
        @eqx.filter_jit
        def discrete_transport(net,d):
            z=potentials(net,encode(net,d),d['points'],times_device,d['amplitudes'],c)
            rate,qx,qe=conservative_advection(z[...,0],z[...,2],d,xp=jnp)
            return z[...,0],rate,qx,qe
        phase,rate,qx,qe=[np.asarray(v,dtype=np.float64) for v in discrete_transport(model,data)]
        nx,ny=c.sample.Nx+1,c.sample.Ny+1;phase=phase.reshape(-1,nx,ny)
        Hx=np.asarray(data['transport_weight_x'],dtype=np.float64)
        He=np.asarray(data['transport_weight_eta'],dtype=np.float64)
        measure=np.asarray(data['transport_measure'],dtype=np.float64)
        flux=np.sum(He[None]*(phase[:,-1]*qx[:,-1]-phase[:,0]*qx[:,0]),axis=1)
        flux+=np.sum(Hx[None]*(phase[:,:,-1]*qe[:,:,-1]-phase[:,:,0]*qe[:,:,0]),axis=1)
        integral=rate@measure;maximum=float(np.max(np.abs(integral-flux)))
        if maximum>2e-5:raise ValueError(f'Training-dtype SBP transport integral mismatch: {maximum}')
        report['conservative_phase_transport']=dict(training_dtype=c.train.compute_dtype,
            advection_integral=integral.tolist(),boundary_phase_flux=flux.tolist(),maximum_gap=maximum,
            continuous_JVP_storage_rate=(dt[...,0]@measure).tolist(),
            dispersed_phase_area=((1-phase.reshape(len(times),-1))/2@measure).tolist(),
            scope='Advection identity only; GMLS diffusion, boundary PDE masks and BDF2-versus-continuous time are separate')
    if 'phase_transport' in c and c.phase_transport.kind=='gmls_conservative':
        from examples.channel_nsch2d.losses import gmls_phase_advection,gmls_first_jets
        @eqx.filter_jit
        def discrete_transport(net,d):
            z=fields(net,encode(net,d),d['points'],times_device,d['amplitudes'],c)
            gx,gy=gmls_first_jets(z,z,d,boundary=False)
            return gmls_phase_advection(z,d),gx[...,2]+gy[...,3]
        rate,div=[np.asarray(a,dtype=np.float64) for a in discrete_transport(model,data)]
        measure=np.asarray(data['transport_measure'],dtype=np.float64)
        report['gmls_phase_transport']=dict(training_dtype=c.train.compute_dtype,
            advection_integral=(rate@measure).tolist(),
            physical_boundary_phase_flux=advective_flux.tolist(),
            quadrature_and_operator_flux_gap=(rate@measure-advective_flux).tolist(),
            maximum_interior_discrete_divergence=float(np.max(np.abs(div[:,raw['interior']>0]))),
            scope='GMLS conservative differential form, not an exact SBP mass identity. Physical velocity remains the unchanged AD curl; continuous-JVP storage is an independent diagnostic.')
    if advection_grid is not None:
        # Refine diagnostic integration while holding the original geometry
        # encoder/context and trained model fixed. No new conditioning tokens.
        grid=tuple(advection_grid)
        if min(grid)<4:raise ValueError('Diagnostic integration grid requires counts>=4')
        fine_points,fine_volume=quadrature(amplitudes,c,*grid)
        @eqx.filter_jit
        def transport(net,d,x,t):
            e=encode(net,d);fn=lambda x,t:potentials(net,e,x,t,d['amplitudes'],c)
            _,ft=jax.jvp(lambda t:fn(x,t),(t,),(jnp.ones_like(t),))
            ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=x.dtype),x.shape)
            ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=x.dtype),x.shape)
            _,fx=jax.jvp(lambda x:fn(x,t),(x,),(ex,))
            _,fy=jax.jvp(lambda x:fn(x,t),(x,),(ey,))
            return ft[...,0],fy[...,2]*fx[...,0]-fx[...,2]*fy[...,0]
        ft,adv=[np.asarray(v,dtype=np.float64) for v in transport(model,data,
            jnp.asarray(fine_points,dtype=c.train.compute_dtype),times_device)]
        report['refined_advection_quadrature']=dict(grid=list(grid),points=len(fine_points),
            original_context_points=len(raw['points']),phase_storage_rate=(ft@fine_volume).tolist(),
            advection_quadrature_gap=(adv@fine_volume-advective_flux).tolist(),
            weak_phase_balance_residual=(ft@fine_volume+advective_flux-c.physical.mobility*chemical_flux).tolist())
        # Independently check AD at the largest transport-gradient queries.
        transport_density=physical[1,:,2]*dx[1,:,0]+physical[1,:,3]*dy[1,:,0]
        ids=np.argsort(np.abs(transport_density))[-3:];probes=raw['points'][ids]
        analytic=np.stack((dx[1,ids,0],dy[1,ids,0]),axis=-1)
        evaluate_phase=eqx.filter_jit(lambda net,d,x:potentials(net,encode(net,d),x,
            times_device[1:2],d['amplitudes'],c)[0,:,0])
        difference=[]
        for step in (.001,.0005):
            offsets=step*np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
            probes_device=jnp.asarray((probes[:,None]+offsets[None]).reshape(-1,2),dtype=c.train.compute_dtype)
            v=np.asarray(evaluate_phase(model,data,probes_device),dtype=np.float64).reshape(3,4)
            fd=np.stack((v[:,0]-v[:,1],v[:,2]-v[:,3]),axis=-1)/(2*step)
            difference.append(dict(step=step,gradient=fd.tolist(),relative_gap=float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic))))
        report['finite_difference_gradient_check']=dict(points=probes.tolist(),time=float(times[1]),
            analytical_gradient=analytic.tolist(),central_differences=difference)
    name='phase-balance.json' if advection_grid is None else 'phase-balance-refined.json'
    (run/name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--advection-grid',type=int,nargs=2)
    args=p.parse_args();diagnose(args.run,args.advection_grid)
