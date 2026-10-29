"""Full-cube direct AC reference, independent of networks/training operators.

Six-axis cubic graph stencil derived from the declared kernel/quadrature.
No octant symmetry, sphere evolution profile or positive-time neural fields.
"""
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.ellipsoids import sample_parameters, signed_distance_numpy


def coefficients(c):
    h = 1/(c.sample.Nx-1)
    if not (c.sample.Nx == c.sample.Ny == c.sample.Nz and h < c.nl.delta < np.sqrt(2)*h):
        raise ValueError('Reference stencil requires a cubic six-neighbor horizon')
    if c.nl.kernel != 'cubic' or c.nl.method != 'sparse':
        raise ValueError('Reference supports the declared cubic sparse graph only')
    n1 = c.physical.M*c.nondim.T_c/c.nondim.L_c**2
    n2 = n1/c.physical.eps_star**2
    n3 = c.physical.lam*c.nondim.T_c/c.nondim.L_c/(np.sqrt(2)*c.physical.eps_star)
    weight = 315/(2*np.pi*c.nl.delta**5)*(1-h/c.nl.delta)**2*h**3
    return n1,n2,n3,weight


def rhs(phi,c):
    n1,n2,n3,weight = coefficients(c)
    p = jnp.pad(phi,1,mode='edge')
    lap = (p[:-2,1:-1,1:-1]+p[2:,1:-1,1:-1]+p[1:-1,:-2,1:-1]
           +p[1:-1,2:,1:-1]+p[1:-1,1:-1,:-2]+p[1:-1,1:-1,2:]-6*phi)
    return n1*weight*lap-n2*(phi**3-phi)-n3*(1-phi**2)


def make_integrator(c,dt):
    interval = 1/(c.operator.snapshots-1)
    steps = round(interval/dt)
    if dt <= 0 or steps < 1 or not np.isclose(steps*dt,interval,rtol=0,atol=1e-12):
        raise ValueError('Reference dt must divide the snapshot interval')
    @jax.jit
    def integrate(initial):
        def step(phi,_):
            k1 = rhs(phi,c)
            k2 = rhs(phi+dt*k1/2,c)
            k3 = rhs(phi+dt*k2/2,c)
            k4 = rhs(phi+dt*k3,c)
            return phi+dt*(k1+2*k2+2*k3+k4)/6,None
        def snapshot(phi,_):
            final,_ = jax.lax.scan(step,phi,None,length=steps)
            return final,final
        _,later = jax.lax.scan(snapshot,initial,None,length=c.operator.snapshots-1)
        return jnp.concatenate([initial[None],later])
    return integrate


def prepare_dataset(c):
    # Only the orchestration layer shares schema/known-IC helpers; rhs above is
    # independently written and never calls the train loss or graph builder.
    from examples.icemelting3d.operator_data import identity,validate_config
    validate_config(c)
    if jax.default_backend() != 'cpu' or not jax.config.x64_enabled:
        raise ValueError('Prepare independent reference with JAX_PLATFORMS=cpu JAX_ENABLE_X64=true')
    folder = Path(c.operator.dataset_path)
    folder.mkdir(parents=True,exist_ok=False)
    n = c.sample.Nx
    axis = np.linspace(0,1,n)
    points = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    indices = np.linspace(0,n-1,c.operator.evaluation_axis_points).round().astype(int)
    evaluation_ids = np.ravel_multi_index(np.meshgrid(indices,indices,indices,indexing='ij'),(n,n,n)).ravel()
    slice_ids = np.arange(n**3).reshape(n,n,n)[:,:,n//2].ravel()
    count = c.operator.monitor_conditions+c.operator.test_conditions
    parameters = np.asarray(sample_parameters(jax.random.key(c.operator.dataset_seed),count,c.initial),dtype=float)
    manifest = dict(identity=identity(c),parameters=parameters.tolist(),
        monitor_indices=list(range(c.operator.monitor_conditions)),
        test_indices=list(range(c.operator.monitor_conditions,count)),
        selection='Declared before solving/training; held-out only, never future-field labels',
        grid_order='XYZ ij for physical graph; CZYX for encoder volumes')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    coarse_fn = make_integrator(c,c.operator.reference_dt)
    fine_fn = make_integrator(c,c.operator.reference_dt/2)
    records = []
    for k,p in enumerate(parameters):
        start = time.perf_counter()
        initial = np.tanh(signed_distance_numpy(points,p,c.initial.reference_sdf_iterations)/(np.sqrt(2)*c.physical.eps_star)).reshape(n,n,n)
        coarse = np.asarray(coarse_fn(jnp.asarray(initial)))
        fine = np.asarray(fine_fn(jnp.asarray(initial)))
        error = np.linalg.norm((fine-coarse).reshape(len(fine),-1),axis=1)/np.linalg.norm(fine.reshape(len(fine),-1),axis=1)
        if (not np.isfinite(coarse).all() or not np.isfinite(fine).all() or not np.isfinite(error).all()
                or fine.min() < -1.000001 or fine.max() > 1.000001
                or error.max() >= c.operator.reference_tolerance):
            raise ValueError(f'Unconverged/range-invalid reference{k}: time error{error.max()}')
        info = dict(maximum_time_refinement_error=float(error.max()),
            seconds=time.perf_counter()-start,phi_min=float(fine.min()),phi_max=float(fine.max()),
            dt=c.operator.reference_dt,refined_dt=c.operator.reference_dt/2,
            scope='Independent full128-cube native graph time convergence, not local continuum convergence')
        np.savez_compressed(folder/f'condition_{k:03d}.npz',parameters=p,time=np.linspace(0,1,len(fine)),
            points=points[evaluation_ids],phi=fine.reshape(len(fine),-1)[:,evaluation_ids],
            coarse=coarse.reshape(len(fine),-1)[:,evaluation_ids],slice_points=points[slice_ids],
            slice_phi=fine.reshape(len(fine),-1)[:,slice_ids],
            solid_fraction=((fine+1)/2).mean(axis=(1,2,3)),positive_fraction=(fine>0).mean(axis=(1,2,3)),
            metadata_json=np.array(json.dumps(info,allow_nan=False)))
        records.append(info)
        print('REFERENCE',k,json.dumps(info),flush=True)
    (folder/'summary.json').write_text(json.dumps(dict(records=records),indent=2,allow_nan=False)+'\n')
    return manifest
