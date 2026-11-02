"""Native GMLS phase balance for interior-chemical elimination; no spatial AD mu."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network,encode
from examples.channel_nsch2d.projected_fields import make_domain_time_fn,make_decoder_fn
from examples.channel_nsch2d.losses import gmls_first_jets,gmls_phase_advection
from examples.channel_nsch2d.chemical_projection import scalar_laplacian


def diagnose(run,c,advection_grid):
    if advection_grid is not None:
        raise ValueError('Projected mu is a native nodal field; refined continuous-AD phase-balance diagnostics are unsupported')
    run=Path(run)
    with np.load(run/'prediction.npz',allow_pickle=False)as f:
        ids=np.maximum(1,np.rint(np.linspace(0,len(f['time'])-1,5)).astype(int))
        amplitudes=f['amplitudes'];times=f['time'][ids]
    raw,_=geometry_data(amplitudes,c)
    data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer)else c.train.compute_dtype)for k,v in raw.items()}
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    domain=make_domain_time_fn(c);decoder=make_decoder_fn(c)
    @eqx.filter_jit
    def evaluate(net,d,ts):
        e=encode(net,d);v,dt=domain(net,e,d,ts)
        bv=decoder(net,e,d['boundary'][:,:2],ts,d['amplitudes'])
        bx,by=gmls_first_jets(v,bv,d,boundary=True)
        gx,gy=gmls_first_jets(v,v,d,boundary=False)
        return v,dt,bv,bx,by,gmls_phase_advection(v,d),gx[...,2]+gy[...,3],scalar_laplacian(v[...,1],d,xp=jnp)
    result=evaluate(model,data,jnp.asarray(times,dtype=c.train.compute_dtype))
    if any(not np.isfinite(a).all()for a in result):raise ValueError('Nonfinite native projected phase balance')
    v,dt,bv,bx,by,advection,div,lap=[np.asarray(a,dtype=np.float64)for a in result]
    # Same declared loss quadrature and GMLS derivative payload as training.
    measure=np.asarray(data['transport_measure'],dtype=np.float64)
    normal=np.asarray(data['boundary'])[:,2:4];weight=np.asarray(data['boundary_weights'],dtype=np.float64)
    storage=dt[...,0]@measure
    un=np.sum(bv[...,2:4]*normal[None],axis=-1)
    advective_flux=(bv[...,0]*un)@weight
    chemical_flux=(bx[...,1]*normal[None,:,0]+by[...,1]*normal[None,:,1])@weight
    volume_advection=advection@measure;volume_laplacian=lap@measure
    weak=storage+advective_flux-c.physical.mobility*chemical_flux
    strong=storage+volume_advection-c.physical.mobility*volume_laplacian
    adv_gap=volume_advection-advective_flux;gauss_gap=volume_laplacian-chemical_flux
    np.testing.assert_allclose(strong,weak+adv_gap-c.physical.mobility*gauss_gap,rtol=0,atol=1e-12)
    report=dict(status='diagnostic_completed_not_an_accuracy_pass',times=times.tolist(),
        diagnostic_time_derivative='Continuous physical time JVP of projected native state',
        chemical_constraint=c.chemical_constraint.to_dict(),spatial_mu_AD_claimed=False,
        phase_storage_rate=storage.tolist(),boundary_advective_phase_flux=advective_flux.tolist(),
        boundary_mu_normal_integral=chemical_flux.tolist(),weak_phase_balance_residual=weak.tolist(),
        integrated_gmls_strong_residual=strong.tolist(),advection_quadrature_gap=adv_gap.tolist(),
        gmls_laplacian_gauss_gap=gauss_gap.tolist(),
        mobility_scaled_laplacian_gauss_gap=(c.physical.mobility*gauss_gap).tolist(),
        maximum_interior_discrete_divergence=float(np.max(np.abs(div[:,raw['interior']>0]))),
        scope='GMLS strong conservative flux and retained boundary traces with the training SBP loss measure; no exact discrete mass identity or extra mass loss claimed')
    (run/'phase-balance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report
