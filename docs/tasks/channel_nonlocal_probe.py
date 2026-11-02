"""CPU consistency probe for original integral kernels on the actual channel.

No training or PDE/BC substitution: tests a correspondence gradient and the
regional pair Laplacian, and exposes missing open/Dirichlet boundary closures.
The gradient is the scalar shape-tensor construction already used by ElasticD;
the positive kernel/target-volume Laplacian matches ExplicitVolumeSparseD.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from examples.channel_nsch2d.configs.train_base import get_config
from examples.channel_nsch2d.problem import quadrature,walls


def probe(nx,ny,delta):
    c=get_config();amp=(.12,.08)
    x,volume=quadrature(amp,c,nx,ny);n=len(x)
    pairs=cKDTree(x).query_pairs(delta,output_type='ndarray')
    i,j=pairs.T;distance=np.linalg.norm(x[j]-x[i],axis=-1)
    active=(distance>0)&(distance<delta);i,j,distance=i[active],j[active],distance[active]
    row=np.r_[i,j];col=np.r_[j,i];distance=np.r_[distance,distance]
    xi=x[col]-x[row]
    weight=120/(np.pi*delta**4)*(1-distance/delta)**2*volume[col]
    moment=np.zeros((n,2,2))
    for a in range(2):
        for b in range(2):np.add.at(moment[:,a,b],row,weight*xi[:,a]*xi[:,b])
    condition=np.linalg.cond(moment)
    if not np.isfinite(condition).all() or condition.max()>1e8:raise ValueError('Singular shape tensors')
    inverse=np.linalg.inv(moment)
    coefficients=weight[:,None]*np.einsum('eab,eb->ea',inverse[row],xi)
    gradients=[]
    for axis in range(2):
        matrix=sparse.coo_matrix((coefficients[:,axis],(row,col)),shape=(n,n)).tocsr()
        matrix-=sparse.diags(np.asarray(matrix.sum(axis=1)).ravel())
        gradients.append(matrix)
    off=sparse.coo_matrix((weight,(row,col)),shape=(n,n)).tocsr()
    lap=off-sparse.diags(np.asarray(off.sum(axis=1)).ravel())
    linear_gap=max(float(np.max(np.abs(gradients[a]@x[:,b]-(a==b)))) for a in range(2) for b in range(2))
    if linear_gap>1e-10:raise ValueError(f'Gradient linear reproduction failed: {linear_gap}')
    rng=np.random.default_rng(42);f=rng.normal(size=n);v=rng.normal(size=(n,2))
    closed_div=-(gradients[0].T@(volume*v[:,0])+gradients[1].T@(volume*v[:,1]))/volume
    adjoint_gap=float(abs(np.dot(volume*f,closed_div)+sum(np.dot(volume*v[:,a],gradients[a]@f) for a in range(2))))
    mass_gap=float(abs(np.dot(volume,lap@f)))
    if adjoint_gap>1e-8 or mass_gap>1e-8:raise ValueError('Pair/adjoint identity failed')
    # Common interior smooth-field consistency, without invoking any BC there.
    edge_x=np.linspace(0,c.geometry.length,8193);b,t=walls(edge_x,amp,c,xp=np)
    edge=np.r_[np.column_stack((edge_x,b)),np.column_stack((edge_x,t)),
        np.column_stack((np.zeros(2049),np.linspace(0,1,2049))),
        np.column_stack((np.full(2049,4.),np.linspace(0,1,2049)))]
    clearance=cKDTree(edge).query(x)[0];interior=clearance>delta*1.001
    if not np.any(interior):raise ValueError('No full-horizon interior queries')
    smooth=np.sin(x[:,0])*np.cos(x[:,1])
    exact_g=np.column_stack((np.cos(x[:,0])*np.cos(x[:,1]),-np.sin(x[:,0])*np.sin(x[:,1])))
    actual_g=np.column_stack([g@smooth for g in gradients])
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    # A local Poiseuille velocity vanishes on straight walls but has nonzero
    # normal derivative. A regional natural-flux closure is not its Dirichlet BC.
    u=6*.2*x[:,1]*(1-x[:,1]);actual_lap=lap@u
    straight_wall=(x[:,0]>.35)&(x[:,0]<.8)&(np.minimum(x[:,1],1-x[:,1])<delta)
    wall_error=relative(actual_lap[straight_wall],np.full(straight_wall.sum(),-12*.2))
    # For v=(x,0), div(v)=1 and its open-boundary integral is the domain area.
    # Bare negative adjoint with no boundary term instead integrates to zero.
    direct_integral=float(np.dot(volume,gradients[0]@x[:,0]))
    missing_boundary_integral=float(-np.sum(gradients[0].T@(volume*x[:,0])))
    return dict(grid=[nx,ny],horizon=delta,points=n,directed_edges=len(row),
        maximum_shape_condition=float(condition.max()),gradient_linear_max_gap=linear_gap,
        pair_mass_max_gap=mass_gap,weighted_adjoint_gap=adjoint_gap,
        full_horizon_gradient_relative_error=relative(actual_g[interior],exact_g[interior]),
        full_horizon_laplacian_relative_error=relative((lap@smooth)[interior],-2*smooth[interior]),
        bare_regional_poiseuille_wall_laplacian_relative_error=wall_error,
        open_linear_velocity_expected_divergence_integral=float(volume.sum()),
        direct_gradient_divergence_integral=direct_integral,
        negative_adjoint_without_boundary_integral=missing_boundary_integral)


if __name__=='__main__':
    reports=[probe(nx,ny,d) for nx,ny,d in [(48,20,.3),(96,40,.2),(128,64,.15)]]
    result=dict(status='consistency_probe_completed_not_a_validated_NS_scheme',checks=reports,
        scope='Original integral-kernel candidates, no spatial AD. Mixed physical boundary closure and pressure stability not supplied or passed.',
        conclusion='First/second neighbor-integral derivatives are constructible; natural regional scalar and closed-adjoint boundaries cannot be silently reused for open no-slip NS.')
    target=Path('/root/autodl-tmp/data/channel_nsch2d/nonlocal-channel-probe.json')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)
