"""Interior constitutive projection; boundary chemical traces remain free.

This explicit research option acts on nodal physical
states, not arbitrary coordinate queries or a continuous chemical extension.
"""
from examples.channel_nsch2d.problem import material


def scalar_laplacian(value,data,*,xp):
    if value.ndim<1 or value.shape[-1]!=len(data['points']):
        raise ValueError('Chemical projection requires an aligned final nodal axis')
    index=data['neighbors'];weight=data['laplacian']
    if index.ndim!=2 or index.shape!=weight.shape or index.shape[0]!=value.shape[-1]:
        raise ValueError('Chemical projection requires aligned nodal Laplacian stencils')
    return xp.sum(weight*(value[...,index]-value[...,None]),axis=-1)


def project_state(value,data,c,*,xp):
    """Eliminate only interior mu; preserve phi, velocity, pressure and all mu_B."""
    if value.ndim<2 or value.shape[-1]!=5 or value.shape[-2]!=len(data['points']):
        raise ValueError('Projection requires nodal states(...,nodes,phi/mu/u/v/p)')
    if data['interior'].shape!=(value.shape[-2],):raise ValueError('Aligned interior mask required')
    A,kappa=material(c);phi=value[...,0]
    chemical=A*(phi**3-phi)-kappa*scalar_laplacian(phi,data,xp=xp)
    mu=xp.where(data['interior'].astype(bool),chemical,value[...,1])
    return xp.stack((phi,mu,value[...,2],value[...,3],value[...,4]),axis=-1)


def project_rate(value,rate,data,c,*,xp):
    """Exact tangent of the nodal projector; not an independent evolution law."""
    if value.shape!=rate.shape or value.ndim<2 or value.shape[-1]!=5:
        raise ValueError('Projection rates must align with five-channel nodal states')
    if value.shape[-2]!=len(data['points'])or data['interior'].shape!=(value.shape[-2],):
        raise ValueError('Projection rates require the same nodal layout and mask')
    A,kappa=material(c);phi=value[...,0];dphi=rate[...,0]
    chemical_rate=A*(3*phi**2-1)*dphi-kappa*scalar_laplacian(dphi,data,xp=xp)
    mu_rate=xp.where(data['interior'].astype(bool),chemical_rate,rate[...,1])
    return xp.stack((dphi,mu_rate,rate[...,2],rate[...,3],rate[...,4]),axis=-1)
