"""Two independent constitutive rows for exactly incompressible2D Cauchy stress.

Rao/Sun/Liu2020, https://arxiv.org/abs/2002.10558, Eqs3--4.
Pressure is -trace(sigma)/2, so no separate trace or continuity residual exists.
"""


def constitutive(stress,dx,dy,viscosity,*,xp):
    if (stress.shape[-1]!=3 or dx.shape!=dy.shape or dx.shape[-1]!=5
            or stress.shape[:-1]!=dx.shape[:-1]):
        raise ValueError('Constitutive rows require aligned3stress and5physical-field channels')
    return xp.stack((.5*(stress[...,0]-stress[...,2])-viscosity*(dx[...,2]-dy[...,3]),
        stress[...,1]-viscosity*(dy[...,2]+dx[...,3])),axis=-1)


def divergence(dx,dy,*,xp):
    if dx.shape!=dy.shape or dx.shape[-1]!=3:raise ValueError('Stress divergence requires aligned three-component jets')
    return xp.stack((dx[...,0]+dy[...,1],dx[...,1]+dy[...,2]),axis=-1)
