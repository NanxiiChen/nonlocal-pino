"""Independent smooth-interface GMLS consistency and chemical cancellation probe."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from examples.channel_nsch2d.configs.train_single_conservative import get_config
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.problem import material
    jax.config.update('jax_enable_x64',True)
    base=get_config();A,kappa=material(base);R=base.initial.radius;eps=base.physical.epsilon
    def phi_lap(x,center):
        delta=x-center;z=jnp.dot(delta,delta)-R**2
        z=z/(2*jnp.sqrt(2.)*eps*R)
        phi=jnp.tanh(z);J=1-phi*phi
        gradient=delta/(jnp.sqrt(2.)*eps*R)
        lap=J*(2/(jnp.sqrt(2.)*eps*R)-2*phi*jnp.dot(gradient,gradient))
        return phi,lap
    mu=lambda x,center:A*(phi_lap(x,center)[0]**3-phi_lap(x,center)[0])-kappa*phi_lap(x,center)[1]
    def exact(x,center):
        phi,lap=phi_lap(x,center)
        return jnp.stack((phi,lap,mu(x,center),jnp.trace(jax.hessian(mu,argnums=0)(x,center))))
    evaluate=jax.jit(jax.vmap(exact,in_axes=(0,None)))
    verify=jax.jit(jax.vmap(lambda x,z:jnp.trace(jax.hessian(lambda p:phi_lap(p,z)[0])(x))-phi_lap(x,z)[1],in_axes=(0,None)))
    rows=[]
    settings=[(3,36,16.),(4,36,16.),(4,48,16.),(4,48,12.)]
    for nx,ny in ((96,40),(192,80)):
        for degree,neighbors,weight in settings:
            c=get_config();c.sample.Nx=nx;c.sample.Ny=ny
            c.nl.degree=degree;c.nl.neighbors=neighbors;c.nl.weight_exponent=weight
            try:data,operator=geometry_data((.12,.08),c)
            except ValueError as error:
                rows.append(dict(grid=[nx,ny],degree=degree,neighbors=neighbors,weight=weight,
                    status='rejected_conditioning',error=str(error)))
                continue
            interior=data['interior']>0;measure=data['volumes'][interior]
            relative=lambda a,b:float(np.sqrt(np.sum((a-b)**2*measure)/np.sum(b*b*measure)))
            L=lambda v:np.sum(data['laplacian']*(v[data['neighbors']]-v[:,None]),axis=1)
            cases=[]
            for center_x in (.8,2.,2.6):
                center=jnp.asarray([center_x,.5]);truth=np.asarray(evaluate(jnp.asarray(data['points']),center))
                gap=float(np.max(np.abs(verify(jnp.asarray(data['points'][::97]),center))))
                if gap>1e-10:raise AssertionError(f'Analytic Laplacian check failed: {gap}')
                phi,lap_phi,chemical,lap_mu=truth.T
                discrete_lap=L(phi);implied=A*(phi**3-phi)-kappa*discrete_lap
                cases.append(dict(center_x=center_x,analytic_laplacian_max_gap=gap,
                    lap_phi_relative=relative(discrete_lap[interior],lap_phi[interior]),
                    implied_mu_relative=relative(implied[interior],chemical[interior]),
                    lap_exact_mu_relative=relative(L(chemical)[interior],lap_mu[interior]),
                    phase_RHS_implied_mu_relative=relative(L(implied)[interior],lap_mu[interior])))
            row=dict(grid=[nx,ny],degree=degree,neighbors=neighbors,weight=weight,
                status='conditioning_and_analytic_checks_passed',operator=operator,cases=cases)
            rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
    report=dict(status='manufactured_constitutive_consistency_probe_completed',rows=rows,
        physical_parameters=dict(A=A,kappa=kappa,epsilon=eps,radius=R),
        scope='Smooth manufactured fields, actual geometry/physical stencil/interior quadrature; no NN, native solution, BC or trained-accuracy claim')
    with Path(output).open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
