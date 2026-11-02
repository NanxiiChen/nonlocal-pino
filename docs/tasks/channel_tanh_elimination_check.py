"""Independently check continuous eliminated-CH tanh identities in two dimensions."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.configs.train_base import get_config
from examples.channel_nsch2d.problem import material


def check(output):
    jax.config.update('jax_enable_x64',True)
    A,kappa=material(get_config());records=[]
    points=jnp.asarray(np.random.default_rng(21).uniform(-1,1,(16,2)))
    for amplitude in (1.,1.25):
        for shift in (-3.,0.,3.):
            latent=lambda x:.5*x[0]**2+.3*jnp.sin(1.3*x[1])+.2*x[0]*x[1]+shift
            phase=lambda x:amplitude*jnp.tanh(latent(x))
            lap=lambda f,x:jnp.trace(jax.hessian(f)(x))
            J=lambda x:amplitude/jnp.cosh(latent(x))**2
            mu=lambda x:A*(phase(x)**3-phase(x))-kappa*lap(phase,x)
            def H(x):
                return -A*amplitude*phase(x)-kappa*lap(latent,x)+2*kappa*phase(x)/amplitude*jnp.sum(jax.grad(latent)(x)**2)
            def comparison(x):
                p=phase(x);grad=jax.grad(latent)(x);s2=lap(latent,x)
                transformed=lap(H,x)-4*p/amplitude*jnp.dot(grad,jax.grad(H)(x))
                transformed+=(-2*p/amplitude*s2+(6*p*p/amplitude**2-2)*jnp.sum(grad**2))*H(x)
                transformed+=A*(amplitude**2-1)*(s2-2*p/amplitude*jnp.sum(grad**2))
                return jnp.stack((mu(x)-(J(x)*H(x)+A*(amplitude**2-1)*p),
                    lap(mu,x)/J(x)-transformed,lap(mu,x)/J(x)))
            values=np.asarray(jax.jit(jax.vmap(comparison))(points))
            gap=float(np.max(np.abs(values[:,1])))
            relative=float(np.linalg.norm(values[:,1])/np.linalg.norm(values[:,2]))
            if not np.isfinite(values).all() or gap>1e-10 or relative>1e-10:
                raise AssertionError(f'Continuous eliminated identity failed: {amplitude},{shift},{gap},{relative}')
            records.append(dict(amplitude=amplitude,latent_shift=shift,
                chemical_identity_max_gap=float(np.max(np.abs(values[:,0]))),
                transformed_laplacian_max_gap=gap,transformed_laplacian_relative_gap=relative))
    # A constant-mobility block identity also holds for any fixed linear
    # Laplacian, without invoking a discrete nonlinear chain rule.
    rng=np.random.default_rng(73);D=rng.normal(size=(24,24));D-=np.diag(D.sum(axis=1))
    phi=1.25*np.tanh(rng.normal(size=24));mu=rng.normal(size=24)
    rate=rng.normal(size=24);advection=rng.normal(size=24);M=get_config().physical.mobility
    r_phi=rate+advection-M*(D@mu)
    r_mu=mu-A*(phi**3-phi)+kappa*(D@phi)
    combined=r_phi+M*(D@r_mu)
    eliminated=rate+advection-M*A*(D@(phi**3-phi))+M*kappa*(D@(D@phi))
    block_gap=float(np.max(np.abs(combined-eliminated)))
    if block_gap>2e-12:raise AssertionError(f'Discrete block elimination failed: {block_gap}')
    report=dict(status='continuous_eliminated_CH_identity_passed',dimension=2,records=records,
        discrete_linear_block_identity_max_gap=block_gap,
        method='Independent physical fourth derivatives versus factored latent expression, float64 AD',
        scope='Algebra only: no mixed-network, GMLS/SBP, boundary, NS coupling or training-accuracy claim')
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    check(parser.parse_args().output)
