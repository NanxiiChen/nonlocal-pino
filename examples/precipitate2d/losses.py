"""Original mixed CH residual with elastic chemical-potential feedback."""
import jax
import jax.numpy as jnp
from examples.precipitate2d.network import predict_fields


def make_loss_fn(points, D, mechanics, c):
    n = c.sample.Nx
    k = 2*jnp.pi*jnp.fft.fftfreq(n,d=c.geometry.length/n)
    k2 = k[:,None]**2+k[None,:]**2

    def negative_lap(v):
        if D is not None:
            return D@v
        return jnp.fft.ifft2(k2*jnp.fft.fft2(v.reshape(n,n))).real.ravel()

    def loss(net,times,initial_mean):
        def residual(t):
            fields, derivative = jax.jvp(lambda time:predict_fields(net,points,time,initial_mean),
                                         (t,),(jnp.ones_like(t),))
            phi,q = fields.T
            elastic,_,_ = mechanics(phi)
            if c.nl.method == 'local_ad':
                # Hessian of each point network output: (N,field,coord,coord).
                # The full-grid phase projection adds a spatially uniform shift;
                # it remains fully differentiated by the time JVP above.
                hessian = jax.vmap(jax.jacfwd(jax.jacrev(net,argnums=0),argnums=0),in_axes=(0,None))(points,t)
                lap_phi,lap_q = -jnp.trace(hessian,axis1=-2,axis2=-1).T
                rphase = derivative[:,0]+c.physical.mobility*lap_q
                rchemical = q-c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*lap_phi-elastic
            else:
                rphase = derivative[:,0]+c.physical.mobility*negative_lap(q)
                rchemical = q-c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*negative_lap(phi)-elastic
            return jnp.stack((jnp.mean(rphase**2),jnp.mean(rchemical**2)))
        parts = jnp.mean(jax.vmap(residual)(times),axis=0)
        return jnp.dot(parts,jnp.asarray([c.loss.phase_weight,c.loss.chemical_weight])), parts
    return loss
