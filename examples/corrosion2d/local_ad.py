"""Explicit local KKS AD controls, including the natural no-flux boundary.

Spatial residual formulas were independently verified in the throughput study;
the six-loss full training objective additionally closes the local boundary.
"""

def spatial_laplacians(net, x, t, cle):
    """Forward-over-reverse spatial Hessians of phi and stable KKS q."""
    import jax
    import jax.numpy as jnp
    def spatial_fields(xx):
        raw = net.net(xx, t)
        physical = net.constraint(raw)
        q = .5*cle*(jnp.tanh(raw[1])-1)
        return jnp.stack([physical[0], q])

    hess = jax.hessian(spatial_fields)(x)
    return jnp.trace(hess[0]), jnp.trace(hess[1])


def local_point(net, x, t, cle):
    """Two scalar time grads plus spatial Hessians, no FD."""
    import jax
    fields = net(x, t)
    pt = jax.grad(lambda tt: net(x, tt)[0])(t)
    ct = jax.grad(lambda tt: net(x, tt)[1])(t)
    lap_p,lap_q=spatial_laplacians(net,x,t,cle)
    return fields[0], fields[1], pt, ct, lap_p, lap_q


def local_residual(net, times, points, nd, kind):
    """Local PDE agrees with the existing D_delta -> -Laplacian branch."""
    import jax
    import jax.numpy as jnp
    n_ch, n1, n2, n3, dc, cle = nd
    if kind == 'local_tensor':
        def values(ts):
            return jax.vmap(lambda t:jax.vmap(net,in_axes=(0,None))(points,t))(ts)
        fields,dt=jax.jvp(values,(times,),(jnp.ones_like(times),))
        lap_p,lap_q=jax.vmap(lambda t:jax.vmap(lambda x:spatial_laplacians(net,x,t,cle))(points))(times)
        data=fields[...,0],fields[...,1],dt[...,0],dt[...,1],lap_p,lap_q
    elif kind == 'local_joint':
        data = jax.vmap(lambda xt: local_point(net,xt[:2],xt[2],cle))(points)
    else:
        raise ValueError(kind)
    p, c, pt, ct, lap_p, lap_q = data
    q = c-dc*(3*p**2-2*p**3)-cle
    ac = pt-n3*lap_p+n2*(4*p**3-6*p**2+2*p)-n1*q*dc*(-6*p**2+6*p)
    ch = ct-n_ch*lap_q
    # All loss reductions use (space,time), matching the production graph path.
    return (ac.T, ch.T) if kind=='local_tensor' else (ac, ch)


def local_loss_factory(config, nd, x_ic, ic, x_zero, x_one, x_neumann, kind):
    import jax
    import jax.numpy as jnp
    from examples.corrosion2d.residuals import normalized_data_fields

    def pde_losses(net, times, points, sub, mask):
        ac, ch = local_residual(net,times,points,nd,kind)
        weights = mask.astype(ac.dtype)[:,None] if kind=='local_tensor' else mask.astype(ac.dtype)
        count = jnp.sum(weights)*(len(times) if kind=='local_tensor' else 1)
        return jnp.stack([config.loss.ac_scale*jnp.sum(jnp.abs(ac)*weights)/count,
                          config.loss.ch_scale*jnp.sum(jnp.abs(ch)*weights)/count])

    def loss_vector(net, times, points, sub, mask):
        pde = pde_losses(net,times,points,sub,mask)
        zero = normalized_data_fields(net,x_zero,times)
        one = normalized_data_fields(net,x_one,times)
        bc_zero = jnp.mean(jnp.sum(zero**2,axis=-1))
        bc_one = jnp.mean(jnp.sum((one-1)**2,axis=-1))
        initial = jax.vmap(net,in_axes=(0,None))(x_ic,jnp.array(0.,dtype=times.dtype))
        ic_loss = jnp.mean(jnp.sum((initial-ic)**2,axis=-1))
        directions=jnp.zeros_like(x_neumann).at[:,1].set(1.)
        _,normal=jax.jvp(lambda xx:normalized_data_fields(net,xx,times),
                         (x_neumann,),(directions,))
        # phi_y=0 and q_y/c_le=s_y=0; outward normal sign drops out in MSE.
        no_flux=jnp.mean(jnp.sum(normal**2,axis=-1))
        return jnp.concatenate([pde,jnp.stack([bc_zero,bc_one,ic_loss,no_flux])])
    return loss_vector, pde_losses
