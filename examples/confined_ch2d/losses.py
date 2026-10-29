"""Mixed CH residuals; local AD has explicit natural-boundary penalties."""
import jax
import jax.numpy as jnp
from examples.confined_ch2d.network import predict_fields


def causal_weights(losses,*,strength,minimum):
    """Stopped exponential prefix weights; every global time retains weight."""
    previous=jnp.concatenate((jnp.zeros(1,dtype=losses.dtype),jnp.cumsum(losses[:-1])))
    return jax.lax.stop_gradient(jnp.maximum(jnp.exp(-strength*previous),minimum))


def make_loss_fn(points,volumes,D,boundary,normals,bweights,c):
    weighted=lambda v:jnp.dot(volumes,v*v)/jnp.sum(volumes)
    bw=lambda v:jnp.dot(bweights,v*v)/jnp.sum(bweights)
    p=c.physical
    eliminated='formulation' in c and c.formulation.chemical=='eliminated'
    if 'phase_metric' in c.loss and c.loss.phase_metric.kind=='mixed_block':
        import equinox as eqx
        from examples.confined_ch2d.time_integration import integration_matrix
        from examples.confined_ch2d.mixed_preconditioning import MixedBlockMetric
        from examples.confined_ch2d.problem import initial_field
        if D is None or eliminated:raise ValueError('Mixed block requires two learned nonlocal fields')
        fixed=jnp.asarray(c.sample.temporal.nodes)
        nodes=jnp.concatenate((jnp.zeros(1,dtype=fixed.dtype),fixed))
        W=jnp.asarray(integration_matrix(nodes,time_scale=c.loss.phase_metric.time_scale))
        metric=MixedBlockMetric(D,volumes,W,mobility=p.mobility,epsilon=p.epsilon)
        phi0=initial_field(points,c,xp=jnp)
        def mixed_loss(net,times,mean):
            if times.shape!=fixed.shape:raise ValueError('Mixed block requires its fixed global time nodes')
            times=eqx.error_if(times,jnp.any(times!=fixed),'Mixed block runtime times differ from fixed nodes')
            query_nodes=jnp.concatenate((jnp.zeros(1,dtype=times.dtype),times))
            values=jax.vmap(lambda t:predict_fields(net,points,t,mean,volumes,D))(query_nodes)
            phi,mu=values[...,0],values[...,1]
            evolution=phi-phi0+p.mobility*jax.vmap(lambda v:D@v)(W@mu)
            chemical=mu-p.beta*(phi**3-phi)-p.epsilon**2*jax.vmap(lambda v:D@v)(phi)
            ephi,emu=metric(evolution,chemical)
            parts=jnp.stack((jnp.mean(jax.vmap(weighted)(ephi)),
                jnp.mean(jax.vmap(weighted)(emu/c.loss.chemical_scale))))
            return jnp.dot(parts,jnp.asarray([c.loss.phase_weight,c.loss.chemical_weight])),parts
        return mixed_loss
    if 'phase_metric' in c.loss and c.loss.phase_metric.kind=='integral_resolvent':
        import equinox as eqx
        from examples.confined_ch2d.preconditioning import SpaceTimeMetric
        from examples.confined_ch2d.time_integration import integration_matrix
        from examples.confined_ch2d.problem import initial_field
        if D is None or not eliminated:raise ValueError('Integral residual requires eliminated nonlocal CH')
        fixed=jnp.asarray(c.sample.temporal.nodes)
        W=jnp.asarray(integration_matrix(jnp.concatenate((jnp.zeros(1),fixed)),time_scale=c.loss.phase_metric.time_scale))
        metric=SpaceTimeMetric(D,volumes,mobility=p.mobility,epsilon=p.epsilon)
        phi0=initial_field(points,c,xp=jnp)
        mu0=p.beta*(phi0**3-phi0)+p.epsilon**2*(D@phi0)
        energy_balance='energy_weight' in c.loss
        if energy_balance:
            E0=jnp.sum(volumes*(.25*p.beta*(phi0**2-1)**2+.5*p.epsilon**2*phi0*(D@phi0)))
            if float(E0)<=0:raise ValueError('Energy balance requires positive initial energy for normalization')
        def integral_loss(net,times,mean):
            if times.shape!=fixed.shape:raise ValueError('Integral residual requires its fixed global nodes')
            times=eqx.error_if(times,jnp.any(times!=fixed),'Integral residual times differ from its quadrature nodes')
            values=jax.vmap(lambda t:predict_fields(net,points,t,mean,volumes,D))(times)
            chemical=jnp.concatenate((mu0[None,:],values[...,1]),axis=0)
            integrated_mu=W[1:]@chemical
            residual=values[...,0]-phi0+p.mobility*jax.vmap(lambda v:D@v)(integrated_mu)
            if 'normalization' in c.loss.phase_metric:
                residual=residual/times[:,None]
            errors=metric.resolvent(residual,times)
            value=jnp.dot(W[-1,1:],jax.vmap(weighted)(errors))/p.terminal_time
            if energy_balance:
                phi=values[...,0];Aphi=jax.vmap(lambda a:D@a)(phi)
                energy=jnp.sum(volumes*(.25*p.beta*(phi**2-1)**2+.5*p.epsilon**2*phi*Aphi),axis=1)
                Amu=jax.vmap(lambda a:D@a)(chemical)
                dissipation=jnp.sum(volumes*chemical*Amu,axis=1)
                defect=(energy-E0+p.mobility*(W[1:]@dissipation))/E0
                energy_loss=jnp.dot(W[-1,1:],defect**2)/p.terminal_time
                return c.loss.phase_weight*value+c.loss.energy_weight*energy_loss,jnp.stack((value,energy_loss))
            return c.loss.phase_weight*value,jnp.stack((value,))
        return integral_loss
    if 'phase_metric' in c.loss and c.loss.phase_metric.kind=='space_time_resolvent':
        if D is None or not eliminated:raise ValueError('Space-time residual metric requires eliminated nonlocal CH')
        from examples.confined_ch2d.preconditioning import SpaceTimeMetric
        metric=SpaceTimeMetric(D,volumes,mobility=p.mobility,epsilon=p.epsilon)
        def global_loss(net,times,mean):
            # Input samples lie strictly in (0,T); explicitly query the terminal
            # residual too. One model and one fully differentiable global loss.
            nodes=jnp.concatenate((times,jnp.asarray([p.terminal_time],dtype=times.dtype)))
            def residual(t):
                fields,dt=jax.jvp(lambda s:predict_fields(net,points,s,mean,volumes,D),(t,),(jnp.ones_like(t),))
                return dt[:,0]+p.mobility*(D@fields[:,1])
            errors=metric(jax.vmap(residual)(nodes),nodes)
            widths=jnp.diff(jnp.concatenate((jnp.zeros(1,dtype=times.dtype),nodes)))
            value=jnp.dot(widths,jax.vmap(weighted)(errors))/p.terminal_time
            parts=jnp.stack((value,))
            return c.loss.phase_weight*value,parts
        return global_loss
    if 'phase_metric' in c.loss:
        if D is None:raise ValueError('Biharmonic graph metric requires the explicit nonlocal operator')
        if c.loss.phase_metric.kind=='neumann_inverse':
            from examples.confined_ch2d.preconditioning import NeumannInverseMetric
            metric=NeumannInverseMetric(D,volumes)
            phase_norm=lambda residual:weighted(metric(residual)/p.mobility)
        else:
            from examples.confined_ch2d.preconditioning import BiharmonicMetric
            metric=BiharmonicMetric(D,volumes,time_scale=c.loss.phase_metric.time_scale,mobility=p.mobility,epsilon=p.epsilon)
            phase_norm=lambda residual:weighted(metric(residual))
    else:phase_norm=weighted
    chemical_scale=c.loss.chemical_scale if 'chemical_scale' in c.loss else 1.
    evolution_scale=c.loss.evolution_scale if 'evolution_scale' in c.loss else chemical_scale
    inverse_jacobian='residual_transform' in c.loss and c.loss.residual_transform.kind=='inverse_tanh_jacobian'
    def loss(net,times,mean):
        if c.sample.include_initial_time:times=jnp.concatenate((jnp.zeros(1,dtype=times.dtype),times))
        def at(t):
            fields,dt=jax.jvp(lambda s:predict_fields(net,points,s,mean,volumes,D),(t,),(jnp.ones_like(t),))
            phi,mu=fields.T
            if D is None:
                hessian=jax.vmap(jax.jacfwd(jax.jacrev(net,argnums=0),argnums=0),in_axes=(0,None))(points,t)
                lap_phi,lap_mu=jnp.trace(hessian,axis1=-2,axis2=-1).T
                parts=[phase_norm(dt[:,0]-p.mobility*lap_mu),weighted(mu-p.beta*(phi**3-phi)+p.epsilon**2*lap_phi)]
                jac=jax.vmap(jax.jacfwd(net,argnums=0),in_axes=(0,None))(boundary,t)
                normal=jnp.einsum('nfc,nc->nf',jac,normals)
                parts += [bw(normal[:,0]),bw(normal[:,1])]
            else:
                evolution=dt[:,0]+p.mobility*(D@mu)
                if inverse_jacobian:
                    from examples.confined_ch2d.tanh_mapping import phase_state
                    _,jacobian,_=phase_state(net,points,t,mean,volumes)
                    evolution=evolution/jacobian
                parts=[phase_norm(evolution/evolution_scale)]
                if not eliminated:parts.append(weighted((mu-p.beta*(phi**3-phi)-p.epsilon**2*(D@phi))/chemical_scale))
            return jnp.stack(parts)
        temporal_parts=jax.vmap(at)(times)
        if 'causal' in c.loss:
            settings=c.loss.causal
            temporal_weights=causal_weights(temporal_parts[:,0],strength=settings.strength,minimum=settings.minimum)
            parts=jnp.sum(temporal_weights[:,None]*temporal_parts,axis=0)/jnp.sum(temporal_weights)
        else:parts=jnp.mean(temporal_parts,axis=0)
        weights=[c.loss.phase_weight]
        if not eliminated:weights.append(c.loss.chemical_weight)
        if D is None:weights += [c.loss.boundary_phi_weight,c.loss.boundary_mu_weight]
        return jnp.dot(parts,jnp.asarray(weights)),parts
    return loss
