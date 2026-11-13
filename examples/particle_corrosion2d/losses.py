"""Original KKS physics on a core/halo graph excluding inert ceramic particles."""
import equinox as eqx
import jax
import jax.numpy as jnp

from examples.particle_corrosion2d.network import decode, encode, physical_fields, time_jets,initial_phase


LOSS_NAMES = ('ac', 'ch', 'mouth_phi', 'mouth_c_over_cle', 'initial')


def loss_names(c):
    if 'boundary' in c:
        return LOSS_NAMES[:2]
    hard = 'phase' in c and 'initial_lifting' in c.phase
    names = LOSS_NAMES[:4] if hard else LOSS_NAMES
    if 'concentration_scale' in c.loss:
        names = names[:3]+('mouth_c_scaled',)+names[4:]
    if 'transport_residual' in c.loss:
        names=names[:1]+('ch_ac_eliminated',)+names[2:]
    return names


def graph_action(values, data):
    """Positive negative-Laplacian on core rows; every halo neighbor is explicit.

    values has shape (Nhalo,Nt). Padded core rows and neighbor slots have zero
    quadrature/edge weights. No transpose or graph diagonal is reconstructed.
    """
    core, neighbors, coefficients = data['core_indices'], data['neighbors'], data['coefficients']
    if (values.ndim != 2 or core.ndim != 1 or neighbors.ndim != 2
            or neighbors.shape != coefficients.shape or len(neighbors) != len(core)):
        raise ValueError('Core/halo operator requires values(Nhalo,Nt), core(Nc), neighbors/coefficients(Nc,K)')
    if 'operator_matrix' in data:
        from .spatial_adjoint import fixed_graph
        return fixed_graph(values,core,neighbors,coefficients,data['operator_matrix'])
    return jnp.sum(coefficients[:, :, None] *
                   (values[core, None, :] - values[neighbors]), axis=1)


def core_values(values,data):
    if 'core_selection' in data:
        from .spatial_adjoint import fixed_gather
        return fixed_gather(values,data['core_indices'],data['core_selection'])
    return values[data['core_indices']]


def spatial_terms(phi,liquid,data,c):
    """Positive G in y_t+G=0; inputs have shape(Nhalo,Nt)."""
    p = c.physical
    q = p.c_le*(liquid-1)
    phi_core, q_core = core_values(phi,data), core_values(q,data)
    hp = 6*phi_core*(1-phi_core)
    gp = 2*phi_core*(1-phi_core)*(1-2*phi_core)
    driving = (p.n_ac3*graph_action(phi, data)+p.n_ac2*gp
               -p.n_ac1*q_core*(p.c_se-p.c_le)*hp)
    ac = p.kappa*driving
    # D(cle*(s-1)) = cle*D(s). Form the liquid differences first: subtracting
    # two q values near -cle can erase small physical gradients in FP32.
    ch = p.n_ch*p.c_le*graph_action(liquid, data)
    return ac, ch


def residuals(raw, fields_t, data, c, fields=None):
    phi=(physical_fields(raw,c) if fields is None else fields)[...,0].T
    ac,ch=spatial_terms(phi,raw[...,1].T,data,c)
    return core_values(fields_t[...,0].T,data)+ac,core_values(fields_t[...,1].T,data)+ch


def eliminated_residuals(raw,raw_t,fields,fields_t,data,c):
    """R_phi,R_c-dc*hprime(phi)*R_phi with the raw liquid time derivative.

    This exact bounded row operation removes the KKS phi_t cross-term without
    subtracting nearly equal physical time derivatives in floating point.
    """
    if raw.shape!=raw_t.shape or fields.shape!=fields_t.shape or raw.shape!=fields.shape:
        raise ValueError('KKS row elimination requires aligned raw/physical fields and time derivatives')
    phi=fields[...,0].T;ga,gc=spatial_terms(phi,raw[...,1].T,data,c)
    core=data['core_indices'];hp=6*phi[core]*(1-phi[core])
    return (fields_t[:,core,0].T+ga,
            c.physical.c_le*raw_t[:,core,1].T+gc-(c.physical.c_se-c.physical.c_le)*hp*ga)


def integral_residuals(raw,fields,data,endpoints,c):
    from .time_integral import simpson_balance
    phi0=initial_phase(data['graph_points'],c)
    h0=phi0**2*(3-2*phi0)
    ga,gc=spatial_terms(fields[...,0].T,raw[...,1].T,data,c)
    ga0,gc0=spatial_terms(phi0[:,None],h0[:,None],data,c)
    ga=ga.at[:,0].set(ga0[:,0]);gc=gc.at[:,0].set(gc0[:,0])
    core=data['core_indices']
    return (simpson_balance(fields[::2,core,0].T,ga,phi0[core],endpoints),
            simpson_balance(fields[::2,core,1].T,gc,c.physical.c_se*h0[core],endpoints))


def weighted_mean(values, weights):
    """Spatial weighted mean, averaged over preceding time axes when present."""
    if values.shape[-1] != len(weights) or weights.ndim != 1:
        raise ValueError('Last value axis must align with the spatial measure')
    return jnp.mean(jnp.sum(values*weights, axis=-1)/jnp.sum(weights))


def condition_losses(net, times, data, c, causal_eps=None, return_aux=False):
    """One geometry, using prescribed initial/boundary targets only."""
    graph, ic, mouth = data['graph_points'], data['ic_points'], data['mouth_points']
    n_graph, n_ic = len(graph), len(ic)
    if (graph.ndim != 2 or graph.shape[1] != 2 or ic.ndim != 2 or ic.shape[1] != 2
            or mouth.ndim != 2 or mouth.shape[1] != 2 or data['initial_ic'].shape != (n_ic, 2)
            or data['pde_weights'].shape != data['core_indices'].shape):
        raise ValueError('Geometry query/initial/measure shapes do not match their declared roles')
    points = jnp.concatenate((graph, ic, mouth), axis=0)
    hard = 'phase' in c and 'initial_lifting' in c.phase
    exact_mouth = 'boundary' in c
    if hard:
        points = jnp.concatenate((graph, mouth), axis=0)
    if exact_mouth:
        points = graph
    cached = net.prepare_encoded(encode(net, data), points)
    evolution = cached if hard or exact_mouth else jnp.concatenate((cached[:n_graph], cached[n_graph+n_ic:]))
    evolution_points = graph if exact_mouth else jnp.concatenate((graph, mouth), axis=0)
    if 'time_form' in c.loss:
        from .time_integral import simpson_queries
        endpoints,queries=simpson_queries(times)
        raw=decode(net,evolution,queries,c,evolution_points)
        all_fields=physical_fields(raw,c,evolution_points,queries)
        ac,ch=integral_residuals(raw[:,:n_graph],all_fields[:,:n_graph],data,endpoints,c)
        fields=all_fields[::2]
    elif 'transport_residual' in c.loss:
        def evaluate(ts):
            raw=decode(net,evolution,ts,c,evolution_points)
            return raw,physical_fields(raw,c,evolution_points,ts)
        (raw,fields),(raw_t,fields_t)=jax.jvp(evaluate,(times,),(jnp.ones_like(times),))
        ac,ch=eliminated_residuals(raw[:,:n_graph],raw_t[:,:n_graph],fields[:,:n_graph],
                                   fields_t[:,:n_graph],data,c)
    else:
        raw, fields, fields_t = time_jets(net, evolution, times, c, evolution_points)
        ac, ch = residuals(raw[:, :n_graph], fields_t[:, :n_graph], data, c, fields[:, :n_graph])
    power = 2 if 'pde_reduction' in c.loss and c.loss.pde_reduction == 'mean_square' else 1
    parts = (
        c.loss.ac_scale*weighted_mean((jnp.abs(ac)**power).T, data['pde_weights']),
        c.loss.ch_scale*weighted_mean((jnp.abs(ch)**power).T, data['pde_weights']))
    diagnostics = jnp.empty((0, len(times)), dtype=times.dtype)
    if 'use_causal' in c.loss and c.loss.use_causal:
        if causal_eps is None:
            raise ValueError('Causal loss requires its dynamic epsilon state')
        from nonlocal_pinn.training.causal import CausalWeightor
        measure = data['pde_weights']/jnp.sum(data['pde_weights'])
        order = jnp.argsort(times)
        segments = jnp.stack((c.loss.ac_scale*jnp.sum(jnp.abs(ac)**power*measure[:, None], axis=0),
                              c.loss.ch_scale*jnp.sum(jnp.abs(ch)**power*measure[:, None], axis=0)))[:, order]
        total = jnp.sum(segments, axis=0)
        temporal_weights = CausalWeightor().compute_weights(total, causal_eps)
        parts = tuple(jnp.mean(segments*temporal_weights[None, :], axis=1))
        diagnostics = jnp.stack((temporal_weights, total))

    def pack(values):
        return (jnp.stack(values), diagnostics) if return_aux else jnp.stack(values)

    if exact_mouth:
        return pack(parts)
    concentration_scale = (c.loss.concentration_scale if 'concentration_scale' in c.loss
                           else 1/c.physical.c_le)
    mouth_fields = fields[:, n_graph:]
    mouth_concentration = (mouth_fields[..., 1]*concentration_scale
                           if 'concentration_scale' in c.loss
                           else mouth_fields[..., 1]/c.physical.c_le)
    parts = parts+(
        weighted_mean(mouth_fields[..., 0]**2, data['mouth_weights']),
        weighted_mean(mouth_concentration**2, data['mouth_weights']))
    if hard:
        return pack(parts)
    zero = jnp.zeros(1, dtype=times.dtype)
    initial_raw=decode(net,cached[n_graph:n_graph+n_ic],zero,c,ic)
    initial = physical_fields(initial_raw,c,ic,zero)[0]
    initial_error = initial-data['initial_ic']
    if 'initial_metric' in c.loss and c.loss.initial_metric=='phase_liquid':
        phi0=data['initial_ic'][:,0]
        h0=phi0**2*(3-2*phi0)
        liquid0=(data['initial_ic'][:,1]-(c.physical.c_se-c.physical.c_le)*h0)/c.physical.c_le
        initial_error=jnp.stack((initial_error[:,0],initial_raw[0,:,1]-liquid0),axis=-1)
    elif 'concentration_scale' in c.loss:
        initial_error = initial_error*jnp.asarray((1., concentration_scale), dtype=initial.dtype)
    else:
        initial_error = initial_error/jnp.asarray((1., c.physical.c_le), dtype=initial.dtype)
    return pack(parts+(weighted_mean(jnp.sum(initial_error**2, axis=-1), data['ic_weights']),))


def make_loss_vector(c):
    """Return loss(net,times,batched_payload); every geometry has its own points."""
    def loss(net, times, payload):
        return jax.vmap(lambda data: condition_losses(net, times, data, c))(payload).mean(axis=0)
    return loss


def make_loss_components(c):
    """Original loss with stopped causal weights and same-forward diagnostics."""
    def loss(net, times, payload, epsilon):
        values, diagnostics = jax.vmap(
            lambda data: condition_losses(net, times, data, c, epsilon, True))(payload)
        return values.mean(axis=0), diagnostics.mean(axis=0)
    return loss


def make_step(loss_vector, optimizer, c, loss_names):
    """Stopped inverse-gradient normalization and one update per full batch."""
    count = len(loss_names)
    bias = jnp.asarray(c.loss.gradnorm_bias, dtype=jnp.float32)
    if bias.shape != (count,):
        raise ValueError('Every declared loss requires an explicit gradnorm_bias')
    from .strategies import parameter_gradient_norm
    norm_kind=c.loss.gradnorm_norm if 'gradnorm_norm' in c.loss else 'l1'

    @eqx.filter_jit
    def step(net, state, weights, times, payload, refresh, active=None):
        values, pullback = eqx.filter_vjp(lambda model: loss_vector(model, times, payload), net)

        def update(_):
            def norm_for_loss(i):
                gradient = pullback(jax.nn.one_hot(i, count, dtype=values.dtype))[0]
                return parameter_gradient_norm(gradient,norm_kind)

            # Sequential map retains one pullback body in the compiled graph.
            # Return only scalar norms, never a stack of full parameter trees.
            raw_norms=jax.lax.map(norm_for_loss,jnp.arange(count,dtype=jnp.int32))
            norms=jnp.clip(raw_norms,1e-6,1e6)
            proposed = jnp.clip(jnp.mean(norms)/norms, 1e-6, c.loss.gradnorm_weight_cap)
            return (jax.lax.stop_gradient(c.train.alpha_gradnorm*proposed+(1-c.train.alpha_gradnorm)*weights),
                    jnp.all(jnp.isfinite(raw_norms)))

        weights,norm_finite=jax.lax.cond(refresh,update,lambda _:(weights,jnp.asarray(True)),operand=None)
        cotangent = weights*bias
        if active is not None:
            if active.shape != (count,) or active.dtype != jnp.bool_:
                raise ValueError('Stagger selector must be a fixed-shape boolean loss vector')
            cotangent = cotangent*active
        gradient = pullback(cotangent)[0]
        finite = (norm_finite & jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(weights)) &
                  jnp.all(weights > 0) & jnp.all(jnp.stack([
                      jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)
                      if eqx.is_array(value)])))
        updates, state = optimizer.update(gradient, state, net)
        return eqx.apply_updates(net, updates), state, weights, values, finite

    return step
