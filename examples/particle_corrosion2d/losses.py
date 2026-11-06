"""Original KKS physics on a core/halo graph excluding inert ceramic particles."""
import equinox as eqx
import jax
import jax.numpy as jnp

from examples.particle_corrosion2d.network import decode, encode, physical_fields, time_jets


LOSS_NAMES = ('ac', 'ch', 'mouth_phi', 'mouth_c_over_cle', 'initial')


def graph_action(values, data):
    """Positive negative-Laplacian on core rows; every halo neighbor is explicit.

    values has shape (Nhalo,Nt). Padded core rows and neighbor slots have zero
    quadrature/edge weights. No transpose or graph diagonal is reconstructed.
    """
    core, neighbors, coefficients = data['core_indices'], data['neighbors'], data['coefficients']
    if (values.ndim != 2 or core.ndim != 1 or neighbors.ndim != 2
            or neighbors.shape != coefficients.shape or len(neighbors) != len(core)):
        raise ValueError('Core/halo operator requires values(Nhalo,Nt), core(Nc), neighbors/coefficients(Nc,K)')
    return jnp.sum(coefficients[:, :, None] *
                   (values[core, None, :] - values[neighbors]), axis=1)


def residuals(raw, fields_t, data, c):
    """Core AC/CH residuals with constant host-metal kinetic multiplier."""
    p = c.physical
    phi = physical_fields(raw, c)[..., 0].T
    q = (p.c_le*(raw[..., 1]-1)).T
    core = data['core_indices']
    phi_core, q_core = phi[core], q[core]
    hp = 6*phi_core*(1-phi_core)
    gp = 2*phi_core*(1-phi_core)*(1-2*phi_core)
    driving = (p.n_ac3*graph_action(phi, data)+p.n_ac2*gp
               -p.n_ac1*q_core*(p.c_se-p.c_le)*hp)
    ac = fields_t[:, core, 0].T+p.kappa*driving
    # D(cle*(s-1)) = cle*D(s). Form the liquid differences first: subtracting
    # two q values near -cle can erase small physical gradients in FP32.
    ch = fields_t[:, core, 1].T+p.n_ch*p.c_le*graph_action(raw[..., 1].T, data)
    return ac, ch


def weighted_mean(values, weights):
    """Spatial weighted mean, averaged over preceding time axes when present."""
    if values.shape[-1] != len(weights) or weights.ndim != 1:
        raise ValueError('Last value axis must align with the spatial measure')
    return jnp.mean(jnp.sum(values*weights, axis=-1)/jnp.sum(weights))


def condition_losses(net, times, data, c):
    """One geometry, using prescribed initial/boundary targets only."""
    graph, ic, mouth = data['graph_points'], data['ic_points'], data['mouth_points']
    n_graph, n_ic = len(graph), len(ic)
    if (graph.ndim != 2 or graph.shape[1] != 2 or ic.ndim != 2 or ic.shape[1] != 2
            or mouth.ndim != 2 or mouth.shape[1] != 2 or data['initial_ic'].shape != (n_ic, 2)
            or data['pde_weights'].shape != data['core_indices'].shape):
        raise ValueError('Geometry query/initial/measure shapes do not match their declared roles')
    points = jnp.concatenate((graph, ic, mouth), axis=0)
    cached = net.prepare_encoded(encode(net, data), points)
    evolution = jnp.concatenate((cached[:n_graph], cached[n_graph+n_ic:]))
    raw, fields, fields_t = time_jets(net, evolution, times, c)
    ac, ch = residuals(raw[:, :n_graph], fields_t[:, :n_graph], data, c)
    initial = physical_fields(decode(net, cached[n_graph:n_graph+n_ic],
                                    jnp.zeros(1, dtype=times.dtype), c)[0], c)
    initial_error = (initial-data['initial_ic']) / jnp.asarray((1., c.physical.c_le), dtype=initial.dtype)
    mouth_fields = fields[:, n_graph:]
    return jnp.stack((
        c.loss.ac_scale*weighted_mean(jnp.abs(ac).T, data['pde_weights']),
        c.loss.ch_scale*weighted_mean(jnp.abs(ch).T, data['pde_weights']),
        weighted_mean(mouth_fields[..., 0]**2, data['mouth_weights']),
        weighted_mean((mouth_fields[..., 1]/c.physical.c_le)**2, data['mouth_weights']),
        weighted_mean(jnp.sum(initial_error**2, axis=-1), data['ic_weights'])))


def make_loss_vector(c):
    """Return loss(net,times,batched_payload); every geometry has its own points."""
    def loss(net, times, payload):
        return jax.vmap(lambda data: condition_losses(net, times, data, c))(payload).mean(axis=0)
    return loss


def make_step(loss_vector, optimizer, c, loss_names):
    """Stopped inverse-L1 gradient normalization and one update per full batch."""
    count = len(loss_names)
    bias = jnp.asarray(c.loss.gradnorm_bias, dtype=jnp.float32)
    if bias.shape != (count,):
        raise ValueError('Every declared loss requires an explicit gradnorm_bias')

    @eqx.filter_jit
    def step(net, state, weights, times, payload, refresh):
        values, pullback = eqx.filter_vjp(lambda model: loss_vector(model, times, payload), net)

        def update(_):
            def norm_for_loss(i):
                gradient = pullback(jax.nn.one_hot(i, count, dtype=values.dtype))[0]
                return sum(jnp.sum(jnp.abs(value)) for value in jax.tree.leaves(gradient)
                           if eqx.is_array(value))

            # Sequential map retains one pullback body in the compiled graph.
            # Return only scalar norms, never a stack of full parameter trees.
            norms = jax.lax.map(norm_for_loss, jnp.arange(count, dtype=jnp.int32))
            norms = jnp.clip(norms, 1e-6, 1e6)
            proposed = jnp.clip(jnp.mean(norms)/norms, 1e-6, c.loss.gradnorm_weight_cap)
            return jax.lax.stop_gradient(c.train.alpha_gradnorm*proposed+(1-c.train.alpha_gradnorm)*weights)

        weights = jax.lax.cond(refresh, update, lambda _: weights, operand=None)
        gradient = pullback(weights*bias)[0]
        finite = (jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(weights)) &
                  jnp.all(weights > 0) & jnp.all(jnp.stack([
                      jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)
                      if eqx.is_array(value)])))
        updates, state = optimizer.update(gradient, state, net)
        return eqx.apply_updates(net, updates), state, weights, values, finite

    return step
