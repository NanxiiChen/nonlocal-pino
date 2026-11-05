"""Original variable-mobility KKS residual; reference fields never enter here."""
import equinox as eqx
import jax
import jax.numpy as jnp

from examples.heterogeneous_corrosion2d.network import physical_fields, prepare
from examples.heterogeneous_corrosion2d.problem import nondim


LOSS_NAMES = ('ac', 'ch', 'mouth_phi', 'mouth_c_over_cle', 'initial')


def graph_action(values, sub_data):
    """Positive graph Laplacian in differences, with exact constant annihilation."""
    row, col, negative_weights, _ = sub_data
    return jax.ops.segment_sum(
        -negative_weights[:, None] * (values[row] - values[col]), row,
        num_segments=values.shape[0])


def residuals(raw, fields_t, kappa, sub_data, c):
    """Return (space,time) AC/CH residuals without mobility division.

    q=c-dc*h(phi)-cle=cle*(raw_liquid-1) is evaluated algebraically to avoid
    cancellation of O(1) mixture fields in the activation-controlled regime.
    kappa multiplies ALL AC driving terms and never the temporal derivative.
    """
    nch, n1, n2, n3, dc, cle = nondim(c)
    fields = physical_fields(raw, c)
    phi, q = fields[..., 0].T, (cle * (raw[..., 1] - 1)).T
    hp = 6 * phi * (1 - phi)
    gp = 2 * phi * (1 - phi) * (1 - 2 * phi)
    driving = n3 * graph_action(phi, sub_data) + n2 * gp - n1 * q * dc * hp
    return fields_t[..., 0].T + kappa[:, None] * driving, fields_t[..., 1].T + nch * graph_action(q, sub_data)


def make_loss_vector(c, n_graph, n_ic, n_bc):
    """Five physical losses over graph core, fixed IC and mouth nodes.

    payload=(images,kappa_graph,queries,initial_ic). The first two arrays vary
    per online material draw; coordinates and IC are shared across the batch.
    There are no soft losses on the remaining no-flux exterior graph boundary.
    """
    if min(n_graph, n_ic, n_bc) < 1:
        raise ValueError('Graph/initial/mouth query blocks must be nonempty')
    cle = c.physical.c_le

    def loss(net, times, payload, sub_data, core_mask):
        images, kappa_graph, queries, initial_ic = payload

        def one(image, kappa):
            cached = prepare(net, image, queries)
            evolution = jnp.concatenate((cached[:n_graph], cached[n_graph+n_ic:]))

            def at_times(ts):
                return net.decode(evolution, ts / c.sample.t_max)

            raw, raw_t = jax.jvp(at_times, (times,), (jnp.ones_like(times),))
            fields, fields_t = jax.jvp(lambda z: physical_fields(z, c), (raw,), (raw_t,))
            ac, ch = residuals(raw[:, :n_graph], fields_t[:, :n_graph], kappa, sub_data, c)
            mask = core_mask[:, None].astype(ac.dtype)
            count = jnp.sum(mask) * len(times)
            initial = physical_fields(net.decode(cached[n_graph:n_graph+n_ic],
                jnp.zeros(1, dtype=times.dtype))[0], c)
            ic_diff = (initial - initial_ic) / jnp.asarray((1., cle), dtype=initial.dtype)
            mouth = fields[:, n_graph:]
            return jnp.stack((
                c.loss.ac_scale * jnp.sum(jnp.abs(ac) * mask) / count,
                c.loss.ch_scale * jnp.sum(jnp.abs(ch) * mask) / count,
                jnp.mean(mouth[..., 0]**2),
                jnp.mean((mouth[..., 1] / cle)**2),
                jnp.mean(jnp.sum(ic_diff**2, axis=-1))))

        return jax.vmap(one)(images, kappa_graph).mean(axis=0)

    return loss


def make_step(loss_vector, optimizer, c, loss_names):
    """Stopped inverse-L1 gradient normalization; one optimizer update per batch."""
    bias = jnp.asarray(c.loss.gradnorm_bias, dtype=jnp.float32)
    if bias.shape != (len(loss_names),):
        raise ValueError('gradnorm_bias must name every declared loss')

    @eqx.filter_jit
    def step(net, state, weights, times, payload, sub_data, mask, refresh):
        values, pullback = eqx.filter_vjp(
            lambda model: loss_vector(model, times, payload, sub_data, mask), net)

        def update(_):
            norms = []
            for i in range(len(loss_names)):
                gradient = pullback(jax.nn.one_hot(i, len(loss_names)))[0]
                norms.append(sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(gradient)
                                 if eqx.is_array(a)))
            norms = jnp.clip(jnp.stack(norms), 1e-6, 1e6)
            proposal = jnp.clip(jnp.mean(norms) / norms, 1e-6, c.loss.gradnorm_weight_cap)
            return jax.lax.stop_gradient(c.train.alpha_gradnorm * proposal +
                                        (1 - c.train.alpha_gradnorm) * weights)

        weights = jax.lax.cond(refresh, update, lambda _: weights, operand=None)
        gradient = pullback(weights * bias)[0]
        finite = (jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(weights)) &
                  jnp.all(weights > 0) & jnp.all(jnp.stack([
                      jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(gradient)
                      if eqx.is_array(a)])))
        updates, state = optimizer.update(gradient, state, net)
        return eqx.apply_updates(net, updates), state, weights, values, finite

    return step
