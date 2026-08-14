"""Optimizer factory: Adam and SOAP (vendored from SOAP_JAX)."""

import jax, jax.numpy as jnp, optax
from optax import GradientTransformation


def make_optimizer(config):
    """Build optimizer from config.  config.train.optimizer = 'adam' | 'soap'."""
    opt_name = config.train.optimizer
    if opt_name == 'rprop':
        inner = optax.rprop(
            learning_rate=config.train.rprop_initial_step_size,
            eta_minus=config.train.rprop_eta_minus,
            eta_plus=config.train.rprop_eta_plus,
            min_step_size=config.train.rprop_min_step_size,
            max_step_size=config.train.rprop_max_step_size,
        )
        return optax.chain(optax.clip_by_global_norm(config.train.grad_clip), inner)
    schedule = optax.exponential_decay(
        init_value=config.train.lr_init,
        transition_steps=config.train.lr_decay_every,
        decay_rate=config.train.lr_decay_rate,
        end_value=config.train.lr_min,
    )
    if opt_name == 'adam':
        inner = optax.adam(schedule)
    elif opt_name == 'soap':
        inner = soap(
            learning_rate=schedule,
            b1=config.train.soap_b1,
            b2=config.train.soap_b2,
            precondition_frequency=config.train.soap_precond_freq,
            max_precond_dim=config.train.soap_max_precond_dim,
            precondition_1d=config.train.soap_precond_1d,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    return optax.chain(optax.clip_by_global_norm(config.train.grad_clip), inner)


# ═══════════════════════════════════════════════════════════════════════════
# Vendored SOAP implementation (from github.com/haydn-jones/SOAP_JAX)
# ═══════════════════════════════════════════════════════════════════════════

def soap(learning_rate, b1=0.95, b2=0.95, shampoo_beta=-1, eps=1e-8,
         weight_decay=0.0, correct_bias=True, precondition_frequency=10,
         max_precond_dim=10000, precondition_1d=False):
    shampoo_beta = shampoo_beta if shampoo_beta >= 0 else b2
    return optax.chain(
        _scale_by_soap(b1, b2, shampoo_beta, eps, correct_bias,
                       precondition_frequency, max_precond_dim, precondition_1d),
        optax.scale_by_learning_rate(learning_rate),
    )


def _scale_by_soap(b1, b2, shampoo_beta, eps, correct_bias,
                   precondition_frequency, max_precond_dim, precondition_1d):
    def init_fn(params):
        exp_avg = jax.tree.map(jnp.zeros_like, params)
        exp_avg_sq = jax.tree.map(jnp.zeros_like, params)
        GG = _tmap(
            lambda p: _init_precond(p, max_precond_dim, precondition_1d), params)
        Q = _tmap(
            lambda p: _init_precond(p, max_precond_dim, precondition_1d), params)
        return dict(count=jnp.zeros([], jnp.int32), exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq, GG=GG, Q=Q)

    def update_fn(updates, state, params=None):
        del params
        count = optax.safe_int32_increment(state['count'])
        state['count'] = count

        def init_step(u, s):
            new_GG = _tmap(
                lambda g, gg: _update_precond(g, gg, shampoo_beta), u, s['GG'])
            new_Q = _tmap(lambda gg: _eigh_Q(gg), new_GG)
            return _tmap(jnp.zeros_like, u), {**s, 'GG': new_GG, 'Q': new_Q}

        def update_step(u, s):
            g_proj = _tmap(lambda g, q: _project(g, q), u, s['Q'])
            exp_avg = _tmap(lambda e, g: e * b1 + g * (1 - b1),
                            s['exp_avg'], g_proj)
            exp_avg_sq = _tmap(
                lambda e, g: e * b2 + jnp.sum(g * g) * (1 - b2),
                s['exp_avg_sq'], g_proj)
            norm_u = _tmap(
                lambda e, es, q: _project_back(e / (jnp.sqrt(es) + eps), q),
                exp_avg, exp_avg_sq, s['Q'])
            if correct_bias:
                eff = count - 1
                bc = jnp.sqrt(1 - b2**eff) / (1 - b1**eff)
                norm_u = _tmap(lambda x: x * bc, norm_u)
            new_GG = _tmap(
                lambda g, gg: _update_precond(g, gg, shampoo_beta), u, s['GG'])
            def refresh():
                return _tmap(
                    lambda es, gg, q: _qr_refresh(gg, q, es), exp_avg_sq, new_GG, s['Q'])
            def keep():
                return _tmap(lambda q, es: (q, es), s['Q'], exp_avg_sq)
            new_Q_es = jax.lax.cond((count - 1) % precondition_frequency == 0,
                                    refresh, keep)
            new_Q = _tmap(lambda _, x: x[0], s['Q'], new_Q_es)
            exp_avg_sq = _tmap(lambda _, x: x[1], s['Q'], new_Q_es)
            exp_avg = _tmap(
                lambda e, oq, nq: _project(_project_back(e, oq), nq),
                exp_avg, s['Q'], new_Q)
            return norm_u, {**s, 'exp_avg': exp_avg, 'exp_avg_sq': exp_avg_sq,
                            'GG': new_GG, 'Q': new_Q}

        updates, state = jax.lax.cond(count == 1, init_step, update_step,
                                      updates, state)
        return updates, state

    return GradientTransformation(init_fn, update_fn)


def _init_precond(p, max_dim, precond_1d):
    if p.ndim == 1:
        if not precond_1d or p.shape[0] > max_dim:
            return _P()
        return _P([jnp.zeros((p.shape[0], p.shape[0]))])
    return _P([jnp.zeros((s, s)) if s <= max_dim else _P.NONE
               for s in p.shape])


class _P:
    """Preconditioner matrices PyTree.  Wraps list of (matrix|NONE) for each tensor dim."""
    NONE = object()  # sentinel for "no preconditioner on this dimension"
    __slots__ = ('matrices',)
    def __init__(self, matrices=None):
        self.matrices = tuple(matrices) if matrices else ()
    def tree_flatten(self):
        mats = [(m if m is not _P.NONE else None) for m in self.matrices]
        return (mats, None)
    @classmethod
    def tree_unflatten(cls, _, mats):
        return cls([m if m is not None else _P.NONE for m in mats])

jax.tree_util.register_pytree_node(_P, _P.tree_flatten, _P.tree_unflatten)


def _is_p(x):
    return isinstance(x, _P)


def _tmap(f, *trees):
    """tree_map that treats _P nodes as leaves."""
    return jax.tree.map(f, *trees, is_leaf=_is_p)


def _update_precond(grad, pp, beta):
    if not pp.matrices:
        return pp
    if grad.ndim == 1:
        g = pp.matrices[0]
        return _P([g * beta + jnp.outer(grad, grad) * (1 - beta)])
    new = []
    for i, g in enumerate(pp.matrices):
        if g is _P.NONE:
            new.append(_P.NONE); continue
        axes = [d for d in range(grad.ndim) if d != i]
        outer = jnp.tensordot(grad, grad, axes=(axes, axes))
        new.append(g * beta + outer * (1 - beta))
    return _P(new)


def _project(grad, pp):
    if not pp.matrices:
        return grad
    if grad.ndim == 1:
        return pp.matrices[0].T @ grad
    for g in pp.matrices:
        if g is _P.NONE:
            grad = jnp.moveaxis(grad, 0, -1)
        else:
            grad = jnp.tensordot(grad, g, axes=((0,), (0,)))
    return grad


def _project_back(grad, pp):
    if not pp.matrices:
        return grad
    if grad.ndim == 1:
        return pp.matrices[0] @ grad
    for g in pp.matrices:
        if g is _P.NONE:
            grad = jnp.moveaxis(grad, -1, 0)
        else:
            grad = jnp.tensordot(grad, g, axes=((0,), (1,)))
    return grad


def _eigh_Q(pp):
    if not pp.matrices:
        return pp
    result = []
    for g in pp.matrices:
        if g is _P.NONE:
            result.append(_P.NONE); continue
        eps = jnp.asarray(1e-30, dtype=g.dtype)
        _, v = jnp.linalg.eigh(g + eps * jnp.eye(g.shape[0], dtype=g.dtype))
        result.append(jnp.flip(v, axis=1))
    return _P(result)


def _qr_refresh(pp, old_pp, es):
    if not pp.matrices:
        return _P(), es
    new_mats = []
    for i, (g, o) in enumerate(zip(pp.matrices, old_pp.matrices)):
        if g is _P.NONE or o is _P.NONE:
            new_mats.append(_P.NONE); continue
        est = jnp.diag(o.T @ g @ o)
        idx = jnp.argsort(est, descending=True)
        es = jnp.take(es, idx, axis=i)
        o_sort = o[:, idx]
        power = g @ o_sort
        q_new, _ = jnp.linalg.qr(power)
        new_mats.append(q_new)
    return _P(new_mats), es
