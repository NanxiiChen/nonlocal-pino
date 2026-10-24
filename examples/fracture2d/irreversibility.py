"""Optional current-driver FE energy with a lumped L1 healing penalty.

The bound belongs to the previous completed load, not the previous outer.
This finite penalty is approximate; maximum healing is a separate stop gate.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.fem_damage import make_energy
from examples.fracture2d.history_training import HistoryDamagePINN


def validate_penalty(c):
    s = c.irreversibility
    if (s.mode != 'current_psi_l1' or not np.isfinite([s.beta, s.healing_tolerance]).all()
            or s.beta <= 0 or not 0 < s.healing_tolerance <= 1e-3
            or c.damage.discretization != 'fem' or c.loss.norm != 'energy'
            or c.arch.type != 'conditioned_history_operator'
            or c.history.replay_enabled or 'replay_sampling' in c.history
            or 'rate_control' not in c.train or c.train.optimizer != 'adam'
            or c.loading.omega != 1. or not c.loading.field_stop
            or c.loading.advance_on_budget):
        raise ValueError('current-psi L1 penalty requires positive beta, healing tolerance in '
                         '(0,1e-3], FEM energy/conditioned Adam/rate control, replay OFF, '
                         'unrelaxed field stopping and no budget advancement')


def make_penalty_terms(operator, volumes, *, gc0, gc1, floor, beta):
    """J_psi/C + beta*gc0*integral relu(d_parent-d)/C; all inputs detached."""
    energy = make_energy(operator, volumes, gc0=gc0, gc1=gc1, floor=floor)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('L1 irreversibility beta must be finite and positive')
    V = jnp.asarray(volumes)

    def terms(damage, psi, parent):
        if parent.shape != V.shape or damage.shape != V.shape or psi.shape != V.shape:
            raise ValueError('penalty fields must share material-node ordering and shape')
        psi, parent = jax.tree.map(jax.lax.stop_gradient, (psi, parent))
        normalization = jnp.maximum(jnp.sum(V*2*psi**2/(gc0+2*psi)),
                                    gc0*jnp.sum(V)*floor)
        healing = jax.nn.relu(parent-damage)
        physical = energy(damage, psi)
        penalty = beta*gc0*jnp.sum(V*healing)/normalization
        return dict(energy_loss=physical, irreversibility_loss=penalty,
                    penalized_loss=physical+penalty,
                    maximum_healing=jnp.max(healing),
                    healing_volume_rms=jnp.sqrt(jnp.sum(V*healing**2)/jnp.sum(V)),
                    healing_volume_fraction=jnp.sum(V*(healing > 0))/jnp.sum(V))
    return terms


def make_penalty_step(objective, unit_optimizer):
    """Same warm Adam/unit-rate protocol, without a psi-only replay batch."""
    @eqx.filter_jit
    def step(net, state, current, replay, key, enabled, rate):
        del replay, key, enabled  # Configuration prohibits replay on this path.
        value, gradient = eqx.filter_value_and_grad(objective)(net, current)
        updates, state = unit_optimizer.update(gradient, state, eqx.filter(net, eqx.is_inexact_array))
        updates = jax.tree.map(lambda x: rate*x, updates)
        return eqx.apply_updates(net, updates), state, value
    return step


class CurrentEnergyDamagePINN(HistoryDamagePINN):
    """Same architecture; current psi condition, fixed load-parent constraint."""

    def __init__(self, c, mechanics, operator):
        validate_penalty(c)
        super().__init__(c, mechanics, operator)
        self.parent_damage = jnp.zeros(self.n, jnp.float64)
        self.terms = eqx.filter_jit(make_penalty_terms(operator, mechanics.volumes,
            gc0=self.gc0, gc1=c.physical.Gc*c.physical.l0,
            floor=c.loss.floor, beta=c.irreversibility.beta))
        image, terms = self.image, self.terms

        def objective(net, context):
            context = jax.tree.map(jax.lax.stop_gradient, context)
            return terms(image(net, context), context['H'], context['d_parent'])['penalized_loss']

        self.value = eqx.filter_jit(objective)
        self.step = make_penalty_step(objective, self.optimizer)
        self.penalty_diagnostics = {}

    def begin_load(self, index, load, *, parent_damage):
        parent = np.asarray(parent_damage)
        if (parent.shape != (self.n,) or parent.dtype != np.float64
                or not np.isfinite(parent).all() or parent.min() < 0 or parent.max() > 1):
            raise ValueError('irreversibility lower bound requires finite FP64 nodal damage in [0,1]')
        super().begin_load(index, load)
        self.parent_damage = jax.lax.stop_gradient(jnp.asarray(parent))

    def context(self, history):
        return dict(super().context(history), d_parent=self.parent_damage)

    def _current_loss(self, context):
        return float(self.value(self.net, context))

    def field_diagnostics(self, damage, psi):
        return {key:float(value) for key, value in
                self.terms(damage, psi, self.parent_damage).items()}

    def solve(self, history, old_damage, load):
        output, value = super().solve(history, old_damage, load)
        self.penalty_diagnostics = self.field_diagnostics(output, history)
        return output, value

    def diagnostics(self):
        return dict(super().diagnostics(), **self.penalty_diagnostics)
