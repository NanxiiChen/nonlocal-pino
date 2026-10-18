"""Optional host-side rate lowering after a failed outer-iteration budget."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class BudgetLearningRate:
    """Monotone rate ladder shared across loads; never resets Adam memory."""

    def __init__(self, settings, *, initial_rate, minimum_rate, decay_rate):
        rates = list(settings.rates)
        if (settings.mode not in ('budget_finishing', 'progress_blocks') or len(rates) < 2
                or not np.isfinite(rates).all() or min(rates) <= 0
                or any(a <= b for a, b in zip(rates, rates[1:]))
                or rates[0] != initial_rate or minimum_rate != initial_rate
                or decay_rate != 1.
                or type(settings.outers_per_rate) is not int
                or settings.outers_per_rate < 1):
            raise ValueError('budget rate control needs descending positive rates, '
                             'a constant initial schedule, and explicit outer block size')
        self.mode, self.rates = settings.mode, rates
        self.outers_per_rate = settings.outers_per_rate
        self.index = 0
        if self.mode == 'progress_blocks':
            if (type(settings.max_blocks_per_load) is not int or settings.max_blocks_per_load < 1
                    or not np.isfinite(settings.minimum_reaction_decrease)
                    or not 0 < settings.minimum_reaction_decrease < 1):
                raise ValueError('progress blocks need an explicit positive block cap and relative reaction decrease in (0,1)')
            self.max_blocks_per_load = settings.max_blocks_per_load
            self.minimum_reaction_decrease = settings.minimum_reaction_decrease

    @property
    def rate(self):
        return self.rates[self.index]

    def lower(self):
        if self.index == len(self.rates)-1:
            return False
        self.index += 1
        return True

    def snapshot(self):
        values = dict(mode=self.mode, rates=self.rates,
                      outers_per_rate=self.outers_per_rate, index=self.index)
        if self.mode == 'progress_blocks':
            values.update(max_blocks_per_load=self.max_blocks_per_load,
                          minimum_reaction_decrease=self.minimum_reaction_decrease)
        return values

    def restore(self, saved):
        expected = self.snapshot()
        if (set(saved) != set(expected)
                or any(saved[k] != expected[k] for k in expected if k != 'index')
                or type(saved['index']) is not int
                or not 0 <= saved['index'] < len(self.rates)):
            raise ValueError('saved budget learning-rate state differs from configuration')
        self.index = saved['index']

    def progress_action(self, *, block, previous_reaction, reaction):
        """Empirical progress heuristic; never a physical convergence gate."""
        if (self.mode != 'progress_blocks' or type(block) is not int or block < 1
                or not np.isfinite([previous_reaction, reaction]).all()
                or previous_reaction <= 0 or reaction < 0):
            raise ValueError('progress decision requires a valid block and finite positive previous reaction')
        decrease = (previous_reaction-reaction)/previous_reaction
        if block >= self.max_blocks_per_load:
            action = 'stop_budget'
        elif decrease > self.minimum_reaction_decrease:
            action = 'continue'
        else:
            action = 'lower' if self.lower() else 'stop_minimum'
        return dict(action=action, relative_decrease=decrease)


def make_rate_step(objective, unit_optimizer, *, old_count, fraction):
    """Scale unit-rate Adam updates with a dynamic scalar, preserving moments."""
    @eqx.filter_jit
    def step(net, state, current, replay, key, enabled, rate):
        batch = replay.batch(current, key, enabled, old_count=old_count, fraction=fraction)
        value, gradient = eqx.filter_value_and_grad(objective)(net, batch)
        updates, state = unit_optimizer.update(gradient, state, eqx.filter(net, eqx.is_inexact_array))
        updates = jax.tree.map(lambda x: rate*x, updates)
        return eqx.apply_updates(net, updates), state, value
    return step


def effective_rate(rate):
    """Match the existing constant Optax schedule's FP32 scalar in FP64 training."""
    return jnp.asarray(rate, jnp.float32).astype(jnp.float64)
