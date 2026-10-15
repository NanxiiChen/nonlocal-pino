"""Local direct mechanics and nonlocal damage PINN; ml_collections entry."""
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags, load_npz
from examples.fracture2d.losses import damage_weights, make_loss_fn
from examples.fracture2d.network import fresh_network, latent_basis_scaling, make_features
from examples.fracture2d.optimization import make_lbfgs, make_optimizer, make_step, optimizer_count
from examples.fracture2d.sampling import (build_finite_sampling_pool, crack_sampling_probabilities,
    make_pool_objective, make_sampling_rows, pack_rows)


class DamagePINN:
    """One network/optimizer across all loads; frozen H within each inner fit."""

    def __init__(self, c, mechanics, operator):
        self.c = c
        B = load_npz(c.data.basis_path)
        if B.shape[0] != mechanics.n_material or B.dtype != np.float64:
            raise ValueError('compact basis and FP64 material node ordering differ')
        self.features, self.feature_info = make_features(operator,
            mechanics.coordinates[:mechanics.n_material], gc0=c.physical.Gc/c.physical.l0,
            length=c.physical.l0, tip=list(c.features.tip), powers=list(c.features.powers),
            roundoff_factor=c.validation.roundoff_factor)
        self.features = eqx.filter_jit(self.features)
        self.net = fresh_network(c, self.feature_info['input_dimension'], B, head=c.arch.output_head)
        self.net, self.scales = latent_basis_scaling(self.net, B, np.ones(B.shape[1]))
        bc, bv = pack_rows(B @ diags(self.scales), width=int(np.max(np.diff(B.indptr))))
        D = operator.to_scipy()
        dc, dv = pack_rows(D, width=int(np.max(np.diff(D.indptr))))
        self.bc, self.bv, self.dc, self.dv = map(jnp.asarray, (bc, bv, dc, dv))
        self.n = mechanics.n_material
        self.image, objective = make_loss_fn(gc0=c.physical.Gc/c.physical.l0,
            gc1=c.physical.Gc*c.physical.l0, gain=c.arch.trunk_gain,
            population_size=self.n, norm=c.loss.norm, floor=c.loss.floor, head=c.arch.output_head)
        self.image, self.value = eqx.filter_jit(self.image), eqx.filter_jit(objective)
        self.pool, self.pool_info = None, None
        if c.sample.mode == 'random':
            rows = make_sampling_rows(D, row_width=dc.shape[1])
            probabilities = np.asarray(crack_sampling_probabilities(
                jnp.asarray(mechanics.coordinates[:self.n]), start=c.sample.start,
                end=c.sample.end, width=c.sample.width, boost=c.sample.boost, power=c.sample.power,
                tip_width=c.sample.tip_width, tip_boost=c.sample.tip_boost, tip_power=c.sample.tip_power))
            rng = np.random.default_rng(np.random.SeedSequence([c.sample.seed, 0]))
            cores = rng.choice(self.n, size=(c.sample.pool_size, c.sample.core_count),
                               replace=True, p=probabilities)
            pool, counts = build_finite_sampling_pool(rows, cores, probabilities)
            self.pool_info = dict(entries=c.sample.pool_size, core_count=c.sample.core_count,
                node_capacity=int(counts.max()), union_min=int(counts.min()),
                union_mean=float(counts.mean()), union_std=float(counts.std()),
                union_variance=float(counts.var()), material_nodes=self.n,
                array_bytes=sum(value.nbytes for value in pool.values()),
                seed=c.sample.seed, resample_every=c.sample.resample_every,
                selection='cyclic', regenerated_during_training=False,
                padding='exact_pool_maximum', support_truncated=False)
            self.pool = jax.device_put(pool)
            jax.block_until_ready(self.pool)
            objective = make_pool_objective(objective, basis_columns=self.bc, basis_values=self.bv)
        self.count = 0
        if c.train.optimizer == 'lbfgs':
            self.state = None
            self.fit, _ = make_lbfgs(self.net, objective, c.train.lbfgs)
        else:
            keys = ('learning_rate', 'decay_steps', 'decay_rate', 'minimum_rate',
                'grad_clip', 'b1', 'b2', 'eps', 'eps_root', 'shampoo_beta',
                'correct_bias', 'precondition_frequency', 'max_precond_dim',
                'precondition_1d', 'weight_decay')
            self.optimizer, self.schedule = make_optimizer(kind=c.train.optimizer,
                **{key: c.train[key] for key in keys})
            self.state = self.optimizer.init(eqx.filter(self.net, eqx.is_inexact_array))
            self.step = make_step(objective, self.optimizer)

    def context(self, history, weights):
        H = jax.lax.stop_gradient(history)
        return dict(H=H, core_H=H, features=jax.lax.stop_gradient(self.features(H)),
            basis_columns=self.bc, basis_values=self.bv,
            core_local=jnp.arange(self.n, dtype=jnp.int32),
            operator_columns=self.dc, operator_values=self.dv,
            p_core=jnp.full(self.n, 1/self.n, jnp.float64), weights=weights)

    def sampled_context(self, full):
        a = self.c.sample
        draw = self.count//a.resample_every
        # The counter also restores the exact slot after checkpoint resume.
        slot = draw % a.pool_size
        return dict(H=full['H'], features=full['features'], weights=full['weights'],
                    pool=self.pool, slot=jnp.asarray(slot, jnp.int32))

    def solve(self, history, old_damage, load):
        if not np.isfinite(np.asarray(history)).all() or float(jnp.max(history)) <= 0:
            raise ValueError('PINN inner fit requires finite positive forcing history')
        ctx = self.context(history, damage_weights(old_damage, self.c.loss))
        if self.c.train.optimizer == 'lbfgs':
            self.net, info = self.fit(self.net, ctx, jnp.asarray(load))
            self.count += info['iterations']
        else:
            for _ in range(self.c.train.inner_steps):
                batch = self.sampled_context(ctx) if self.c.sample.mode == 'random' else ctx
                self.net, self.state, loss, grad = self.step(
                    self.net, self.state, batch, jnp.asarray(load))
                self.count += 1
            if not np.isfinite([float(loss), float(grad)]).all():
                raise FloatingPointError('nonfinite damage loss/gradient')
            if optimizer_count(self.state, kind=self.c.train.optimizer) != self.count:
                raise RuntimeError('optimizer memory count changed across sampling/loads')
        # Global check only after a complete inner solve; no full metric per update.
        output = self.image(self.net, ctx)
        value = float(self.value(self.net, ctx, jnp.asarray(load)))
        if not np.isfinite(value) or not np.isfinite(np.asarray(output)).all():
            raise FloatingPointError('nonfinite full damage field/objective')
        return output, value

    def save(self, folder):
        eqx.tree_serialise_leaves(Path(folder)/'network.eqx', self.net)
        if self.state is not None:
            eqx.tree_serialise_leaves(Path(folder)/'optimizer.eqx', self.state)

    def restore(self, folder, count):
        self.net = eqx.tree_deserialise_leaves(Path(folder)/'network.eqx', self.net)
        if self.state is not None:
            self.state = eqx.tree_deserialise_leaves(Path(folder)/'optimizer.eqx', self.state)
            if optimizer_count(self.state, kind=self.c.train.optimizer) != count:
                raise ValueError('saved optimizer count differs')
        self.count = count


def run_training(config):
    from examples.fracture2d.loading import run
    return run(config, damage_solver='pinn')


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'train_base.py'),
                                   'Fracture2D PINN configuration', lock_config=True)
    def _main(_):
        run_training(flags.FLAGS.config)
    app.run(_main)


if __name__ == '__main__':
    main()
