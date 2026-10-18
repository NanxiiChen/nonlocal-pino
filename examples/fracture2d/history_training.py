"""Online token damage network with replay collected from its own loading path."""
import json
from functools import partial
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.history_network import (
    HistoryOperator, HistoryReplay, build_patches, make_context)
from examples.fracture2d.history_probe import make_loss, make_step
from examples.fracture2d.optimization import make_optimizer, optimizer_count
from examples.fracture2d.sampling import pack_rows
from examples.fracture2d.storage import atomic_fields, atomic_json


def replay_active(update, *, enabled, every, inner_steps, current_only_steps):
    """Host schedule within one frozen-H fit; Adam state persists throughout."""
    return enabled and update % every == 0 and update < inner_steps-current_only_steps


class HistoryDamagePINN:
    """Physical history is owned by stagger; replay H is training input only."""

    def __init__(self, c, mechanics, operator):
        h = c.history
        self.conditioned = c.arch.type == 'conditioned_history_operator'
        self.energy = c.loss.norm == 'energy'
        if (c.arch.type not in ('history_operator', 'conditioned_history_operator') or c.sample.mode != 'full'
                or c.loss.norm != ('energy' if self.energy else 'l2' if self.conditioned else 'l1')
                or (self.energy and (not self.conditioned or c.damage.discretization != 'fem'))
                or c.loss.weight_mode != 'uniform'
                or c.train.optimizer != 'adam' or c.arch.output_head != 'sigmoid'
                or type(h.replay_enabled) is not bool or type(h.replay_capacity) is not int
                or h.replay_capacity < 2*round(c.loading.u_final/c.loading.du)
                or type(h.replay_batch) is not int or h.replay_batch < 1
                or not np.isfinite(h.replay_fraction) or not 0 < h.replay_fraction < 1):
            raise ValueError('online history model requires full/uniform/Adam/sigmoid, L1 (original), L2 (conditioned) or FEM energy (conditioned), and fixed replay capacity for two snapshots per load')
        if self.conditioned:
            bootstraps = ((h.bootstrap_energy_steps,) if self.energy else
                          (h.bootstrap_l1_steps, h.bootstrap_l2_steps))
            if (any(type(v) is not int or v < 1 for v in bootstraps)
                    or type(h.normalized) is not bool or type(h.local_history) is not bool):
                raise ValueError('conditioned history needs explicit positive bootstrap lengths and Boolean input modes')
        if (type(h.replay_every) is not int or h.replay_every < 1
                or type(c.train.inner_steps) is not int or c.train.inner_steps < 1
                or type(h.current_only_steps) is not int
                or not 0 <= h.current_only_steps <= c.train.inner_steps):
            raise ValueError('invalid replay cadence or current-only finishing length')
        self.completed_replay = 'replay_sampling' in h
        if self.completed_replay and (
                h.replay_sampling != 'completed_without_replacement'
                or not h.replay_enabled or not self.energy
                or 'rate_control' not in c.train or h.replay_batch > h.replay_capacity):
            raise ValueError('completed replay requires explicit FEM energy rate control and a valid fixed batch')
        self.c, self.n = c, mechanics.n_material
        self.gc0 = c.physical.Gc/c.physical.l0
        self.coarse = build_patches(mechanics.coordinates[:self.n], mechanics.volumes, cells=h.coarse_cells)
        self.fine = build_patches(mechanics.coordinates[:self.n], mechanics.volumes, cells=h.fine_cells)
        if type(h.fine_tokens) is not int or not 0 < h.fine_tokens <= int(self.fine.valid.sum()):
            raise ValueError('fine token budget exceeds nonempty patch count')
        low = mechanics.coordinates[:self.n].min(0)
        span = np.ptp(mechanics.coordinates[:self.n], axis=0)
        points = jnp.asarray(2*(mechanics.coordinates[:self.n]-low)/span-1)
        loss_factory, step_factory, network = make_loss, make_step, HistoryOperator
        network_modes = {}
        if self.conditioned:
            from examples.fracture2d.history_conditioning import (
                ConditionedHistoryOperator, make_conditioned_loss, make_conditioned_step)
            loss_factory, step_factory, network = make_conditioned_loss, make_conditioned_step, ConditionedHistoryOperator
            network_modes = dict(normalized=h.normalized, local_history=h.local_history)
        common = dict(coarse=self.coarse, fine=self.fine, points=points, gc0=self.gc0,
                      gc1=c.physical.Gc*c.physical.l0, floor=c.loss.floor)
        if self.energy:
            from examples.fracture2d.fem_damage import make_energy_loss
            image, objective = make_energy_loss(**common, operator=operator, volumes=mechanics.volumes)
            if self.completed_replay:
                from examples.fracture2d.fem_damage import make_shared_energy_loss
                shared_functions = make_shared_energy_loss(**common,
                    operator=operator, volumes=mechanics.volumes)
            step_factory = make_step
        else:
            D = operator.to_scipy()
            columns, values = pack_rows(D, width=int(np.max(np.diff(D.indptr))))
            image, objective = loss_factory(**common,
                columns=jnp.asarray(columns), values=jnp.asarray(values))
        self.image, self.value = eqx.filter_jit(image), eqx.filter_jit(objective)
        self.net = network(width=h.width, point_depth=h.point_depth,
            query_depth=h.query_depth, head_depth=h.head_depth, heads=h.heads,
            initial_damage=c.arch.initial_damage, key=jax.random.PRNGKey(c.train.seed), **network_modes)
        keys = ('learning_rate', 'decay_steps', 'decay_rate', 'minimum_rate',
            'grad_clip', 'b1', 'b2', 'eps', 'eps_root', 'shampoo_beta',
            'correct_bias', 'precondition_frequency', 'max_precond_dim',
            'precondition_1d', 'weight_decay')
        settings = {key:c.train[key] for key in keys}
        self.rate_control = None
        if 'rate_control' in c.train:
            from examples.fracture2d.learning_rate import BudgetLearningRate, make_rate_step
            if not self.energy or (c.history.replay_enabled and not self.completed_replay):
                raise ValueError('rate control requires FEM energy and explicit completed-history replay policy')
            self.rate_control = BudgetLearningRate(c.train.rate_control,
                initial_rate=c.train.learning_rate, minimum_rate=c.train.minimum_rate,
                decay_rate=c.train.decay_rate)
            settings.update(learning_rate=1., minimum_rate=1.)
            step_factory = (partial(make_rate_step, shared_functions=shared_functions)
                            if self.completed_replay else make_rate_step)
        self.optimizer, _ = make_optimizer(kind='adam', **settings)
        self.state = self.optimizer.init(eqx.filter(self.net, eqx.is_inexact_array))
        self.step = step_factory(objective, self.optimizer, old_count=h.replay_batch, fraction=h.replay_fraction)
        self.replay = HistoryReplay(capacity=h.replay_capacity, nodes=self.n, fine_count=h.fine_tokens)
        self.count, self.load_index, self.load = 0, 0, 0.
        self.phase, self.start_recorded = 'zero', False
        self.persisted_count = 0
        self.store = Path(c.output.root)/'replay_store'
        self.pool_info = None
        self.pre_update_loss, self.training_loss = 0., 0.

    def context(self, history):
        return make_context(history, self.fine, gc0=self.gc0, count=self.c.history.fine_tokens)

    def begin_load(self, index, load):
        if (type(index) is not int or index != self.load_index+1
                or self.phase not in ('zero', 'endpoint')
                or int(self.replay.count) != 2*(index-1)
                or not np.isfinite(load) or load <= self.load):
            raise ValueError('history replay load order/count differs from physical loading')
        self.load_index, self.load = index, load
        self.phase, self.start_recorded = 'in_load', False

    def _current_loss(self, context):
        batch = dict(H=context['H'][None], fine_ids=context['fine_ids'][None],
            weights=jnp.ones(1, jnp.float64))
        if self.conditioned and not self.energy:
            batch['l2'] = jnp.asarray(True)
        return float(self.value(self.net, batch))

    def solve(self, history, old_damage, load):
        del old_damage
        if self.phase != 'in_load' or load != self.load:
            raise ValueError('history solve must follow begin_load at the current load')
        context = self.context(history)
        if self.completed_replay:
            context = dict(context, replay_count=jnp.asarray(2*(self.load_index-1), jnp.int32))
        if not self.start_recorded:
            self.pre_update_loss = self._current_loss(context)
        bootstrap = self.conditioned and self.count == 0
        steps = self.c.train.inner_steps
        if bootstrap:
            steps = (self.c.history.bootstrap_energy_steps if self.energy else
                     self.c.history.bootstrap_l1_steps+self.c.history.bootstrap_l2_steps)
        rate_args = ()
        if self.rate_control is not None:
            from examples.fracture2d.learning_rate import effective_rate
            rate_args = (effective_rate(self.rate_control.rate),)
        for update in range(steps):
            key = jax.random.fold_in(jax.random.PRNGKey(self.c.train.seed+1729), self.count)
            active = replay_active(update, enabled=self.c.history.replay_enabled,
                every=self.c.history.replay_every, inner_steps=steps,
                current_only_steps=self.c.history.current_only_steps)
            args = ((jnp.asarray(not bootstrap or update >= self.c.history.bootstrap_l1_steps),)
                    if self.conditioned and not self.energy else ())
            if self.rate_control is not None:
                args = rate_args
            self.net, self.state, value = self.step(self.net, self.state, context,
                self.replay, key, jnp.asarray(active), *args)
            self.count += 1
            if bootstrap and self.count % 128 == 0:
                print(json.dumps(dict(event='history_bootstrap', load=load,
                    optimizer_count=self.count, total_updates=steps,
                    objective=('energy' if self.energy else 'l2' if bool(args[0]) else 'l1'),
                    loss=float(value))), flush=True)
        self.training_loss = float(value)
        if optimizer_count(self.state, kind='adam') != self.count:
            raise RuntimeError('online history Adam counter changed across outer updates')
        output = self.image(self.net, context)
        current_loss = self._current_loss(context)
        if not np.isfinite([self.training_loss, current_loss, self.pre_update_loss]).all() or not np.isfinite(np.asarray(output)).all():
            raise FloatingPointError('nonfinite online history damage/loss')
        if not self.start_recorded:
            # This H has now been encountered/trained; it is not replayed as
            # an old input during its first fit. Only once per physical load.
            self.replay = self.replay.append(context)
            self.start_recorded = True
        return output, current_loss

    def end_load(self, synchronized_history):
        if (self.phase != 'in_load' or not self.start_recorded
                or int(self.replay.count) != 2*self.load_index-1):
            raise ValueError('history endpoint is missing its first-outer snapshot')
        self.replay = self.replay.append(self.context(synchronized_history))
        self.phase = 'endpoint'

    def diagnostics(self):
        values = dict(replay_snapshots=int(self.replay.count),
                    replay_training_loss=self.training_loss, replay_pre_update_loss=self.pre_update_loss)
        if self.rate_control is not None:
            values.update(learning_rate=self.rate_control.rate,
                          learning_rate_index=self.rate_control.index)
        return values

    def save(self, folder):
        folder = Path(folder)
        if self.rate_control is not None:
            atomic_json(folder/'rate_control.json', self.rate_control.snapshot())
        eqx.tree_serialise_leaves(folder/'network.eqx', self.net)
        eqx.tree_serialise_leaves(folder/'optimizer.eqx', self.state)
        self.store.mkdir(parents=True, exist_ok=True)
        count = int(self.replay.count)
        for index in range(self.persisted_count, count):
            atomic_fields(self.store/f'snapshot_{index:04d}.npz',
                H=np.asarray(self.replay.histories[index]),
                fine_ids=np.asarray(self.replay.fine_ids[index]))
        self.persisted_count = count
        atomic_json(folder/'replay_state.json', dict(store=str(self.store),
            capacity=self.c.history.replay_capacity, snapshots=count,
            load_index=self.load_index, load=self.load, phase=self.phase,
            start_recorded=self.start_recorded, optimizer_count=self.count,
            pre_update_loss=self.pre_update_loss, training_loss=self.training_loss))

    def restore(self, folder, count):
        folder = Path(folder)
        if self.rate_control is not None:
            self.rate_control.restore(json.loads((folder/'rate_control.json').read_text()))
        saved = json.loads((folder/'replay_state.json').read_text())
        snapshots, capacity = saved['snapshots'], self.c.history.replay_capacity
        if (saved['store'] != str(self.store) or saved['capacity'] != capacity
                or type(snapshots) is not int or not 0 <= snapshots <= capacity
                or saved['optimizer_count'] != count or saved['phase'] != 'endpoint'
                or saved['start_recorded'] is not True
                or snapshots != 2*saved['load_index']):
            raise ValueError('saved endpoint replay manifest is inconsistent')
        histories = np.zeros(self.replay.histories.shape, np.float64)
        fine_ids = np.zeros(self.replay.fine_ids.shape, np.int32)
        for index in range(snapshots):
            path = self.store/f'snapshot_{index:04d}.npz'
            with np.load(path, allow_pickle=False) as data:
                H, ids = data['H'], data['fine_ids']
                if (H.shape != histories.shape[1:] or H.dtype != np.float64
                        or ids.shape != fine_ids.shape[1:] or ids.dtype != np.int32
                        or not np.isfinite(H).all() or H.min() < 0 or H.max() <= 0
                        or ids.min() < 0 or ids.max() >= len(self.fine.valid)
                        or not np.all(np.asarray(self.fine.valid)[ids])):
                    raise ValueError(f'invalid replay snapshot: {path}')
                histories[index], fine_ids[index] = H, ids
        self.net = eqx.tree_deserialise_leaves(folder/'network.eqx', self.net)
        self.state = eqx.tree_deserialise_leaves(folder/'optimizer.eqx', self.state)
        if optimizer_count(self.state, kind='adam') != count:
            raise ValueError('saved history optimizer counter differs')
        self.replay = eqx.tree_at(lambda b:(b.histories,b.fine_ids,b.count), self.replay,
            (jnp.asarray(histories),jnp.asarray(fine_ids),jnp.asarray(snapshots,jnp.int32)))
        self.count, self.persisted_count = count, snapshots
        self.load_index, self.load = saved['load_index'], saved['load']
        self.phase, self.start_recorded = saved['phase'], saved['start_recorded']
        self.pre_update_loss, self.training_loss = saved['pre_update_loss'], saved['training_loss']
