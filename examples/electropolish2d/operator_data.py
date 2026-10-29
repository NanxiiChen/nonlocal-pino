"""Random prescribed interfaces and strictly aligned same-graph references.

Only initial fields enter training. Existing independent physical-field BDF
integration supplies a small evaluation subset; no operator training labels.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.corrosion2d.residual_experiment import (
    boundary_values, generate_reference, grid, initial_fields, load_reference,
)


def dataset_spec(c):
    surface = c.surface.to_dict()
    if 'seed' in surface:
        del surface['seed']  # per-sample identities are stored, not strategy seeds
    return json.loads(json.dumps(dict(
        physical=c.physical.to_dict(), nondim=c.nondim.to_dict(), surface=surface,
        Nx=c.sample.Nx, Ny=c.sample.Ny, delta=c.nl.delta, kernel=c.nl.kernel,
        image_shape=list(c.operator.image_shape),
        train_conditions=c.operator.train_conditions,
        test_conditions=c.operator.test_conditions,
        eval_train_conditions=c.operator.eval_train_conditions,
        initial='existing piecewise-linear rough-profile SDF; exact nodal Dirichlet override',
        reference_snapshots=c.experiment.reference_snapshots,
        reference_error_tolerance=c.experiment.reference_error_tolerance)))


def condition_config(c, surface_id, path):
    resolved = c.copy_and_resolve_references()
    with resolved.surface.unlocked():
        resolved.surface.seed = int(surface_id)
    resolved.experiment.reference_path = str(path)
    return resolved



def training_condition_indices(c):
    """Actual eligible inputs; an omitted optional subset preserves the full set."""
    if 'training_indices' not in c.operator:
        return np.arange(c.operator.train_conditions, dtype=np.int32)
    requested = c.operator.training_indices
    indices = np.asarray(requested)
    if (indices.ndim != 1 or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or any(isinstance(i, (bool, np.bool_)) for i in requested)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= c.operator.train_conditions)):
        raise ValueError('operator.training_indices must be a nonempty unique integer subset '
                         'of the source training partition; held-out IDs are forbidden')
    return indices.astype(np.int32)


def sample_condition_indices(key, eligible, batch_size):
    """Draw fixed-size condition batches; arange population matches legacy draws."""
    positions = jax.random.choice(key, len(eligible), (batch_size,), replace=False)
    return jnp.asarray(eligible)[positions]


def condition_split(index, source_train_count, selected):
    """Keep excluded source-training monitors separate from fitted inputs."""
    if index >= source_train_count:
        return 'test'
    return 'train' if index in selected else 'unused_train'


def validate_config(c):
    o = c.operator
    for name in ('train_conditions', 'test_conditions', 'condition_batch', 'ic_points'):
        value = o[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'operator.{name} must be a positive integer')
    if not 1 <= o.eval_train_conditions <= o.train_conditions:
        raise ValueError('Need at least one training evaluation condition, no more than train set')
    if 'online' in o:
        from examples.electropolish2d.online_conditions import validate_online_config
        validate_online_config(c)
    if (('online' not in o and o.condition_batch > len(training_condition_indices(c)))
            or o.ic_points > c.sample.Nx * c.sample.Ny):
        raise ValueError('Condition batch/IC count exceeds dataset/grid')
    if o.concentration_map != 'linear_liquid':
        raise ValueError('Random-interface operator requires unrestricted linear_liquid KKS map')
    if 'local_input' in c.arch and c.arch.local_input not in ('initial', 'zero'):
        raise ValueError('arch.local_input must be initial or zero')
    if o.reference_action not in ('prepare', 'load'):
        raise ValueError('operator.reference_action must be prepare or load')
    if (o.image_shape[0] != 1 or c.nl.method not in ('sparse', 'local_ad')
            or c.experiment.case != 'electropolish2d' or c.experiment.formulation != 'phase'
            or c.experiment.data_loss_mode != 'physical'
            or c.experiment.transport_evaluation != 'physical_difference'
            or c.experiment.ch_norm != 'l1'):
        raise ValueError('Operator pilot requires the declared rectangular physical KKS graph/original residual')
    if c.nl.method == 'local_ad':
        if ('local_input' not in c.arch or c.arch.local_input != 'zero'
                or c.loss.use_causal or o.reference_action != 'load'):
            raise ValueError('Local operator requires explicit zero query features, noncausal training and reference load')
        if ('neumann_scale' not in c.loss or not np.isfinite(c.loss.neumann_scale)
                or c.loss.neumann_scale <= 0):
            raise ValueError('Local operator requires explicit positive loss.neumann_scale')
    if c.train.compute_dtype != 'float32' or c.train.gpu_policy != 'ordinary':
        raise ValueError('Operator pilot explicitly uses ordinary FP32')
    if not isinstance(c.loss.use_causal, bool):
        raise ValueError('loss.use_causal must be boolean')
    if c.loss.use_causal:
        causal = c.causal
        if (isinstance(causal.n_chunks, bool) or not isinstance(causal.n_chunks, int) or causal.n_chunks != 0
                or not all(np.isfinite(causal[name]) for name in (
                    'eps_init', 'max_eps', 'step_size', 'max_last_weight', 'min_mean_weight'))
                or not 0 < causal.eps_init <= causal.max_eps
                or causal.step_size <= 1
                or not 0 < causal.max_last_weight < 1
                or not 0 < causal.min_mean_weight < 1):
            raise ValueError('Operator causal requires per-time n_chunks=0 and valid explicit eps thresholds')
    if (c.loss.stagger_steps or c.loss.t_ramp_steps
            or c.loss.bias_schedule or c.loss.iface_gamma_init
            or c.loss.iface_gamma_decay_steps or c.loss.iface_gamma_decay_to):
        raise ValueError('Operator pilot does not implement stagger/curriculum/interface schedules')
    if (c.sample.K_core < 1 or c.sample.K_pool < 1 or c.sample.Nt < 1
            or c.train.iterations < 1 or c.train.gradnorm_update_every < 1
            or c.log.eval_every < 1 or c.log.log_every < 1
            or not 0 < c.sample.t_min < 1 or not c.log.save_ckpt or c.log.run_diag):
        raise ValueError('Invalid sampling/training/logging settings')
    if (c.physical.c_se != 1.0 or not 0 < c.physical.c_le < 1
            or any(not np.isfinite(c.physical[k]) or c.physical[k] <= 0
                   for k in ('D', 'A', 'L', 'alpha_phi', 'omega_phi'))
            or min(c.nondim.L_c, c.nondim.T_c) <= 0
            or c.loss.ac_scale <= 0 or c.loss.ch_scale <= 0):
        raise ValueError('Invalid physical KKS parameters/scales')
    bias = np.asarray(c.loss.gradnorm_bias)
    n_losses = 6 if c.nl.method == 'local_ad' else 5
    if bias.shape != (n_losses,) or np.any(bias <= 0) or not np.isfinite(bias).all():
        raise ValueError(f'Expected {n_losses} finite positive loss biases')


def prepare_dataset(c):
    validate_config(c)
    if jax.default_backend() != 'cpu' or not jax.config.x64_enabled:
        raise ValueError('Preparation requires JAX_PLATFORMS=cpu JAX_ENABLE_X64=true')
    root = Path(c.operator.dataset_path)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(c.train.seed)
    count = c.operator.train_conditions + c.operator.test_conditions
    # Distinct identities; train/test partition is fixed before solving references.
    ids = rng.choice(2**31, size=count, replace=False)
    _, _, points = grid(c)
    _, height, width = c.operator.image_shape
    xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, .5, height), indexing='xy')
    sensors = np.column_stack([xx.ravel(), yy.ravel()])
    zero, one, boundary = boundary_values(c, points)
    sensor_zero, sensor_one, sensor_boundary = boundary_values(c, sensors)
    images, initial = [], []
    eval_indices = list(range(c.operator.eval_train_conditions)) + list(range(c.operator.train_conditions, count))
    reference_paths = []
    for i, identity in enumerate(ids):
        path = root / f'condition_{i:03d}' / 'reference.npz'
        local = condition_config(c, identity, path)
        fields = initial_fields(local, points)
        fields[zero | one] = boundary[zero | one]
        sensor_fields = initial_fields(local, sensors)
        sensor_fields[sensor_zero | sensor_one] = sensor_boundary[sensor_zero | sensor_one]
        if not np.isfinite(fields).all() or fields.min() < -3e-7 or fields.max() > 1+3e-7:
            raise ValueError(f'Invalid prescribed initial fields for sample {i}')
        initial.append(fields.astype(np.float32))
        images.append(sensor_fields[:, 0].reshape(1, height, width).astype(np.float32))
        # Reference generator prescribes exactly this same SDF/BC/grid/operator.
        if i in eval_indices:
            generate_reference(local)
            reference_paths.append(str(path))
    np.savez_compressed(root / 'conditions.npz', surface_ids=ids,
                        images=np.stack(images), initial=np.stack(initial), points=points)
    metadata = dict(spec=dataset_spec(c), dataset_seed=c.train.seed,
                    train_indices=list(range(c.operator.train_conditions)),
                    test_indices=list(range(c.operator.train_conditions, count)),
                    eval_indices=eval_indices, reference_paths=reference_paths)
    (root / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return load_dataset(c)


def load_dataset(c):
    validate_config(c)
    root = Path(c.operator.dataset_path)
    metadata = json.loads((root / 'dataset.json').read_text())
    if metadata['spec'] != dataset_spec(c):
        raise ValueError(f'Condition dataset does not match requested physical/input specification: {root}')
    with np.load(root / 'conditions.npz') as f:
        data = {k: f[k] for k in f.files}
    count = c.operator.train_conditions + c.operator.test_conditions
    _, _, points = grid(c)
    if (data['images'].shape != (count, *c.operator.image_shape)
            or data['initial'].shape != (count, len(points), 2)
            or data['surface_ids'].shape != (count,)
            or len(np.unique(data['surface_ids'])) != count
            or not np.array_equal(data['points'], points)):
        raise ValueError('Condition shapes, identities or grid ordering mismatch')
    for name in ('images', 'initial'):
        a = data[name]
        if not np.isfinite(a).all() or a.min() < -3e-7 or a.max() > 1+3e-7:
            raise ValueError(f'Invalid condition data {name}')
    train_ids = list(range(c.operator.train_conditions))
    test_ids = list(range(c.operator.train_conditions, count))
    expected_eval = train_ids[:c.operator.eval_train_conditions] + test_ids
    if (metadata['train_indices'] != train_ids or metadata['test_indices'] != test_ids
            or metadata['eval_indices'] != expected_eval
            or len(metadata['reference_paths']) != len(expected_eval)):
        raise ValueError('Train/test/evaluation partition mismatch')
    zero, one, boundary = boundary_values(c, points)
    if not np.array_equal(data['initial'][:, zero | one],
                          np.broadcast_to(boundary[zero | one].astype(np.float32),
                                          data['initial'][:, zero | one].shape)):
        raise ValueError('Condition initial Dirichlet traces mismatch')
    refs = []
    for index, path in zip(expected_eval, metadata['reference_paths']):
        ref = load_reference(condition_config(c, data['surface_ids'][index], path))
        expected = np.stack([ref['phi'][0].ravel(), ref['c'][0].ravel()], -1)
        if not np.allclose(expected, data['initial'][index], rtol=0, atol=3e-7):
            raise ValueError(f'Initial condition/reference mismatch for sample {index}')
        refs.append(ref)
    return data, metadata, refs
