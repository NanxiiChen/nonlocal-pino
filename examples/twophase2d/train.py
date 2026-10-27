"""Canonical cold-start entry for the simple prescribed-flow phase-field pilot."""
from pathlib import Path
import sys
if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from absl import app, flags
from ml_collections import config_flags
import equinox as eqx
import jax
import jax.numpy as jnp
import json
import numpy as np
import shutil
import time
from datetime import datetime, timezone

from examples.twophase2d.problem import initial_field, validate_config, load_reference
from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.operators import PeriodicLaplacian
from examples.twophase2d.losses import make_loss_fn
from examples.twophase2d.fields import predict_fields
from nonlocal_pinn.training.runtime import with_run_seed, configure_gpu_policy
from nonlocal_pinn.training.optimizers import make_optimizer

_CONFIG = config_flags.DEFINE_config_file('config', str(Path(__file__).parent/'configs/train_base.py'))
_SEED = flags.DEFINE_integer('seed', None, 'Run provenance, not a strategy parameter')


def _build_network(config, key):
    with jax.enable_x64(False):net = PhaseNetwork(config, key)
    dtype = np.dtype(config.train.compute_dtype)
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, net)


def _make_step(loss_fn, optimizer):
    @eqx.filter_jit
    def step(net, state, times, initial_mean):
        (loss, parts), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(net, times, initial_mean)
        updates, state = optimizer.update(grads, state, net)
        net = eqx.apply_updates(net, updates)
        return net, state, loss, parts
    return step


def train(config):
    validate_config(config)
    configure_gpu_policy(config.train.gpu_policy)
    jax.config.update('jax_enable_x64', config.train.compute_dtype == 'float64')
    jax.config.update('jax_default_matmul_precision', config.train.matmul_precision)
    reference = load_reference(config)
    dtype = np.dtype(config.train.compute_dtype)
    points = jnp.asarray(reference['points'], dtype=dtype)
    kn, kt = jax.random.split(jax.random.PRNGKey(config.train.seed))
    net = _build_network(config, kn)
    D = None if config.nl.method == 'local_ad' else PeriodicLaplacian(
        config.sample.Nx, config.geometry.length, config.nl.delta, dtype=dtype)
    loss_fn = make_loss_fn(points, D, config)
    optimizer = make_optimizer(config)
    state = optimizer.init(eqx.filter(net, eqx.is_array))
    step_fn = _make_step(loss_fn, optimizer)
    initial_mean = jnp.mean(initial_field(points, config, xp=jnp))
    name = config.log.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-'+config.nl.method
    run = Path(config.log.log_dir)/name
    run.mkdir(parents=True, exist_ok=False)
    (run/'config.json').write_text(config.to_json(indent=2)+'\n')
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(Path(__file__).parent, run/'source/examples/twophase2d', ignore=shutil.ignore_patterns('__pycache__'))
    for relative in ('src/nonlocal_pinn/training/runtime.py', 'src/nonlocal_pinn/training/optimizers.py',
                     'src/nonlocal_pinn/nn/arch.py', 'src/nonlocal_pinn/nn/embedding.py'):
        target = run/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, target)
    eqx.tree_serialise_leaves(run/'model_initial.eqx', net)
    target_key = 'local' if D is None else 'nonlocal'
    target = reference['phi_'+target_key]
    times_eval = jnp.asarray(reference['time'], dtype=dtype)
    evaluate = eqx.filter_jit(lambda model: jax.vmap(lambda t: predict_fields(model, points, t, config))(times_eval))
    start = time.perf_counter()
    timing = []

    def assess(completed):
        physical = np.asarray(evaluate(net), dtype=np.float64).reshape(len(times_eval), config.sample.Ny, config.sample.Nx, 2)
        phi, q = physical[..., 0], physical[..., 1]
        if not np.isfinite(physical).all():raise FloatingPointError('Nonfinite predicted physical field')
        relative = lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
        gas = np.mean((1-phi)/2, axis=(1, 2))
        refgas = np.mean((1-target)/2, axis=(1, 2))
        mask = np.abs(target[-1]) < .9
        record = dict(step=completed, elapsed_seconds=time.perf_counter()-start,
            terminal_rel_l2_phi=relative(phi[-1], target[-1]), pooled_rel_l2_phi=relative(phi, target),
            per_time_rel_l2_phi=[relative(a, b) for a, b in zip(phi, target, strict=True)],
            terminal_rel_l2_phi_local=relative(phi[-1], reference['phi_local'][-1]),
            pooled_rel_l2_phi_local=relative(phi, reference['phi_local']),
            terminal_interface_rel_l2_phi=relative(phi[-1][mask], target[-1][mask]),
            terminal_q_rms_error=float(np.sqrt(np.mean((q[-1]-reference['q_'+target_key][-1])**2))),
            gas_area_relative_drift_max=float(np.max(np.abs(gas-refgas[0]))/refgas[0]),
            gas_area_reference_difference_terminal=float(abs(gas[-1]-refgas[-1])/refgas[-1]),
            phase_min=float(phi.min()), phase_max=float(phi.max()))
        with (run/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(f'[{config.nl.method} {completed}] phiL2={record["terminal_rel_l2_phi"]:.6g}, area drift={record["gas_area_relative_drift_max"]:.6g}', flush=True)
        return record, phi, q

    metrics, phi, q = assess(0)
    with (run/'loss.jsonl').open('x') as log:
        for completed in range(1, config.train.iterations+1):
            kt, sample_key = jax.random.split(kt)
            # Common FP32 time draws, promoted for an explicitly selected FP64 run.
            times = jax.random.uniform(sample_key, (config.sample.Nt,), dtype=jnp.float32)*config.physical.terminal_time
            if 'time_rule' in config.sample:
                # Keep a few representable values away from bin boundaries;
                # the Nt factor also covers rounding when adding a bin index.
                margin = 4*np.finfo(np.float32).eps*config.sample.Nt
                draws = jax.random.uniform(sample_key, (config.sample.Nt,),
                    minval=margin, maxval=1-margin,
                    dtype=jnp.float32)
                times = (jnp.arange(config.sample.Nt)+draws)*config.physical.terminal_time/config.sample.Nt
            times = times.astype(dtype)
            before = time.perf_counter()
            net, state, loss, parts = step_fn(net, state, times, initial_mean)
            loss.block_until_ready()
            timing.append(time.perf_counter()-before)
            if not np.isfinite(float(loss)):raise FloatingPointError(f'Loss diverged at update{completed}')
            if completed == 1 or completed % 100 == 0:
                log.write(json.dumps(dict(step=completed, total=float(loss), parts=np.asarray(parts).tolist()), allow_nan=False)+'\n');log.flush()
            if completed % config.log.eval_every == 0 or completed == config.train.iterations:
                metrics, phi, q = assess(completed)
            if completed % config.log.checkpoint_every == 0:
                eqx.tree_serialise_leaves(run/f'ckpt_{completed:06d}.eqx', net)
    eqx.tree_serialise_leaves(run/'model_final.eqx', net)
    refinement = None
    total_updates = config.train.iterations
    if 'refinement' in config.train:
        from examples.twophase2d.refinement import refine
        eqx.tree_serialise_leaves(run/'model_soap.eqx', net)
        np.savez_compressed(run/'prediction_soap.npz', points=reference['points'],
                            time=reference['time'], phi=phi, q=q)

        def refinement_assess(candidate, iteration):
            nonlocal net, metrics, phi, q
            net = candidate
            metrics, phi, q = assess(config.train.iterations+iteration)

        net, refinement = refine(net, loss_fn, initial_mean, config, run, refinement_assess)
        total_updates += refinement['iterations']
        if metrics['step'] != total_updates:
            metrics, phi, q = assess(total_updates)
        eqx.tree_serialise_leaves(run/'model_final.eqx', net)
    np.savez_compressed(run/'prediction.npz', points=reference['points'], time=reference['time'], phi=phi, q=q)
    summary = dict(updates=total_updates, soap_updates=config.train.iterations,
        refinement=refinement, method=config.nl.method,
        reference_path=config.reference.path, compute_dtype=config.train.compute_dtype,
        gpu_policy=config.train.gpu_policy, cold_start=True,
        wall_seconds=time.perf_counter()-start, first_update_seconds=timing[0],
        mean_update_seconds_after_100=float(np.mean(timing[100:])) if len(timing)>100 else None,
        final_metrics=metrics)
    (run/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def _main(_):train(with_run_seed(_CONFIG.value, _SEED.value))


if __name__ == '__main__':app.run(_main)
