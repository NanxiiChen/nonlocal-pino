"""Full-bank production-loss/SOAP GPU feasibility probe; no training promotion."""
import argparse
import importlib
import json
from pathlib import Path
import time

from ml_collections import ConfigDict
from nonlocal_pinn.training.runtime import configure_gpu_policy, with_run_seed


def probe(preset, prior, output, updates):
    if updates < 12 or output.exists():
        raise ValueError("Require at least12updates and a new output file")
    c = with_run_seed(importlib.import_module(preset).get_config(), None)
    baseline = with_run_seed(importlib.import_module(
        "examples.confined_ch2d.configs.train_operator_smallscale10_best").get_config(), None)
    changed = c.to_dict()
    changed["arch"]["width"] = 96
    assert changed == baseline.to_dict()
    configure_gpu_policy(c.train.gpu_policy)
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from examples.confined_ch2d.operator_training import build_network, load_bank, make_loss, operator_temporal_samples
    from examples.confined_ch2d.operator_problem import validate
    from nonlocal_pinn.training.optimizers import make_optimizer
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU feasibility requires a GPU, no CPU fallback")
    validate(c)
    start = time.perf_counter()
    print("GPU_PROBE loading full520geometry bank", flush=True)
    _, _, bank = load_bank(c)
    assert set(bank) == {"points", "volumes", "initial", "neighbors", "coefficients", "metric", "condition", "condition_weights"}
    keys = jax.random.split(jax.random.key(c.train.seed), 3)
    net = build_network(c, keys[0])
    old = ConfigDict(json.loads((prior / "config.json").read_text()))
    old_net = eqx.tree_deserialise_leaves(prior / "model_initial.eqx", build_network(old, jax.random.key(0)))
    np.testing.assert_array_equal(net.frequencies, old_net.frequencies)
    del old_net
    opt = make_optimizer(c)
    state = opt.init(eqx.filter(net, eqx.is_array))
    loss = make_loss(c)
    @eqx.filter_jit
    def step(model, state, batch, times):
        value, gradient = eqx.filter_value_and_grad(loss)(model, batch, times)
        finite = jnp.isfinite(value) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a)]))
        change, state = opt.update(gradient, state, model)
        return eqx.apply_updates(model, change), state, value, finite
    with np.load(prior / "training_draws.npz", allow_pickle=False) as f:
        prior_draws = f["condition_ids"][:updates]
    if prior_draws.shape != (updates, c.operator.condition_batch):
        raise ValueError("Prior must contain the matched cold sampling prefix")
    durations, losses = [], []
    print("GPU_PROBE compiling first production update", flush=True)
    for iteration in range(1, updates + 1):
        ids = jax.random.randint(jax.random.fold_in(keys[1], iteration), (c.operator.condition_batch,), 0, c.operator.train_conditions)
        np.testing.assert_array_equal(ids, prior_draws[iteration-1])
        batch = jax.tree.map(lambda a: a[ids], bank)
        time_key = jax.random.fold_in(keys[2], iteration)
        times, _ = operator_temporal_samples(c, iteration, time_key)
        expected, _ = operator_temporal_samples(baseline, iteration, time_key)
        np.testing.assert_array_equal(times, expected)
        tick = time.perf_counter()
        net, state, value, finite = step(net, state, batch, times)
        value.block_until_ready()
        durations.append(time.perf_counter()-tick)
        if not bool(finite):
            raise FloatingPointError(f"Nonfinite probe loss/gradient at {iteration}")
        losses.append(float(value))
        print(json.dumps(dict(probe_step=iteration, loss=losses[-1], seconds=durations[-1])), flush=True)
    memory = jax.devices()[0].memory_stats()
    ratio = memory["peak_bytes_in_use"] / memory["bytes_limit"]
    result = dict(status="passed" if ratio < .95 else "insufficient_memory_headroom", preset=preset,
                  configuration_change={"arch.width": [96, 128]}, updates=updates,
                  parameters=sum(a.size for a in jax.tree.leaves(net) if eqx.is_array(a))-net.frequencies.size,
                  bank_shapes={key: list(value.shape) for key, value in bank.items()},
                  fixed_fourier_identity=True, geometry_and_time_sampling_match_prior=True,
                  no_future_labels_in_step=True, all_losses_and_gradients_finite=True,
                  first_update_seconds=durations[0], warm_mean_seconds=float(np.mean(durations[2:])),
                  loss=losses, memory=memory, active_peak_fraction=ratio, wall_seconds=time.perf_counter()-start,
                  scope="Independent feasibility updates, discarded. Subsequent full training restarts from the common fresh runtime seed; not an accuracy or convergence result.")
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if result["status"] != "passed":
        raise RuntimeError("Less than5% allocator headroom; do not start long run")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    a = parser.parse_args()
    probe(a.config, a.prior.resolve(), a.output.resolve(), a.updates)
