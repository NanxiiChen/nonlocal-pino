"""Actual B4/Nt32 update and 64-case evaluation preflight, no geometry/PDE changes."""
import argparse
import importlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-active-gib', type=float, required=True)
    a = p.parse_args()
    c = importlib.import_module(a.config).get_config()
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from nonlocal_pinn.training.optimizers import make_optimizer
    from examples.particle_corrosion2d.execution import configure_execution, execution_provenance
    from examples.particle_corrosion2d.train import load_bank
    from examples.particle_corrosion2d.online import pad_initial_bank
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.losses import make_step, make_loss_vector, LOSS_NAMES
    from examples.particle_corrosion2d.spatial_adjoint import prepare_adjoint
    from examples.particle_corrosion2d.sampling import sample_times
    from examples.particle_corrosion2d.evaluate import load_references, make_evaluator
    configure_gpu_policy(c.train.gpu_policy); configure_execution(c)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    if jax.devices()[0].platform != 'gpu': raise ValueError('Real GPU probe required')
    bank, manifest = load_bank(a.inputs, c); bank = pad_initial_bank(bank, c)
    if not np.isfinite(a.max_active_gib) or not 0 < a.max_active_gib <= 85:
        raise ValueError('Explicit active-memory ceiling must be positive and at most85GiB')
    # Match the formal trainer's persistent GPU bank, not merely one host batch.
    gpu_bank = {key: jnp.asarray(value) for key, value in bank.items()}
    @jax.jit
    def gather(bank_arrays, ids, pools):
        data = {k: bank_arrays[k][ids] for k in manifest['fixed_keys']}
        data.update({k: bank_arrays[k][ids, pools] for k in manifest['graph_keys']})
        return jax.vmap(prepare_adjoint)(data)
    jax.block_until_ready(gpu_bank)
    rng = np.random.default_rng(42)
    ids = rng.integers(len(manifest['geometries']), size=(c.train.iterations, c.operator.condition_batch), dtype=np.int32)
    pools = rng.integers(c.sample.K_pool, size=ids.shape, dtype=np.int32)
    times = sample_times(rng, c)
    net = build_network(c, jax.random.PRNGKey(42))
    opt = make_optimizer(c); state = opt.init(eqx.filter(net, eqx.is_array))
    step = make_step(make_loss_vector(c), opt, c, LOSS_NAMES)
    weights = jnp.ones(5, dtype=jnp.float32)
    records = []
    for i in range(26):
        payload = gather(gpu_bank, jnp.asarray(ids[i]), jnp.asarray(pools[i])); jax.block_until_ready(payload)
        start = time.monotonic_ns()
        net, state, weights, parts, finite = step(net, state, weights, jnp.asarray(times[i]), payload,
                                                jnp.asarray(i % c.train.gradnorm_update_every == 0))
        jax.block_until_ready((net, state, weights, parts, finite))
        if not bool(finite): raise FloatingPointError(f'Nonfinite update {i+1}')
        row = dict(step=i+1, seconds=(time.monotonic_ns()-start)/1e9,
                   losses=np.asarray(parts).tolist(), weights=np.asarray(weights).tolist())
        records.append(row)
        if i in (0, 25): print(json.dumps(row), flush=True)
    refs, _ = load_references(a.reference, c)
    start = time.monotonic_ns(); predictions, metrics = make_evaluator(c, refs)(net)
    evaluation_seconds = (time.monotonic_ns()-start)/1e9
    for predicted in predictions:
        if not np.isfinite(predicted).all(): raise FloatingPointError('Nonfinite full-cohort prediction')
    memory = jax.devices()[0].memory_stats()
    acceptable = memory['peak_bytes_in_use'] <= a.max_active_gib*1024**3
    result = dict(status='passed' if acceptable else 'memory_rejected',
        memory_ceiling_gib=a.max_active_gib, persistent_gpu_bank=True,
        bank_bytes=sum(value.nbytes for value in bank.values()),
        config=c.to_dict(), inputs=str(a.inputs), reference=str(a.reference),
        steps=26, records=records, memory=memory, execution=execution_provenance(),
        evaluation_seconds=evaluation_seconds, test_count=len(refs), metrics=metrics,
        parameters=sum(v.size for v in jax.tree.leaves(net) if eqx.is_array(v)),
        scope='Finite preflight, not a trained-accuracy result; costs excluded from cold-training total')
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if not acceptable:
        raise MemoryError('Preflight exceeds declared active allocation ceiling; measurements retained')
    print('PASSED '+str(a.output), flush=True)


if __name__ == '__main__': main()
