"""Repeat the identical cold ceramic update; never advance a training trajectory."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def compare(first, second):
    if len(first) != len(second):
        raise ValueError('Replay output tree changed')
    count, maximum = 0, 0.
    for x, y in zip(first, second, strict=True):
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError('Replay shape or dtype changed')
        count += int(np.count_nonzero(x != y))
        maximum = max(maximum, float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if x.size else 0.)
    return dict(different_entries=count, maximum_absolute_difference=maximum, identical=count == 0)


def run(source, audit, output, policy, repeats):
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    flags = configure_gpu_policy(policy)
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.losses import make_loss_vector, make_step, LOSS_NAMES
    from nonlocal_pinn.training.optimizers import make_optimizer
    if repeats < 2 or any(device.platform != 'gpu' for device in jax.devices()):
        raise ValueError('At least two replays on an otherwise idle GPU are required')
    receipt = json.loads((audit/'audit.json').read_text())
    if receipt['status'] != 'passed' or Path(receipt['new']) != source:
        raise ValueError('The input audit does not qualify this source run')
    output.mkdir(parents=True, exist_ok=False)
    c = ConfigDict(json.loads((source/'config.json').read_text()))
    c.train.gpu_policy = policy
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    runtime = json.loads((source/'runtime.json').read_text())
    net = build_network(c, jax.random.PRNGKey(runtime['seed']))
    net = eqx.tree_deserialise_leaves(source/'model_initial.eqx', net)
    optimizer = make_optimizer(c)
    state = eqx.tree_deserialise_leaves(source/'optimizer_initial.eqx', optimizer.init(eqx.filter(net, eqx.is_array)))
    with np.load(audit/'first_batch.npz') as inputs:
        times = jnp.asarray(inputs['times'])
        plain = {k:jnp.asarray(inputs[k]) for k in inputs.files if k != 'times'}
    marked = dict(plain, pde_core_points=jax.vmap(lambda x,i:x[i])(plain['graph_points'],plain['core_indices']))
    weights = jnp.ones(len(LOSS_NAMES), dtype=jnp.float32)
    refresh = jnp.asarray(True)
    step = make_step(make_loss_vector(c), optimizer, c, LOSS_NAMES)
    lowered = [step.lower(net, state, weights, times, payload, refresh).as_text() for payload in (plain, marked)]
    rows = []
    first_global = None
    for label, payload in (('plain',plain),('marked',marked),('plain_again',plain)):
        first_group = None
        for index in range(repeats):
            tick = time.perf_counter()
            result = step(net, state, weights, times, payload, refresh)
            jax.block_until_ready(result)
            seconds = time.perf_counter()-tick
            if not bool(result[-1]):
                raise FloatingPointError('Cold replay produced a nonfinite training update')
            arrays = [np.asarray(a) for a in jax.tree.leaves(result) if eqx.is_array(a)]
            if first_group is None:first_group = arrays
            if first_global is None:first_global = arrays
            rows.append(dict(group=label, index=index, seconds=seconds,
                loss_parts=np.asarray(result[3]).tolist(), weights=np.asarray(result[2]).tolist(),
                versus_group_first=compare(first_group,arrays), versus_plain_first=compare(first_global,arrays)))
            print(json.dumps(rows[-1],allow_nan=False),flush=True)
    report = dict(status='passed',source=str(source),input_audit=str(audit),gpu_policy=policy,
        xla_flags=flags,jax_version=jax.__version__,devices=[str(d) for d in jax.devices()],
        stablehlo_identical=lowered[0] == lowered[1],repeats_per_group=repeats,records=rows,
        within_group_nondeterminism=any(not row['versus_group_first']['identical'] for row in rows),
        peak_active_bytes=jax.devices()[0].memory_stats()['peak_bytes_in_use'],
        scope='All calls start from exactly the same network, full optimizer, weights, times and physical batch',
        limitation='Single-update repeatability; this does not isolate the cause of the complete20k accuracy gap')
    (output/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'config.json').write_text(c.to_json_best_effort()+'\n')
    for label,text in zip(('plain','marked'),lowered,strict=True):
        (output/(label+'.mlir')).write_text(text)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--input-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-policy',choices=('ordinary','deterministic'),required=True)
    parser.add_argument('--repeats',type=int,required=True)
    args=parser.parse_args()
    run(args.source_run,args.input_audit,args.output,args.gpu_policy,args.repeats)
