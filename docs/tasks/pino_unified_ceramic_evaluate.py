"""Final-weight common64 evaluation for new rolling spatial controls."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ml_collections import ConfigDict


def evaluate(run, reference, output):
    import equinox as eqx
    import jax
    from examples.particle_corrosion2d.evaluate import load_references, make_evaluator
    from examples.particle_corrosion2d.network import build_network
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    if json.loads((run/'summary.json').read_text())['steps'] != c.train.iterations:
        raise ValueError('Only completed fixed-budget final checkpoints are evaluated')
    if json.loads((reference/'validation.json').read_text())['status'] != 'passed':
        raise ValueError('Local reference validation has not passed')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    refs, manifest = load_references(reference, c)
    if len(refs) != 64 or manifest['backend'] != 'local_p1_lumped_fem':
        raise ValueError('Exactly64common local-FEM cases required')
    root = Path(__file__).resolve().parents[2]
    for source in ('examples/particle_corrosion2d/network.py', 'src/nonlocal_pinn/nn/point_conditional.py',
                   'src/nonlocal_pinn/nn/conditional.py'):
        if (root/source).read_bytes() != (run/'source'/source).read_bytes():
            raise ValueError(f'Inference source changed since training: {source}')
    model = eqx.tree_deserialise_leaves(run/'model_final.eqx', build_network(c, jax.random.PRNGKey(0)))
    started = time.perf_counter()
    predictions, physical = make_evaluator(c, refs)(model)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, (prediction, ref) in enumerate(zip(predictions, refs, strict=True)):
        p, y = prediction.astype(np.float64), ref['fields'].astype(np.float64)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f'Invalid physical prediction in case{i}')
        numerator = np.sum((p-y)**2, axis=(0, 1))
        denominator = np.sum(y*y, axis=(0, 1))
        if np.any(denominator <= 0):
            raise ValueError('Channel relative error denominator must be positive')
        errors = np.sqrt(numerator/denominator)
        records.append(dict(index=i, phi=float(errors[0]), concentration=float(errors[1]),
            channel_mean=float(errors.mean()), numerator=numerator.tolist(), denominator=denominator.tolist(),
            physical_weighted=physical['conditions'][i]))
        np.savez_compressed(output/f'prediction_{i:03d}.npz', prediction=prediction,
                            points=ref['points'], times=ref['times'])
    values = np.array([[r[k] for k in ('phi', 'concentration', 'channel_mean')] for r in records])
    statistics = {key: dict(mean=float(values[:, j].mean()), std=float(values[:, j].std(ddof=1)))
                  for j, key in enumerate(('phi', 'concentration', 'channel_mean'))}
    result = dict(status='passed', run=str(run), reference=str(reference), checkpoint='model_final.eqx',
        count=64, steps=c.train.iterations, statistics=statistics, records=records,
        norm='Per-channel Euclidean norm over all space-time evaluation nodes, then channel/sample means',
        supplementary_norm='Original physical-volume/trapezoidal-time scores retained per case',
        evaluation_seconds=time.perf_counter()-started, training_performed=False)
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(statistics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.reference, args.output)
