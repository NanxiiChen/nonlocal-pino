"""Replay midpoint and late checkpoints on the same eight held-out ellipsoids."""
import argparse
import json
from pathlib import Path
import time


def finish(root):
    while True:
        journal = json.loads((root/'journal.json').read_text())
        if journal['status'] != 'running':
            break
        time.sleep(30)
    if journal['status'] != 'complete':
        raise RuntimeError('Cold training queue did not complete')
    import equinox as eqx
    import jax
    import numpy as np
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.icemelting3d.operator_data import build_network, load_dataset
    from examples.icemelting3d.operator_training import make_predictor, field_metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    jax.config.update('jax_enable_x64', False)
    stages = journal['stages']
    all_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for stage in stages:
        run = Path(stage['path'])
        c = ConfigDict(json.loads((run/'config.json').read_text()))
        configure_gpu_policy(c.train.gpu_policy)
        jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
        manifest, references = load_dataset(c)
        template = build_network(c, jax.random.key(0))
        predict = make_predictor(c)
        rows = []
        for step in (5000, 9000, 9250, 9500, 9750, 10000):
            checkpoint = run/('model_final.eqx' if step == 10000 else f'model_{step:06d}.eqx')
            net = eqx.tree_deserialise_leaves(checkpoint, template)
            values = []
            for index in manifest['test_indices']:
                ref = references[index]
                predicted = predict(net, ref['parameters'], ref['points'], ref['time'])
                scores = field_metrics(predicted, ref['phi'])
                if step == 10000:
                    with np.load(run/f'prediction_{index:03d}.npz') as stored:
                        np.testing.assert_allclose(predicted, stored['prediction'], rtol=0, atol=2e-5)
                values.append(dict(index=index, **scores))
            score = dict(step=step, checkpoint=str(checkpoint), conditions=values,
                         pooled_phi=float(np.mean([v['pooled_phi'] for v in values])),
                         terminal_phi=float(np.mean([v['terminal_phi'] for v in values])))
            rows.append(score)
            print(stage['mode'], step, score['pooled_phi'], score['terminal_phi'], flush=True)
        history = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        monitor = [h for h in history if h['split'] == 'monitor' and h['step'] > 0]
        axes[0].plot([h['step'] for h in monitor], [100*h['pooled_phi'] for h in monitor], label=stage['mode'])
        axes[1].plot([r['step'] for r in rows if r['step'] >= 9000],
                     [100*r['pooled_phi'] for r in rows if r['step'] >= 9000], 'o-', label=stage['mode'])
        # Match the physical sampling prefix, not just its nominal RNG seed.
        parent = Path('/root/autodl-tmp/tf-logs/icemelting3d')/f'pino-spatial-pools-20261111-01-icemelting3d-s43-{stage["mode"]}'
        with np.load(parent/'sampling.npz') as a, np.load(run/'sampling.npz') as b:
            for key in ('parameters', 'times', 'draw_ids'):
                np.testing.assert_array_equal(a[key], b[key][:5000])
            for key in ('core_ids', 'actual_nodes'):
                np.testing.assert_array_equal(a[key], b[key][:len(a[key])])
        all_rows.append(dict(mode=stage['mode'], run=str(run), checkpoints=rows,
                            original5000sampling_prefix_equal=True))
    for ax, title in zip(axes, ('Two fixed monitoring inputs', 'Eight test inputs, late checkpoints'), strict=True):
        ax.set_title(title)
        ax.set_xlabel('Optimizer update')
        ax.set_ylabel('Pooled phase relative L2 (%)')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle('Melting cold 10,000 updates, runtime seed 43')
    fig.savefig(root/'convergence.png', dpi=160)
    plt.close(fig)
    (root/'checkpoint-evaluation.json').write_text(json.dumps(dict(status='passed', seed=43, test_inputs=8,
        records=all_rows, scope='Same eight reused initial ellipsoids, native reference, fixed1024query chunks; late checkpoints are diagnostic, final model remains budget endpoint.'), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    finish(parser.parse_args().root)
