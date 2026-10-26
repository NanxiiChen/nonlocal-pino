"""Select a saved phi checkpoint and render the requested 3-by-5 CPU field panel."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from ml_collections import ConfigDict
import numpy as np

from examples.corrosion2d.residual_experiment import build_model, load_reference

OUT = Path(__file__).with_suffix('')


def relative_l2(value, target):
    norm = np.linalg.norm(target)
    if norm <= 0:
        raise ValueError('Reference field must have a positive norm')
    return float(np.linalg.norm(value-target)/norm)


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('Run with JAX_PLATFORMS=cpu')
    records = []
    for filename in ('electropolish-optimization-controls.json',
                     'electropolish-optimization-validation.json'):
        records.extend(json.loads(OUT.with_name(filename).read_text()))
    assert len(records) == 9 and all(v['returncode'] == 0 for v in records)
    candidates = []
    for record in records:
        run = Path(record['path'])
        history = {v['step']: v for v in map(json.loads, (run/'metrics.jsonl').read_text().splitlines())}
        checkpoints = sorted(run.glob('ckpt_*.eqx'))
        assert len(checkpoints) == 5
        for checkpoint in checkpoints:
            step = int(checkpoint.stem.split('_')[1])
            score = history[step]['terminal_rel_l2_phi']
            assert np.isfinite(score) and score > 0
            candidates.append({'checkpoint': str(checkpoint), 'step': step,
                               'terminal_rel_l2_phi': score})
    ranked = sorted(candidates, key=lambda v: (v['terminal_rel_l2_phi'], v['checkpoint']))
    selected = ranked[0]
    checkpoint = Path(selected['checkpoint'])
    run = checkpoint.parent
    config = ConfigDict(json.loads((run/'config.json').read_text()))
    assert config.experiment.case == 'electropolish2d' and config.nondim.T_c == 5
    reference = load_reference(config)
    times = np.array([0., .25, .5, .75, 1.])
    indices = []
    for time in times:
        found = np.flatnonzero(np.isclose(reference['time'], time, rtol=0, atol=1e-12))
        assert len(found) == 1
        indices.append(int(found[0]))
    np.testing.assert_allclose(reference['physical_time'][indices], times*config.nondim.T_c,
                               atol=1e-12, rtol=0)
    X, Y = reference['X'], reference['Y']
    assert X.shape == Y.shape == (51, 101)
    x = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=jnp.float32)
    ts = jnp.asarray(reference['time'], dtype=jnp.float32)
    key = jax.random.split(jax.random.PRNGKey(config.train.seed))[0]
    model = eqx.tree_deserialise_leaves(checkpoint, build_model(config, key))

    @eqx.filter_jit
    def predict(net):
        return jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(x, t))(ts)

    prediction = np.asarray(predict(model)).reshape(len(ts), *X.shape, 2)[..., 0].astype(float)
    assert prediction.shape == reference['phi'].shape
    assert np.all(np.isfinite(prediction)) and prediction.min() >= 0 and prediction.max() <= 1
    terminal = relative_l2(prediction[-1], reference['phi'][-1])
    np.testing.assert_allclose(terminal, selected['terminal_rel_l2_phi'], atol=2e-6, rtol=0)
    # The selected saved checkpoint is the final model, so its full saved field is available.
    assert selected['step'] == config.train.iterations == 5000
    with np.load(run/'prediction.npz') as saved:
        for name in ('X', 'Y', 'time'):
            np.testing.assert_array_equal(saved[name], reference[name])
        replay_difference = float(np.max(np.abs(prediction-saved['phi'])))
        np.testing.assert_allclose(prediction, saved['phi'], atol=5e-6, rtol=0)
    target = reference['phi'][indices]
    predicted = prediction[indices]
    error = np.abs(predicted-target)
    l2 = [relative_l2(p, r) for p, r in zip(predicted, target)]
    maximum_error = float(error.max())
    assert maximum_error > 0 and np.all(np.isfinite(error))
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 5, figsize=(18, 6.8))
    fig.subplots_adjust(left=.13, right=.93, top=.86, bottom=.12, wspace=.12, hspace=.36)
    for row, fields in enumerate((target, predicted, error)):
        for column, time in enumerate(times):
            ax = axes[row, column]
            im = ax.pcolormesh(X, Y, fields[column], shading='nearest',
                              cmap='viridis' if row < 2 else 'magma',
                              vmin=0, vmax=1 if row < 2 else maximum_error,
                              rasterized=True)
            ax.set(aspect='equal', xlim=(0, 1), ylim=(0, .5),
                   xticks=[0, .5, 1], yticks=[0, .25, .5])
            if row == 0:
                ax.set_title(f'$t^* = {time:.2f}$', pad=9)
            if column != 0:
                ax.tick_params(labelleft=False)
            if row < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('$x^*$', labelpad=3)
            if row == 1:
                ax.text(.03, .06, f'Rel. L2 = {l2[column]:.2e}', transform=ax.transAxes,
                        fontsize=9, color='black',
                        bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 2})
            if column == 0:
                ax.set_ylabel('$y^*$', labelpad=3)
            if row == 1 and column == 4:
                field_image = im
            if row == 2 and column == 4:
                error_image = im
    for row, label in enumerate(('Reference\nsolution', 'Predicted\nfield', 'Absolute\nerror')):
        position = axes[row, 0].get_position()
        fig.text(.025, (position.y0+position.y1)/2, label, ha='left', va='center', fontsize=13)
    upper = axes[0, -1].get_position()
    middle = axes[1, -1].get_position()
    lower = axes[2, -1].get_position()
    field_axis = fig.add_axes([.945, middle.y0, .012, upper.y1-middle.y0])
    fig.colorbar(field_image, cax=field_axis, label=r'$\phi$', ticks=[0, .25, .5, .75, 1])
    error_axis = fig.add_axes([.945, lower.y0, .012, lower.height])
    cb = fig.colorbar(error_image, cax=error_axis, label=r'$|\phi_{\mathrm{PINN}}-\phi_{\mathrm{ref}}|$')
    cb.locator = MaxNLocator(nbins=4)
    cb.formatter = ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()
    fig.suptitle('Electropolish: reference, best saved prediction, and absolute error',
                 fontsize=17, y=.98)
    fig.text(.5, .91, 'Original residual | random IC/BC | seed 7 | checkpoint: 5000 updates',
             ha='center', fontsize=11)
    fig.text(.5, .025, r'$t^*=t/(5\,\mathrm{s})$; one checkpoint for all times; shared color scales across columns',
             ha='center', fontsize=11)
    for suffix in ('png', 'svg'):
        fig.savefig(OUT.with_suffix('.'+suffix), dpi=200, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    np.savez_compressed(OUT.with_suffix('.npz'), X=X, Y=Y, time=times,
                        physical_time=times*config.nondim.T_c,
                        reference_phi=target, predicted_phi=predicted, absolute_error=error)
    result = {'selection_rule': 'Lowest terminal phi relative L2 among45 saved checkpoints in current9-run cohort',
              'selected': selected, 'ranked_checkpoints': ranked,
              'reference_path': config.experiment.reference_path,
              'config_path': str(run/'config.json'), 'cpu_replay_max_abs_difference': replay_difference,
              'nondimensional_times': times.tolist(), 'physical_times_seconds': (times*config.nondim.T_c).tolist(),
              'relative_l2_phi_by_time': l2, 'terminal_phi_relative_l2_cpu': terminal,
              'max_absolute_error_by_time': error.max(axis=(1, 2)).tolist(),
              'color_limits': {'field': [0, 1], 'absolute_error': [0, maximum_error]},
              'note': 'One checkpoint for every column. Best by terminal phi only, not pooled error or transport. CPU replay.'}
    OUT.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('selected', 'relative_l2_phi_by_time',
          'max_absolute_error_by_time', 'cpu_replay_max_abs_difference')}, indent=2))


if __name__ == '__main__':
    main()
