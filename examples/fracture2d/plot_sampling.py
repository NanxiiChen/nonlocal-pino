"""CPU-friendly visualization of the production finite core+halo sampler."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np
from scipy.sparse import load_npz

from examples.fracture2d.sampling import (
    crack_sampling_probabilities, make_sampling_rows, sample_core_halo,
    validate_probabilities)


def plot_sampling(c, *, output, pool_slot, dpi):
    """Reproduce the training pool without storing its large padded arrays."""
    if (c.sample.mode != 'random' or type(pool_slot) is not int
            or not 0 <= pool_slot < c.sample.pool_size or dpi < 1
            or c.sample.core_count < 1 or c.sample.resample_every < 1):
        raise ValueError('require random mode, valid pool slot, positive counts/DPI')
    jax.config.update('jax_enable_x64', True)
    with np.load(c.data.cloud, allow_pickle=False) as cloud:
        points = cloud['coordinates']
    root = Path(c.mesh.prepared_dir)
    with np.load(root/'mesh.npz', allow_pickle=False) as mesh:
        n = int(mesh['n_material'])
        if not np.array_equal(points, mesh['coordinates'][:n]):
            raise ValueError('cloud and prepared material node ordering differ')
    D = load_npz(root/'damage_operator.npz').tocsr()
    if (points.dtype != np.float64 or not np.isfinite(points).all()
            or points.shape != (n, 2) or D.shape != (n, n)):
        raise ValueError('invalid material points or prepared operator shape')
    s = c.sample
    p = np.asarray(crack_sampling_probabilities(
        jnp.asarray(points), start=s.start, end=s.end, width=s.width,
        boost=s.boost, power=s.power, tip_width=s.tip_width,
        tip_boost=s.tip_boost, tip_power=s.tip_power))
    validate_probabilities(p, population_size=n)
    a, b = np.asarray(s.start), np.asarray(s.end)

    def raw_density(x):
        direction = b-a
        t = np.clip((x-a)@direction / np.dot(direction, direction), 0., 1.)
        r2 = np.sum((x-(a+t[:, None]*direction))**2, axis=1)
        rt2 = np.sum((x-a)**2, axis=1)
        return (1+s.boost*(1+r2/s.width**2)**(-s.power/2)
                + s.tip_boost*(1+rt2/s.tip_width**2)**(-s.tip_power/2))

    raw = raw_density(points)
    np.testing.assert_allclose(p, raw/raw.sum(), rtol=2e-14, atol=1e-18)
    rng = np.random.default_rng(np.random.SeedSequence([s.seed, 0]))
    cores = rng.choice(n, size=(s.pool_size, s.core_count), replace=True, p=p)
    rows = make_sampling_rows(D, row_width=int(np.diff(D.indptr).max()))
    counts = np.array([np.unique(np.concatenate(
        (core, rows.columns[core][rows.values[core] != 0]))).size for core in cores])
    batch = sample_core_halo(rows, cores[pool_slot], node_capacity=int(counts.max()))
    union = batch['global_ids'][batch['node_mask']]
    unique_core = np.unique(cores[pool_slot])
    halo = np.setdiff1d(union, unique_core)
    np.testing.assert_array_equal(union[batch['core_local']], cores[pool_slot])
    assert len(unique_core)+len(halo) == counts[pool_slot]
    enrichment = n*p
    tip_mask = np.linalg.norm(points-a, axis=1) <= s.tip_width
    stats = dict(sample=c.sample.to_dict(), cloud=str(c.data.cloud),
        operator=str(root/'damage_operator.npz'), pool_slot=pool_slot,
        material_nodes=n, core_draws=s.core_count, unique_core=len(unique_core),
        halo_only=len(halo), batch_union=len(union), padded_forward=int(counts.max()),
        union_min=int(counts.min()), union_mean=float(counts.mean()),
        union_std=float(counts.std()), uniform_enrichment_range=[
            float(enrichment.min()), float(enrichment.max())],
        tip_radius=s.tip_width, tip_nodes=int(tip_mask.sum()),
        tip_expected_draws=float(s.core_count*p[tip_mask].sum()),
        tip_actual_draws=int(tip_mask[cores[pool_slot]].sum()),
        probability_sum=float(p.sum()), sampling_weights_detached=True)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout='constrained')
    ax = axes[0, 0]
    sc = ax.scatter(*points.T, c=enrichment, s=1.5, linewidths=0,
                    cmap='magma', norm=LogNorm(enrichment.min(), enrichment.max()),
                    rasterized=True)
    fig.colorbar(sc, ax=ax, label='Per-node probability / uniform probability (N p)')
    ax.set_title('(a) Actual normalized sampling density')

    def scatter_batch(ax, *, zoom):
        ax.scatter(*points.T, s=1 if zoom else .25, c='#dedede',
                   linewidths=0, rasterized=True)
        ax.scatter(*points[halo].T, s=10 if zoom else 2, c='#69b5e6',
                   alpha=.65, linewidths=0, rasterized=True, label='Halo only')
        ax.scatter(*points[unique_core].T, s=18 if zoom else 7, c='#d83237',
                   linewidths=0, alpha=.9, rasterized=True, label='Core (unique)')
        ax.legend(loc='upper right', framealpha=.95)

    ax = axes[0, 1]
    scatter_batch(ax, zoom=False)
    ax.set_title(f'(b) Pool slot {pool_slot}: {len(unique_core):,} core + {len(halo):,} halo')
    ax = axes[1, 0]
    scatter_batch(ax, zoom=True)
    ax.add_patch(Circle(a, s.tip_width, fill=False, color='#313131',
                        linestyle='--', linewidth=1.2))
    ax.set_xlim(a[0]-.07, a[0]+.13)
    ax.set_ylim(a[1]-.10, a[1]+.10)
    ax.set_title(f'(c) Notch-tip zoom; dashed radius = {s.tip_width:g}')
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.plot([points[:, 0].min(), a[0]], [a[1], a[1]],
                color='#202020', lw=1.5, label='_notch')
        ax.plot(*np.stack((a, b)).T, ls=':', color='#207c64', lw=1.1)
        ax.plot(*a, marker='x', color='#00836e', ms=7)
        ax.set(xlabel='x', ylabel='y', aspect='equal')
    ax = axes[1, 1]
    offset = np.linspace(-.15, .15, 1001)
    for fraction in (0., .3, .7):
        center = a+fraction*(b-a)
        profile = np.tile(center, (len(offset), 1))
        profile[:, 1] += offset
        ax.plot(offset, n*raw_density(profile)/raw.sum(), lw=2,
                label=f'x = {center[0]:.2f}')
    ax.axhline(1., color='grey', ls='--', lw=1, label='Uniform sampling')
    ax.axvline(0., color='#bbbbbb', lw=.8)
    ax.set(xlabel=f'Vertical offset from sampling segment: y - {a[1]:g}',
           ylabel='Per-node probability / uniform probability (N p)',
           title='(d) Transverse probability profiles')
    ax.grid(alpha=.18)
    ax.legend()
    fig.suptitle(
        f'Fracture2D core + halo | segment boost {s.boost:g}, tip boost {s.tip_boost:g}\n'
        f'{s.core_count:,} draws/batch; {s.pool_size:,} fixed batches; '
        f'switch every {s.resample_every} Adam update(s)\n'
        f'This batch: {len(union):,} unique nodes; padded forward: {counts.max():,} / {n:,}',
        fontsize=12)
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('output must have .png suffix')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    fig.savefig(output.with_suffix('.pdf'), dpi=dpi)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), **stats), indent=2))


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/
        'train_core_halo.py'), 'Sampling configuration', lock_config=True)
    flags.DEFINE_string('output', None, 'Output PNG path; PDF/JSON saved alongside')
    flags.mark_flag_as_required('output')
    flags.DEFINE_integer('pool_slot', 0, 'Zero-based batch index in the fixed pool')
    flags.DEFINE_integer('dpi', 180, 'Figure resolution')

    def _main(_):
        f = flags.FLAGS
        plot_sampling(f.config, output=f.output, pool_slot=f.pool_slot, dpi=f.dpi)
    app.run(_main)


if __name__ == '__main__':
    main()
