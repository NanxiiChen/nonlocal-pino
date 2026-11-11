"""Measure exact stored spatial supports, without training or solving a PDE."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from examples.channel_nsch2d.configs.train_default import get_config as channel_config
from examples.channel_nsch2d.problem import walls
from examples.confined_ch2d.configs.train_operator import get_config as spinodal_config
from examples.confined_ch2d.operator_metrics import linear_stable_resolvent
from examples.confined_ch2d.operator_problem import fixed_initial

channel = Path('/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time18-nt36-cold20k-lr1e5-04')
spinodal = Path('/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1')
counts = (128, 256, 512, 1024)


def summary(seq):
    a = np.asarray(seq)
    return dict(min=float(a.min()), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max()))


def channel_stats():
    c = channel_config()
    data = {}
    for filename in ('known_inputs.npz', 'known_inputs_final.npz'):
        with np.load(channel / filename, allow_pickle=False) as f:
            bank = {k: f[k] for k in ('points', 'interior', 'volumes', 'transport_measure', 'amplitudes', 'neighbors',
                                    'laplacian', 'gradient_x', 'gradient_y', 'boundary', 'boundary_neighbors',
                                    'boundary_gradient_x', 'boundary_gradient_y')}
        x = np.linspace(0, c.geometry.length, 4097)
        results = {mode: {k: [] for k in counts} for mode in ('area', 'wall_strata', 'local_tiles')}
        wall_fractions = []
        bc_counts = []
        for i, (points, parameters, mask, weights) in enumerate(zip(bank['points'], bank['amplitudes'], bank['interior'], bank['transport_measure'], strict=True)):
            lower, upper = walls(x, parameters, c, xp=np)
            wall_points = np.column_stack((np.tile(x, 2), np.r_[lower, upper]))
            near = cKDTree(wall_points).query(points)[0] <= c.nl.delta
            free = np.flatnonzero(mask)
            wall = free[near[free]]
            bulk = free[~near[free]]
            prob = weights.astype(np.float64) / weights[free].sum(dtype=np.float64)
            wall_fractions.append(prob[wall].sum())
            nb = bank['neighbors'][i]
            active = np.logical_or.reduce([bank[k][i] != 0 for k in ('laplacian', 'gradient_x', 'gradient_y')])
            bn = bank['boundary_neighbors'][i]
            ba = (bank['boundary_gradient_x'][i] != 0) | (bank['boundary_gradient_y'][i] != 0)
            bc_support = np.unique(bn[ba])
            bc_counts.append(len(bc_support))
            # Disjoint 8x8 mapped-index tiles. Choose by each tile's summed
            # physical quadrature weight and average within that tile by area.
            ix, iy = np.divmod(free, c.sample.Ny + 1)
            tile_ids = (ix // 8) * ((c.sample.Ny + 8) // 8) + iy // 8
            tiles = [free[tile_ids == j] for j in np.unique(tile_ids)]
            tile_prob = np.asarray([prob[tile].sum() for tile in tiles])
            np.testing.assert_allclose(tile_prob.sum(), 1, rtol=0, atol=1e-12)
            for j in range(32):
                for k in counts:
                    rng = np.random.default_rng(np.random.SeedSequence([20261111, i, j, k]))
                    area = rng.choice(free, k, replace=True, p=prob[free])
                    strat = np.r_[rng.choice(wall, k // 2, replace=True, p=prob[wall] / prob[wall].sum()),
                                  rng.choice(bulk, k // 2, replace=True, p=prob[bulk] / prob[bulk].sum())]
                    tile_draws = rng.choice(len(tiles), k // 64, replace=True, p=tile_prob)
                    clustered = np.concatenate([tiles[tile] for tile in tile_draws])
                    for mode, core in (('area', area), ('wall_strata', strat), ('local_tiles', clustered)):
                        support = np.union1d(core, nb[core][active[core]])
                        union = np.union1d(support, bc_support)
                        if j == 0:
                            # Compare the stored operator rows before and after
                            # compact indexing. Zero slots use any finite index.
                            v = np.sin(points[:, 0]) * np.cos(points[:, 1])
                            compact = v[union]
                            old_neighbors = nb[core]
                            new_neighbors = np.searchsorted(union, np.where(active[core], old_neighbors, union[0]))
                            np.testing.assert_array_equal(union[new_neighbors][active[core]], old_neighbors[active[core]])
                            for name in ('laplacian', 'gradient_x', 'gradient_y'):
                                coeff = bank[name][i][core]
                                old = np.sum(coeff * (v[old_neighbors] - v[core, None]), axis=1)
                                new = np.sum(coeff * (compact[new_neighbors] - compact[np.searchsorted(union, core), None]), axis=1)
                                np.testing.assert_array_equal(old, new)
                            new_boundary = np.searchsorted(union, np.where(ba, bn, union[0]))
                            np.testing.assert_array_equal(union[new_boundary][ba], bn[ba])
                        results[mode][k].append([len(union), len(support), int(near[core].sum()),
                                                len(np.unique(core)), len(core)])
        data[filename] = dict(domain_nodes=len(points), pde_targets=len(free), separate_boundary_queries=len(bank['boundary'][0]),
            sampled_geometries=len(bank['points']), draws_per_geometry=32, wall_band=c.nl.delta,
            wall_weight=summary(wall_fractions), bc_halo=summary(bc_counts),
            modes={mode: {k: {name: summary(np.asarray(rows)[:, axis]) for axis, name in
                       enumerate(('full_union', 'pde_only', 'wall_cores', 'unique_cores', 'drawn_cores'))} for k, rows in by_k.items()}
                   for mode, by_k in results.items()})
    return data


def spinodal_stats():
    c = spinodal_config()
    manifest = json.loads((spinodal / 'manifest.json').read_text())
    p = c.physical
    out = []
    for i in manifest['test_indices']:
        entry = manifest['conditions'][i]
        with np.load(spinodal / entry['file'], allow_pickle=False) as data:
            v, nb, coeff, points = [data[k] for k in ('volumes', 'neighbors', 'coefficients', 'points')]
            initial = fixed_initial(points, c, xp=np)
            metric, _ = linear_stable_resolvent(v, nb, coeff, initial, p.beta, p.epsilon, p.mobility, p.terminal_time)
        nonzero = (metric.astype(np.float32) != 0).sum(axis=1)
        out.append(dict(case=i, nodes=len(v), nonzero_row_entries_min=int(nonzero.min()),
                        nonzero_row_entries_max=int(nonzero.max())))
    return out


if __name__ == '__main__':
    print(json.dumps(dict(channel=channel_stats(), spinodal=spinodal_stats()), indent=2))
