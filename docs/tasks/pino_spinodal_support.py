"""Known-input two-hop support and quadrature audit on actual Spinodal domains."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(output):
    from examples.confined_ch2d.configs.train_operator import get_config
    from examples.confined_ch2d.operator_core_halo import VolumeGroups
    from examples.confined_ch2d.operator_potential import compact_payload
    from examples.confined_ch2d.operator_problem import fixed_initial
    c = get_config()
    root = Path(c.operator.dataset_path)
    manifest = json.loads((root/'manifest.json').read_text())
    indices = manifest['train_indices'][:8]+manifest['test_indices']
    # Parameters are chosen before inspecting coverage or prediction errors.
    settings = [(k, size, width, walls) for k, size in ((64, 32), (128, 32), (128, 64), (256, 64))
                for width in (.12, .18) for walls in (0, 1)]
    records = []
    checks = 0
    for index in indices:
        path = root/manifest['conditions'][index]['file']
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ('points', 'volumes', 'neighbors', 'coefficients',
                                               'condition', 'condition_weights')}
        data['initial'] = fixed_initial(data['points'], c, xp=np)
        n = len(data['points'])
        nb, co = data['neighbors'], data['coefficients']
        rng = np.random.default_rng(12345+index)
        chi = rng.normal(size=n)
        D = lambda field: np.sum(co*(field[:, None]-field[nb]), axis=1)
        alpha = c.physical.epsilon**2/c.physical.beta
        phi = data['initial']+alpha*D(chi)
        mu = c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*D(phi)
        np.testing.assert_allclose(data['volumes']@phi, data['volumes']@data['initial'], atol=1e-12)
        for k, size, width, walls in settings:
            sampler = VolumeGroups(data, c, cell_width=width, wall_band=c.nl.delta)
            counts, estimates = [], []
            f = np.exp(-np.sum((data['points']-np.array([.42, .58]))**2, axis=1)/.01)
            exact = np.dot(data['volumes'], f)/data['volumes'].sum()
            for _ in range(16):
                core, weights = sampler.draw(core_count=k, group_size=size, wall_groups=walls, rng=rng)
                small, report = compact_payload(data, core, weights, inner_capacity=n, query_capacity=n)
                # Independent NumPy reconstruction checks the two actual rows.
                # Nearest-neighbor lookup is safe only after exact support closure.
                q = np.unique(np.r_[core, nb[core][co[core] != 0]])
                q = np.union1d(q, nb[q][co[q] != 0])
                z = np.pad(chi[q], (0, n-len(q)), mode='edge')
                p = small['inner_initial']+alpha*np.sum(small['inner_coefficients']*(
                    z[small['inner_indices'], None]-z[small['inner_neighbors']]), axis=1)
                pcore = p[small['core_inner_indices']]
                chem = c.physical.beta*(pcore**3-pcore)+c.physical.epsilon**2*np.sum(
                    small['core_coefficients']*(pcore[:, None]-p[small['core_neighbors']]), axis=1)
                np.testing.assert_array_equal(pcore, phi[core])
                np.testing.assert_array_equal(chem, mu[core])
                np.testing.assert_allclose(weights.sum(), 1., atol=1e-14)
                checks += 1
                counts.append(report)
                estimates.append(float(weights@f[core]-exact))
            records.append(dict(index=index, core=k, group_size=size, cell_width=width,
                wall_groups=walls, physical_nodes=n, counts=counts, synthetic_errors=estimates))
    summaries = []
    for k, size, width, walls in settings:
        rows = [r for r in records if (r['core'],r['group_size'],r['cell_width'],r['wall_groups']) == (k,size,width,walls)]
        counts = [count for r in rows for count in r['counts']]
        errors = [value for r in rows for value in r['synthetic_errors']]
        summaries.append(dict(core=k,group_size=size,cell_width=width,wall_groups=walls,
            median_inner=float(np.median([r['inner'] for r in counts])),
            maximum_inner=max(r['inner'] for r in counts),
            median_query=float(np.median([r['query'] for r in counts])),
            maximum_query=max(r['query'] for r in counts),
            median_query_fraction=float(np.median([r['query']/r['physical_nodes'] for r in counts])),
            synthetic_rmse=float(np.sqrt(np.mean(np.square(errors))))))
    result = dict(status='passed', source=str(root), geometry_indices=indices, checked_payloads=checks,
        support_hops=2, training_labels_loaded=False, scope='Support/quadrature only; not neural accuracy',
        settings=summaries, records=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key!='records'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
