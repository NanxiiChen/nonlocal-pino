"""Freeze and independently solve a new64geometry particle evaluation cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import secrets
import shutil
import time

import numpy as np
from ml_collections import ConfigDict
from scipy import sparse

from examples.particle_corrosion2d.geometry import sample_geometry, GeometrySamplingExhausted
from examples.particle_corrosion2d.online import geometry_identity
from examples.particle_corrosion2d.inputs import conditioning
from examples.particle_corrosion2d.mesh import build_mesh
from examples.particle_corrosion2d.solver import solve

REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def solve_case(job):
    output, config, kind, index, record = job
    c = ConfigDict(config)
    c.mesh.boundary_points = 256
    c.nl.visibility_points = 256
    spacing = .0075 if kind == 'space' else .01
    rtol, atol, max_step = (1e-8, 1e-10, .003125) if kind == 'time' else (2e-8, 2e-10, .00625)
    mesh = build_mesh(c, record, spacing)
    matrix = sparse.diags(1/mesh['volumes'])@mesh['stiffness']
    times = np.linspace(0, c.sample.t_max, 11)
    fields, info = solve(c, matrix, mesh['points'], mesh['mouth'], times, rtol, atol, max_step)
    folder = Path(output) if kind == 'base' else Path(output)/'refinement'
    file = f'case_{index:03d}.npz' if kind == 'base' else f'{kind}_{index:03d}.npz'
    np.savez_compressed(folder/file, fields=fields, points=mesh['points'],
        triangles=mesh['triangles'], volumes=mesh['volumes'], mouth=mesh['mouth'],
        times=times, physical_time=times*c.nondim.T_c, **conditioning(c, record))
    result = dict(index=index, kind=kind, file=str((folder/file).relative_to(output)), geometry=record,
                  mesh=mesh['mesh_info'], solver=info)
    write((folder/file).with_suffix('.json'), result)
    return result


def interpolate(source, points):
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    from scipy.spatial import cKDTree
    tri = Triangulation(source['points'][:, 0], source['points'][:, 1], source['triangles'])
    inside = tri.get_trifinder()(points[:, 0], points[:, 1]) >= 0
    distance, ids = cKDTree(source['points']).query(points)
    exact = distance < 1e-12
    if not np.all(inside | exact):
        raise ValueError('Refinement query outside physical mesh; extrapolation forbidden')
    result = np.empty((len(source['times']), len(points), 2))
    for i in range(len(source['times'])):
        for channel in range(2):
            values = np.ma.asarray(LinearTriInterpolator(tri, source['fields'][i, :, channel])(
                points[:, 0], points[:, 1]))
            result[i, :, channel] = values.filled(np.nan)
            result[i, exact, channel] = source['fields'][i, ids[exact], channel]
    if not np.isfinite(result).all():
        raise ValueError('Incomplete physical P1 interpolation')
    return result


def sensitivity(base, reference):
    t, volumes = base['times'], base['volumes']
    tw = np.r_[(t[1]-t[0])/2, (t[2:]-t[:-2])/2, (t[-1]-t[-2])/2]
    weights = tw[:, None]*volumes[None, :]
    return {name: float(np.sqrt(np.sum(weights*(base['fields'][..., k]-reference[..., k])**2)/
                               np.sum(weights*reference[..., k]**2))) for k, name in enumerate(('phi', 'c'))}


def collect_exclusions(study):
    journal = read(study/'journal.json')
    excluded, paths = set(), []
    for run_record in journal['runs']:
        if run_record['case'] != 'particle_corrosion2d':
            continue
        run = Path(run_record['run'])
        runtime = read(run/'runtime.json')
        bank = Path(runtime['inputs'])/'manifest.json'
        paths.append(str(bank))
        excluded.update(geometry_identity(r) for r in read(bank)['geometries'])
        refresh = run/'online_refresh.jsonl'
        paths.append(str(refresh))
        for line in refresh.read_text().splitlines():
            for r in json.loads(line)['records']:
                excluded.add(geometry_identity(r['geometry']))
        reference = Path(runtime['reference'])/'manifest.json'
        paths.append(str(reference))
        excluded.update(geometry_identity(r['geometry']) for r in read(reference)['cases'])
    return excluded, sorted(set(paths))


def run(study, output, workers, cohort):
    if not 1 <= workers <= 16:
        raise ValueError('Use1..16independent single-thread reference workers')
    output.mkdir(parents=True, exist_ok=False)
    (output/'refinement').mkdir()
    source = next(r for r in read(study/'journal.json')['runs'] if r['case'] == 'particle_corrosion2d')
    old_reference = Path(read(Path(source['run'])/'runtime.json')['reference'])
    config = read(old_reference/'manifest.json')['config']
    c = ConfigDict(config)
    excluded, exclusion_paths = collect_exclusions(study)
    started = time.perf_counter()
    if cohort is None:
        seed = secrets.randbits(32)
        rng = np.random.default_rng(seed)
        geometries, rejections = [], []
        seen = set(excluded)
        for attempt in range(6400):
            try:
                record = sample_geometry(c, rng)
            except GeometrySamplingExhausted as error:
                rejections.append(dict(attempt=attempt, reason=str(error)))
                continue
            identity = geometry_identity(record)
            if identity in seen:
                rejections.append(dict(attempt=attempt, reason='exact_existing_geometry'))
                continue
            seen.add(identity)
            geometries.append(record)
            if len(geometries) == 64:
                break
    else:
        original = read(cohort)
        if original['config'] != config or original['count'] != 64:
            raise ValueError('Replay cohort configuration/count differs from the original study')
        geometries, seed, rejections = original['geometries'], original['seed'], original['rejections']
        identities = {geometry_identity(r) for r in geometries}
        if len(identities) != 64 or identities & excluded:
            raise ValueError('Replay cohort has duplicate or training/development geometries')
    if len(geometries) != 64:
        raise RuntimeError('Declared64case cohort not generated within6400attempts')
    plan = dict(status='frozen_before_solves_and_inference', created_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed, count=64, config=config, geometries=geometries, rejections=rejections,
        excluded_unique_geometries=len(excluded), exclusion_sources=exclusion_paths,
        training_geometry_overlap=0, old_development_overlap=0,
        spatial_check_indices=[0, 21, 42, 63], temporal_check_indices=[0],
        spacing=.01, boundary_points=256, refinement_spacing=.0075,
        spatial_phi_limit=.005, temporal_phi_limit=.0001,
        scheme='Same local P1/lumped-mass KKS BDF; polygon fixed for spatial refinement',
        purpose='Evaluation only; no reference labels in training or new model selection',
        replay_cohort=None if cohort is None else str(cohort))
    write(output/'cohort.json', plan)
    for src in (REPO/'examples/particle_corrosion2d').glob('*.py'):
        target = output/'source'/src.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    own = output/'source/docs/tasks/particle_reference64.py'
    own.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, own)
    print('COHORT_FROZEN', seed, len(excluded), flush=True)
    jobs = [(str(output), config, 'base', i, r) for i, r in enumerate(geometries)]
    jobs += [(str(output), config, 'space', i, geometries[i]) for i in plan['spatial_check_indices']]
    jobs += [(str(output), config, 'time', 0, geometries[0])]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        pending = [pool.submit(solve_case, job) for job in jobs]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            write(output/'progress.json', dict(completed=len(rows), total=len(jobs), seconds=time.perf_counter()-started,
                                              latest=dict(kind=row['kind'], index=row['index'], solver=row['solver'])))
            print('REFERENCE', len(rows), len(jobs), row['kind'], row['index'], row['solver']['seconds'], flush=True)
    checks = []
    for i in plan['spatial_check_indices']:
        base = arrays(output/f'case_{i:03d}.npz')
        refined = arrays(output/f'refinement/space_{i:03d}.npz')
        np.testing.assert_array_equal(base['times'], refined['times'])
        error = sensitivity(base, interpolate(refined, base['points']))
        checks.append(dict(kind='space', index=i, errors=error))
        if error['phi'] > plan['spatial_phi_limit']:
            raise ValueError(f'Spatial reference sensitivity exceeds0.5% at case{i}: {error}')
    base = arrays(output/'case_000.npz')
    refined = arrays(output/'refinement/time_000.npz')
    for name in ('points', 'triangles', 'times'):
        np.testing.assert_array_equal(base[name], refined[name])
    error = sensitivity(base, refined['fields'])
    if error['phi'] > plan['temporal_phi_limit']:
        raise ValueError(f'Temporal reference sensitivity exceeds0.01%: {error}')
    checks.append(dict(kind='time', index=0, errors=error))
    write(output/'validation.json', dict(status='passed', checks=checks, cohort=str(output/'cohort.json'),
        qualification='Spatial checks on4predeclared geometries, temporal check on1; not a whole-cohort continuum certificate'))
    manifest = dict(config=config, cases=sorted((r for r in rows if r['kind'] == 'base'), key=lambda r:r['index']),
        backend='local_p1_lumped_fem', actual_spacing=.01, actual_boundary_points=256, count=64,
        cohort=str(output/'cohort.json'), validation=str(output/'validation.json'), seconds=time.perf_counter()-started,
        use='Fresh64geometry evaluation only, generated after training without model-based selection')
    write(output/'manifest.json', manifest)
    print('REFERENCES_COMPLETE', output, manifest['seconds'], checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--cohort', type=Path,
                        help='Replay this frozen cohort.json instead of drawing new geometries')
    args = parser.parse_args()
    run(args.study.resolve(strict=True), args.output.resolve(), args.workers,
        None if args.cohort is None else args.cohort.resolve(strict=True))
