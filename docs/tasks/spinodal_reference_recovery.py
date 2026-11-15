"""Explicit bounded mesh refinement of an incomplete Spinodal reference bank.

Run in fenicsenv with the GPU idle. Qualified prior fields are checked and copied,
never solved again. Every new trajectory uses the unchanged mixed-P2 solver.
"""
import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from docs.tasks.electro_threeway_queue import read
from docs.tasks.spinodal_local_reference import relative, write

ROOT = Path(__file__).resolve().parents[2]
TAGS = ('space-coarse', 'time-coarse', 'fine')
NUMERICAL_SOURCES = (
    'examples/confined_ch2d/solver.py', 'examples/confined_ch2d/geometry.py',
    'examples/confined_ch2d/problem.py', 'examples/confined_ch2d/native_reference.py',
    'examples/confined_ch2d/operator_problem.py',
)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def inside(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Reference paths must remain within their explicit bank')
    return root/path


def checks(data, solutions):
    coarse, time_coarse, fine = solutions
    for solution in solutions:
        np.testing.assert_array_equal(solution['points'], data['points'])
        for key in ('phi', 'mu'):
            if solution[key].shape != data[key].shape:
                raise ValueError('Refinement fields do not share the native query/time shape')
        if any(not np.isfinite(solution[key]).all() for key in ('phi', 'mu', 'mass', 'energy')):
            raise ValueError('Nonfinite local reference fields or invariants')
    result = {}
    for name, other in (('space', coarse), ('time', time_coarse)):
        for field, start, label in (('phi', 0, 'phase'), ('mu', 1, 'chemical_positive')):
            result[name+'_'+label] = max(relative(a, b, data['volumes'])
                for a, b in zip(other[field][start:], fine[field][start:], strict=True))
    result['mass_drift'] = float(np.max(np.abs(fine['mass']-fine['mass'][0])))
    result['energy_increase'] = float(max(0., np.max(np.diff(fine['energy']))))
    result['native_local_phase'] = relative(data['phi'], fine['phi'], data['volumes'])
    result['native_local_chemical'] = relative(data['mu'], fine['mu'], data['volumes'])
    return result


def passes(values, c):
    return (max(values['space_phase'], values['time_phase']) < c.reference.phase_tolerance
        and max(values['space_chemical_positive'], values['time_chemical_positive']) < c.reference.chemical_tolerance
        and values['mass_drift'] < 1e-8 and values['energy_increase'] < 1e-8)


def spatial_failure_only(values, c):
    return (values['time_phase'] < c.reference.phase_tolerance
        and values['time_chemical_positive'] < c.reference.chemical_tolerance
        and values['mass_drift'] < 1e-8 and values['energy_increase'] < 1e-8
        and (values['space_phase'] >= c.reference.phase_tolerance
             or values['space_chemical_positive'] >= c.reference.chemical_tolerance))


def plan(prior, specification, expected, finer_mesh, c):
    if prior['status'] != 'failed' or prior['specification'] != specification:
        raise ValueError('Recovery requires a failed bank with identical prescribed setup')
    if prior['indices'] != expected or [x['index'] for x in prior['cases']] != expected[:len(prior['cases'])]:
        raise ValueError('Previous cohort must be an ordered prefix of the complete reserved cohort')
    if not finer_mesh or not np.isfinite(finer_mesh).all() or any(h <= 0 for h in finer_mesh):
        raise ValueError('Explicit finite positive refinement mesh sizes required')
    last = c.reference.mesh_sizes[-1]
    for h in finer_mesh:
        if h >= last:
            raise ValueError('Additional mesh sizes must strictly decrease below the original fine size')
        last = h
    expected_thresholds = dict(phase=c.reference.phase_tolerance,
        chemical_positive=c.reference.chemical_tolerance, mass=1e-8, energy_increase=1e-8)
    if prior['refinement_thresholds'] != expected_thresholds:
        raise ValueError('Recovery must retain the original reference tolerances')
    return expected_thresholds


def prepare(c, source, output, finer_mesh):
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('FEM reference recovery requires idle GPU: '+occupied)
    from examples.confined_ch2d.operator_problem import case_config, specification, validate, valid_holes
    from examples.confined_ch2d.solver import fem
    validate(c)
    native_root = Path(c.operator.dataset_path)
    bank, prior = read(native_root/'manifest.json'), read(source/'manifest.json')
    spec = specification(c)
    if bank['status'] != 'passed' or bank['specification'] != spec:
        raise ValueError('Native bank must retain the same qualified physical specification')
    thresholds = plan(prior, spec, bank['test_indices'], finer_mesh, c)
    if source.resolve() == output.resolve():
        raise ValueError('Recovery must preserve the failed bank in a separate directory')
    for name in NUMERICAL_SOURCES:
        if (source/'source'/name).read_bytes() != (ROOT/name).read_bytes():
            raise ValueError('Reference numerical implementation changed: '+name)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        native_bank=str(native_root), specification=spec, cases=[], indices=bank['test_indices'],
        kind=prior['kind'], initial_chemical_convention=prior['initial_chemical_convention'],
        refinement_thresholds=thresholds,
        recovery=dict(source=str(source), finer_mesh_sizes=list(finer_mesh),
            verified_unchanged_numerical_sources=list(NUMERICAL_SOURCES), reused_cases=[], attempts=[]))
    sources = [Path(__file__), ROOT/'docs/tasks/spinodal_local_reference.py',
        ROOT/'docs/tasks/spinodal_threeway_support.py', *sorted((ROOT/'examples/confined_ch2d').rglob('*.py'))]
    manifest['sources'] = [str(p.relative_to(ROOT)) for p in sources]
    for p in sources:
        target = output/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write(output/'manifest.json', manifest)
    previous = {x['index']: x for x in prior['cases']}
    begin = time.perf_counter()
    try:
        for index in bank['test_indices']:
            record = bank['conditions'][index]
            if record['index'] != index:
                raise ValueError('Native condition ordering changed')
            data = arrays(inside(native_root, record['file']))
            np.testing.assert_array_equal(data['holes'], record['holes'])
            np.testing.assert_array_equal(data['time'], np.linspace(0., c.physical.terminal_time, c.reference.snapshots))
            if not valid_holes(data['holes'], c):
                raise ValueError('Invalid prescribed reference geometry')
            cc = case_config(c, data['holes'], index)
            directory = output/f'case_{index:04d}'
            directory.mkdir()
            cc.reference.root = str(directory)
            if index in previous:
                old = previous[index]
                old_directory = source/f'case_{index:04d}'
                paths = [old_directory/(tag+'-fields.npz') for tag in TAGS]
                solutions = [arrays(path) for path in paths]
                values = checks(data, solutions)
                for key, value in values.items():
                    np.testing.assert_allclose(value, old['checks'][key], rtol=2e-12, atol=1e-14)
                target = arrays(inside(source, old['file']))
                for key in ('points', 'volumes', 'time', 'holes'):
                    np.testing.assert_array_equal(target[key], data[key])
                for key in ('phi', 'mu', 'mass', 'energy'):
                    np.testing.assert_array_equal(target[key], solutions[-1][key])
                mesh_sizes = list(old['mesh_sizes'])
                if old['status'] == 'passed':
                    if not passes(values, c):
                        raise ValueError('Prior passed reference fails independent recheck')
                    shutil.copyfile(inside(source, old['file']), output/old['file'])
                    for tag, path in zip(TAGS, paths, strict=True):
                        shutil.copyfile(path, directory/(tag+'-fields.npz'))
                    row = copy.deepcopy(old)
                    row['reused_from'] = str(source)
                    manifest['cases'].append(row)
                    manifest['recovery']['reused_cases'].append(index)
                    write(output/'manifest.json', manifest)
                    continue
                if old['status'] != 'failed' or not spatial_failure_only(values, c):
                    raise ValueError('This bounded recovery handles spatial refinement failure only')
            else:
                mesh_sizes = list(c.reference.mesh_sizes)
                solutions = [fem(cc, mesh_sizes[0], c.reference.refined_dt, data['points'], 'initial-space'),
                    fem(cc, mesh_sizes[1], c.reference.dt, data['points'], 'initial-time'),
                    fem(cc, mesh_sizes[1], c.reference.refined_dt, data['points'], 'initial-fine')]
                paths = [directory/(tag+'-fields.npz') for tag in ('initial-space', 'initial-time', 'initial-fine')]
                values = checks(data, solutions)
            manifest['recovery']['attempts'].append(dict(index=index, mesh_sizes=mesh_sizes,
                paths=list(map(str, paths)), checks=values, passed=bool(passes(values, c))))
            write(output/'manifest.json', manifest)
            for level, h in enumerate(finer_mesh):
                if passes(values, c):
                    break
                if not spatial_failure_only(values, c) or h >= mesh_sizes[-1]:
                    raise ValueError('Unexpected non-spatial failure or non-refining mesh request')
                old_fine, old_path = solutions[-1], paths[-1]
                coarse_tag, fine_tag = f'refined-{level}-time', f'refined-{level}-fine'
                solutions = [old_fine, fem(cc, h, c.reference.dt, data['points'], coarse_tag),
                    fem(cc, h, c.reference.refined_dt, data['points'], fine_tag)]
                paths = [old_path, directory/(coarse_tag+'-fields.npz'), directory/(fine_tag+'-fields.npz')]
                mesh_sizes = [mesh_sizes[-1], h]
                values = checks(data, solutions)
                manifest['recovery']['attempts'].append(dict(index=index, mesh_sizes=mesh_sizes,
                    paths=list(map(str, paths)), checks=values, passed=bool(passes(values, c))))
                write(output/'manifest.json', manifest)
            passed = passes(values, c)
            for tag, path in zip(TAGS, paths, strict=True):
                shutil.copyfile(path, directory/(tag+'-fields.npz'))
            fine = solutions[-1]
            target = output/f'condition_{index:04d}.npz'
            np.savez_compressed(target, phi=fine['phi'], mu=fine['mu'], points=data['points'],
                volumes=data['volumes'], time=data['time'], holes=data['holes'], mass=fine['mass'], energy=fine['energy'])
            row = dict(index=index, file=target.name, status='passed' if passed else 'failed',
                checks=values, mesh_sizes=mesh_sizes, time_steps=[c.reference.dt, c.reference.refined_dt],
                resolution={name: {key: float(sol[key]) if key == 'seconds' else int(sol[key])
                    for key in ('vertices', 'dofs', 'seconds')}
                    for name, sol in zip(('space_coarse', 'time_coarse', 'fine'), solutions, strict=True)})
            manifest['cases'].append(row)
            write(output/'manifest.json', manifest)
            print(json.dumps(row), flush=True)
            if not passed:
                raise ValueError(f'Explicit mesh ladder exhausted for geometry{index}; preserve failed evidence')
        manifest['status'] = 'passed'
        write(output/'manifest.json', manifest)
        from docs.tasks.spinodal_threeway_support import reference_checks
        audit = reference_checks(c.to_dict())
        write(output/'independent-gate-audit.json', audit)
        manifest.update(seconds=time.perf_counter()-begin, finished_utc=datetime.now(timezone.utc).isoformat(),
            independent_gate_audit='independent-gate-audit.json')
        write(output/'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', seconds=time.perf_counter()-begin)
        write(output/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--finer-mesh', type=float, nargs='+', required=True)
    args = parser.parse_args()
    config = importlib.import_module(args.config).get_config()
    prepare(config, args.source, Path(config.operator.comparison_reference.path), args.finer_mesh)
