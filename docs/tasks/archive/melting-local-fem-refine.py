"""Sequential MPI reference refinement under the host's 12-CPU quota."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from examples.icemelting3d.configs.solver_base import get_config

c = get_config()
OUT = Path(__file__).resolve().parent
ROOT = Path(c.reference.root)
ROOT.mkdir(parents=True, exist_ok=True)
records = []
process_record = OUT/'melting-local-fem-processes.json'
commands = json.loads(process_record.read_text()) if process_record.exists() else []


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(index, n, dt):
    stem = f'case{index:03d}-imex-n{n}-dt{str(dt).replace(".", "p")}'
    path = ROOT / (stem+'.npz')
    if path.exists():
        with np.load(path) as fields:
            info = json.loads(str(fields['metadata_json']))
        assert info['range_gate_pass'] and info['scheme'].startswith('imex_bdf2')
        assert info['mesh_subdivisions'] == n and info['dt'] == dt
        assert info['degree'] == c.mesh.degree and info['quadrature_degree'] == c.mesh.quadrature_degree
        training = json.loads(Path(c.reference.training_config).read_text())
        for key in ('physical', 'nondim', 'initial'):
            assert info['problem'][key] == training[key], (path, key)
        manifest = json.loads(Path(c.reference.manifest).read_text())
        assert info['problem']['parameters'] == manifest['parameters'][index]
        return path
    source = ROOT / 'source' / stem
    source.mkdir(parents=True, exist_ok=False)
    for name in ('solver.py', 'ellipsoids.py', 'configs/solver_base.py'):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path('examples/icemelting3d')/name, target)
    write(source/'controls.json', json.loads(c.to_json()))
    command = ['mpirun', '--allow-run-as-root', '-np', '12', sys.executable,
               'examples/icemelting3d/solver.py', '--config', 'examples.icemelting3d.configs.solver_base',
               '--index', str(index), '--subdivisions', str(n), '--dt', str(dt), '--output', str(path)]
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    start = time.perf_counter()
    with (OUT/('melting-local-fem-'+stem+'.txt')).open('x') as stream:
        result = subprocess.run(command, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    commands.append(dict(command=command, seconds=time.perf_counter()-start, returncode=result.returncode))
    write(OUT/'melting-local-fem-processes.json', commands)
    if result.returncode:
        raise RuntimeError(f'FEM failed, inspect {stem}: exit{result.returncode}')
    print('SOLVED', stem, commands[-1]['seconds'], flush=True)
    return path


def compare(coarse_path, fine_path):
    report = {}
    with np.load(coarse_path) as coarse, np.load(fine_path) as fine:
        for key in ('points', 'slice_points', 'parameters', 'time'):
            np.testing.assert_array_equal(coarse[key], fine[key])
        for key in ('phi', 'slice_phi'):
            for field in ('phase', 'solid'):
                a, b = coarse[key], fine[key]
                if field == 'solid':
                    a, b = (a+1)/2, (b+1)/2
                norm = np.linalg.norm(b, axis=1)
                assert (norm > 0).all()
                error = np.linalg.norm(a-b, axis=1)/norm
                report[key+'_'+field] = dict(by_time=error.tolist(), maximum=float(error.max()))
    report['passed'] = all(item['maximum'] < (c.reference.solid_tolerance if key.endswith('solid')
                                             else c.reference.phase_tolerance)
                           for key, item in report.items())
    report['coarse'] = str(coarse_path)
    report['fine'] = str(fine_path)
    return report


for index in c.reference.indices:
    previous = None
    space_checks = []
    for n in c.reference.mesh_levels:
        current = run(index, n, c.reference.time_steps[0])
        if previous is not None:
            check = compare(previous, current)
            space_checks.append(check)
            write(OUT/f'melting-local-fem-case{index:03d}-space.json', space_checks)
            print('SPACE', index, n, json.dumps(check), flush=True)
            if check['passed']:
                break
        previous = current
    else:
        raise RuntimeError(f'No spatial convergence for case{index}; preserve results, do not call reference validated')
    time_checks = []
    for dt in c.reference.time_steps[1:]:
        finer = run(index, n, dt)
        check = compare(current, finer)
        time_checks.append(check)
        write(OUT/f'melting-local-fem-case{index:03d}-time.json', time_checks)
        print('TIME', index, n, dt, json.dumps(check), flush=True)
        if check['passed']:
            accepted = finer
            break
        current = finer
    else:
        raise RuntimeError(f'No temporal convergence for case{index}; preserve results, do not call reference validated')
    records.append(dict(index=index, path=str(accepted), space_checks=space_checks,
                        time_checks=time_checks, status='passed'))
    write(OUT/'melting-local-fem-validation.json', dict(status='in_progress', conditions=records))

write(OUT/'melting-local-fem-validation.json', dict(status='passed', conditions=records,
      phase_tolerance=c.reference.phase_tolerance, solid_tolerance=c.reference.solid_tolerance,
      scope='Matching local CG2 FEM at common diagnostic/slice points; validated against adjacent meshes/time steps, no neural retraining.'))
print('ALL_REFERENCES_PASSED', flush=True)
