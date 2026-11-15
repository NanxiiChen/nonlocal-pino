"""Independent local mixed-P2 cohort; run only in fenicsenv with GPU idle."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import shutil
import subprocess
from pathlib import Path
import time

import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def relative(a, b, volumes):
    if (np.shape(a) != np.shape(b) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(volumes).all()
            or np.any(np.asarray(volumes) <= 0)):
        raise ValueError('Finite matching physical fields and positive quadrature required')
    den = float(np.sum(b*b*volumes))
    if den <= 0 or not np.isfinite(den):
        raise ValueError('Positive finite reference norm required')
    return float(np.sqrt(np.sum((a-b)**2*volumes)/den))


def prepare(c, output):
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('Local FEM solve requires idle GPU; active PIDs: '+occupied)
    from examples.confined_ch2d.operator_problem import case_config, specification, validate, valid_holes
    from examples.confined_ch2d.solver import fem
    validate(c)
    native_root = Path(c.operator.dataset_path)
    bank = json.loads((native_root/'manifest.json').read_text())
    if bank['status']!='passed' or bank['specification']!=specification(c):
        raise ValueError('Native bank physics/prior must match before independent solving')
    expected=list(range(c.operator.train_conditions,c.operator.train_conditions+c.operator.test_conditions))
    if bank['test_indices']!=expected:raise ValueError('Unexpected common reference cohort')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        native_bank=str(native_root), specification=specification(c), cases=[],
        indices=bank['test_indices'], kind='independent_mixed_p2_bdf2',
        initial_chemical_convention='Exact analytic continuum IC derivatives at t0, not a compatible Neumann trace')
    root=Path(__file__).resolve().parents[2]
    sources=[Path(__file__),*sorted((root/'examples/confined_ch2d').rglob('*.py'))]
    for source in sources:
        target=output/'source'/source.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    manifest['sources']=[str(path.relative_to(root)) for path in sources]
    manifest['refinement_thresholds']=dict(phase=c.reference.phase_tolerance,
        chemical_positive=c.reference.chemical_tolerance,mass=1e-8,energy_increase=1e-8)
    write(output/'manifest.json', manifest)
    begin=time.perf_counter()
    try:
        for index in bank['test_indices']:
            record=bank['conditions'][index];relative_path=Path(record['file'])
            if record['index']!=index or relative_path.is_absolute() or '..' in relative_path.parts:
                raise ValueError('Invalid ordered reference source identity/path')
            with np.load(native_root/relative_path) as f:
                data={key:f[key] for key in f.files}
            np.testing.assert_array_equal(data['holes'],record['holes'])
            if not valid_holes(data['holes'],c):raise ValueError('Invalid prescribed reference geometry')
            cc=case_config(c,data['holes'],index)
            cc.reference.root=str(output/f'case_{index:04d}')
            h0,h1=c.reference.mesh_sizes
            for dt in (c.reference.dt,c.reference.refined_dt):
                steps=c.physical.terminal_time/dt
                if abs(steps-round(steps))>1e-10 or round(steps)%(c.reference.snapshots-1):
                    raise ValueError('Reference time steps must exactly resolve every requested snapshot')
            np.testing.assert_array_equal(data['time'],np.linspace(0,c.physical.terminal_time,c.reference.snapshots))
            coarse=fem(cc,h0,c.reference.refined_dt,data['points'],'space-coarse')
            coarse_time=fem(cc,h1,c.reference.dt,data['points'],'time-coarse')
            fine=fem(cc,h1,c.reference.refined_dt,data['points'],'fine')
            for solution in (coarse,coarse_time,fine):
                if any(not np.isfinite(solution[key]).all() for key in ('phi','mu','mass','energy')):
                    raise FloatingPointError('Nonfinite local FEM fields or invariants')
                if solution['phi'].shape!=data['phi'].shape or solution['mu'].shape!=data['mu'].shape:
                    raise ValueError('Local/native snapshot shapes differ')
                np.testing.assert_array_equal(solution['points'],data['points'])
            checks={}
            for label,first in (('space',coarse),('time',coarse_time)):
                checks[label+'_phase']=max(relative(a,b,data['volumes']) for a,b in zip(first['phi'],fine['phi'],strict=True))
                checks[label+'_chemical_positive']=max(relative(a,b,data['volumes']) for a,b in zip(first['mu'][1:],fine['mu'][1:],strict=True))
            checks['mass_drift']=float(np.max(np.abs(fine['mass']-fine['mass'][0])))
            checks['energy_increase']=float(max(0.,np.max(np.diff(fine['energy']))))
            checks['native_local_phase']=relative(data['phi'],fine['phi'],data['volumes'])
            checks['native_local_chemical']=relative(data['mu'],fine['mu'],data['volumes'])
            passed=(max(checks['space_phase'],checks['time_phase'])<c.reference.phase_tolerance
                and max(checks['space_chemical_positive'],checks['time_chemical_positive'])<c.reference.chemical_tolerance
                and checks['mass_drift']<1e-8 and checks['energy_increase']<1e-8)
            target=output/f'condition_{index:04d}.npz'
            np.savez_compressed(target,phi=fine['phi'],mu=fine['mu'],points=data['points'],
                volumes=data['volumes'],time=data['time'],holes=data['holes'],
                mass=fine['mass'],energy=fine['energy'])
            row=dict(index=index,file=target.name,status='passed' if passed else 'failed',
                checks=checks,mesh_sizes=list(c.reference.mesh_sizes),
                time_steps=[c.reference.dt,c.reference.refined_dt],
                resolution={name:{key:solution[key] for key in ('vertices','dofs','seconds')}
                    for name,solution in (('space_coarse',coarse),('time_coarse',coarse_time),('fine',fine))})
            manifest['cases'].append(row);write(output/'manifest.json',manifest)
            print(json.dumps(row),flush=True)
            if not passed:
                raise ValueError(f'Local reference refinement failed for geometry{index}; retain evidence and refine explicitly')
        manifest.update(status='passed',seconds=time.perf_counter()-begin,finished_utc=datetime.now(timezone.utc).isoformat())
        write(output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}',seconds=time.perf_counter()-begin)
        write(output/'manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=importlib.import_module(args.config).get_config()
    prepare(config,Path(config.operator.comparison_reference.path))
