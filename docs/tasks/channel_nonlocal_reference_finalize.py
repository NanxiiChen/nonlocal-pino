"""Finite CPU reference validation: await one existing solve, halve dt once, compare."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
if hasattr(os, 'sched_getaffinity'):
    cores = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, cores[-4:])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from docs.tasks.channel_nonlocal_comparison import compare


def alive(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(')', 1)[1].split()[0] != 'Z'


def difference(actual, target):
    with np.load(Path(actual)/'solution.npz') as z, np.load(Path(target)/'solution.npz') as r:
        if json.loads(str(z['identity_json'])) != json.loads(str(r['identity_json'])):
            raise ValueError('Refinement requires the same frozen physical problem')
        for key in ('points', 'time', 'amplitudes'):
            np.testing.assert_allclose(z[key], r[key], atol=1e-12, rtol=0)
        a, b = z['fields'], r['fields']
    norm = lambda x, y: float(np.linalg.norm(x-y)/np.linalg.norm(y))
    return dict(phi=norm(a[..., 0], b[..., 0]), terminal_phi=norm(a[-1, :, 0], b[-1, :, 0]),
        velocity=norm(a[1:, :, 2:4], b[1:, :, 2:4]), pressure=norm(a[1:, :, 4], b[1:, :, 4]))


def finalize(a):
    output = Path(a.output); output.mkdir(parents=True, exist_ok=False)
    record = dict(status='waiting_for_existing_fine_solver', fine=a.fine, coarse=a.coarse,
        solver_pid=a.solver_pid, time_refinement=a.time_output,
        scope='One existing fine solve, one half-dt solve and saved-field comparisons; no FEM/training/retries/default changes')
    def save():
        temp = output/'status.partial'; temp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temp.replace(output/'status.json')
    save(); (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    try:
        expected = Path(a.fine).resolve()
        cmdline = Path(f'/proc/{a.solver_pid}/cmdline').read_text()
        if str(expected) not in cmdline or 'channel_nonlocal_reference.py' not in cmdline:
            raise ValueError('Existing solver PID does not match the exact requested trajectory')
        start = time.monotonic()
        while alive(a.solver_pid):
            if time.monotonic()-start > a.max_wait_seconds:
                raise TimeoutError('Finite wait expired; existing solver is retained, no retry')
            time.sleep(5)
        fine = json.loads((expected/'report.json').read_text())
        if fine['status'] != 'direct_baseline_completed_NOT_refinement_validated':
            raise RuntimeError(f'Fine solve failed: {fine["status"]}')
        if (fine['grid'] != [192, 80] or fine['dt'] != .025 or fine['terminal'] != 6
                or fine['construction']['pressure_layout'] != 'interior'
                or fine['construction']['pressure_stride'] != 1):
            raise ValueError('Finite half-dt workflow requires its exact explicit 192x80/.025/interior parent')
        record['fine_comparison'] = compare(a.fine, a.fem, a.pino, output/'fine-comparison', a.pino_label)
        record['space_difference'] = difference(a.coarse, a.fine)
        record['status'] = 'one_explicit_half_dt_CPU_solve'; save()
        command = [sys.executable, '-u', str(Path(__file__).with_name('channel_nonlocal_reference.py')),
            '--saved-config', str(expected/'config.json'), '--grid', '192', '80',
            '--pressure-stride', '1', '--pressure-layout', 'interior', '--dt', '0.0125',
            '--terminal', '6', '--query-grid', '96', '40', '--tolerance', '1e-8',
            '--max-newton', '16', '--linear-solver', 'gmres_lu', '--lu-refresh-steps', '20',
            '--output', a.time_output]
        record['time_refinement_arguments'] = command; save()
        with (output/'time-refinement.log').open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record['time_difference'] = difference(a.fine, a.time_output)
        record['refined_comparison'] = compare(a.time_output, a.fem, a.pino,
            output/'time-refined-comparison', a.pino_label)
        limits = dict(phi=.025, terminal_phi=.025, velocity=.025, pressure=.08)
        record['explicit_refinement_limits'] = limits
        record['space_gate_passed'] = all(record['space_difference'][k] <= v for k, v in limits.items())
        record['time_gate_passed'] = all(record['time_difference'][k] <= v for k, v in limits.items())
        records = [json.loads(line) for line in (expected/'steps.jsonl').read_text().splitlines()]
        config = json.loads((expected/'config.json').read_text())
        flux = config['physical']['mean_velocity']*config['geometry']['height']
        record['fine_maximum_section_flux_error_over_driving_Q'] = max(
            r['maximum_section_flux_error'] for r in records)/flux
        record['flux_gate_passed'] = record['fine_maximum_section_flux_error_over_driving_Q'] <= .025
        record['reference_gate_passed'] = all(record[k] for k in ('space_gate_passed', 'time_gate_passed', 'flux_gate_passed'))
        record['status'] = 'finite_reference_comparisons_completed_gates_explicit'
        save()
    except Exception as error:
        record.update(status='FAILED_no_validated_reference', error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fine', 'coarse', 'fem', 'pino', 'pino-label', 'time-output', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--solver-pid', type=int, required=True)
    p.add_argument('--max-wait-seconds', type=int, required=True)
    args = p.parse_args()
    if args.solver_pid < 1 or args.max_wait_seconds < 1:
        raise ValueError('Positive actual solver PID and finite wait required')
    finalize(args)
