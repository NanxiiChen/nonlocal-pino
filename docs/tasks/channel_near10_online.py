"""Advance audited final single cases under explicitly provisional pressure acceptance."""
import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def accepted(metrics, pressure_limit):
    if not math.isfinite(pressure_limit) or not .1 <= pressure_limit <= .11:
        raise ValueError('This user-approved provisional pressure study permits10%-11% only')
    values = [metrics[k] for k in ('phi', 'terminal_phi', 'velocity', 'pressure')]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Acceptance requires finite nonnegative physical errors')
    return all(metrics[k] < .1 for k in ('phi', 'terminal_phi', 'velocity')) and metrics['pressure'] <= pressure_limit


def validate_candidate(run, online_config, pressure_limit):
    """Replay saved-field scalars and compare recipes; never fit reference values."""
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.problem import identity, validate_config

    run = Path(run)
    if not run.is_absolute():
        raise ValueError('A completed candidate requires an explicit absolute run path')
    read = lambda name: json.loads((run / name).read_text())
    c = read('config.json')
    summary = read('summary.json')
    physical = read('validation.json')
    audit = read('single-validation.json')
    dense = read('dense-fem-evaluation.json')
    if (summary['status'] != 'completed' or c['operator']['train_conditions'] != 1
            or physical['status'] != 'replay_and_hard_constraints_passed'
            or audit['status'] != 'independent_replay_and_diagnostics_completed'
            or not physical['actual_draws_reproduced_from_global_keys']
            or dense['status'] != 'dense_FEM_evaluation_completed'
            or dense['run'] != str(run)
            or not (run / 'model_final.eqx').is_file()):
        raise ValueError(f'Candidate is missing actual complete final audits: {run}')
    gate = json.loads(Path(dense['reference_validation']).read_text())
    if gate['status'] != 'passed' or gate['identity'] != json.loads(json.dumps(identity(ConfigDict(c)))):
        raise ValueError('Candidate dense reference lacks matching accepted physical identity')
    with np.load(run / 'dense-fem-prediction.npz', allow_pickle=False) as z:
        fields, target = z['fields'], z['reference']
        with np.load(gate['reference'], allow_pickle=False) as ref:
            for key in ('points', 'time', 'amplitudes'):
                np.testing.assert_array_equal(z[key], ref[key])
            np.testing.assert_array_equal(target, ref['fields'])
        if fields.shape != target.shape or not np.isfinite(fields).all():
            raise ValueError('Invalid actual candidate dense field arrays')
        relative = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        metrics = dict(phi=relative(fields[..., 0], target[..., 0]),
                       terminal_phi=relative(fields[-1, :, 0], target[-1, :, 0]),
                       velocity=relative(fields[1:, :, 2:4], target[1:, :, 2:4]),
                       pressure=relative(fields[1:, :, 4], target[1:, :, 4]))
    gap = max(abs(metrics[k] - dense['metrics'][k]) for k in metrics)
    if gap > 2e-6:
        raise ValueError(f'Candidate dense metric replay failed: {gap}')
    module = online_config.replace('/', '.').removesuffix('.py')
    online = importlib.import_module(module).get_config()
    # Qualification used this actual dense gate. Older staged single configs
    # still name their384-query monitoring root; new online monitoring uses3840.
    online.reference.root = str(Path(dense['reference_validation']).resolve().parent)
    online.train.seed = c['train']['seed']
    validate_config(online)
    before, after = copy.deepcopy(c), json.loads(online.to_json())
    for config in (before, after):
        del config['log']
        del config['reference']['root']  # Explicit monitoring-only root change above.
        for key in ('train_conditions', 'condition_batch', 'online', 'validation_manifest'):
            if key in config['operator']:
                del config['operator'][key]
        for key in ('iterations', 'continuation', 'refinement'):
            if key in config['train']:
                del config['train'][key]
    if before != after:
        raise ValueError('Online recipe changes more than declared geometry sampling/budget/origin')
    if (online.operator.train_conditions != 16 or online.operator.condition_batch != 2
            or online.train.iterations != 20000 or online.sample.Nt != 4
            or 'continuation' in online.train or 'refinement' in online.train):
        raise ValueError('This study requires the declared cold20k/16pool/B2/Nt4online recipe')
    return dict(run=str(run), online_config=online_config, metrics=metrics,
                pressure_limit=pressure_limit, provisional_accepted=accepted(metrics, pressure_limit),
                strict_all_fields_below10=all(v < .1 for v in metrics.values()),
                saved_dense_metric_max_gap=gap, recipe_parity=True,
                seed=c['train']['seed'], reference_root=online.reference.root,
                original_monitoring_root=c['reference']['root'],
                single_was_continued='continuation' in c['train'],
                scope='Final budget checkpoint only; new online run starts cold, not optimizer continuation')


def main(tag, after_pid, previous_progress, candidates, pressure_limit):
    root = Path(__file__).resolve().parents[2]
    data = Path('/root/autodl-tmp/data/channel_nsch2d')
    run = Path('/root/autodl-tmp/tf-logs/channel_nsch2d') / f'{tag}-online-01'
    progress = data / f'{tag}-progress.json'
    if progress.exists() or run.exists():
        raise FileExistsError('Provisional follow-up requires fresh direct progress/run paths')
    accepted(dict(phi=0., terminal_phi=0., velocity=0., pressure=0.), pressure_limit)
    with (data / f'{tag}-driver-source.py').open('xb') as source:
        source.write(Path(__file__).read_bytes())

    def record(stage, **extra):
        progress.write_text(json.dumps(dict(stage=stage, **extra), indent=2, allow_nan=False) + '\n')
        print(stage, flush=True)

    record('waiting_for_fixed_budget_single_driver', pid=after_pid,
           pressure_limit=pressure_limit, strict_limit=.1)
    while Path(f'/proc/{after_pid}').exists():
        time.sleep(2)
    previous = json.loads(Path(previous_progress).read_text())
    if previous['stage'] == 'online_training_and_input_replay_completed_unseen_validation_required':
        record('strict_pipeline_proceeded_no_duplicate_training', goal_complete=False)
        return
    if previous['stage'] != 'single_geometry_target_still_unmet':
        raise ValueError('Previous driver must have completed its actual training and independent gates')
    reports = [validate_candidate(path, preset, pressure_limit) for path, preset in candidates]
    eligible = [report for report in reports if report['provisional_accepted']]
    if not eligible:
        record('no_qualified_near10_single_case', candidates=reports, goal_complete=False)
        return
    # The explicit order is current final control, then the named audited fallback.
    selected = eligible[0]
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('A GPU process remains active; do not overlap training')
    record('cold_online_training_under_provisional_pressure_acceptance',
           run=str(run), selected=selected, candidates=reports, goal_complete=False)
    env = dict(os.environ, JAX_PLATFORMS='cuda')
    python = str(root / '.venv/bin/python')

    def launch(arguments, label):
        with (data / f'{tag}-{label}.log').open('x') as log:
            subprocess.run([python, *arguments], cwd=root, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    launch(['-m', 'examples.channel_nsch2d.train', f'--config={selected["online_config"]}',
            f'--config.reference.root={selected["reference_root"]}',
            f'--config.log.run_name={run.name}', f'--seed={selected["seed"]}'], 'online-train')
    for module in ('validate_run', 'diagnose_phase'):
        launch(['-m', f'examples.channel_nsch2d.{module}', '--run', str(run)], f'online-{module}')
    record('online_training_and_input_replay_completed_unseen_validation_required',
           run=str(run), selected=selected, candidates=reports, pressure_limit=pressure_limit,
           strict_limit=.1, goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--after-pid', required=True, type=int)
    parser.add_argument('--previous-progress', required=True)
    parser.add_argument('--candidate', action='append', nargs=2, required=True,
                        metavar=('FINAL_RUN', 'MATCHING_ONLINE_CONFIG'))
    parser.add_argument('--pressure-limit', required=True, type=float)
    args = parser.parse_args()
    main(args.tag, args.after_pid, args.previous_progress, args.candidate, args.pressure_limit)
