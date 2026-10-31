"""Exclusive declared capacity probe -> cold fixed budget -> common validations."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from docs.tasks.confined_ch_capacity_cases import capacity_case


def main(prefix, prior, case='width192'):
    spec = capacity_case(case)
    root = Path(__file__).resolve().parents[2]
    base = Path('/root/autodl-tmp/tf-logs/confined_ch2d')
    python = str(root / '.venv/bin/python')
    if spec['token'] not in prefix or '/' in prefix:
        raise ValueError(f'Expected a fresh explicit {spec["token"]} prefix')
    label = f'width{spec["width"]}-bank1024' if spec['control'] == 'bank_growth' else f'width{spec["width"]}'
    run = base / prefix.replace(spec['token'],
        f'-confined-geometry-b16-t16-{label}-fp32-{spec["iterations"] // 1000}k-')
    journal, probe = base / (prefix + '-process.json'), base / (prefix + '-probe')
    receipts = root / 'docs/tasks'
    initial = receipts / (spec['receipt'] + '-initial-audit.json')
    preflight = receipts / (spec['receipt'] + '-preflight.json')
    audit = receipts / (spec['receipt'] + '-audit.json')
    results = receipts / (spec['receipt'] + '-results.json')
    for path in (run, journal, probe, initial, preflight, audit, results):
        if path.exists():
            raise FileExistsError(path)
    prior_result = json.loads((receipts / (spec['prior_receipt'] + '-results.json')).read_text())
    prior_audit = json.loads((receipts / (spec['prior_receipt'] + '-audit.json')).read_text())
    config_audit = json.loads((receipts / (spec['receipt'] + '-config.json')).read_text())
    if not (prior_result['status'] == 'complete' and prior_result['run'] == prior_audit['run'] == str(prior)
            and prior_audit['optimizer_state_finite'] and prior_audit['condition_draws_match_global_seed']
            and config_audit['status'] == 'passed' and config_audit['prior'] == str(prior)):
        raise RuntimeError('Completed baseline/configuration audits are required')
    if spec['control'] == 'bank_growth':
        growth = json.loads((receipts / (spec['receipt'] + '-bank.json')).read_text())
        dense = json.loads((Path(spec['dense_reference']) / 'validation.json').read_text())
        if not (growth['status'] == dense['status'] == 'passed'
                and growth['training_conditions'] == 1024 and growth['source_training_conditions'] == 512
                and growth['test_indices'] == dense['indices'] and growth['new_operator_only_inputs'] == 512):
            raise RuntimeError('Exact input-bank growth and mapped dense-reference audits required')
    if case == 'width192' and 'required_free_disk_bytes' not in config_audit:
        # The completed legacy192 receipt predates checkpoint storage estimates.
        required_disk = 4 * 2**30
        storage_policy = 'Explicit legacy192 fourGiB gate; original receipt retained'
    else:
        required_disk = config_audit['required_free_disk_bytes']
        storage_policy = 'Audited paired model/SOAP checkpoints plus declared reserve'
    if shutil.disk_usage(base).free < required_disk:
        raise RuntimeError(f'Need {required_disk / 2**30:.2f}GiB free for paired states and margin; no automatic artifact removal')
    used = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if len(used) != 1 or int(used[0]) >= 1024:
        raise RuntimeError('Expected one idle current GPU; refuse competing work')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env.pop('JAX_PLATFORMS', None)
    cpu = dict(env, JAX_PLATFORMS='cpu')
    record = dict(status='starting', runner_pid=os.getpid(), runner_session=os.getsid(0),
        run=str(run), prior=str(prior), probe=str(probe), case=case,
        requested_steps=spec['iterations'], required_free_disk_bytes=required_disk, stages=[],
        storage_policy=storage_policy,
        started_utc=datetime.now(timezone.utc).isoformat(),
        scope='Declared192/1024input-bank/cold50k control; same initial arrays/time stream and all physics/optimizer settings. Common30k/native reference, no default promotion.'
            if spec['control'] == 'bank_growth' else
            'Declared backbone/Fourier capacity and budget control, four heads/SOAP cap512. SameGPU FP32 native benchmark; common-budget comparisons, no default promotion.')

    def write():
        temporary = journal.with_name(journal.name + '.partial')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, journal)

    write()

    def execute(stage, command, environment, monitor=False):
        log = base / (prefix + '-' + stage + '.log')
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            entry = dict(stage=stage, pid=child.pid, command=command, log=str(log), started_utc=datetime.now(timezone.utc).isoformat())
            record['stages'].append(entry)
            record.update(status=stage, active_child_pid=child.pid)
            write()
            try:
                while True:
                    try:
                        code = child.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        metrics = run / 'metrics.jsonl'
                        if monitor and metrics.exists():
                            rows = [json.loads(s) for s in metrics.read_text().splitlines(keepends=True) if s.endswith('\n')]
                            if rows:
                                record['latest_evaluation'] = {k: v for k, v in rows[-1].items() if k != 'conditions'}
                                if not initial.exists():
                                    command_initial = [python, '-m', 'docs.tasks.confined_ch_capacity_probe', '--action', 'initial',
                                        '--prior', str(prior), '--run', str(run), '--probe', str(probe), '--output', str(initial), '--case', case]
                                    with (base / (prefix + '-initial-audit.log')).open('x') as audit_log:
                                        subprocess.run(command_initial, cwd=root, env=cpu, stdout=audit_log,
                                            stderr=subprocess.STDOUT, check=True)
                                    record['initial_audit'] = str(initial)
                        record['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
                        write()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=60)
                raise
        entry.update(exit_code=code, completed_utc=datetime.now(timezone.utc).isoformat())
        record['active_child_pid'] = None
        write()
        if code:
            raise RuntimeError(f'{stage} exited{code}; inspect{log}')

    try:
        helper = [python, '-m', 'docs.tasks.confined_ch_capacity_probe', '--case', case]
        execute('gpu_probe', helper + ['--action', 'probe', '--prior', str(prior), '--output', str(probe)], env)
        execute('probe_cpu', helper + ['--action', 'cpu', '--prior', str(prior), '--output', str(probe)], cpu)
        measured = json.loads((probe / 'probe.json').read_text())
        measured['cpu_replay'] = json.loads((probe / 'cpu_audit.json').read_text())
        measured['probe_path'] = str(probe)
        assert measured['status'] == measured['cpu_replay']['status'] == 'passed'
        preflight.write_text(json.dumps(measured, indent=2, allow_nan=False) + '\n')
        preset = spec['config'].replace('.', '/') + '.py'
        execute('training', [python, 'examples/confined_ch2d/train.py', '--config=' + preset,
            '--config.log.run_name=' + run.name], env, monitor=True)
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['final_metrics']['step'] == spec['iterations'] and summary['start_step'] == 0
        assert json.loads(initial.read_text())['status'] == 'passed'
        dataset = json.loads((run / 'config.json').read_text())['operator']['dataset_path']
        dense_reference = spec['dense_reference'] if spec['control'] == 'bank_growth' else '/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation'
        stages = [
            ('native', [python, '-m', 'examples.confined_ch2d.operator_evaluate', '--run', str(run), '--output', str(run / 'evaluation')]),
            ('dense61', [python, 'docs/tasks/confined-fourier-dense-evaluate.py', '--run', str(run),
                '--dense', dense_reference,
                '--step', str(spec['iterations']), '--output', str(run / 'dense_evaluation')]),
            ('geometry', [python, '-m', 'examples.confined_ch2d.operator_geometry_effect', '--run', str(run),
                '--dataset', dataset, '--output', str(run / 'geometry_effect')]),
            ('report', [python, 'docs/tasks/confined-fourier-training-report.py', '--run', str(run), '--output', str(run / 'training_report')]),
            ('cpu_audit', [python, 'docs/tasks/confined-fourier-audit.py', '--run', str(run), '--prior', str(prior),
                '--output', str(audit), '--config', spec['config']]),
        ]
        if spec['control'] == 'bank_growth':
            stages[-1][1].append('--bank-growth')
        common_budget = json.loads((prior / 'summary.json').read_text())['final_metrics']['step']
        if common_budget != spec['iterations']:
            marker = json.loads((run / f'checkpoint_{common_budget:06d}.json').read_text())
            assert marker['step'] == common_budget
            stages.insert(2, ('dense61_common_budget', [python, 'docs/tasks/confined-fourier-dense-evaluate.py',
                '--run', str(run), '--dense', dense_reference,
                '--step', str(common_budget), '--output', str(run / f'dense_evaluation_{common_budget:06d}')]))
        for stage, command in stages:
            execute(stage, command, cpu)
        from examples.confined_ch2d.operator_history import training_records
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        comparisons, histories = [], []
        for width, path, color in ((spec['baseline_width'], prior, 'C0'), (spec['width'], run, 'C1')):
            bank_count = json.loads((path / 'config.json').read_text())['operator']['train_conditions']
            rows = training_records(path, 'metrics.jsonl')
            histories.append({r['step']: r for r in rows})
            timing = json.loads((path / 'summary.json').read_text())
            comparisons.append(dict(width=width, training_geometries=bank_count, run=str(path), native=json.loads((path / 'evaluation/metrics.json').read_text()),
                dense61=json.loads((path / 'dense_evaluation/metrics.json').read_text()),
                wall_seconds=timing['wall_seconds'], warm_update_seconds=timing['mean_update_seconds_after_100'],
                memory=timing['device_memory_stats']))
            for axis, x in zip(axes, ([r['step'] for r in rows], [r['elapsed_seconds'] / 60 for r in rows]), strict=True):
                for split, style in (('test', '-'), ('train', '--')):
                    axis.plot(x, [100 * r[split + '_phi_pooled'] for r in rows], style, color=color,
                        label=f'Width{width}/N{bank_count}, {"reserved dev" if split == "test" else "train16 monitors"}')
        for axis, label in zip(axes, ('Updates', 'Elapsed training minutes'), strict=True):
            axis.set(xlabel=label, ylabel='Phase relative L2 (%)', yscale='log')
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        fig.savefig(run / 'capacity_comparison.png', dpi=180)
        fig.savefig(run / 'capacity_comparison.pdf')
        plt.close(fig)
        a, b = histories
        matched = [dict(step=step, baseline_phi=a[step]['test_phi_pooled'], candidate_phi=b[step]['test_phi_pooled'],
            baseline_terminal=a[step]['test_phi_terminal'], candidate_terminal=b[step]['test_phi_terminal'])
            for step in sorted(set(a) & set(b))]
        intermediate = None
        if common_budget != spec['iterations']:
            intermediate = dict(step=common_budget,
                online={k: v for k, v in b[common_budget].items() if k != 'conditions'},
                dense61=json.loads((run / f'dense_evaluation_{common_budget:06d}/metrics.json').read_text()))
        result = dict(status='complete', run=str(run), prior=str(prior), controls=comparisons,
            candidate_at_baseline_budget=intermediate,
            matched_steps=matched, preflight=measured,
            scope='Cold192/1024/50k versus192/512/30k; identical fresh model arrays/time stream, expanded geometry sampling support. Common30kcheckpoint separates bank size from extra updates. Same8reserved physical domains with audited draw-index mapping. One seed/reused development cohort; no FEM/default/seed-stability claim.'
                if spec['control'] == 'bank_growth' else
                'Fixed declared cold FP32 budget, common actual geometry draw prefix and temporal sampler; joint backbone/Fourier growth with SOAP coverage retained. Common-budget checkpoint separates capacity from extra updates. One initialization/eight reused native development geometries; no FEM/seed-stability/isolated-head claim or default promotion.')
        results.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        record.update(status='complete', results=str(results), completed_utc=datetime.now(timezone.utc).isoformat())
        write()
    except Exception as error:
        record.update(status='failed', error=repr(error), failed_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--case', choices=('width192', 'width256_50k', 'width192_bank1024_50k'), default='width192',
                        help='Legacy invocation retains width192; new case is explicit')
    args = parser.parse_args()
    main(args.prefix, args.prior.resolve(), args.case)
