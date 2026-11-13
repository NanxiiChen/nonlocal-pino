"""Direct-path validation of fixed versus rolling input controls (CPU only)."""
import ast
import json
from pathlib import Path

import numpy as np


SHARED_SOURCES = (
    'examples/particle_corrosion2d/network.py',
    'examples/particle_corrosion2d/losses.py',
    'examples/particle_corrosion2d/inputs.py',
    'examples/particle_corrosion2d/problem.py',
    'examples/particle_corrosion2d/geometry.py',
    'examples/particle_corrosion2d/mesh.py',
    'examples/particle_corrosion2d/operators.py',
    'src/nonlocal_pinn/nn/point_conditional.py',
    'src/nonlocal_pinn/nn/conditional.py',
    'src/nonlocal_pinn/training/optimizers.py',
)


def compatible_control_source(source, prior, current):
    """Only the documented geometry admission/retry instrumentation may differ.

    All geometry maps, validation thresholds/formulas, PDEs, operators and
    network functions remain source-identical. Existing fixed input arrays are
    reused, not regenerated with the repaired sampler.
    """
    from .compare import _same_file
    if _same_file(prior, current):
        return True
    if source != 'examples/particle_corrosion2d/geometry.py':
        return False
    def except_sampler(path):
        tree = ast.parse(Path(path).read_text())
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'sample_geometry']
        if len(matches) != 1:
            return None
        sampler = matches[0]
        # The optional counter does not consume RNG values or enter the PDE.
        if sampler.args.kwonlyargs:
            if ([argument.arg for argument in sampler.args.kwonlyargs] != ['diagnostics']
                    or len(sampler.args.kw_defaults) != 1
                    or not isinstance(sampler.args.kw_defaults[0], ast.Constant)
                    or sampler.args.kw_defaults[0].value is not None):
                return None
            sampler.args.kwonlyargs = []
            sampler.args.kw_defaults = []
        sampler.body = [ast.Pass()]
        retry = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'GeometrySamplingExhausted']
        if len(retry) > 1:
            return None
        if retry:
            if (len(retry[0].bases) != 1 or not isinstance(retry[0].bases[0], ast.Name)
                    or retry[0].bases[0].id != 'RuntimeError'):
                return None
            tree.body.remove(retry[0])
        return ast.dump(tree, include_attributes=False)
    try:
        before, after = except_sampler(prior), except_sampler(current)
    except SyntaxError:
        return False
    return before is not None and before == after


def validate_fixed_control(c, control, inputs, reference):
    """Permit only adding the explicit rolling-input record to a completed run."""
    from .compare import _same_file
    control = Path(control).resolve(strict=True)
    prior = json.loads((control/'config.json').read_text())
    current = json.loads(c.to_json())
    if 'online' not in current['operator'] or 'online' in prior['operator']:
        raise ValueError('Sampling control requires rolling current and fixed prior configurations')
    current['operator'].pop('online')
    if current != prior:
        raise ValueError('Fixed/rolling control permits only operator.online to differ')
    runtime = json.loads((control/'runtime.json').read_text())
    summary = json.loads((control/'summary.json').read_text())
    if (summary['steps'] != c.train.iterations or not runtime['cold_initialization']
            or runtime['training_uses_reference_targets']):
        raise ValueError('Sampling control must be a completed cold physics-only run')
    if runtime['inputs'] != str(Path(inputs).resolve()) or runtime['reference'] != str(Path(reference).resolve()):
        raise ValueError('Sampling control must share the initial bank and reference paths')
    root = Path(__file__).resolve().parents[2]
    for source in SHARED_SOURCES:
        if not compatible_control_source(source, control/'source'/source, root/source):
            raise ValueError(f'Shared numerical/model source differs from fixed control: {source}')
    return runtime


def validate_control_result(run, control):
    """Check actual cold arrays and common random draws after rolling training."""
    from ml_collections import ConfigDict
    from .compare import _same_file
    run, control = Path(run).resolve(strict=True), Path(control).resolve(strict=True)
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    runtime = json.loads((run/'runtime.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    if (summary['steps'] != c.train.iterations or not (run/'model_final.eqx').is_file()
            or not (run/'optimizer_final.eqx').is_file()):
        raise ValueError('Only a completed fixed-budget rolling run may pass this audit')
    prior = validate_fixed_control(c, control, runtime['inputs'], runtime['reference'])
    if runtime['seed'] != prior['seed'] or runtime['sampling_control'] != str(control):
        raise ValueError('Rolling control seed/provenance differs')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if not _same_file(run/name, control/name):
            raise ValueError(f'Cold initialization differs: {name}')
    source_changes = []
    for source in SHARED_SOURCES:
        if not compatible_control_source(source, control/'source'/source, run/'source'/source):
            raise ValueError(f'Actual run source differs: {source}')
        if not _same_file(run/'source'/source, control/'source'/source):
            source_changes.append(dict(source=source, function='sample_geometry',
                reason='Candidate admission enforces existing coarse/refined geometry checks and records rejection counts; a dedicated search-budget exception supports same-RNG retries. Physical thresholds and maps are unchanged.'))
    with np.load(run/'sampling.npz', allow_pickle=False) as actual, np.load(control/'sampling.npz', allow_pickle=False) as expected:
        if set(actual.files) != {'condition_ids', 'pool_ids', 'times'} or set(actual.files) != set(expected.files):
            raise ValueError('Unexpected common sampling schema')
        for name in actual.files:
            value = actual[name]
            shape = ((c.train.iterations, c.sample.Nt) if name == 'times'
                     else (c.train.iterations, c.operator.condition_batch))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid common sampling shape/values: {name}')
            if name == 'times':
                if not np.issubdtype(value.dtype, np.floating) or np.any(value < 0) or np.any(value > c.sample.t_max):
                    raise ValueError('Physical time draws outside the declared interval')
            else:
                bound = c.operator.geometry_count if name == 'condition_ids' else c.sample.K_pool
                if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0) or np.any(value >= bound):
                    raise ValueError(f'Invalid common sampled index: {name}')
            if actual[name].dtype != expected[name].dtype or not np.array_equal(actual[name], expected[name]):
                raise ValueError(f'Common slot/pool/time draws differ: {name}')
    result = dict(run=str(run), control=str(control), identical_cold_model=True,
                  identical_cold_optimizer=True, identical_slot_pool_time_draws=True,
                  identical_shared_sources=[source for source in SHARED_SOURCES
                      if source not in {item['source'] for item in source_changes}],
                  input_sampler_source_changes=source_changes,
                  allowed_config_changes=['operator.online'],
                  distinction='Slot IDs match, but rolling geometries and spatial revisions intentionally change their physical inputs.')
    (run/'fixed_control_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def _rebuild(arguments):
    from .online import build_input
    slot, config, seed, draw, revision, initial = arguments
    data, metadata = build_input(config, seed, draw, revision, initial)
    return slot, data, metadata


def audit_online_run(run, workers):
    """Replay installed identities and reconstruct the entire final input pool."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from ml_collections import ConfigDict
    from .compare import _same_file
    from .online import (GEOMETRY_NAMESPACE, SPATIAL_NAMESPACE, geometry_identity,
                         pad_initial_bank, plan_events, validate_payload,
                         _worker_environment, _cpu_worker)
    from .full_grid import is_full,prepare_initial_full_bank
    run = Path(run).resolve(strict=True)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('Positive integer reconstruction workers required')
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    runtime = json.loads((run/'runtime.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    if summary['steps'] != c.train.iterations or not runtime['rolling_inputs']:
        raise ValueError('Online audit requires the completed declared rolling budget')
    root = Path(__file__).resolve().parents[2]
    full_sources=('examples/particle_corrosion2d/full_grid.py',) if is_full(c) else ()
    if c.nl.method == 'fem_weak':
        full_sources += ('examples/particle_corrosion2d/fem_weak.py',)
    sampling_sources=('src/nonlocal_pinn/training/spatial_sampling.py',) if 'core_draw' in c.sample else ()
    for source in ('examples/particle_corrosion2d/online.py',
                   'examples/particle_corrosion2d/train.py', *SHARED_SOURCES,*full_sources,*sampling_sources):
        if not _same_file(run/'source'/source, root/source):
            raise ValueError(f'Input reconstruction source changed: {source}')
    bankpath = Path(runtime['inputs'])
    manifest = json.loads((bankpath/'manifest.json').read_text())
    with np.load(bankpath/'inputs.npz', allow_pickle=False) as arrays:
        original = {name: arrays[name] for name in arrays.files}
    if c.nl.method == 'fem_weak':
        from .fem_weak import prepare_initial_bank
        initial = prepare_initial_bank(original,c,manifest,workers)
    else:
        initial = (prepare_initial_full_bank(original,c,manifest,workers) if is_full(c)
                   else pad_initial_bank(original,c))
    with np.load(run/'online_initial_inputs.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(initial):
            raise ValueError('Initial rolling bank schema differs')
        for name, values in initial.items():
            if values.dtype != saved[name].dtype or not np.array_equal(values, saved[name]):
                raise ValueError(f'Initial padding changed physical inputs: {name}')
    forbidden = {geometry_identity(record) for record in json.loads((run/'excluded_geometries.json').read_text())}
    capacity = c.operator.geometry_count
    ids, versions = -1-np.arange(capacity, dtype=np.int64), np.zeros(capacity, dtype=np.int64)
    records = list(manifest['geometries'])
    if any(geometry_identity(record) in forbidden for record in records):
        raise ValueError('Initial pool overlaps evaluation inputs')
    journals = [json.loads(line) for line in (run/'online_refresh.jsonl').read_text().splitlines()]
    events = plan_events(c, capacity)
    if len(events) != len(journals):
        raise ValueError('Missing or extra rolling installation events')
    known = {int(draw): record for draw, record in zip(ids, records, strict=True)}
    installed_geometry = installed_spatial = 0
    by_step = {}
    for event, journal in zip(events, journals, strict=True):
        if (journal['event'] != event['index'] or journal['completed_updates'] != event['completed_updates']
                or journal['seed'] != runtime['seed']
                or journal['geometry_namespace'] != GEOMETRY_NAMESPACE
                or journal['spatial_namespace'] != SPATIAL_NAMESPACE
                or journal['spatial_merged_into_geometry'] != event['spatial_merged_into_geometry']):
            raise ValueError('Installed FIFO event differs from the explicit schedule')
        tasks = event['tasks']
        for name, key in (('slots', 'slot'), ('draw_ids', 'draw_id'), ('spatial_versions', 'spatial_version')):
            if journal[name] != [task[key] for task in tasks]:
                raise ValueError(f'Installed event {name} differ')
        if len(journal['records']) != len(tasks):
            raise ValueError('Incomplete per-slot input records')
        for task, actual in zip(tasks, journal['records'], strict=True):
            if any(actual[name] != value for name, value in task.items()):
                raise ValueError('Installed slot metadata differs from its scheduled identity')
            draw, revision = task['draw_id'], task['spatial_version']
            expected_seed = [runtime['seed'], SPATIAL_NAMESPACE, int(draw >= 0),
                             -1-draw if draw < 0 else draw, revision]
            if (actual['spatial_seed_words'] != expected_seed or actual['seed'] != runtime['seed']
                    or actual['pde_solves'] != 0 or actual['fem_solves'] != 0 or actual['evolved_fields_generated']):
                raise ValueError('Input provenance includes invalid randomness or evolved training fields')
            identity = geometry_identity(actual['geometry'])
            if identity in forbidden:
                raise ValueError('Rolling geometry overlaps evaluation inputs')
            if draw in known and geometry_identity(known[draw]) != identity:
                raise ValueError('A retained geometry changed during spatial refresh')
            known[draw] = actual['geometry']
            slot = task['slot']
            ids[slot], versions[slot], records[slot] = draw, revision, actual['geometry']
            installed_geometry += int(task['kind'] == 'geometry')
            installed_spatial += int(task['kind'] == 'spatial')
        by_step[event['completed_updates']] = journal
    state = json.loads((run/f'online_pool_{c.train.iterations:06d}.json').read_text())
    if (state['draw_ids'] != ids.tolist() or state['spatial_versions'] != versions.tolist()
            or state['geometries'] != records or state['installed_events'] != len(events)):
        raise ValueError('Final pool identity state differs from installed event replay')
    with np.load(run/'sampling.npz', allow_pickle=False) as arrays:
        slots = arrays['condition_ids']
        if slots.shape != (c.train.iterations, c.operator.condition_batch):
            raise ValueError('Incomplete actual slot sampling trace')
    with np.load(run/'online_sampling.npz', allow_pickle=False) as arrays:
        actual_ids, actual_versions = arrays['geometry_draw_ids'], arrays['spatial_versions']
    if actual_ids.shape != slots.shape or actual_versions.shape != slots.shape:
        raise ValueError('Incomplete consumed geometry/spatial identity trace')
    replay_ids, replay_versions = -1-np.arange(capacity, dtype=np.int64), np.zeros(capacity, dtype=np.int64)
    for step in range(c.train.iterations):
        if step in by_step:
            event = by_step[step]
            replay_ids[event['slots']] = event['draw_ids']
            replay_versions[event['slots']] = event['spatial_versions']
        if (not np.array_equal(actual_ids[step], replay_ids[slots[step]])
                or not np.array_equal(actual_versions[step], replay_versions[slots[step]])):
            raise ValueError(f'Actual update {step+1} used a stale/wrong geometry or spatial revision')
    with np.load(run/'online_final_inputs.npz', allow_pickle=False) as arrays:
        final = {name: arrays[name] for name in arrays.files}
    jobs = []
    for slot in range(capacity):
        validate_payload({name: value[slot] for name, value in final.items()}, c)
        draw, revision = int(ids[slot]), int(versions[slot])
        if draw < 0 and revision == 0:
            for name in initial:
                if not np.array_equal(initial[name][slot], final[name][slot]):
                    raise ValueError(f'An unrefreshed initial slot changed: {slot}/{name}')
        else:
            jobs.append((slot, c.to_dict(), runtime['seed'], draw, revision,
                         manifest['geometries'][-1-draw] if draw < 0 else None))
    with _worker_environment():
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_cpu_worker) as pool:
            for slot, data, metadata in pool.map(_rebuild, jobs):
                if geometry_identity(metadata['geometry']) != geometry_identity(records[slot]):
                    raise ValueError(f'Final geometry cannot be reconstructed: {slot}')
                for name, value in data.items():
                    if value.dtype != final[name].dtype or not np.array_equal(value, final[name][slot]):
                        raise ValueError(f'Final input cannot be reconstructed exactly: {slot}/{name}')
    identities = np.column_stack((actual_ids.ravel(), actual_versions.ravel()))
    result = dict(status='passed', run=str(run), updates=c.train.iterations,
        installed_events=len(events), installed_new_geometries=installed_geometry,
        installed_spatial_refreshes=installed_spatial,
        distinct_sampled_geometries=int(len(np.unique(actual_ids))),
        distinct_sampled_geometry_spatial_versions=int(len(np.unique(identities, axis=0))),
        exact_final_slots_reconstructed=len(jobs), unchanged_initial_slots=capacity-len(jobs),
        evaluation_overlap=0, future_training_fields=0,
        scope='All consumed identities and all final physical input arrays audited; intermediate payloads are not stored.')
    (run/'online_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--workers', required=True, type=int)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    if args.control is not None:
        print(json.dumps(validate_control_result(args.run, args.control)), flush=True)
    print(json.dumps(audit_online_run(args.run, args.workers)), flush=True)


if __name__ == '__main__':
    main()
