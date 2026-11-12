"""Check complete compact supports over fresh online channel geometry lifetimes.

This constructs known inputs only. It neither solves a PDE nor qualifies neural
accuracy. Selected draw IDs span the actual planned training sequence; every
pool set/revision while that geometry remains installed is checked.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.configs.train_stratified_spatial import get_config
from examples.channel_nsch2d.online import build_input
from examples.channel_nsch2d.spatial_sampling import (
    GRAPH_KEYS, ROW_WEIGHTS, _build, validate_spatial)

SETTINGS = (('bulk512', 512, 0, 10240), ('wall512', 512, 8, 10240),
            ('wall1024', 1024, 16, 12288))


def check_case(config, draw_id):
    started = time.perf_counter()
    c = ConfigDict(config)
    data, construction = build_input(config, draw_id)
    size = c.operator.online.refresh_size
    every = c.operator.online.refresh_every
    # before_step consumes refreshes before updates, excluding the final endpoint.
    installed = (draw_id // size + 1) * every
    replaced = installed + c.operator.train_conditions // size * every
    stop = min(replaced, c.train.iterations)
    stride = c.sample.spatial.refresh_every
    revisions = list(range(installed // stride, (stop - 1) // stride + 1))
    if every != stride or not revisions:
        raise ValueError('This audit requires the declared aligned refresh clocks and consumed draws')
    field = np.sin(data['points'][:, 0]) * np.cos(2 * data['points'][:, 1])
    records = {}
    checks = 0
    for label, cores, walls, capacity in SETTINGS:
        c.sample.spatial.core_count = cores
        c.sample.spatial.wall_groups = walls
        c.sample.spatial.node_capacity = capacity
        validate_spatial(c)
        counts = []
        mass_errors = []
        for revision in revisions:
            sets = _build(data, c, draw_id, revision, tiles=())
            for payload, indices, count in sets:
                if len(indices) != cores or not np.all(data['interior'][indices]):
                    raise ValueError('Invalid sampled PDE rows')
                counts.append(count)
                # Audit cast weights in FP64: an FP32 reduction adds its own rounding.
                mass = payload['core_weights'].sum(dtype=np.float64)
                np.testing.assert_allclose(mass, 1., rtol=1e-7)
                mass_errors.append(float(abs(mass-1.)))
                cf = np.sin(payload['points'][:, 0]) * np.cos(2 * payload['points'][:, 1])
                for key in ROW_WEIGHTS:
                    np.testing.assert_array_equal(payload[key], data[key][indices])
                    expected = np.sum(data[key][indices] * (
                        field[data['neighbors'][indices]] - field[indices, None]), axis=-1)
                    actual = np.sum(payload[key] * (
                        cf[payload['neighbors']] - cf[payload['core_indices'], None]), axis=-1)
                    np.testing.assert_array_equal(actual, expected)
                    checks += 1
                # Every retained boundary derivative uses the same source values.
                for key in ('boundary_gradient_x', 'boundary_gradient_y'):
                    active = data[key] != 0
                    np.testing.assert_array_equal(payload[key], data[key])
                    np.testing.assert_array_equal(
                        cf[payload['boundary_neighbors']][active],
                        field[data['boundary_neighbors']][active])
                for key in data:
                    if key not in GRAPH_KEYS:
                        np.testing.assert_array_equal(payload[key], data[key])
        records[label] = dict(capacity=capacity, core_count=cores, wall_groups=walls,
            pool_sets=len(counts), halo_counts=counts, maximum_halo=max(counts),
            minimum_margin=capacity-max(counts), maximum_weight_mass_error=max(mass_errors))
    return dict(draw_id=draw_id, installed_update=installed, replaced_update=replaced,
        revisions=revisions, construction=construction, settings=records,
        exact_derivative_checks=checks, seconds=time.perf_counter()-started)


def run(output, count, workers):
    if count < 2 or workers < 1:
        raise ValueError('At least two geometries and one CPU worker required')
    c = get_config()
    c.train.seed = 42
    c.train.iterations = 20000
    draws = ((c.train.iterations-1) // c.operator.online.refresh_every
             * c.operator.online.refresh_size)
    if count > draws:
        raise ValueError('Requested cohort exceeds actual consumed geometry draws')
    selected = np.linspace(0, draws-1, count, dtype=int).tolist()
    if len(set(selected)) != count:
        raise ValueError('Duplicate prescribed audit draw')
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(c.to_json_best_effort()+'\n')
    sources = (
        'docs/tasks/pino_channel_fresh_support.py',
        'examples/channel_nsch2d/online.py',
        'examples/channel_nsch2d/data.py',
        'examples/channel_nsch2d/geometry.py',
        'examples/channel_nsch2d/operators.py',
        'examples/channel_nsch2d/spatial_sampling.py',
        'src/nonlocal_pinn/training/spatial_sampling.py')
    for source in sources:
        target=output/'source'/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    started=time.perf_counter()
    records=[]
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(check_case, c.to_dict(), index):index for index in selected}
        for future in as_completed(futures):
            try:
                record=future.result()
            except Exception as error:
                for pending in futures: pending.cancel()
                (output/'failure.json').write_text(json.dumps(dict(status='failed',
                    draw_id=futures[future],error=repr(error)),indent=2)+'\n')
                raise
            records.append(record)
            with (output/'cases.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            print(json.dumps(dict(completed=len(records),draw_id=record['draw_id'],
                maxima={k:v['maximum_halo'] for k,v in record['settings'].items()})),flush=True)
    statistics={}
    for label, _, _, capacity in SETTINGS:
        values=np.concatenate([r['settings'][label]['halo_counts'] for r in records])
        statistics[label]=dict(pool_sets=len(values),capacity=capacity,
            minimum_halo=int(values.min()),median_halo=float(np.median(values)),
            maximum_halo=int(values.max()),minimum_margin=int(capacity-values.max()),
            maximum_weight_mass_error=max(r['settings'][label]['maximum_weight_mass_error'] for r in records))
    report=dict(status='passed',date=datetime.now(timezone.utc).isoformat(),
        config=str(output/'config.json'),geometries=count,draw_ids=selected,
        consumed_online_draws=draws,workers=workers,statistics=statistics,
        exact_derivative_checks=sum(r['exact_derivative_checks'] for r in records),
        complete_boundary_supports=True,complete_encoder_context=True,
        pde_solves=0,fem_solves=0,training_performed=False,seconds=time.perf_counter()-started,
        limitation='Finite selected cohort/lifetimes; runtime overflow remains an explicit error, not a population guarantee',
        timing_scope='CPU-only diagnostic concurrent with ceramic wall training; not a training speed measurement')
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--geometries',type=int,required=True)
    parser.add_argument('--workers',type=int,required=True)
    args=parser.parse_args()
    run(args.output,args.geometries,args.workers)
