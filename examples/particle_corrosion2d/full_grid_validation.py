"""CPU-only pairing and independent complete-row quadrature audits."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict
from scipy import sparse

from .compare import _same_file
from .full_grid import validate_full
from .online import FIXED_KEYS, GRAPH_KEYS, _cpu_worker, _worker_environment
from .online_validation import SHARED_SOURCES


def _read(path):
    return json.loads(Path(path).read_text())


def _receipt(run, c):
    receipt = _read(run/'online_validation.json')
    if (receipt['status'] != 'passed' or Path(receipt['run']).resolve() != run
            or receipt['updates'] != c.train.iterations or receipt['evaluation_overlap'] != 0
            or receipt['future_training_fields'] != 0
            or receipt['exact_final_slots_reconstructed']+receipt['unchanged_initial_slots']
            != c.operator.geometry_count):
        raise ValueError('A matching completed label-free online input audit is required')
    return receipt


def validate_full_control(c, control, inputs, reference):
    """Allow only sample.full_grid relative to a qualified rolling core control."""
    validate_full(c)
    control = Path(control).resolve(strict=True)
    prior = _read(control/'config.json')
    current = json.loads(c.to_json())
    if 'full_grid' in prior['sample'] or 'online' not in prior['operator']:
        raise ValueError('Full-grid control must be an online core/halo run')
    current['sample'].pop('full_grid')
    if current != prior:
        raise ValueError('Full-grid pairing permits only sample.full_grid to differ')
    runtime, summary = _read(control/'runtime.json'), _read(control/'summary.json')
    if (summary['steps'] != c.train.iterations or not runtime['cold_initialization']
            or not runtime['rolling_inputs'] or runtime['training_uses_reference_targets']
            or not (control/'model_final.eqx').is_file()
            or not (control/'optimizer_final.eqx').is_file()):
        raise ValueError('Control must be a completed cold physics-only rolling run')
    if (runtime['inputs'] != str(Path(inputs).resolve(strict=True))
            or runtime['reference'] != str(Path(reference).resolve(strict=True))):
        raise ValueError('Full-grid pairing requires identical initial-bank/reference paths')
    _receipt(control, ConfigDict(prior))
    root = Path(__file__).resolve().parents[2]
    for source in SHARED_SOURCES:
        if not _same_file(control/'source'/source, root/source):
            raise ValueError(f'Full-grid shared source differs: {source}')
    return runtime


def audit_full_graph(c, mesh, matrix, graph):
    """Check one saved graph against an independently supplied physical CSR.

    Unlike the producer, this constructs a sparse exchange matrix from the
    saved arrays. Coverage, physical volumes and every retained CSR coefficient
    are checked separately; matching a few test-vector actions is insufficient.
    """
    validate_full(c)
    capacity, width = c.sample.full_grid.nodes, c.sample.full_grid.neighbors
    points, volumes, mouth = mesh['points'], mesh['volumes'], mesh['mouth']
    n = len(points)
    if (n > capacity or points.shape != (n, 2) or volumes.shape != (n,)
            or mouth.shape != (n,) or mouth.dtype != np.bool_
            or not np.isfinite(points).all() or not np.isfinite(volumes).all()
            or np.any(volumes <= 0) or np.all(mouth)):
        raise ValueError('Invalid physical mesh for full-grid mathematical audit')
    shapes = dict(graph_points=(capacity, 2), core_indices=(capacity,),
                  neighbors=(capacity, width), coefficients=(capacity, width),
                  pde_weights=(capacity,))
    if set(graph) != set(shapes):
        raise ValueError('Unexpected full graph schema')
    for name, shape in shapes.items():
        value = graph[name]
        dtype = np.dtype('int32' if name in ('core_indices', 'neighbors') else 'float32')
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid saved full graph shape/dtype/values: {name}')
    if not np.array_equal(graph['graph_points'][:n], points.astype(np.float32)):
        raise ValueError('Full graph must retain every mesh node in physical order')
    for name in ('core_indices', 'neighbors'):
        if np.any(graph[name] < 0) or np.any(graph[name] >= capacity):
            raise ValueError(f'Out-of-range full graph index: {name}')
    active = graph['pde_weights'] > 0
    expected_active = np.zeros(capacity, dtype=bool)
    expected_active[:n] = ~mouth
    if not np.array_equal(active, expected_active):
        raise ValueError('Complete PDE coverage must include each non-mouth node exactly once')
    physical_weight = np.zeros(capacity, dtype=np.float32)
    physical_weight[:n] = np.where(mouth, 0., volumes)
    if not np.array_equal(graph['pde_weights'], physical_weight):
        raise ValueError('PDE weights must equal physical lumped volumes, with zero mouth/padding weights')
    if (not np.array_equal(graph['core_indices'][active], np.flatnonzero(active))
            or np.any(graph['core_indices'][~active] != 0)):
        raise ValueError('Full PDE rows must represent their own nodes once; inactive cores are zero')
    coefficients, neighbors = graph['coefficients'], graph['neighbors']
    if np.any(coefficients < 0) or np.any(coefficients[~active] != 0):
        raise ValueError('Inactive rows must have zero coefficients; active exchange must be nonnegative')
    if not sparse.issparse(matrix) or matrix.shape != (n, n) or not np.isfinite(matrix.data).all():
        raise ValueError('Independent native operator must be a finite square sparse matrix')
    expected = -matrix.astype(np.float32).tocsr()
    expected.setdiag(0)
    expected.eliminate_zeros()
    if np.any(expected.data < 0):
        raise ValueError('Independent operator has the wrong negative-Laplacian sign')
    expected = (sparse.diags((~mouth).astype(np.float32))@expected).tocsr()
    expected.eliminate_zeros()
    rows, columns = np.nonzero(coefficients)
    destinations = neighbors[rows, columns]
    if np.any(destinations >= n) or np.any(destinations == rows):
        raise ValueError('Nonzero full-row edges must connect distinct actual mesh nodes')
    actual = sparse.coo_matrix((coefficients[rows, columns].astype(np.float64),
                                (rows, destinations)), shape=(n, n)).tocsr()
    actual.sum_duplicates()
    if actual.nnz != len(rows):
        raise ValueError('Duplicate physical neighbor in a complete PDE row')
    delta = actual-expected.astype(np.float64)
    delta.eliminate_zeros()
    if delta.nnz:
        raise ValueError('Saved full rows omit or alter native CSR coefficients')
    return dict(nodes=n, free_pde_nodes=int(active.sum()), mouth_nodes=int(mouth.sum()),
        padding_nodes=capacity-n, edges=int(actual.nnz),
        physical_free_volume=float(volumes[~mouth].sum()),
        float32_quadrature_sum=float(graph['pde_weights'].sum(dtype=np.float64)),
        complete_free_row_coverage=True, physical_volume_weights=True,
        full_native_csr_coefficients_exact=True, mouth_neighbors_retained=True)


def _audit_slot(arguments):
    kind, slot, config, record, graph = arguments
    from .mesh import build_mesh
    from .operators import build_operator
    c = ConfigDict(config)
    mesh = build_mesh(c, record, c.mesh.spacing)
    matrix = build_operator(c, record, mesh['points'], mesh['volumes'])
    return dict(kind=kind, slot=slot, **audit_full_graph(c, mesh, matrix, graph))


def _common_sampling(run, control, c):
    pairs = (('sampling.npz', ('condition_ids', 'pool_ids', 'times')),
             ('online_sampling.npz', ('geometry_draw_ids', 'spatial_versions')))
    for filename, names in pairs:
        with np.load(run/filename, allow_pickle=False) as full, np.load(control/filename, allow_pickle=False) as core:
            if set(full.files) != set(names) or set(core.files) != set(names):
                raise ValueError(f'Unexpected common sampling schema: {filename}')
            for name in names:
                shape = (c.train.iterations, c.sample.Nt if name == 'times' else c.operator.condition_batch)
                actual, previous = full[name], core[name]
                if actual.shape != shape or not np.isfinite(actual).all():
                    raise ValueError(f'Invalid full common sampling shape/values: {name}')
                if actual.dtype != previous.dtype or not np.array_equal(actual, previous):
                    raise ValueError(f'Actual common sampling differs: {name}')
                if name == 'times':
                    if actual.dtype != np.float32 or np.any(actual < 0) or np.any(actual > c.sample.t_max):
                        raise ValueError('Invalid full physical time draws')
                elif not np.issubdtype(actual.dtype, np.integer):
                    raise ValueError(f'Integer common sampling required: {name}')
                elif name in ('condition_ids', 'pool_ids'):
                    bound = c.operator.geometry_count if name == 'condition_ids' else c.sample.K_pool
                    if np.any(actual < 0) or np.any(actual >= bound):
                        raise ValueError(f'Common sampling index outside declared capacity: {name}')


def _common_installed_geometry(run, control):
    journals = [[json.loads(line) for line in (path/'online_refresh.jsonl').read_text().splitlines()]
                for path in (run, control)]
    if len(journals[0]) != len(journals[1]):
        raise ValueError('Full/core installed event counts differ')
    event_keys = ('event', 'completed_updates', 'slots', 'draw_ids', 'spatial_versions',
                  'seed', 'geometry_namespace', 'spatial_namespace', 'spatial_merged_into_geometry')
    record_keys = ('slot', 'draw_id', 'spatial_version', 'kind', 'geometry', 'seed',
                   'geometry_namespace', 'spatial_seed_words', 'generation_diagnostics')
    requests = 0
    for full, core in zip(*journals, strict=True):
        if any(full[name] != core[name] for name in event_keys):
            raise ValueError('Full/core installed geometry/revision event differs')
        if len(full['records']) != len(core['records']):
            raise ValueError('Full/core installed slot record count differs')
        for a, b in zip(full['records'], core['records'], strict=True):
            if any(a[name] != b[name] for name in record_keys):
                raise ValueError('Full/core geometry, IC seed or sampling-generation record differs')
            requests += 1
    return len(journals[0]), requests


def validate_full_result(run, control):
    """Audit a completed full run once, including initial/final mathematical rows."""
    run, control = Path(run).resolve(strict=True), Path(control).resolve(strict=True)
    c = ConfigDict(_read(run/'config.json'))
    runtime, summary = _read(run/'runtime.json'), _read(run/'summary.json')
    prior = validate_full_control(c, control, runtime['inputs'], runtime['reference'])
    if (summary['steps'] != c.train.iterations or not (run/'model_final.eqx').is_file()
            or not (run/'optimizer_final.eqx').is_file() or not runtime['cold_initialization']
            or runtime['training_uses_reference_targets'] or not runtime['rolling_inputs']
            or runtime['pde_sampling'] != 'full_grid' or runtime['pool_ids_used_for_pde'] is not False
            or runtime['full_grid_control'] != str(control) or runtime['seed'] != prior['seed']):
        raise ValueError('Completed cold full-grid run with matching control provenance is required')
    audit = _receipt(run, c)
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if not _same_file(run/name, control/name):
            raise ValueError(f'Full/core cold initialization differs: {name}')
    for source in SHARED_SOURCES:
        if not _same_file(run/'source'/source, control/'source'/source):
            raise ValueError(f'Actual full/core shared source differs: {source}')
    _common_sampling(run, control, c)
    events, requests = _common_installed_geometry(run, control)
    manifest = _read(Path(runtime['inputs'])/'manifest.json')
    initial_records = manifest['geometries']
    full_state = _read(run/f'online_pool_{c.train.iterations:06d}.json')
    core_state = _read(control/f'online_pool_{c.train.iterations:06d}.json')
    for name in ('completed_updates', 'seed', 'draw_ids', 'spatial_versions', 'geometries', 'installed_events'):
        if full_state[name] != core_state[name]:
            raise ValueError(f'Full/core final pool state differs: {name}')
    jobs = []
    for kind, records in (('initial', initial_records), ('final', full_state['geometries'])):
        if len(records) != c.operator.geometry_count:
            raise ValueError('Full-grid graph audit requires every declared geometry slot')
        filename = f'online_{kind}_inputs.npz'
        with np.load(run/filename, allow_pickle=False) as full, np.load(control/filename, allow_pickle=False) as core:
            if set(full.files) != set(FIXED_KEYS+GRAPH_KEYS) or set(core.files) != set(full.files):
                raise ValueError('Unexpected full/core input schema')
            for name in FIXED_KEYS:
                a, b = full[name], core[name]
                if a.dtype != b.dtype or not np.array_equal(a, b):
                    raise ValueError(f'Full/core {kind} conditioning/IC/BC inputs differ: {name}')
            arrays = {name: full[name] for name in GRAPH_KEYS}
            if any(value.shape[:2] != (c.operator.geometry_count, 1) for value in arrays.values()):
                raise ValueError('Full-grid arrays require exactly one graph per geometry')
            jobs.extend((kind, slot, c.to_dict(), record,
                         {name: value[slot, 0] for name, value in arrays.items()})
                        for slot, record in enumerate(records))
    workers = c.operator.online.workers
    with _worker_environment():
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_cpu_worker) as pool:
            mathematical = list(pool.map(_audit_slot, jobs))
    result = dict(status='passed', run=str(run), control=str(control),
        identical_cold_model=True, identical_cold_optimizer=True,
        identical_slot_pool_time_draws=True, identical_online_geometry_revision_traces=True,
        identical_shared_sources=list(SHARED_SOURCES), common_fixed_input_keys=list(FIXED_KEYS),
        initial_final_fixed_inputs_exact=True, matched_installation_events=events,
        matched_installed_payload_identities=requests, online_input_audit=str(run/'online_validation.json'),
        audited_updates=audit['updates'], allowed_config_changes=['sample.full_grid'],
        pool_ids_used_for_pde=False,
        mathematical_graph_audit=dict(workers=workers, initial_slots=c.operator.geometry_count,
            final_slots=c.operator.geometry_count, total_slots=len(mathematical), records=mathematical,
            scope='Independent mesh/CSR rebuild verifies every free row once, physical-volume weights, complete neighbors and zero mouth/padding PDE measure.'),
        intermediate_input_scope=('All consumed IDs, installed geometries and IC random seeds match; '
            'initial/final fixed arrays are directly equal. Intermediate payload arrays are not stored.'),
        pde_solves=0, reference_targets_used=False)
    (run/'full_grid_control_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    import argparse
    from .online_validation import audit_online_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--workers', required=True, type=int,
                        help='Online reconstruction workers; independent row audit uses operator.online.workers')
    args = parser.parse_args()
    audit_online_run(args.run, args.workers)
    result = validate_full_result(args.run, args.control)
    print(json.dumps(dict(status=result['status'], run=result['run'],
        mathematical_slots=result['mathematical_graph_audit']['total_slots'])), flush=True)


if __name__ == '__main__':
    main()
