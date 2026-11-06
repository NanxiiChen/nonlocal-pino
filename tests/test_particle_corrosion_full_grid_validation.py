"""Independent full-row corruption checks and strict paired-control provenance."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy import sparse

from examples.particle_corrosion2d.configs.train_full_grid import get_config
from examples.particle_corrosion2d.full_grid import full_graph
from examples.particle_corrosion2d.full_grid_validation import (
    _common_installed_geometry, _common_sampling, audit_full_graph,
    validate_full_control, validate_full_result)
from examples.particle_corrosion2d.online_validation import SHARED_SOURCES


class FullRowMathematicalTests(unittest.TestCase):
    def setUp(self):
        self.c = get_config()
        self.c.sample.full_grid.nodes = 8
        self.c.sample.full_grid.neighbors = 4
        points = np.column_stack((np.linspace(.1, .9, 5), np.full(5, .1)))
        volumes = np.array([.02, .04, .08, .16, .32])
        kernel = 1/(1+np.linalg.norm(points[:, None]-points[None], axis=-1))
        np.fill_diagonal(kernel, 0)
        exchange = kernel*volumes[None]
        self.matrix = sparse.csr_matrix(np.diag(exchange.sum(axis=1))-exchange)
        self.mesh = dict(points=points, volumes=volumes,
                         mouth=np.array([True, False, False, False, False]))
        self.graph = full_graph(self.c, self.mesh, self.matrix)

    def test_complete_volume_weighted_rows_include_mouth_neighbors(self):
        result = audit_full_graph(self.c, self.mesh, self.matrix, self.graph)
        self.assertEqual(result['free_pde_nodes'], 4)
        self.assertEqual(result['padding_nodes'], 3)
        self.assertEqual(result['edges'], 16)
        self.assertAlmostEqual(result['physical_free_volume'], .6)
        active = self.graph['pde_weights'] > 0
        self.assertTrue(np.all(np.any((self.graph['neighbors'][active] == 0)
                                     & (self.graph['coefficients'][active] > 0), axis=1)))
        self.assertEqual(self.graph['pde_weights'][0], 0)

    def test_uniform_weights_cannot_replace_physical_volumes(self):
        self.graph['pde_weights'][1:5] = 1
        with self.assertRaisesRegex(ValueError, 'physical lumped volumes'):
            audit_full_graph(self.c, self.mesh, self.matrix, self.graph)

    def test_missing_free_node_and_active_padding_are_rejected(self):
        for row in (2, 7):
            with self.subTest(row=row):
                graph = {key: value.copy() for key, value in self.graph.items()}
                graph['pde_weights'][row] = 0 if row == 2 else 1
                with self.assertRaisesRegex(ValueError, 'Complete PDE coverage'):
                    audit_full_graph(self.c, self.mesh, self.matrix, graph)

    def test_duplicate_physical_core_is_rejected(self):
        self.graph['core_indices'][2] = 1
        with self.assertRaisesRegex(ValueError, 'their own nodes once'):
            audit_full_graph(self.c, self.mesh, self.matrix, self.graph)

    def test_dropping_reservoir_neighbor_does_not_pass_row_audit(self):
        column = np.flatnonzero(self.graph['neighbors'][1] == 0)[0]
        self.graph['coefficients'][1, column] = 0
        with self.assertRaisesRegex(ValueError, 'omit or alter native CSR'):
            audit_full_graph(self.c, self.mesh, self.matrix, self.graph)

    def test_even_one_coefficient_bit_change_is_detected(self):
        value = self.graph['coefficients'][1, 0]
        self.graph['coefficients'][1, 0] = np.nextafter(value, np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, 'omit or alter native CSR'):
            audit_full_graph(self.c, self.mesh, self.matrix, self.graph)


class FullControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.control, self.run = root/'core', root/'full'
        self.inputs, self.reference = root/'inputs', root/'reference'
        for path in (self.control, self.run, self.inputs, self.reference):
            path.mkdir()
        self.c = get_config()
        self.c.train.iterations = 4
        self.c.operator.online.geometry_every = 2
        self.c.operator.online.spatial_every = 1
        prior = json.loads(self.c.to_json())
        prior['sample'].pop('full_grid')
        self.write(self.control/'config.json', prior)
        self.write(self.run/'config.json', json.loads(self.c.to_json()))
        runtime = dict(seed=193, cold_initialization=True, rolling_inputs=True,
                       training_uses_reference_targets=False,
                       inputs=str(self.inputs), reference=str(self.reference))
        self.write(self.control/'runtime.json', runtime)
        self.write(self.run/'runtime.json', dict(runtime, full_grid_control=str(self.control),
                                               pde_sampling='full_grid', pool_ids_used_for_pde=False))
        for path in (self.control, self.run):
            self.write(path/'summary.json', dict(steps=4))
            self.write(path/'online_validation.json', dict(status='passed', run=str(path),
                updates=4, exact_final_slots_reconstructed=64, unchanged_initial_slots=0,
                evaluation_overlap=0, future_training_fields=0))
            for name in ('model_initial.eqx', 'optimizer_initial.eqx', 'model_final.eqx', 'optimizer_final.eqx'):
                (path/name).write_bytes(name.encode()+b'\x00paired byte fixture')
            for source in SHARED_SOURCES:
                destination = path/'source'/source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(__file__).resolve().parents[1]/source, destination)
        self.draws = dict(condition_ids=np.arange(16, dtype=np.int32).reshape(4, 4),
                          pool_ids=np.zeros((4, 4), dtype=np.int32),
                          times=np.full((4, self.c.sample.Nt), .4, dtype=np.float32))
        self.online = dict(geometry_draw_ids=-1-self.draws['condition_ids'].astype(np.int64),
                           spatial_versions=np.zeros((4, 4), dtype=np.int64))
        for path in (self.control, self.run):
            np.savez(path/'sampling.npz', **self.draws)
            np.savez(path/'online_sampling.npz', **self.online)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data, allow_nan=False)+'\n')

    def test_accepts_only_full_grid_addition(self):
        result = validate_full_control(self.c, self.control, self.inputs, self.reference)
        self.assertEqual(result['seed'], 193)
        _common_sampling(self.run, self.control, self.c)

    def test_rejects_physics_core_count_and_schedule_changes(self):
        changes = ((self.c.physical, 'kappa'), (self.c.sample, 'K_core'),
                   (self.c.operator.online, 'geometry_every'))
        for group, name in changes:
            with self.subTest(name=name):
                value = group[name]
                group[name] = 2*value
                with self.assertRaisesRegex(ValueError, 'only sample.full_grid'):
                    validate_full_control(self.c, self.control, self.inputs, self.reference)
                group[name] = value

    def test_unqualified_control_and_source_changes_rejected(self):
        receipt_path = self.control/'online_validation.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['status'] = 'failed'
        self.write(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, 'online input audit'):
            validate_full_control(self.c, self.control, self.inputs, self.reference)
        receipt['status'] = 'passed'
        self.write(receipt_path, receipt)
        source = self.control/'source'/SHARED_SOURCES[1]
        source.write_bytes(source.read_bytes()+b'\n# numerical source drift\n')
        with self.assertRaisesRegex(ValueError, 'shared source differs'):
            validate_full_control(self.c, self.control, self.inputs, self.reference)

    def test_cold_state_and_unused_pool_draws_stay_paired(self):
        path = self.run/'optimizer_initial.eqx'
        path.write_bytes(path.read_bytes()+b' altered')
        with self.assertRaisesRegex(ValueError, 'cold initialization differs'):
            validate_full_result(self.run, self.control)
        changed = {name: value.copy() for name, value in self.draws.items()}
        changed['pool_ids'][1, 2] = 1
        np.savez(self.run/'sampling.npz', **changed)
        with self.assertRaisesRegex(ValueError, 'sampling differs: pool_ids'):
            _common_sampling(self.run, self.control, self.c)

    def test_stale_consumed_spatial_revision_is_rejected(self):
        changed = {name: value.copy() for name, value in self.online.items()}
        changed['spatial_versions'][3, 0] = 1
        np.savez(self.run/'online_sampling.npz', **changed)
        with self.assertRaisesRegex(ValueError, 'sampling differs: spatial_versions'):
            _common_sampling(self.run, self.control, self.c)

    def test_changed_installed_geometry_is_rejected(self):
        record = dict(slot=0, draw_id=0, spatial_version=0, kind='geometry',
            geometry={'known_test_parameters': [1., 2.]}, seed=193,
            geometry_namespace=271828, spatial_seed_words=[193, 161803, 1, 0, 0],
            generation_diagnostics={'sampling_calls': 1})
        event = dict(event=0, completed_updates=2, slots=[0], draw_ids=[0],
            spatial_versions=[0], seed=193, geometry_namespace=271828,
            spatial_namespace=161803, spatial_merged_into_geometry=[], records=[record])
        for path in (self.control, self.run):
            self.write(path/'online_refresh.jsonl', event)
        self.assertEqual(_common_installed_geometry(self.run, self.control), (1, 1))
        record['geometry']['known_test_parameters'][0] = 1.1
        self.write(self.run/'online_refresh.jsonl', event)
        with self.assertRaisesRegex(ValueError, 'geometry, IC seed'):
            _common_installed_geometry(self.run, self.control)


if __name__ == '__main__':
    unittest.main()
