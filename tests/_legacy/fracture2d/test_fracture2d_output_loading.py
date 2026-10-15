"""A bounded-head transfer cannot promote a rejected/changed physical parent."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from examples.fracture2d._legacy.configs.output_damage_loading import get_config
from examples.fracture2d._legacy.local_damage_trial import sha
from examples.fracture2d._legacy.output_damage_loading import parent_state


class OutputLoadingTest(unittest.TestCase):
    def test_accepted_ancestry_and_rejected_parent_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = None
            for index in (1, 2):
                folder = root/f'state_{index:04d}'
                folder.mkdir()
                np.savez_compressed(folder/'fields.npz', q=np.zeros(6), d=np.zeros(3), H=np.ones(3)*index)
                (folder/'network.eqx').write_bytes(bytes([index]))
                report = dict(index=index, status='ready', physical_state_promoted=True,
                    parent_sha256=previous, state_sha256=sha(folder/'fields.npz'),
                    network_sha256=sha(folder/'network.eqx'),
                    outer_convergence=dict(converged=True, checks=dict(AT2={'pass': True})))
                (folder/'result.json').write_text(json.dumps(report))
                previous = report['state_sha256']
            c = get_config()
            c.head_loading.parent = str(folder)
            c.head_loading.parent_field_sha256 = previous
            c.head_loading.parent_network_sha256 = report['network_sha256']
            c.head_loading.indices = [3, 4]
            fields, accepted = parent_state(c)
            np.testing.assert_array_equal(fields['H'], np.full(3, 2.))
            self.assertEqual(accepted['index'], 2)
            report['status'], report['physical_state_promoted'] = 'inner_rejected', False
            (folder/'result.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'prefix fails'):
                parent_state(c)
            report['status'], report['physical_state_promoted'] = 'ready', True
            (folder/'result.json').write_text(json.dumps(report))
            (folder/'network.eqx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'prefix fails'):
                parent_state(c)


if __name__ == '__main__':
    unittest.main()
