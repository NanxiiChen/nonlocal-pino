"""Fork only own-neural accepted states and retain immutable ancestry."""
import json
from pathlib import Path
import tempfile
import unittest
from ml_collections import ConfigDict
import numpy as np
from examples.fracture2d._legacy.compact_damage_fork import prepare
from examples.fracture2d._legacy.local_damage_trial import sha
import examples.fracture2d._legacy.compact_damage_fork as fork_module


class CompactForkTest(unittest.TestCase):
    def test_accepted_prefix_copy_and_rejected_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, target = Path(directory)/'parent', Path(directory)/'fork'
            folder = parent/'state_0001'
            folder.mkdir(parents=True)
            c = ConfigDict(dict(data=dict(end_state_index=2, reference_indices=[1, 2], load_step=.1),
                output=dict(resume=False), physical=dict(E=1.),
                loading=dict(output_dir=str(parent), methods=['residual_mse'], lbfgs=dict(maxiter=2))))
            identity = dict(schema='own-zero-compact-neural-loading-v1', own_zero_initial_state=True,
                direct_targets=False, damage_correction=False, config=c.to_dict(),
                runner_sha256='declared_parent_code_hash')
            (parent/'identity.json').write_text(json.dumps(identity))
            np.savez(folder/'fields.npz', q=np.zeros(4), d=np.ones(2)*.1, H=np.ones(2),
                d_parent=np.zeros(2), H_parent=np.zeros(2))
            (folder/'network.eqx').write_bytes(b'fake serialization bytes')
            report = dict(index=1, load=.1, status='ready', physical_state_promoted=True,
                outer_convergence=dict(converged=True, checks={'current': {'pass': True}}),
                parent_sha256=None, state_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'))
            (folder/'result.json').write_text(json.dumps(report))
            c.loading.output_dir = str(target)
            c.loading.methods = ['residual_l16']
            c.loading.fork = ConfigDict(dict(parent_root=str(parent), prefix_length=1,
                parent_identity_sha256=sha(parent/'identity.json'), runner_sha256=sha(fork_module.__file__)))
            prepare(c, resume=False)
            self.assertTrue(c.output.resume)
            self.assertEqual(sha(target/'state_0001/fields.npz'), report['state_sha256'])
            provenance = json.loads((target/'fork.json').read_text())
            self.assertEqual(provenance['parent_numerical_runner_sha256'], identity['runner_sha256'])
            self.assertEqual(json.loads((target/'identity.json').read_text())['runner_sha256'], sha(fork_module.run.__code__.co_filename))
            prepare(c, resume=True)
            report['status'] = 'inner_rejected'
            (folder/'result.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'rejected'):
                prepare(c, resume=True)


if __name__ == '__main__':
    unittest.main()
