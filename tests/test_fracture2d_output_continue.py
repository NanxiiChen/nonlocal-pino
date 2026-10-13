"""Continuation checkpoints preserve ancestry and reject changed candidates."""
import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.compact_damage_features import Readout
from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.output_damage_continue import accepted_state, save_pending, load_pending


class OutputContinueTest(unittest.TestCase):
    def test_pending_generation_and_accepted_only_restart(self):
        jax.config.update('jax_enable_x64', True)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            net = Readout(jnp.asarray([.1, .2]))
            q, d, H = jnp.zeros(6), jnp.ones(3)*.1, jnp.ones(3)
            for sweep in range(3):
                save_pending(folder, net, q, d, H+sweep, index=35, load=.0035,
                    completed_sweep=sweep, consecutive=sweep, reaction=4., mechanical={'converged': True},
                    parent_sha='parent', last_image=None if sweep==0 else dict(H_used=H, damage_image=d))
            restored, rq, rd, rh, meta, last = load_pending(folder, net,
                index=35, load=.0035, parent_sha='parent')
            np.testing.assert_array_equal(restored.coefficients, net.coefficients)
            np.testing.assert_array_equal(rh, H+2)
            np.testing.assert_array_equal(last['damage_image'], d)
            self.assertEqual(meta['completed_sweep'], 2)
            self.assertFalse((folder/'pending_0000_fields.npz').exists())
            self.assertTrue((folder/'pending_0001_fields.npz').exists())
            with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                load_pending(folder, net, index=36, load=.0036, parent_sha='parent')
            np.savez_compressed(folder/'fields.npz', q=rq, d=rd, H=rh)
            (folder/'network.eqx').write_bytes(b'weights')
            report = dict(status='ready', physical_state_promoted=True, parent_sha256='parent',
                state_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'),
                outer_convergence=dict(converged=True, checks={'AT2': {'pass': True}}))
            (folder/'result.json').write_text(json.dumps(report))
            accepted_state(folder, 'parent')
            report['status'], report['physical_state_promoted'] = 'inner_rejected', False
            (folder/'result.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'rejected/changed/unready'):
                accepted_state(folder, 'parent')
            (folder/meta['network_file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                load_pending(folder, net, index=35, load=.0035, parent_sha='parent')


if __name__ == '__main__':
    unittest.main()
