"""Reproduced initializer-domain failure and declared-fork ancestry checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.bounded_damage_output import make_image
from examples.fracture2d.configs.output_damage_safe_continue import get_config
from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.output_damage_safe_continue import bounded_initializer, verify_parent


class SafeInitializerTest(unittest.TestCase):
    def test_changed_history_domain_and_physical_parameter_gradient(self):
        jax.config.update('jax_enable_x64', True)
        frozen = lambda H, load: H*load
        safe = bounded_initializer(frozen, epsilon=1e-10)
        D = jnp.asarray([[2., -2.], [-2., 2.]])
        B = jnp.asarray([[1., .2], [.3, 1.]])
        options = dict(mode='sigmoid', gc0=2., gc1=.3, floor=1e-8,
                       initialization_epsilon=1e-10)
        old, _ = make_image(frozen, lambda b, H, load: B@b, jnp.diag(D), **options)
        new, anchors = make_image(safe, lambda b, H, load: B@b, jnp.diag(D), **options)
        beta = jnp.asarray([.1, -.2])
        H = jnp.asarray([.2, .5])
        np.testing.assert_array_equal(old(beta, H, .8), new(beta, H, .8))
        H = jnp.asarray([2., 9.])
        self.assertTrue(np.isfinite(anchors(H, 1.)[0]).all())
        damage = new(beta, H, 1.)
        self.assertGreater(float(damage.min()), 0.)
        self.assertLess(float(damage.max()), 1.)
        for H, load in [(H, 1.), (H*3, 1.2)]:
            def loss(b):
                d = new(b, H, load)
                r = (2+2*H)*d+.3*(D@d)-2*H
                return jnp.mean(r*r)
            direction = jnp.asarray([.3, -.7])
            eps = 1e-5
            fd = (loss(beta+eps*direction)-loss(beta-eps*direction))/(2*eps)
            np.testing.assert_allclose(fd, jnp.dot(jax.grad(loss)(beta), direction),
                                       rtol=1e-7, atol=1e-10)

    def test_fork_refuses_rejected_parent_and_changed_archived_code(self):
        c = get_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root/'state_0035'
            parent.mkdir()
            previous = dict(q=np.zeros(4), d=np.zeros(2), H=np.zeros(2))
            np.savez(parent/'fields.npz', **previous, d_parent=previous['d'], H_parent=previous['H'])
            (parent/'network.eqx').write_bytes(b'network')
            report = dict(index=35, load=35*c.data.load_step, status='ready',
                physical_state_promoted=True, parent_sha256='prior', state_sha256=sha(parent/'fields.npz'),
                network_sha256=sha(parent/'network.eqx'),
                outer_convergence=dict(converged=True, checks={'AT2': {'pass': True}}))
            (parent/'result.json').write_text(json.dumps(report))
            c.continuation.parent = str(parent)
            c.continuation.first_index = 36
            c.continuation.parent_field_sha256 = report['state_sha256']
            c.continuation.parent_network_sha256 = report['network_sha256']
            original = c.to_dict()
            del original['continuation']['initializer']
            original['continuation']['first_index'] = 35
            (root/'code_snapshot').mkdir()
            archived = root/'code_snapshot'/'output_damage_continue.py'
            archived.write_bytes(b'archived runner')
            identity = dict(schema='certified-sigmoid-own-history-continuation-v1',
                config=original, own_history=True, direct_targets=False, damage_correction=False,
                source={}, code={'output_damage_continue.py': sha(archived)})
            (root/'identity.json').write_text(json.dumps(identity))
            audit = dict(pass_replay=True, independent_numpy_network_replay=True,
                own_history=True, parent_unchanged=True, total_ready_loads=35,
                reports=[dict(index=35, status='ready')])
            (root/'audit.json').write_text(json.dumps(audit))
            (root/'audit_script.py').write_text('audit')
            (root/'compilation_parity.json').write_text(json.dumps(dict(pass_parity=True)))
            with patch('examples.fracture2d.output_damage_continue.verify_parent',
                       return_value=(previous, {'state_sha256': 'prior'}, {})):
                verify_parent(c)
                report['status'] = 'inner_rejected'
                (parent/'result.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'rejected/changed/unready'):
                    verify_parent(c)
                report['status'] = 'ready'
                (parent/'result.json').write_text(json.dumps(report))
                archived.write_bytes(b'changed runner')
                with self.assertRaisesRegex(ValueError, 'archived code changed'):
                    verify_parent(c)


if __name__ == '__main__':
    unittest.main()
