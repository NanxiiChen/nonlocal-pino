"""Independent CPU/JAX correction trajectory and hard budget validation."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from pinn.operators import ExplicitVolumeSparseD
from examples.fracture2d.damage_correction import make_corrector

jax.config.update('jax_enable_x64', True)


class DamageCorrectionTest(unittest.TestCase):
    def test_checked_iteration_matches_cpu_and_exact_solve(self):
        x = np.array([[0., 0.], [.5, 0.], [0., .4]])
        volumes = np.array([1., 2., .5])
        edges = np.array([[i, j] for i in range(3) for j in range(3) if i != j], np.int64)
        op = ExplicitVolumeSparseD(x, volumes, edges, 1., 'cubic')
        controls = dict(omega=.8, maxiter=2048, check_every=8, residual_limit=1e-10, range_atol=1e-10)
        solve = make_corrector(op, volumes, 2., 3., controls, roundoff_factor=1024.)
        H, initial = np.array([.1, .8, .3]), np.array([.1, .2, .4])
        result, info = solve(jnp.asarray(H), jnp.asarray(initial))
        A = np.diag(2+2*H)+3*op.to_scipy().toarray()
        expected = np.linalg.solve(A, 2*H)
        np.testing.assert_allclose(result, expected, atol=1e-9)
        cpu = initial.copy()
        for _ in range(info['iterations']):
            cpu -= .8*(A@cpu-2*H)/np.diag(A)
        np.testing.assert_allclose(result, cpu, rtol=1e-13, atol=1e-14)
        controls['maxiter'] = 8
        truncated = make_corrector(op, volumes, 2., 3., controls, roundoff_factor=1024.)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            truncated(jnp.asarray(H), jnp.asarray(initial))
        with self.assertRaisesRegex(ValueError, 'bounded initial'):
            solve(jnp.asarray(H), jnp.full(3, 1.1))


if __name__ == '__main__':
    unittest.main()
