import unittest

import jax.numpy as jnp
import equinox as eqx
import numpy as np

from examples.elasticity2d.losses import loss_pde, make_loss_fn
from nonlocal_pinn.operators import ElasticD


def _grid(n=13):
    q = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(q, q, indexing="xy")
    return np.stack([X.ravel(), Y.ravel()], axis=-1)


class _ResidualOperator:
    def elastic_matvec(self, u, lam, mu):
        del lam, mu
        return jnp.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])


class _ZeroNet:
    def __call__(self, x, t):
        del t
        return jnp.zeros(2, dtype=x.dtype)


class _LinearNet(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __call__(self, x, t):
        del t
        return self.weight @ x + self.bias


class ElasticDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = _grid()
        cls.dV = (1.0 / 12.0) ** 2
        cls.operator = ElasticD(cls.x, 0.24, dV=cls.dV, kernel="cubic")

    def test_affine_displacement_reproduces_gradient(self):
        A = np.array([[1.2, -0.3], [0.4, 0.7]], dtype=np.float32)
        u = self.x @ A.T
        row = np.asarray(self.operator._row_idx)
        col = np.asarray(self.operator._col_idx)
        xi = np.asarray(self.operator._xi)
        weights = np.asarray(self.operator._w_pos)
        du = u[col] - u[row]

        raw = np.zeros((len(self.x), 2, 2), dtype=np.float64)
        np.add.at(raw, row, weights[:, None, None] *
                  du[:, :, None] * xi[:, None, :])
        gradient = raw @ np.asarray(self.operator._K_inv)
        np.testing.assert_allclose(
            gradient, np.broadcast_to(A, gradient.shape),
            rtol=2e-5, atol=2e-5)

    def test_force_state_conserves_linear_and_angular_momentum(self):
        rng = np.random.default_rng(7)
        u = jnp.asarray(rng.normal(size=self.x.shape), dtype=jnp.float32)
        residual = np.asarray(self.operator.elastic_matvec(u, 1.3, 0.7))
        resultant = self.dV * residual.sum(axis=0)
        moment = self.dV * np.sum(
            self.x[:, 0] * residual[:, 1] - self.x[:, 1] * residual[:, 0])

        np.testing.assert_allclose(resultant, 0.0, atol=2e-4)
        np.testing.assert_allclose(moment, 0.0, atol=2e-4)

    def test_rigid_translation_has_zero_residual(self):
        u = jnp.tile(jnp.array([[0.37, -0.22]], dtype=jnp.float32),
                     (len(self.x), 1))
        residual = np.asarray(self.operator.elastic_matvec(u, 1.3, 0.7))
        np.testing.assert_allclose(residual, 0.0, atol=2e-6)

    def test_interior_quadratic_has_navier_cauchy_sign(self):
        x = _grid(n=31)
        h = 1.0 / 30.0
        operator = ElasticD(x, 0.2, dV=h * h, kernel="cubic")
        u = jnp.stack([x[:, 0] ** 2, x[:, 0] * x[:, 1]], axis=-1)
        residual = np.asarray(operator.elastic_matvec(u, 1.3, 0.7))
        center = (31 // 2) * 31 + 31 // 2
        np.testing.assert_allclose(residual[center], [7.4, 0.0],
                                   rtol=2e-5, atol=2e-5)

    def test_subgraph_matches_full_residual_on_core(self):
        rng = np.random.default_rng(11)
        u = jnp.asarray(rng.normal(size=self.x.shape), dtype=jnp.float32)
        full = np.asarray(self.operator.elastic_matvec(u, 1.3, 0.7))
        all_idx, sub_edges, core_mask, _ = self.operator.sub_sample(3, 10)
        sub = np.asarray(self.operator.elastic_matvec_sub(
            u[all_idx], sub_edges, core_mask, 1.3, 0.7))
        np.testing.assert_allclose(
            sub[core_mask], full[np.asarray(all_idx)[core_mask]],
            rtol=2e-5, atol=2e-5)

    def test_row_mask_excludes_prescribed_rows(self):
        points = jnp.zeros((3, 2), dtype=jnp.float32)
        row_mask = jnp.array([True, False, True])
        value = loss_pde(
            _ZeroNet(), points, _ResidualOperator(), 1.0, 1.0,
            row_mask=row_mask)
        np.testing.assert_allclose(float(value), 3.0, atol=1e-6)

    def test_loss_factories_accept_full_subgraph_and_local_keywords(self):
        net = _LinearNet(
            weight=jnp.array([[1.0, 0.2], [-0.1, 0.8]], dtype=jnp.float32),
            bias=jnp.zeros(2, dtype=jnp.float32))
        row_mask = jnp.asarray(self.x[:, 0] > 0.1)
        full_loss = make_loss_fn(
            self.x, self.operator, 1.3, 0.7, pde_mask=row_mask)
        full = full_loss(
            net, jnp.ones(1), jnp.ones(1), False,
            x_points=self.x, sub_data=None, core_mask=None,
            row_mask=row_mask)
        self.assertTrue(np.isfinite(float(full[0])))

        idx, edges, core, _ = self.operator.sub_sample(5, 10)
        sub = full_loss(
            net, jnp.ones(1), jnp.ones(1), False,
            x_points=self.x[idx], sub_data=edges, core_mask=core,
            row_mask=row_mask[idx])
        self.assertTrue(np.isfinite(float(sub[0])))

        local_loss = make_loss_fn(
            None, None, 1.3, 0.7,
            x_bc_top=self.x[:4], x_bc_bottom=self.x[4:8],
            x_bc_hole=self.x[8:12], hole_center=(0.5, 0.5, 0.2, 0.2))
        local = local_loss(
            net, jnp.ones(4), jnp.ones(4), False,
            x_points=self.x[:20], sub_data=None, core_mask=None,
            row_mask=None)
        self.assertTrue(np.isfinite(float(local[0])))


if __name__ == "__main__":
    unittest.main()
