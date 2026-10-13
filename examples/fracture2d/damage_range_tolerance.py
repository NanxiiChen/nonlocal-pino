"""Explicit numerical damage-range tolerance for the unchanged local FEM law."""
import numpy as np

from examples.fracture2d.local_mechanics import LocalMechanics, QUADRATURE


class TolerantLocalMechanics(LocalMechanics):
    def __init__(self, *args, damage_range_atol, **kwargs):
        if not np.isfinite(damage_range_atol) or not 0 < damage_range_atol < 1:
            raise ValueError('damage_range_atol must be explicit finite and between0 and1')
        super().__init__(*args, **kwargs)
        self.damage_range_atol = damage_range_atol

    def factors(self, damage):
        if (damage.shape != (self.n_material,) or damage.dtype != np.float64
                or not np.all(np.isfinite(damage))
                or np.min(damage) < -self.damage_range_atol
                or np.max(damage) > 1+self.damage_range_atol):
            raise ValueError('P1 material damage violates explicit numerical range tolerance')
        factors = np.full(len(self.triangles), 1+self.k_res)
        values = damage[self.triangles[:self.n_material_cells]] @ QUADRATURE.T
        factors[:self.n_material_cells] = np.mean((1-values)**2, axis=1)+self.k_res
        return factors
