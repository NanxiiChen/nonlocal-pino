"""Independent tetrahedral assembly and weak/tanh residual checks."""
from itertools import permutations, product
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.fem_weak import build_cube, validate_comparison
from examples.icemelting3d.losses import _tanh_difference_ratio
from examples.icemelting3d.spatial_sampling import core_subgraph
from examples.icemelting3d.configs.train_operator_compare_fem import get_config


def matrix(graph):
    rows = np.repeat(np.arange(graph._N), 6)
    out = np.zeros((graph._N, graph._N))
    np.add.at(out, (rows,graph._nei_idx_np.ravel()), -graph._nei_w_np.ravel())
    out[np.diag_indices(graph._N)] += graph._nei_w_np.sum(axis=1)
    return out


class MeltingFEMOperatorTests(unittest.TestCase):
    def test_independent_element_assembly_and_energy(self):
        n=4; axis=np.linspace(0,1,n)
        points=np.stack(np.meshgrid(axis,axis,axis,indexing='xy'),-1).reshape(-1,3)
        graph,info=build_cube(n);mass=np.zeros(n**3);stiffness=np.zeros((n**3,n**3))
        for origin in product(range(n-1),repeat=3):
            for order in permutations(range(3)):
                vertices=[np.array(origin)]
                for component in order:
                    vertex=vertices[-1].copy();vertex[component]+=1;vertices.append(vertex)
                ids=np.array([v[1]*n*n+v[0]*n+v[2] for v in vertices])
                x=points[ids];jacobian=(x[1:]-x[0]).T
                volume=abs(np.linalg.det(jacobian))/6
                reference=np.vstack((-np.ones(3),np.eye(3)))
                gradients=reference@np.linalg.inv(jacobian)
                mass[ids]+=volume/4
                stiffness[np.ix_(ids,ids)]+=volume*(gradients@gradients.T)
        actual=graph.mass[:,None]*matrix(graph)
        np.testing.assert_allclose(graph.mass,mass,rtol=2e-14,atol=1e-16)
        np.testing.assert_allclose(actual,stiffness,rtol=2e-14,atol=1e-14)
        np.testing.assert_allclose(actual,actual.T,rtol=2e-14,atol=1e-14)
        np.testing.assert_allclose(actual.sum(axis=1),0.,atol=2e-15)
        self.assertAlmostEqual(mass.sum(),1.)
        f=points@np.array([1.,2.,-3.])+2
        self.assertAlmostEqual(float(f@actual@f),14.,places=12)
        interior=np.all((points>0)&(points<1),axis=1)
        np.testing.assert_allclose((matrix(graph)@np.sum(points**2,axis=1))[interior],-6.,atol=1e-12)
        self.assertEqual(info['tetrahedra'],6*(n-1)**3)

    def test_complete_core_rows_and_transformed_weak_residual(self):
        graph,_=build_cube(5);d=matrix(graph)
        cores=np.array([0,1,24,62,124],dtype=np.int32)
        ids,sub,mask=core_subgraph(graph,cores)
        row,col,values,_=sub
        u=np.linspace(-1.7,2.1,125);phi=np.tanh(u);ut=np.cos(u)
        expected=ut+(d@phi)/(1-phi**2)-.3*phi+.7
        with jax.enable_x64():
            ratio=np.asarray(_tanh_difference_ratio(jnp.asarray(u[ids[row]]),jnp.asarray(u[ids[col]])))
        diffusion=np.bincount(row,weights=-values*ratio,minlength=len(ids))
        actual=ut[ids]+diffusion-.3*phi[ids]+.7
        np.testing.assert_allclose(actual[mask],expected[cores],rtol=1e-12,atol=1e-12)

    def test_explicit_control_does_not_change_native_reference_identity(self):
        from examples.icemelting3d.operator_data import identity,validate_config
        from examples.icemelting3d.configs.train_operator import get_config as original
        c=get_config();validate_config(c)
        self.assertEqual(identity(c),identity(original()))
        c.operator.spatial_discretization='unknown'
        with self.assertRaises(ValueError):validate_comparison(c)


if __name__=='__main__':unittest.main()
