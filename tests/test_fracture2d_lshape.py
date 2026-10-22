"""Nonconvex domain, finite-patch constraints and exact cold-loading resume."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from ml_collections import ConfigDict
from examples.fracture2d.mesh import l_panel_boundary, load_geometry, prepare
from examples.fracture2d.storage import atomic_json
from examples.fracture2d_lshape.solver import run_reference
from examples.fracture2d_lshape.train import run_training
from tests.test_fracture2d_fem_damage import configuration


def panel_config(root):
    c = configuration(root)
    c.boundary = ConfigDict(dict(mode='l_shaped_panel', corner_x=.5, corner_y=.5,
                                load_start=.94, load_end=1.))
    c.geometry.width = 1.
    c.geometry.height = 1.
    c.geometry.grip_extent = 0.
    xy = np.array([[0.,0.],[.5,0.],[0.,.5],[.5,.5],[1.,.5],
                   [0.,1.],[.5,1.],[1.,1.]], np.float64)
    tri = np.array([[0,1,3],[0,3,2],[2,3,6],[2,6,5],[3,4,7],[3,7,6]], np.int64)
    cloud = dict(coordinates=xy, triangles=tri, volumes=np.bincount(
        tri.ravel(), weights=np.full(tri.size, .125/3), minlength=8),
        bottom=xy[:,1]==0., top=(xy[:,0]==1.) & (xy[:,1]==.5), anchor_particle=np.int64(0))
    c.data.cloud = str(root/'panel_cloud.npz')
    np.savez(c.data.cloud, **cloud)
    c.mesh.prepared_dir = str(root/'panel_mesh')
    with contextlib.redirect_stdout(io.StringIO()):
        prepare(c)
    # This fixture checks geometry/lifecycle only; production uses actual UFL.
    atomic_json(Path(c.mesh.prepared_dir)/'fenics_parity.json', {'pass': True})
    return c, cloud


class LPanelTests(unittest.TestCase):
    def test_concave_boundary_and_free_horizontal_loading_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, cloud = panel_config(Path(tmp))
            m, op, identity = load_geometry(c)
            np.testing.assert_array_equal(m.constrained, [0,1,2,3,9])
            self.assertTrue(m.free[8])
            self.assertEqual(m.load_direction[9], 1.)
            self.assertAlmostEqual(float(m.volumes.sum()), .75)
            self.assertEqual(identity['config']['schema'], 'p1-local-l-panel-amor-lumped-v1')
            self.assertIsNone(identity['operator_path'])
            cloud['coordinates'][7] = [1.,.4]
            with self.assertRaisesRegex(ValueError, 'outside the material'):
                l_panel_boundary(cloud, c)
            c.boundary.load_start = .4
            with self.assertRaisesRegex(ValueError, 'geometry setting differs: boundary'):
                load_geometry(c)

    def test_l_panel_natural_damage_operator_and_nonzero_free_equilibrium(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = panel_config(Path(tmp))
            m, op, _ = load_geometry(c)
            np.testing.assert_allclose(op.stiffness@np.ones(8), 0., atol=1e-14)
            H = np.full(8, 2.)
            g0, g1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
            d = spsolve(diags(m.volumes*(g0+2*H))+g1*op.stiffness, 2*m.volumes*H)
            np.testing.assert_allclose(d, 2*H/(g0+2*H), rtol=1e-13)
            q = 1e-4*m.load_direction
            K = m.matrix(np.zeros(8))
            q[m.free] = spsolve(K[m.free][:,m.free], -(K@q)[m.free])
            np.testing.assert_allclose((K@q)[m.free], 0., atol=1e-12)
            self.assertGreater(float((K@q)@m.load_direction), 0.)
            self.assertGreater(np.linalg.norm(q[m.free]), 0.)

    def test_direct_and_neural_use_new_domain_and_exact_endpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = panel_config(root)
            c.output.root = str(root/'panel_reference')
            with contextlib.redirect_stdout(io.StringIO()):
                run_reference(c)
            c.output.reference_root = c.output.root
            c.output.root = str(root/'panel_neural')
            with contextlib.redirect_stdout(io.StringIO()):
                run_training(c)
            output = Path(c.output.root)
            d = json.loads((output/'result.json').read_text())
            self.assertTrue(d['complete'])
            with np.load(output/'state_0000/fields.npz') as z:
                for key in ('q','d','H'):
                    np.testing.assert_array_equal(z[key], 0.)
            old = (output/'state_0002/network.eqx').read_bytes()
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()):
                run_training(c)
            self.assertEqual(old, (output/'state_0002/network.eqx').read_bytes())


if __name__ == '__main__':
    unittest.main()
