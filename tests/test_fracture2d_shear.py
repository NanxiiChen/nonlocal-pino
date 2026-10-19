"""Material-only shear BCs, analytical patch and coupled entry regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.sparse.linalg import spsolve
from ml_collections import ConfigDict
from examples.fracture2d.mesh import geometry_identity, load_geometry, prepare, read_cloud, shear_boundary
from examples.fracture2d_shear.train import run_training
from examples.fracture2d_shear.solver import run_reference
from examples.fracture2d.storage import atomic_json
from tests.test_fracture2d_fem_damage import configuration


def shear_config(root):
    c = configuration(root)
    c.boundary = ConfigDict(dict(mode='horizontal_clamped_edges'))
    c.geometry.grip_extent = 0.
    c.mesh.prepared_dir = str(root/'shear_mesh')
    with contextlib.redirect_stdout(io.StringIO()):
        prepare(c)
    # Manufactured patch only; production requires actual independent UFL.
    atomic_json(Path(c.mesh.prepared_dir)/'fenics_parity.json', {'pass': True})
    return c


class ShearTests(unittest.TestCase):
    def test_original_material_domain_edge_constraints_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = shear_config(Path(tmp));m,op,identity=load_geometry(c)
            self.assertEqual(len(m.coordinates),m.n_material)
            self.assertEqual(len(m.triangles),m.n_material_cells)
            np.testing.assert_array_equal(m.load_direction,[0.,0.,0.,0.,1.,0.,1.,0.])
            np.testing.assert_array_equal(m.constrained,np.arange(8))
            self.assertEqual(identity['config']['boundary']['mode'],'horizontal_clamped_edges')
            self.assertFalse((Path(c.mesh.prepared_dir)/'damage_operator.npz').exists())
            archive=Path(c.mesh.prepared_dir)/'mesh.npz'
            with np.load(archive) as z:values={k:z[k].copy() for k in z.files}
            values['load_direction']=np.roll(values['load_direction'],1)
            np.savez(archive,**values)
            with self.assertRaisesRegex(ValueError,'boundary differs'):load_geometry(c)
            cloud=read_cloud(c.data.cloud);cloud['top'][2]=False
            with self.assertRaises(ValueError):shear_boundary(cloud)
            c.geometry.grip_extent=.048
            with self.assertRaisesRegex(ValueError,'zero grip extent'):geometry_identity(c)

    def test_affine_shear_reaction_energy_driver_and_natural_scalar_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = shear_config(Path(tmp));m,op,_=load_geometry(c)
            U=1e-4;q=np.column_stack((U*m.coordinates[:,1],np.zeros(4))).ravel()
            d=np.full(4,.2);K=m.matrix(d);force=K@q
            expected=m.mu*U*((1-.2)**2+c.physical.k_res)
            self.assertAlmostEqual(float(force@m.load_direction),expected,places=11)
            np.testing.assert_allclose(m.driver(q),.5*m.mu*U**2,rtol=1e-14)
            self.assertAlmostEqual(float(q@K@q),U*expected,places=13)
            np.testing.assert_allclose(op.stiffness@np.ones(4),0.,atol=1e-14)
            g0,g1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
            H=m.driver(q);from scipy.sparse import diags
            direct=spsolve(diags(m.volumes*(g0+2*H))+g1*op.stiffness,2*m.volumes*H)
            np.testing.assert_allclose(direct,2*H/(g0+2*H),rtol=1e-13)

    def test_cold_reference_and_pinn_share_shear_identity_and_reject_tension_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=shear_config(root)
            c.output.root=str(root/'reference');c.output.reference_root=''
            with contextlib.redirect_stdout(io.StringIO()):run_reference(c)
            reference=Path(c.output.root)
            c.output.root=str(root/'pinn');c.output.reference_root=str(reference)
            with contextlib.redirect_stdout(io.StringIO()):run_training(c)
            output=Path(c.output.root)
            with np.load(output/'state_0000/fields.npz') as z:
                for name in ('q','d','H'):np.testing.assert_array_equal(z[name],0.)
            report=json.loads((output/'result.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['current']['optimizer_count'],15)
            a=json.loads((output/'identity.json').read_text())
            b=json.loads((reference/'identity.json').read_text())
            self.assertEqual(a['geometry'],b['geometry'])
            c.output.resume=True
            original=(output/'state_0002/network.eqx').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):run_training(c)
            self.assertEqual(original,(output/'state_0002/network.eqx').read_bytes())
            b['geometry']['config'].pop('boundary')
            atomic_json(reference/'identity.json',b)
            with self.assertRaisesRegex(ValueError,'matched P1 direct reference'):
                with contextlib.redirect_stdout(io.StringIO()):run_training(c)


if __name__=='__main__':unittest.main()
