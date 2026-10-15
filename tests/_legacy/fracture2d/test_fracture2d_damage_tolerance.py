"""Range tolerance preserves the FEM law and same-load accepted ancestry."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.fracture2d._legacy.damage_range_tolerance import TolerantLocalMechanics
from examples.fracture2d._legacy.local_mechanics import LocalMechanics, QUADRATURE
from examples.fracture2d._legacy.local_damage_trial import sha
from examples.fracture2d._legacy.linear_damage_continue import recovery_fields


def mechanics():
    return TolerantLocalMechanics(
        np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]],np.float64),
        np.array([[0,1,2],[1,3,2]],np.int64),4,2,
        np.array([1/6,1/3,1/3,1/6],np.float64),E=10.,nu=.3,k_res=1e-8,
        constrained=np.array([0,1,2],np.int64),load_direction=np.zeros(8,np.float64),
        damage_range_atol=2e-6)


class DamageToleranceTest(unittest.TestCase):
    def test_unchanged_quadrature_and_explicit_bounds(self):
        m = mechanics()
        d = np.array([0.,.2,.5,1.],np.float64)
        np.testing.assert_array_equal(m.factors(d),LocalMechanics.factors(m,d))
        d[0] = -1.2832056783e-6
        expected = np.mean((1-d[m.triangles]@QUADRATURE.T)**2,axis=1)+m.k_res
        np.testing.assert_array_equal(m.factors(d),expected)
        self.assertTrue(np.isfinite(m.matrix(d).data).all())
        with self.assertRaises(ValueError):
            LocalMechanics.factors(m,d)
        for value in (-2.01e-6,1+2.01e-6,np.nan):
            invalid = d.copy(); invalid[0] = value
            with self.assertRaises(ValueError):
                m.factors(invalid)

    def test_same_load_recovery_rejects_promotion_and_tampering(self):
        m = mechanics()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent,failed = root/'state_0052',root/'state_0053'
            parent.mkdir();failed.mkdir()
            q,H = np.zeros(8,np.float64),np.zeros(4,np.float64)
            dp = np.zeros(4,np.float64)
            d = np.array([-1.28e-6,.01,.02,.03],np.float64)
            np.savez(parent/'fields.npz',q=q,d=dp,H=H)
            np.savez(failed/'pending_0122_fields.npz',q=q,d=d,H=H)
            np.savez(failed/'fields.npz',q=q,d=d,H=H,H_used=H,d_parent=dp,H_parent=H)
            for p in (parent/'network.eqx',failed/'network.eqx',failed/'pending_0122_network.eqx'):
                p.write_bytes(b'checksum-fixture')
            parent_sha = sha(parent/'fields.npz')
            def write(p,x):
                p.write_text(json.dumps(x))
            pr = dict(status='ready',physical_state_promoted=True,
                outer_convergence=dict(converged=True),state_sha256=parent_sha,
                network_sha256=sha(parent/'network.eqx'))
            fr = dict(status='inner_rejected',physical_state_promoted=False,inner_calls=123,
                accepted_inner_sweeps=122,parent_sha256=parent_sha,
                state_sha256=sha(failed/'fields.npz'),network_sha256=sha(failed/'network.eqx'))
            meta = dict(completed_sweep=122,index=53,load=.0053,has_image=True,
                parent_sha256=parent_sha,fields_file='pending_0122_fields.npz',
                network_file='pending_0122_network.eqx',fields_sha256=sha(failed/'pending_0122_fields.npz'),
                network_sha256=sha(failed/'pending_0122_network.eqx'),reaction=0.,
                mechanical=dict(convergence_limit=1e-10))
            write(parent/'result.json',pr);write(failed/'result.json',fr);write(failed/'pending.json',meta)
            parents,pending,recovery = recovery_fields(root,m,parent_index=52,completed_outer=122,load=.0053)
            np.testing.assert_array_equal(parents['d'],dp)
            np.testing.assert_array_equal(pending['d'],d)
            self.assertEqual(recovery['completed_sweep'],122)
            write(failed/'result.json',{**fr,'physical_state_promoted':True})
            with self.assertRaisesRegex(ValueError,'ancestry'):
                recovery_fields(root,m,parent_index=52,completed_outer=122,load=.0053)
            write(failed/'result.json',fr)
            (failed/'pending_0122_fields.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):
                recovery_fields(root,m,parent_index=52,completed_outer=122,load=.0053)
