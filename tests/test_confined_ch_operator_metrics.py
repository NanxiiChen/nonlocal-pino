"""Independent spectra/Jacobian checks for the input-only CH residual norm."""
import unittest
import numpy as np
from examples.confined_ch2d.operator_metrics import linear_stable_resolvent


class OperatorMetricTests(unittest.TestCase):
    def test_closed_form_complete_graph_stable_and_growing_modes(self):
        v=np.ones(3);neighbors=np.array([[1,2],[0,2],[0,1]])
        coefficients=np.ones((3,2));initial=np.array([.1,-.1,0.])
        identity=np.eye(3);constant=np.ones((3,3))/3
        growing,report=linear_stable_resolvent(v,neighbors,coefficients,initial,1.,.2,.5,2.)
        np.testing.assert_allclose(growing,identity,atol=1e-15)
        self.assertEqual(report['growing_modes'],2)
        stable,report=linear_stable_resolvent(v,neighbors,coefficients,initial,1.,1.,.5,2.)
        np.testing.assert_allclose(stable,constant+(identity-constant)/7,atol=1e-15)
        self.assertEqual(report['growing_modes'],0)

    def test_nonuniform_volume_jacobian_and_weighted_symmetry(self):
        v=np.array([.1,.2,.15]);neighbors=np.array([[1,2],[0,2],[0,1]])
        conductance=np.array([[.2,.3],[.2,.4],[.3,.4]])
        coefficients=conductance/v[:,None];initial=np.full(3,.1)
        beta,eps,M,T=1.,.4,.5,2.
        P,_=linear_stable_resolvent(v,neighbors,coefficients,initial,beta,eps,M,T)
        A=np.array([[.5,-.2,-.3],[-.2,.6,-.4],[-.3,-.4,.7]])/v[:,None]
        root=np.sqrt(v);S=root[:,None]*A/root[None,:]
        eigenvalues,U=np.linalg.eigh(S)
        rhs=lambda f:-M*A@(beta*(f**3-f)+eps**2*A@f)
        for value,u in zip(eigenvalues,U.T,strict=True):
            mode=u/root;delta=1e-6
            derivative=(rhs(initial+delta*mode)-rhs(initial-delta*mode))/(2*delta)
            rate=M*value*(eps**2*value+beta*(3*.1**2-1))
            np.testing.assert_allclose(derivative,-rate*mode,atol=2e-9,rtol=1e-8)
            np.testing.assert_allclose(P@mode,mode/(1+T*max(rate,0)),atol=1e-14)
        weighted=root[:,None]*P/root[None,:]
        np.testing.assert_allclose(weighted,weighted.T,atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(weighted).min(),0.)
        np.testing.assert_allclose(P@np.ones(3),1.,atol=1e-14)

    def test_invalid_non_self_adjoint_graph_and_physical_scales(self):
        v=np.ones(3);neighbors=np.array([[1,2],[0,2],[0,1]])
        co=np.ones((3,2));initial=np.zeros(3)
        with self.assertRaises(ValueError):
            linear_stable_resolvent(v,neighbors,co*np.array([[2.,1.],[1.,1.],[1.,1.]]),initial,1.,.2,.5,2.)
        with self.assertRaises(ValueError):linear_stable_resolvent(v,neighbors,co,initial,1.,.2,.5,0.)


if __name__=='__main__':unittest.main()
