"""Independent GMLS first moments and exact Newtonian viscous divergence."""
import unittest
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative import get_config
from examples.channel_nsch2d.problem import quadrature
from docs.tasks.channel_gmls_stress_probe import gradient_stencils,gradient,viscous_divergence


class GMLSStress(unittest.TestCase):
    def test_polynomials_viscous_identity_and_smooth_refinement(self):
        c=get_config();nu=c.physical.viscosity
        for amplitudes in ((0.,0.),(.12,.08)):
            errors=[]
            for nx,ny in ((24,12),(48,24)):
                q=ConfigDict(c.to_dict());q.sample.Nx=nx;q.sample.Ny=ny
                points,_=quadrature(amplitudes,q,nx,ny);x,y=points.T
                ids,weights,report=gradient_stencils(points,q)
                values=np.stack((np.ones(len(x)),x,y,x**3,x*y*y),axis=-1)[None]
                exact=np.stack((np.zeros((len(x),2)),np.tile([1.,0.],(len(x),1)),np.tile([0.,1.],(len(x),1)),
                    np.stack((3*x*x,np.zeros(len(x))),axis=-1),np.stack((y*y,2*x*y),axis=-1)),axis=1)
                np.testing.assert_allclose(gradient(values,ids,weights)[0],exact,atol=2e-11,rtol=2e-12)
                tau=nu*np.stack((8*x*y,2*(x*x-y*y),-8*x*y),axis=-1)[None]
                np.testing.assert_allclose(viscous_divergence(tau,ids,weights)[0],nu*np.stack((4*y,-4*x),axis=-1),atol=2e-12,rtol=2e-12)
                k,l=1.3,2.7
                velocity=np.stack((l*np.sin(k*x)*np.cos(l*y),-k*np.cos(k*x)*np.sin(l*y)),axis=-1)
                normal=2*nu*k*l*np.cos(k*x)*np.cos(l*y)
                shear=nu*(k*k-l*l)*np.sin(k*x)*np.sin(l*y)
                tau=np.stack((normal,shear,-normal),axis=-1)[None]
                truth=-nu*(k*k+l*l)*velocity
                errors.append(np.linalg.norm(viscous_divergence(tau,ids,weights)[0]-truth)/np.linalg.norm(truth))
                self.assertLess(report['polynomial_error'],5e-9)
            self.assertLess(errors[1],errors[0]/4)
            self.assertLess(errors[1],.002)


if __name__=='__main__':unittest.main()
