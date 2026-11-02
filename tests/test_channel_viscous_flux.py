"""Mapped viscous-flux identity, GCL and curved-grid consistency."""
import unittest
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_conservative import get_config
from examples.channel_nsch2d.problem import quadrature
from examples.channel_nsch2d.transport import build_transport
from docs.tasks.channel_viscous_flux_probe import mapped_flux_divergence


class ViscousFlux(unittest.TestCase):
    def test_constant_flux_and_manufactured_flat_curved_stress(self):
        c=get_config();nu=c.physical.viscosity
        for a in ((0.,0.),(.12,.08)):
            errors=[]
            for nx,ny in ((48,24),(96,48)):
                q=ConfigDict(c.to_dict());q.sample.Nx=nx;q.sample.Ny=ny
                points,_=quadrature(a,q,nx,ny);data=dict(points=points,**build_transport(points,a,q))
                x,y=points.T;one=np.ones((1,len(x),1))
                constant=mapped_flux_divergence(2*one,-.7*one,data,xp=np)
                np.testing.assert_allclose(constant,0.,atol=8e-12,rtol=0)
                # psi=x^2*y^2, u=(2x^2*y,-2x*y^2), exact div(u)=0.
                tau_xx=8*nu*x*y;tau_xy=2*nu*(x*x-y*y)
                result=mapped_flux_divergence(np.stack((tau_xx,tau_xy),axis=-1)[None],
                    np.stack((tau_xy,-tau_xx),axis=-1)[None],data,xp=np)[0]
                exact=nu*np.stack((4*y,-4*x),axis=-1)
                errors.append(np.linalg.norm(result-exact)/np.linalg.norm(exact))
                if a==(0.,0.):np.testing.assert_allclose(result,exact,atol=5e-12,rtol=1e-12)
            if a!=(0.,0.):
                self.assertLess(errors[1],errors[0]/4)
                self.assertLess(errors[1],.003)


if __name__=='__main__':unittest.main()
