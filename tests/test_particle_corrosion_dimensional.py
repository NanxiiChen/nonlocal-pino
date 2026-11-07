"""Independent SI equations, stale-cache rejection and legacy compatibility."""
import unittest

import numpy as np
from scipy import sparse

from examples.particle_corrosion2d.configs.train_base import get_config as legacy_config
from examples.particle_corrosion2d.configs.train_physical import get_config
from examples.particle_corrosion2d.dimensional import (
    derive_config, has_dimensional_inputs, require_matching_units, validate_dimensional)
from examples.particle_corrosion2d.problem import validate
from examples.particle_corrosion2d.solver import make_system


class ParticleDimensionalTests(unittest.TestCase):
    def test_material_groups_geometry_and_mobility_are_consistent(self):
        c = get_config()
        validate(c)
        np.testing.assert_allclose([c.physical.n_ch,c.physical.n_ac1,c.physical.n_ac2,c.physical.n_ac3],
                                   [1.7,1070.,176.,.103],rtol=1e-14)
        np.testing.assert_allclose([c.geometry.length,c.geometry.height,c.geometry.pit_radius],
                                   [1.,.6,.12],rtol=1e-14)
        self.assertAlmostEqual(c.geometry.interface_k,29.22959000805237)
        self.assertEqual(c.physical.kappa,1.)
        self.assertEqual(c.sample.Nt,32)

    def test_changed_physical_inputs_cannot_leave_stale_rates(self):
        c = get_config()
        before = [c.physical.n_ac1,c.physical.n_ac2,c.physical.n_ac3,c.physical.n_ch]
        width = c.geometry.interface_k
        c.nondim.T_c *= 2
        with self.assertRaisesRegex(ValueError,'Stale dimensional'):
            validate(c)
        derive_config(c);validate(c)
        np.testing.assert_allclose([c.physical.n_ac1,c.physical.n_ac2,c.physical.n_ac3,c.physical.n_ch],2*np.array(before))
        self.assertEqual(c.geometry.interface_k,width)
        c.physical.kappa=2
        with self.assertRaisesRegex(ValueError,'actual matrix mobility'):
            derive_config(c)
        c=get_config();c.geometry.interface_k=30
        with self.assertRaisesRegex(ValueError,'Stale dimensional geometry'):
            validate_dimensional(c)

    def test_reference_rhs_matches_independently_dimensional_equations(self):
        c=get_config();p=c.physical
        matrix=np.array([[1.3,-1.,-.3],[-1.,1.4,-.4],[-.3,-.4,.7]])
        points=np.array([[.5,0],[.5,.1],[.6,.2]])
        mouth=np.array([True,False,False])
        rhs,_,_,_=make_system(c,sparse.csr_matrix(matrix),points,mouth)
        state=np.array([.25,.8,.2,.7])
        phi=np.array([0.,.25,.8]);conc=np.array([0.,.2,.7])
        q=conc-(p.c_se-p.c_le)*phi**2*(3-2*phi)-p.c_le
        physical_laplacian=-matrix/c.nondim.L_c**2
        phi_rate=p.L*(p.alpha_phi*(physical_laplacian@phi)
            -p.omega_phi*(2*phi*(1-phi)*(1-2*phi))
            +2*p.A*(p.c_se-p.c_le)*q*(6*phi*(1-phi)))
        conc_rate=p.D*(physical_laplacian@q)
        expected=c.nondim.T_c*np.r_[phi_rate[~mouth],conc_rate[~mouth]]
        np.testing.assert_allclose(rhs(.2,state),expected,rtol=1e-13,atol=1e-13)

    def test_units_and_legacy_metadata_cannot_be_silently_mixed(self):
        legacy=legacy_config();validate(legacy)
        self.assertFalse(has_dimensional_inputs(legacy))
        require_matching_units(legacy.to_dict(),legacy)
        c=get_config();require_matching_units(c.to_dict(),c)
        with self.assertRaisesRegex(ValueError,'parameterization differs'):
            require_matching_units(legacy.to_dict(),c)
        other=get_config();other.nondim.T_c=2;derive_config(other)
        with self.assertRaisesRegex(ValueError,'dimensional nondim'):
            require_matching_units(other.to_dict(),c)
        legacy.physical.D=8.5e-10
        with self.assertRaisesRegex(ValueError,'requires all SI fields'):
            validate_dimensional(legacy)

    def test_pino_residual_matches_si_oracle_without_extra_time_factor(self):
        import jax
        import jax.numpy as jnp
        from examples.particle_corrosion2d.losses import residuals
        c=get_config();p=c.physical;lc=c.nondim.L_c;tc=c.nondim.T_c
        edges=np.array([[0.,1.3,.2],[1.3,0.,.4],[.2,.4,0.]])
        matrix=np.diag(edges.sum(axis=1))-edges
        rng=np.random.default_rng(203)
        raw=rng.normal(size=(4,3,2));dt=rng.normal(size=(4,3,2))
        phi=.5*(1+np.tanh(raw[...,0]));q=p.c_le*(raw[...,1]-1)
        hp=6*phi*(1-phi);gp=2*phi*(1-phi)*(1-2*phi)
        ac_si=dt[...,0]/tc+p.L*(p.alpha_phi*(matrix@phi.T).T/lc**2
            +p.omega_phi*gp-2*p.A*(p.c_se-p.c_le)*q*hp)
        ch_si=dt[...,1]/tc+p.D*(matrix@q.T).T/lc**2
        with jax.enable_x64():
            payload=dict(core_indices=jnp.arange(3),
                neighbors=jnp.asarray(np.tile(np.arange(3),(3,1))),coefficients=jnp.asarray(edges))
            ac,ch=residuals(jnp.asarray(raw),jnp.asarray(dt),payload,c)
            np.testing.assert_allclose(ac,tc*ac_si.T,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(ch,tc*ch_si.T,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
