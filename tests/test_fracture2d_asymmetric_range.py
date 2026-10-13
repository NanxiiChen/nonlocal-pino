"""Relaxing negative tolerance must not relax upper bounds or physical gates."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.asymmetric_damage_range import (
    AsymmetricLocalMechanics,make_asymmetric_lbfgs,evaluate_asymmetric,
)
from examples.fracture2d.compact_damage_features import Readout
from examples.fracture2d.configs.linear_damage_relaxed_continue import get_config
from examples.fracture2d.configs.linear_damage_continue import get_config as original
from examples.fracture2d.linear_damage_relaxed_continue import validate_recovery_configuration
from examples.fracture2d.local_mechanics import QUADRATURE
from tests.test_fracture2d_damage_tolerance import mechanics


class AsymmetricRangeTest(unittest.TestCase):
    def test_signed_fem_and_distinct_bounds(self):
        m=mechanics()
        a=AsymmetricLocalMechanics(m.coordinates,m.triangles,m.n_material,m.n_material_cells,m.volumes,
            E=10.,nu=.3,k_res=1e-8,constrained=m.constrained,load_direction=m.load_direction,
            negative_tolerance=.001,positive_tolerance=2e-6)
        d=np.array([-.001,.2,.5,1+2e-6],np.float64)
        expected=np.mean((1-d[a.triangles]@QUADRATURE.T)**2,axis=1)+a.k_res
        np.testing.assert_array_equal(a.factors(d),expected)
        for bad in (-.001001,1+2.01e-6,np.nan):
            invalid=d.copy();invalid[0]=bad
            with self.assertRaises(ValueError):a.factors(invalid)

    def test_inner_early_stop_keeps_upper_constraint(self):
        jax.config.update('jax_enable_x64',True)
        net=Readout(jnp.array([-.0001,.8],jnp.float64))
        def objective(n,H,load):return jnp.mean((n.coefficients-jnp.array([0.,.8]))**2)
        def metrics(n,H,load):return dict(AT2_relative=jnp.asarray(.0005),
            damage_min=n.coefficients.min(),damage_max=n.coefficients.max())
        fit,_,_=make_asymmetric_lbfgs(net,objective,metrics)
        controls=dict(maxiter=8,check_every=1,memory=10,max_linesearch=20,gtol=1e-12,ftol=0.,
            residual_limit=.001,negative_tolerance=.001,positive_tolerance=2e-6)
        _,r=fit(net,jnp.zeros(2),jnp.asarray(.0053),controls,progress=lambda r:None)
        self.assertEqual(r['iterations'],0);self.assertEqual(r['status'],'residual_reached')
        invalid=Readout(jnp.array([0.,1+.0001],jnp.float64))
        _,r=fit(invalid,jnp.zeros(2),jnp.asarray(.0053),controls,progress=lambda r:None)
        self.assertFalse(r['rows'][0]['ready'])
        self.assertGreater(r['iterations'],0)
        self.assertEqual(r['status'],'residual_reached')

    def test_outer_checks_remain_physical_and_consecutive(self):
        c=get_config()
        report=dict(damage_min=-.0001,damage_max=.9,fixed_H_image_damage_min=-.0001,
            fixed_H_image_damage_max=.9,fixed_H_image_AT2_relative=.0005,coupled_AT2_relative=.0005,
            damage_change=1e-6,q_change_over_load=1e-5,H_change_relative=1e-5,
            reaction_change_relative=1e-5,healing=.0001,drift_count=0,history_decrease=0.,
            mechanics=dict(converged=True,free_residual_norm=1e-9,convergence_limit=1e-8))
        def check(r):return evaluate_asymmetric(r,c.outer_stop,c.reporting,sweep=3,
            previous_consecutive=2,negative_tolerance=.001,positive_tolerance=2e-6)
        self.assertTrue(check(report)['converged'])
        for name,value in [('damage_min',-.001001),('damage_max',1+.0001),
                           ('coupled_AT2_relative',.002),('healing',.002),('drift_count',1)]:
            r=check({**report,name:value});self.assertFalse(r['converged']);self.assertEqual(r['consecutive'],0)

    def test_configuration_changes_only_declared_budget_and_range(self):
        c=get_config();identity=dict(config=original().to_dict())
        validate_recovery_configuration(c,identity)
        c.reporting.coupled_AT2_relative_limit=.01
        with self.assertRaisesRegex(ValueError,'differs beyond'):
            validate_recovery_configuration(c,identity)
