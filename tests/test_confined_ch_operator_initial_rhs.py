"""Known initial RHS inputs: independent CH formula, masking and no labels."""
import json
from pathlib import Path
import tempfile
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_initial_rhs import get_config
from examples.confined_ch2d.operator_conditioning import initial_operator_fields
from examples.confined_ch2d.operator_problem import validate,fixed_initial,sample_geometries,condition,case_config,specification
from examples.confined_ch2d.operator_training import build_network,fields,make_loss,query_phase,load_bank
from examples.confined_ch2d.operators import RegionalLaplacian
from test_confined_ch_operator import small_fixture


class InitialRHSConditioningTests(unittest.TestCase):
    def test_known_columns_match_independent_initial_time_derivative(self):
        c=get_config();validate(c)
        initial=np.array([-.25,.1,.2]);v=np.array([.1,.2,.15])
        A=np.array([[5.,-2.,-3.],[-1.,3.,-2.],[-2.,-8/3,14/3]])
        data=dict(initial=initial,neighbors=np.array([[1,2],[0,2],[0,1]]),
            coefficients=np.array([[2.,3.],[1.,2.],[2.,8/3]]),phi=np.full((5,3),999.),mu=np.full((5,3),123.))
        p=c.physical
        def rhs(_,u):return -p.mobility*A@(p.beta*(u**3-u)+p.epsilon**2*A@u)
        result=initial_operator_fields(data,c)
        np.testing.assert_array_equal(result[:,0],initial)
        np.testing.assert_allclose(result[:,1],p.terminal_time*rhs(0,initial),rtol=1e-14,atol=1e-15)
        self.assertLess(abs(v@result[:,1]),1e-14)
        h=1e-6
        integration=solve_ivp(rhs,(0,h),initial,rtol=1e-12,atol=1e-14)
        self.assertTrue(integration.success)
        np.testing.assert_allclose((integration.y[:,-1]-initial)/h,result[:,1]/p.terminal_time,rtol=2e-6,atol=1e-10)
        data['phi'][:]=np.nan;data['mu'][:]=np.nan
        np.testing.assert_array_equal(initial_operator_fields(data,c),result)
        broken=dict(data,neighbors=np.array([[3,2],[0,2],[0,1]]))
        with self.assertRaisesRegex(ValueError,'aligned finite'):initial_operator_fields(broken,c)
        broken=ConfigDict(c.to_dict());del broken.operator['initial_rhs_condition']
        broken.operator.initial_rhs_condition=1
        with self.assertRaisesRegex(ValueError,'boolean'):validate(broken)
        broken=ConfigDict(c.to_dict());broken.arch.input_dims=(5,3)
        with self.assertRaisesRegex(ValueError,'input_dims'):validate(broken)
        broken=ConfigDict(c.to_dict());broken.operator.augmentation='axis_reflections'
        with self.assertRaisesRegex(ValueError,'unaugmented'):validate(broken)

    def test_known_input_preserves_hard_constraints_queries_padding_and_backprop(self):
        with jax.enable_x64(True):
            c=get_config();_,d=small_fixture();d=dict(d)
            c.arch.width=8;c.arch.heads=2;c.arch.condition_depth=1;c.arch.decoder_depth=1
            c.arch.spatial_features=2;c.arch.time_features=2
            d['initial']=jnp.where(d['volumes']>0,fixed_initial(d['points'],c,xp=jnp),0.)
            d['domain_fields']=jnp.asarray(initial_operator_fields(d,c))
            model=build_network(c,jax.random.key(7));times=jnp.array([0.,.3,2.4])
            value=fields(model,d,times,c)
            np.testing.assert_array_equal(value[0,:,0],d['initial'])
            np.testing.assert_allclose(value[:,:,0]@d['volumes'],d['initial']@d['volumes'],atol=1e-14,rtol=0)
            np.testing.assert_allclose(query_phase(model,d,d['points'][:3],times,c),value[:,:3,0],atol=1e-14,rtol=0)
            changed=dict(d,domain_fields=d['domain_fields'].at[-1].set(jnp.array([1e9,-1e9])))
            np.testing.assert_array_equal(fields(model,changed,times,c),value)
            batch=jax.tree.map(lambda a:a[None],d)
            loss,gradient=eqx.filter_jit(eqx.filter_value_and_grad(make_loss(c)))(model,batch,times)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(gradient) if eqx.is_array(a)))
            self.assertGreater(float(jnp.linalg.norm(gradient.lifts[1].layers[0].weight[:,-1])),0.)

    def test_loader_constructs_known_fields_for_unlabeled_inputs(self):
        from examples.confined_ch2d.configs.train_operator_heterogeneous import get_config as base
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            c=base();c.operator.initial_rhs_condition=True;c.arch.input_dims=(5,4)
            c.operator.train_conditions=2;c.operator.test_conditions=1;c.operator.local_validation_indices=(2,)
            c.operator.dataset_path=directory;root=Path(directory)
            x=np.array([[.46,.65],[.5,.65],[.54,.65]]);v=np.array([.1,.2,.15])
            holes=sample_geometries(c)[0];f,w=condition(holes,c)
            op=RegionalLaplacian(x,v,case_config(c,holes,0));initial=fixed_initial(x,c,xp=np)
            common=dict(points=x,volumes=v,holes=holes,condition=f,condition_weights=w,initial=initial,
                neighbors=np.asarray(op.neighbors),coefficients=np.asarray(op.coefficients),metric=np.eye(3))
            for i in range(3):
                labels={} if i==1 else dict(time=np.linspace(0,c.physical.terminal_time,c.reference.snapshots),
                    phi=np.broadcast_to(initial,(c.reference.snapshots,3)),mu=np.full((c.reference.snapshots,3),123.))
                np.savez(root/f'{i}.npz',**common,**labels)
            manifest=dict(format='operator_only_v1',status='passed',specification=specification(c),train_indices=[0,1],test_indices=[2],
                monitor_train_indices=[0],conditions=[dict(index=i,file=f'{i}.npz',holes=holes.tolist(),has_reference=i!=1) for i in range(3)])
            (root/'manifest.json').write_text(json.dumps(manifest))
            _,refs,bank=load_bank(c)
            np.testing.assert_array_equal(bank['domain_fields'][1],initial_operator_fields(common,c))
            self.assertNotIn('phi',refs[1]);self.assertNotIn('mu',refs[1])
            self.assertNotIn('phi',bank);self.assertNotIn('mu',bank)
            self.assertEqual(bank['domain_fields'].shape,(3,3,2))
            np.savez(root/'1.npz',**common,domain_fields=np.zeros((3,2)))
            with self.assertRaisesRegex(ValueError,'derived from known inputs'):load_bank(c)


if __name__=='__main__':unittest.main()
