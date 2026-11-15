"""Independent joint-feature, JVP and gradient checks for2D/3Dtime fusion."""
import io
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator
from examples.electropolish2d.temporal_query import TemporalQueryOperator
from examples.icemelting3d.operator_losses import latent_fields
from docs.tasks.melting_test64_reference import reference_error


def model(dim):
    return ConditionalFieldOperator(image_shape=(1,)+(4,)*dim, patch_shape=(2,)*dim,
        extent=(1.,)*dim, width=12, encoder_depth=1, heads=2, mlp_ratio=2,
        decoder_depth=2, spatial_features=3, spatial_scale=1., time_features=2,
        time_scale=1., local_dim=0, out_dim=1, conditioning='cross_attention',
        norm_eps=1e-5, key=jax.random.key(42))


class TimeFusionTests(unittest.TestCase):
    def test_joint_query_equals_single_concatenated_affine_map_and_time_jvp(self):
        with jax.enable_x64(True):
            for dim in (2,3):
                net = model(dim)
                joint = TemporalQueryOperator(net, 'joint_query', key=jax.random.key(14))
                image = jnp.linspace(-1,1,4**dim).reshape((1,)+(4,)*dim)
                points = jnp.arange(1., 1+3*dim).reshape(3,dim)/(3*dim+1)
                local = jnp.zeros((3,0)); times = jnp.array([.2,.7])
                normalized = points/jnp.asarray(net.extent)
                phase = jnp.pi*(normalized@net.spatial_frequencies)
                spatial = jnp.concatenate((normalized,jnp.sin(phase),jnp.cos(phase)), axis=-1)
                weight = jnp.concatenate((net.spatial_projection.weight, joint.time_projection.weight), axis=1)
                encoded = net.encode(image)
                expected = []
                for t in times:
                    phase_t = jnp.pi*t*jnp.asarray(net.time_frequencies)
                    tf = jnp.concatenate((jnp.array([t]),jnp.sin(phase_t),jnp.cos(phase_t)))
                    features = jnp.concatenate((spatial,jnp.broadcast_to(tf,(len(points),len(tf)))),axis=1)
                    q = features@weight.T+net.spatial_projection.bias
                    hidden = net.cross_attention(q, encoded)
                    for layer in net.layers:
                        hidden = jax.nn.gelu(jax.vmap(layer)(hidden))
                    expected.append(jax.vmap(net.output)(hidden))
                raw = joint(image,points,times,local)
                np.testing.assert_allclose(raw,expected,rtol=2e-12,atol=2e-12)
                prepared = joint.prepare_encoded(joint.encode(image),points,local)
                np.testing.assert_array_equal(joint.decode(prepared,times),raw)
                fn = lambda ts:joint(image,points,ts,local)
                _, tangent = jax.jvp(fn,(times,),(jnp.ones_like(times),))
                h = 1e-5; finite = (fn(times+h)-fn(times-h))/(2*h)
                np.testing.assert_allclose(tangent,finite,rtol=3e-6,atol=1e-8)
                jac = jax.jacfwd(fn)(times)
                np.testing.assert_allclose(jac[0,:,:,1],0,atol=1e-14)
                np.testing.assert_allclose(jac[1,:,:,0],0,atol=1e-14)
                self.assertIsNone(joint.backbone.time_film)
                a,b = io.BytesIO(),io.BytesIO()
                eqx.tree_serialise_leaves(a,eqx.tree_at(lambda n:n.time_film,net,None))
                eqx.tree_serialise_leaves(b,joint.backbone)
                self.assertEqual(a.getvalue(),b.getvalue())

    def test_joint_melting_hard_ic_and_differentiated_pde_gradient(self):
        with jax.enable_x64(True):
            net = TemporalQueryOperator(model(3),'joint_query',key=jax.random.key(3))
            image = jnp.linspace(-1,1,64).reshape(1,4,4,4)
            x = jnp.array([[.2,.3,.4],[.6,.7,.8]])
            initial = jnp.array([-.4,.8]);times=jnp.array([0.,.3,.8])
            u,ut = latent_fields(net,net.encode(image),x,initial,times)
            np.testing.assert_array_equal(u[0],initial)
            def objective(n):
                u,ut=latent_fields(n,n.encode(image),x,initial,times)
                return jnp.mean((ut+.2*jnp.tanh(u))**2)
            grad = eqx.filter_grad(objective)(net)
            direction = jnp.ones_like(net.time_projection.weight)/net.time_projection.weight.size**.5
            self.assertGreater(float(jnp.linalg.norm(grad.time_projection.weight)),1e-8)
            h=1e-5
            plus=eqx.tree_at(lambda n:n.time_projection.weight,net,net.time_projection.weight+h*direction)
            minus=eqx.tree_at(lambda n:n.time_projection.weight,net,net.time_projection.weight-h*direction)
            np.testing.assert_allclose(jnp.sum(grad.time_projection.weight*direction),
                (objective(plus)-objective(minus))/(2*h),rtol=2e-5,atol=1e-8)

    def test_melting_builder_keeps_zero_head_and_legacy_initial_arrays(self):
        from examples.icemelting3d.configs.train_operator_compare_nonlocal import get_config
        from examples.icemelting3d.operator_data import build_network,validate_config
        c=get_config();c.arch.image_shape=(1,4,4,4);c.arch.patch_shape=(2,2,2)
        c.arch.width=12;c.arch.heads=2;c.arch.encoder_depth=1;c.arch.spatial_features=3
        old=build_network(c,jax.random.key(42));c.arch.temporal_query='joint_query';validate_config(c)
        new=build_network(c,jax.random.key(42))
        for a,b in zip(jax.tree.leaves(eqx.tree_at(lambda n:n.time_film,old,None)),
                       jax.tree.leaves(new.backbone),strict=True):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(new.backbone.output.weight,0)
        c.operator.spatial_discretization='fem_p1_lumped'
        with self.assertRaisesRegex(ValueError,'time-fusion'):validate_config(c)

    def test_reference_gate_uses_entire_volume_before_query_selection(self):
        fine=np.full((5,3,3,3),.5);coarse=fine.copy();coarse[2,1,1,1]+=.001
        expected=np.linalg.norm((fine-coarse).reshape(5,-1),axis=1)/np.linalg.norm(fine.reshape(5,-1),axis=1)
        np.testing.assert_array_equal(reference_error(coarse,fine,.1),expected)
        with self.assertRaisesRegex(ValueError,'Time-refinement'):reference_error(coarse,fine,1e-5)
        invalid=fine.copy();invalid[1,1,1,1]=1.1
        with self.assertRaisesRegex(ValueError,'Invalid full-cube'):reference_error(coarse,invalid,.1)


if __name__=='__main__':unittest.main()
