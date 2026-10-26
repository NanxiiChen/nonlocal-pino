"""Focused algebra/gradient/reference checks for coupled residual experiments."""

import unittest
import json
from pathlib import Path
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.corrosion2d import losses as corrosion
from examples.electropolish2d import losses as electropolish
from examples.corrosion2d.configs.residual_pilot import get_config
from examples.corrosion2d.residual_experiment import boundary_error, boundary_values, grid, graph_matrix, load_reference, make_step, reference_spec, stagger_phase, training_initial_data, training_boundary_data, initial_fields
from examples.corrosion2d.residuals import bounded_phase_weight, compute_pde, make_loss_vector, make_normalized_data_loss_vector, normalized_data_fields, phase_only_constraint, tanh_edge_ratio
from examples.corrosion2d.residuals import compute_transport_pde, graph_edge_action, make_transport_loss_vector, masked_ch_loss
from examples.corrosion2d.residual_experiment import make_training_losses, validate_loss_schedule
from nonlocal_pinn.nn.constraints import ConstrainedForward, kks_constraint
from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.optimizers import make_optimizer


class Raw(eqx.Module):
    weights: jax.Array

    def __call__(self, x, t):
        return self.weights @ jnp.array([1.0, x[0], x[1], t, t * x[0]])


class KKSResidualTests(unittest.TestCase):
    def test_late_rms_losses_preserve_other_components_and_soap_state(self):
        c=get_config('corrosion2d','phase',42);c.physical.c_le=.035
        c.train.iterations=6;c.train.gradnorm_update_every=1
        c.experiment.ch_rms_after=3;c.experiment.transport_evaluation='kks_raw_edges'
        c.experiment.data_loss_mode='s_bc'
        args=(c,self.nd,self.x,jnp.full((len(self.x),2),.25),self.x[:2],self.x[-2:])
        initial,final=make_training_losses(*args)
        mask=jnp.ones(len(self.x),dtype=bool)
        a=initial(self.net,self.times,self.x,self.sub,mask)
        b=final(self.net,self.times,self.x,self.sub,mask)
        np.testing.assert_array_equal(a[jnp.array([0,2,3,4])],b[jnp.array([0,2,3,4])])
        self.assertGreater(float(b[1]),float(a[1]))
        optimizer=make_optimizer(c);state=optimizer.init(eqx.filter(self.net,eqx.is_array))
        steps=[make_step(loss,optimizer,c) for loss in (initial,final)]
        net=self.net;weights=jnp.ones(5)
        for i in range(4):
            net,state,weights,values,finite=steps[int(i>=3)](net,state,weights,self.times,
                    self.x,self.sub,mask,jnp.array(True),jnp.ones(5,dtype=bool))
            self.assertTrue(bool(finite));self.assertTrue(np.all(np.asarray(weights)>0))
        counts=[int(x) for x in jax.tree.leaves(state) if hasattr(x,'dtype') and x.ndim==0 and jnp.issubdtype(x.dtype,jnp.integer)]
        self.assertIn(4,counts)  # the RMS step continues the existing schedule/state
        c.experiment.ch_rms_after=0
        unchanged,same=make_training_losses(*args);self.assertIs(unchanged,same)
        for switch,norm in ((True,'l1'),(-1,'l1'),(6,'l1'),(3,'rms')):
            with c.ignore_type():c.experiment.ch_rms_after=switch
            c.experiment.ch_norm=norm
            with self.assertRaises(ValueError):validate_loss_schedule(c)

    def test_raw_kks_edge_transport_matches_independent_double_precision_graph(self):
        # Float64 independent replay checks the same model, graph and physical c_t.
        with jax.enable_x64():
            x=self.x.astype(jnp.float64);times=self.times.astype(jnp.float64)
            net=eqx.tree_at(lambda n:n.net.weights,self.net,self.net.net.weights.astype(jnp.float64))
            sub=tuple(a.astype(jnp.float64) if jnp.issubdtype(a.dtype,jnp.floating) else a for a in self.sub)
            f=lambda ts:jax.vmap(lambda t:jax.vmap(net,in_axes=(0,None))(x,t))(ts)
            fields,deriv=jax.jvp(f,(times,),(jnp.ones_like(times),))
            p=np.asarray(fields[...,0]);c=np.asarray(fields[...,1])
            q=c-self.nd[-2]*(3*p*p-2*p*p*p)-self.nd[-1]
            expected=np.asarray(deriv[...,1]).T+self.nd[0]*(graph_matrix(self.op,len(x))@q.T)
            legacy,_=compute_pde(net,x,times,sub,self.nd,'phase',.5)
            for form in ('phase','latent','bounded'):
                ac,ch=compute_transport_pde(net,x,times,sub,self.nd,form,.5,
                            transport_evaluation='kks_raw_edges',bounded_eta=.1)
                old_ac,_=compute_pde(net,x,times,sub,self.nd,form,.5,bounded_eta=.1)
                np.testing.assert_array_equal(ac,old_ac)
                # Stored graph diag is FP32; compare direct edge accumulation in FP64.
                row,col,vals,_=map(np.asarray,sub)
                independent=np.zeros_like(q.T)
                np.add.at(independent,row,-vals[:,None]*(q.T[row]-q.T[col]))
                np.testing.assert_allclose(ch,np.asarray(deriv[...,1]).T+self.nd[0]*independent,
                                           rtol=1e-11,atol=2e-13)
                np.testing.assert_allclose(ch,expected,rtol=2e-4,atol=3e-6)
            np.testing.assert_array_equal(graph_edge_action(jnp.full((len(x),2),-.035),sub),0.)

    def test_ch_rms_mask_outlier_gradient_and_zero_are_correct(self):
        mask=jnp.array([[1.],[1.],[0.]])
        values=jnp.array([[1.,2.],[3.,20.],[1e6,1e6]])
        count=4
        loss=lambda v:masked_ch_loss(v,mask,count,norm='rms',rms_floor=1e-12)
        expected=np.sqrt((1+4+9+400)/4)
        np.testing.assert_allclose(loss(values),expected,rtol=1e-6)
        grad=np.asarray(jax.grad(loss)(values))
        np.testing.assert_allclose(grad[:2],np.asarray(values[:2])/(count*expected),rtol=1e-6)
        np.testing.assert_array_equal(grad[2],0.)
        self.assertAlmostEqual(float(grad[1,1]/grad[0,0]),20.,places=5)
        self.assertEqual(float(loss(jnp.zeros_like(values))),0.)
        np.testing.assert_array_equal(jax.grad(loss)(jnp.zeros_like(values)),0.)

    def test_transport_factory_legacy_identity_gradient_and_invalid_settings(self):
        c=get_config('corrosion2d','phase',42);c.physical.c_le=.035
        c.experiment.data_loss_mode='s_bc'
        args=(c,self.nd,self.x,jnp.full((len(self.x),2),.25),self.x[:2],self.x[-2:])
        old=make_normalized_data_loss_vector(*args,normalize_initial=False)
        fresh=make_transport_loss_vector(*args)
        mask=jnp.ones(len(self.x),dtype=bool)
        call=lambda f,n:f(n,self.times,self.x,self.sub,mask)
        np.testing.assert_array_equal(call(old,self.net),call(fresh,self.net))
        for field,value,message in (('transport_evaluation','invalid','Unknown transport'),
                                    ('ch_norm','invalid','Unknown CH norm'),
                                    ('ch_rms_floor',0.,'finite and positive')):
            original=c.experiment[field];c.experiment[field]=value
            with self.assertRaisesRegex(ValueError,message):make_transport_loss_vector(*args)
            c.experiment[field]=original
        c.experiment.transport_evaluation='kks_raw_edges';c.experiment.ch_norm='rms'
        fresh=make_transport_loss_vector(*args)
        actual=eqx.filter_jit(lambda n:call(fresh,n))(self.net)
        np.testing.assert_allclose(actual[jnp.array([0,2,3,4])],call(old,self.net)[jnp.array([0,2,3,4])],
                                   rtol=1e-6,atol=1e-7)
        g=eqx.filter_grad(lambda n:call(fresh,n)[1])(self.net).net.weights
        direction=jnp.zeros_like(g).at[1,2].set(1.)
        eps=1e-3
        plus=eqx.tree_at(lambda n:n.net.weights,self.net,self.net.net.weights+eps*direction)
        minus=eqx.tree_at(lambda n:n.net.weights,self.net,self.net.net.weights-eps*direction)
        np.testing.assert_allclose(g[1,2],(call(fresh,plus)[1]-call(fresh,minus)[1])/(2*eps),rtol=.01,atol=1e-4)
        c.arch.use_hard_constraint=False
        with self.assertRaisesRegex(ValueError,'KKS output'):make_transport_loss_vector(*args)

    def test_reference_rejects_parameter_time_coordinate_and_boundary_mismatches(self):
        c = get_config("corrosion2d", "phase", 42)
        c.sample.Nx, c.sample.Ny, c.nl.delta = 7, 5, 0.3
        c.experiment.reference_snapshots = 3
        xx, yy, points = grid(c)
        zero, one, _ = boundary_values(c, points)
        fields = np.full((3, len(points), 2), 0.5)
        fields[:, zero] = 0
        fields[:, one] = 1
        shape = (3, *xx.shape)
        arrays = dict(X=xx, Y=yy, time=np.array([0.0, 0.5, 1.0]),
            physical_time=np.array([0.0, 0.5, 1.0]) * c.nondim.T_c,
            phi=fields[..., 0].reshape(shape), c=fields[..., 1].reshape(shape),
            graph_row=np.asarray(self.op._row_idx), graph_col=np.asarray(self.op._col_idx),
            graph_weight=-np.asarray(self.op._vals),
            metadata_json=np.array(json.dumps({"spec": reference_spec(c), "refinement_max_relative_l2": 1e-9})))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            c.experiment.reference_path = str(path)
            np.savez(path, **arrays)
            load_reference(c)
            c.physical.L *= 2
            with self.assertRaisesRegex(ValueError, "model/BC/grid mismatch"):
                load_reference(c)
            c.physical.L /= 2
            for key, value, message in (
                ("time", np.array([0.0, 0.5, 0.5]), "snapshots must be ordered"),
                ("X", xx[:, ::-1], "coordinate values/order"),
                ("phi", np.full(shape, 0.5), "Dirichlet field values"),
            ):
                invalid = dict(arrays)
                invalid[key] = value
                np.savez(path, **invalid)
                with self.assertRaisesRegex(ValueError, message):
                    load_reference(c)

    def test_boundary_metric_uses_graph_mask_on_spatial_grid(self):
        zero = np.array([True, False, False, False, False, False])
        one = np.array([False, False, False, False, False, True])
        prediction = np.full((2, 2, 3, 2), 0.4)
        prediction[:, 0, 0] = 0
        prediction[:, 1, 2] = 1
        self.assertEqual(boundary_error(prediction, zero, one), 0.0)
        prediction[1, 1, 2, 1] = 0.8
        self.assertAlmostEqual(boundary_error(prediction, zero, one), 0.2)

    def setUp(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 0.5, 5))
        self.x = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=-1))
        self.op = build_D(self.x, 0.3, method="sparse", kernel="cubic")
        self.sub = (self.op._row_idx, self.op._col_idx, self.op._vals, self.op._diag)
        self.net = ConstrainedForward(
            Raw(jnp.array([[0.3, -0.7, 0.5, 0.2, -0.1], [-0.2, 0.1, 0.5, -0.4, 0.3]])),
            lambda r: kks_constraint(r, 1.0, 0.035),
        )
        self.times = jnp.array([0.2, 0.7])
        self.nd = (0.8, 1.2, 0.9, 0.3, 0.965, 0.035)

    def test_production_residual_equivalence_and_chain_rule(self):
        phase, conc = compute_pde(self.net, self.x, self.times, self.sub, self.nd, "phase", 0.5)
        latent, unchanged_conc = compute_pde(self.net, self.x, self.times, self.sub, self.nd, "latent", 0.5)
        fields = jax.vmap(lambda t: jax.vmap(self.net, in_axes=(0, None))(self.x, t))(self.times)
        phi = fields[..., 0].T
        np.testing.assert_allclose(phase, 4 * phi * (1 - phi) * latent, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(conc, unchanged_conc)
        for module in (corrosion, electropolish):
            a, b, _ = module._compute_pde(self.net, self.x, self.times, self.op, self.nd)
            np.testing.assert_allclose(a, phase, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(b, conc, rtol=2e-5, atol=2e-5)

    def test_latent_gradient_matches_finite_difference_and_saturation_is_finite(self):
        def value(net):
            a, b = compute_pde(net, self.x, self.times, self.sub, self.nd, "latent", 0.5)
            return jnp.mean(a**2 + b**2)
        grad = eqx.filter_grad(value)(self.net)
        direction = jnp.zeros_like(self.net.net.weights).at[0, 3].set(1.0)
        eps = 1e-3
        plus = eqx.tree_at(lambda n: n.net.weights, self.net, self.net.net.weights + eps * direction)
        minus = eqx.tree_at(lambda n: n.net.weights, self.net, self.net.net.weights - eps * direction)
        finite_difference = (value(plus) - value(minus)) / (2 * eps)
        np.testing.assert_allclose(grad.net.weights[0, 3], finite_difference, rtol=0.02, atol=0.002)
        for offset in (-40.0, 40.0):
            weights = self.net.net.weights.at[0, 0].set(offset)
            saturated = eqx.tree_at(lambda n: n.net.weights, self.net, weights)
            self.assertTrue(np.isfinite(float(value(saturated))))
            self.assertTrue(np.all(np.isfinite(np.asarray(eqx.filter_grad(value)(saturated).net.weights))))
        np.testing.assert_allclose(tanh_edge_ratio(jnp.array(40.), jnp.array(39.8)), 0.5 * (np.exp(0.4) - 1), rtol=2e-5)

    def test_direct_concentration_map_preserves_physical_residual_and_chain_rule(self):
        net = ConstrainedForward(self.net.net, phase_only_constraint)
        fields = jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(self.x, t))(self.times)
        phi = fields[..., 0].T
        phase, ch = compute_pde(net, self.x, self.times, self.sub, self.nd, "phase", 0.5)
        latent, same_ch = compute_pde(net, self.x, self.times, self.sub, self.nd, "latent", 0.5)
        actual, actual_ch, _ = corrosion._compute_pde(net, self.x, self.times, self.op, self.nd)
        np.testing.assert_allclose(phase, actual, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(ch, actual_ch, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(ch, same_ch)
        np.testing.assert_allclose(phase, 4 * phi * (1 - phi) * latent, rtol=2e-5, atol=2e-5)
        raw = jnp.array([40., -20.])
        self.assertEqual(float(phase_only_constraint(raw)[1]), -20.)
        jac = jax.jacfwd(phase_only_constraint)(raw)
        self.assertEqual(float(jac[1, 1]), 1.)
        grad = eqx.filter_grad(lambda n: jnp.mean(compute_pde(n, self.x, self.times,
            self.sub, self.nd, "latent", 0.5)[0]**2))(net)
        self.assertTrue(all(np.all(np.isfinite(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)))

    def test_compiled_soap_steps_with_and_without_weight_refresh(self):
        config = get_config("corrosion2d", "latent", 42)
        config.loss.ac_scale = 1.0
        loss = make_loss_vector(config, self.nd, self.x[:3], jnp.ones((3, 2)) * 0.25, self.x[:2], self.x[-2:])
        optimizer = make_optimizer(config)
        state = optimizer.init(eqx.filter(self.net, eqx.is_array))
        step = make_step(loss, optimizer, config)
        net, weights = self.net, jnp.ones(5)
        for refresh, active in ((True, [True,False,True,True,True]),
                                (False, [False,True,True,True,True]),
                                (False, [True,True,True,True,True])):
            net, state, weights, values, finite = step(net, state, weights, self.times,
                self.x, self.sub, jnp.ones(len(self.x), dtype=bool), jnp.asarray(refresh), jnp.asarray(active))
            self.assertTrue(bool(finite))
            self.assertTrue(np.all(np.asarray(weights) > 0))
            self.assertTrue(np.all(np.isfinite(np.asarray(values))))

    def test_production_stagger_weights_survive_switch_without_refresh_or_repeated_bias(self):
        loss = corrosion.make_loss_fn(self.x, self.op, self.x[:2], self.x[-2:], self.nd,
                    x_ic=self.x, alpha=.9, ac_scale=1., ch_scale=1., gradnorm_cap=10000, causal=None)
        initial = jnp.full((len(self.x),), .25)
        bias = jnp.array([2.,3.,1.,1.,1.])
        def call(state, ac, ch, refresh):
            return loss(self.net, self.times, initial, initial, jnp.array(1.),
                        jnp.array(ac), jnp.array(ch), state, bias, refresh,
                        jnp.array(0.), x_points=self.x)
        _, ac_aux, _, stored = call(jnp.ones(5), True, False, True)
        self.assertTrue(np.all(np.asarray(stored)>0))
        self.assertEqual(float(ac_aux['weight/ch']),0.)
        for ac, ch in ((False,True),(True,False),(False,True),(True,True)):
            _, aux, _, returned = call(stored,ac,ch,False)
            np.testing.assert_array_equal(returned,stored)
            np.testing.assert_allclose(aux['weight/ac'],stored[0]*bias[0] if ac else 0.)
            np.testing.assert_allclose(aux['weight/ch'],stored[1]*bias[1] if ch else 0.)
            self.assertGreater(float(aux['weight/bc_pit']),0.)
            self.assertGreater(float(aux['weight/ic']),0.)

    def test_research_stagger_gradient_mask_and_schedule(self):
        config=get_config('corrosion2d','phase',42)
        config.loss.gradnorm_bias=(2.,3.,1.,1.,1.)
        def components(net,times,x,sub,mask):
            w=net.net.weights
            return jnp.stack([jnp.sum(w[0]**2),jnp.sum(w[1]**2),
                              0*jnp.sum(w),0*jnp.sum(w),0*jnp.sum(w)])
        class GradientRecorder:
            def update(self,grad,state,params):
                return jax.tree.map(lambda a:jnp.zeros_like(a),grad),grad
        step=make_step(components,GradientRecorder(),config)
        weights=jnp.array([.7,1.3,1.,1.,1.])
        for ac,ch in ((True,False),(False,True),(True,True)):
            active=jnp.array([ac,ch,True,True,True])
            _, grad, returned, _, finite=step(self.net,None,weights,self.times,self.x,self.sub,
                                jnp.ones(len(self.x),dtype=bool),jnp.array(False),active)
            expected=self.net.net.weights*jnp.array([2*weights[0]*2*ac,2*weights[1]*3*ch])[:,None]
            np.testing.assert_allclose(grad.net.weights,expected,rtol=1e-6,atol=1e-7)
            np.testing.assert_array_equal(returned,weights)
            self.assertTrue(bool(finite))
        self.assertEqual([stagger_phase(i,1000,25) for i in (0,24,25,49,50,999,1000)],
                         [0,0,1,1,0,1,2])
        self.assertTrue(all(stagger_phase(i,0,25)==2 for i in range(5)))

    def test_normalized_data_preserves_pdes_and_explicit_bc_ic_targets(self):
        c=get_config('corrosion2d','phase',42)
        c.physical.c_le=.035
        p0=jnp.linspace(.1,.9,len(self.x))
        ic=jnp.stack([p0,3*p0**2-2*p0**3],axis=-1)
        args=(c,self.nd,self.x,ic,self.x[:2],self.x[-2:])
        old=make_loss_vector(*args)
        bc=make_normalized_data_loss_vector(*args,normalize_initial=False)
        both=make_normalized_data_loss_vector(*args,normalize_initial=True)
        mask=jnp.ones(len(self.x),dtype=bool)
        a,b,d=[eqx.filter_jit(f)(self.net,self.times,self.x,self.sub,mask) for f in (old,bc,both)]
        np.testing.assert_array_equal(a[:2],b[:2])
        np.testing.assert_array_equal(a[:2],d[:2])
        np.testing.assert_array_equal(a[-1],b[-1])
        mapped=normalized_data_fields(self.net,self.x,self.times)
        expected_zero=jnp.mean(jnp.sum(mapped[:,:2]**2,axis=-1))
        expected_one=jnp.mean(jnp.sum((mapped[:,-2:]-1)**2,axis=-1))
        np.testing.assert_allclose(d[2:4],jnp.array([expected_zero,expected_one]),rtol=1e-6)
        def ic_value(n):
            out=normalized_data_fields(n,self.x,jnp.zeros(1))[0]
            return jnp.mean(jnp.sum((out-ic)**2,axis=-1))
        np.testing.assert_allclose(d[-1],ic_value(self.net),rtol=1e-6)
        grad=eqx.filter_grad(lambda n:both(n,self.times,self.x,self.sub,mask)[-1])(self.net)
        direct=eqx.filter_grad(ic_value)(self.net)
        np.testing.assert_allclose(grad.net.weights,direct.net.weights,rtol=1e-6,atol=1e-7)

    def test_normalized_bc_resolves_small_drive_scale_but_not_exact_saturation(self):
        c=get_config('corrosion2d','phase',42);c.physical.c_le=.035
        p0=jnp.full((len(self.x),),.25)
        ic=jnp.stack([p0,3*p0**2-2*p0**3],axis=-1)
        args=(c,self.nd,self.x,ic,self.x[:2],self.x[-2:])
        old=make_loss_vector(*args)
        new=make_normalized_data_loss_vector(*args,normalize_initial=False)
        mask=jnp.ones(len(self.x),dtype=bool)
        for raw_v in (-4.,-20.):
            w=jnp.zeros_like(self.net.net.weights).at[0,0].set(8.).at[1,0].set(raw_v)
            n=eqx.tree_at(lambda m:m.net.weights,self.net,w)
            g0=eqx.filter_grad(lambda m:old(m,self.times,self.x,self.sub,mask)[3])(n).net.weights[1,0]
            g1=eqx.filter_grad(lambda m:new(m,self.times,self.x,self.sub,mask)[3])(n).net.weights[1,0]
            if raw_v == -4.:
                self.assertGreater(abs(float(g1)),0.)
                np.testing.assert_allclose(g1/g0,1/.035**2,rtol=.002)
            else:
                self.assertEqual(float(g0),0.);self.assertEqual(float(g1),0.)

    def test_matched_graph_initial_data_prescribes_corner_and_keeps_free_ic(self):
        c=get_config('corrosion2d','phase',42)
        c.experiment.initial_sampling='matched_graph'
        points=np.asarray(self.x)
        x,ic=training_initial_data(c,points,points[:3])
        zero,one,_=boundary_values(c,points)
        target=initial_fields(c,points)
        np.testing.assert_array_equal(x,self.x)
        np.testing.assert_array_equal(np.asarray(ic)[~(zero|one)],target[~(zero|one)])
        np.testing.assert_array_equal(np.asarray(ic)[zero],0.)
        np.testing.assert_array_equal(np.asarray(ic)[one],1.)
        q=np.asarray(ic)[:,1]-(1-c.physical.c_le)*(3*np.asarray(ic)[:,0]**2-2*np.asarray(ic)[:,0]**3)-c.physical.c_le
        np.testing.assert_allclose(q[zero],-c.physical.c_le,atol=1e-7)
        np.testing.assert_allclose(q[one],0.,atol=1e-7)
        c.experiment.initial_sampling='random_interior'
        xr,ir=training_initial_data(c,points,points[2:5])
        np.testing.assert_array_equal(xr,points[2:5])
        np.testing.assert_array_equal(ir,initial_fields(c,points[2:5]))
        c.experiment.initial_sampling='invalid'
        with self.assertRaisesRegex(ValueError,'Unknown initial sampling'):
            training_initial_data(c,points,points[:3])

    def test_matched_graph_boundary_coordinates_are_exact_reference_nodes(self):
        c=get_config('corrosion2d','phase',42);c.experiment.boundary_sampling='matched_graph'
        points=np.asarray(self.x);zero,one,_=boundary_values(c,points)
        z,o=training_boundary_data(c,points,points[:2],points[-2:])
        np.testing.assert_array_equal(z,points[zero]);np.testing.assert_array_equal(o,points[one])
        c.experiment.boundary_sampling='random'
        z,o=training_boundary_data(c,points,points[:2],points[-2:])
        np.testing.assert_array_equal(z,points[:2]);np.testing.assert_array_equal(o,points[-2:])
        c.experiment.boundary_sampling='invalid'
        with self.assertRaisesRegex(ValueError,'Unknown boundary sampling'):
            training_boundary_data(c,points,points[:2],points[-2:])

    def test_normalized_data_rejects_incompatible_mapping_and_initial_values(self):
        c=get_config('corrosion2d','phase',42)
        args=(self.nd,self.x,jnp.zeros((len(self.x),2)),self.x[:2],self.x[-2:])
        c.arch.use_hard_constraint=False
        with self.assertRaisesRegex(ValueError,'KKS output'):
            make_normalized_data_loss_vector(c,*args,normalize_initial=False)
        c.arch.use_hard_constraint=True;c.physical.c_se=.9
        with self.assertRaisesRegex(ValueError,'c_se=1'):
            make_normalized_data_loss_vector(c,*args,normalize_initial=False)
        c.physical.c_se=1.
        bad=jnp.full((len(self.x),2),.25)
        with self.assertRaisesRegex(ValueError,'declared c_0'):
            make_normalized_data_loss_vector(c,self.nd,self.x,bad,self.x[:2],self.x[-2:],normalize_initial=True)

    def test_bounded_weight_and_physical_residual_identity(self):
        phase, ch = compute_pde(self.net, self.x, self.times, self.sub, self.nd, "phase", .5)
        fields = jax.vmap(lambda t: jax.vmap(self.net, in_axes=(0, None))(self.x, t))(self.times)
        phi = fields[..., 0].T
        for eta in (1., .1, .01):
            bounded, same_ch = compute_pde(self.net, self.x, self.times, self.sub, self.nd,
                                          "bounded", .5, bounded_eta=eta)
            np.testing.assert_allclose(bounded * (4 * phi * (1 - phi) + eta),
                                       (1 + eta) * phase, rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(ch, same_ch)
            weights = np.asarray(bounded_phase_weight(jnp.array([0., .25, .5, .75, 1.]), eta))
            self.assertTrue(np.all(weights >= 1))
            np.testing.assert_allclose(weights[[0, -1]], (1 + eta) / eta)
            self.assertEqual(float(weights[2]), 1.)
        for eta in (None, 0., -1., np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                bounded_phase_weight(phi, eta)

    def test_bounded_gradient_and_compiled_soft_loss(self):
        config = get_config("corrosion2d", "bounded", 42)
        config.loss.ac_scale = 1.
        def value(net):
            ac, _ = compute_pde(net, self.x, self.times, self.sub, self.nd,
                                "bounded", .5, bounded_eta=.1)
            return jnp.mean(ac**2)
        grad = eqx.filter_jit(eqx.filter_grad(value))(self.net)
        direction = jnp.zeros_like(self.net.net.weights).at[0, 3].set(1.)
        eps = 1e-3
        plus = eqx.tree_at(lambda n: n.net.weights, self.net, self.net.net.weights + eps * direction)
        minus = eqx.tree_at(lambda n: n.net.weights, self.net, self.net.net.weights - eps * direction)
        np.testing.assert_allclose(grad.net.weights[0, 3], (value(plus) - value(minus)) / (2 * eps),
                                   rtol=.02, atol=.002)
        bounded_loss = make_loss_vector(config, self.nd, self.x[:3], jnp.full((3, 2), .25), self.x[:2], self.x[-2:])
        config.experiment.formulation = "phase"
        phase_loss = make_loss_vector(config, self.nd, self.x[:3], jnp.full((3, 2), .25), self.x[:2], self.x[-2:])
        mask = jnp.ones(len(self.x), dtype=bool)
        actual = eqx.filter_jit(bounded_loss)(self.net, self.times, self.x, self.sub, mask)
        expected = eqx.filter_jit(phase_loss)(self.net, self.times, self.x, self.sub, mask)
        np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-6, atol=1e-7)
        self.assertGreater(float(actual[-1]), .01)

    def test_soft_ic_is_preserved_and_boundary_spec_is_explicit(self):
        config = get_config("corrosion2d", "latent", 42)
        config.loss.ac_scale = 1.0
        ic = jnp.ones((3, 2)) * 0.25
        losses = make_loss_vector(config, self.nd, self.x[:3], ic, self.x[:2], self.x[-2:])
        vals = losses(self.net, self.times, self.x, self.sub, jnp.ones(len(self.x), dtype=bool))
        self.assertGreater(float(vals[-1]), 0.01)
        grad = eqx.filter_grad(lambda n: losses(n, self.times, self.x, self.sub, jnp.ones(len(self.x), dtype=bool))[-1])(self.net)
        self.assertGreater(float(jnp.linalg.norm(grad.net.weights)), 1e-4)
        for case in ("corrosion2d", "electropolish2d"):
            c = get_config(case, "phase", 42)
            xx, yy, points = grid(c)
            zero, one, values = boundary_values(c, points)
            self.assertFalse(np.any(zero & one))
            self.assertTrue(np.all(values[one] == 1))
            if case == "corrosion2d":
                self.assertTrue(np.all(one.reshape(xx.shape)[-1]))
                self.assertFalse(np.any(zero.reshape(xx.shape)[-1]))
            else:
                self.assertTrue(np.all(zero.reshape(xx.shape)[-1]))
                self.assertTrue(np.all(one.reshape(xx.shape)[0]))
            spec = reference_spec(c)
            c.physical.L *= 2
            self.assertNotEqual(spec, reference_spec(c))
        matrix = graph_matrix(self.op, len(self.x))
        np.testing.assert_allclose(matrix @ np.ones(len(self.x)), 0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
