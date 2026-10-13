"""Fixed-own-H head comparison; no diagnostic field becomes a physical parent."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz

from examples.fracture2d.cached_damage_optimization import make_cached_lbfgs
from examples.fracture2d.conditioned_damage import make_features
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import load_geometry, sha
from examples.fracture2d.output_damage_continue import atomic_fields
from examples.fracture2d.plain_damage_head import fresh_network, initial_scales, initialization, make_problem


def run(c):
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or list(c.head.heads) != ['sigmoid','exp_square','identity']
            or list(c.head.methods) != ['residual_l32','residual_l16']
            or c.head.load != .0001 or c.head.initial_damage != 1e-5
            or c.head.lbfgs.residual_limit != .001
            or c.head.lbfgs.range_tolerance != 1e-10
            or c.head.lbfgs.maxiter != 2048):
        raise ValueError('head comparison requires declared first-load original gates and2048-stage budget')
    source = Path(c.head.source_dir)
    protected = {}
    for name,digest in [('initial_physical_fields.npz',c.head.source_fields_sha256),
            ('identity.json',c.head.source_identity_sha256),('audit.json',c.head.source_audit_sha256),
            ('zero_fields.npz',c.head.source_zero_sha256)]:
        path = source/name
        if sha(path) != digest:
            raise ValueError(f'head comparison own-zero source changed: {path}')
        protected[str(path)] = digest
    original = json.loads((source/'identity.json').read_text())
    audit = json.loads((source/'audit.json').read_text())
    if (not audit['pass_replay'] or not original['own_zero_initial_state']
            or not original['fresh_network'] or original['old_weights']
            or original['direct_targets'] or original['damage_correction']):
        raise ValueError('head comparison requires independently audited own-zero neural source')
    for key in ('physical','geometry','nl','trial','conditioned','validation','arch','train'):
        if original['config'][key] != c[key].to_dict():
            raise ValueError(f'head comparison source/config mismatch: {key}')
    mechanics,operator,geometry = load_geometry(c)
    with np.load(source/'initial_physical_fields.npz',allow_pickle=False) as f:
        q,d,H = [f[key].copy() for key in ('q','d','H')]
    with np.load(source/'zero_fields.npz',allow_pickle=False) as f:
        for key in f.files:
            np.testing.assert_array_equal(f[key],np.zeros_like(f[key]))
    np.testing.assert_array_equal(d,np.zeros(mechanics.n_material))
    np.testing.assert_array_equal(H,mechanics.driver(q))
    if q.shape != mechanics.load_direction.shape or H.shape != (mechanics.n_material,):
        raise ValueError('head comparison source fields/material ordering differ')
    np.testing.assert_array_equal(q[mechanics.constrained],
        c.head.load*mechanics.load_direction[mechanics.constrained])
    basis_path = Path(c.cold.basis_path)
    if sha(basis_path) != c.cold.basis_sha256:
        raise ValueError('head comparison compact geometry changed')
    protected[str(basis_path)] = sha(basis_path)
    for name in ('identity.json','mesh.npz','damage_operator.npz','fenics_parity.json'):
        path = Path(c.trial.geometry_dir)/name
        protected[str(path)] = sha(path)
    matrix = load_npz(basis_path)
    gc0,gc1 = c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    features,info = make_features(operator,mechanics.coordinates[:mechanics.n_material],
        gc0=gc0,length=c.physical.l0,tip=list(c.conditioned.tip),powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    s = dict(matrix=matrix,operator=operator,features=lambda known,load:features(known),
        reference_load=c.head.load)
    names = ['plain_damage_head_trial.py','plain_damage_head.py','plain_sigmoid_damage.py',
        'cached_damage_optimization.py','conditioned_damage.py','compact_damage_features.py',
        'compact_damage_loading.py','local_damage_trial.py','local_mechanics.py',
        'output_damage_continue.py','split_pinn_gradients.py','train.py']
    code = {name:sha(Path(__file__).with_name(name)) for name in names}
    root = Path(c.head.output_dir)
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'identity.json',dict(schema='plain-head-fixed-own-H-v1',config=c.to_dict(),
        source=protected,code=code,geometry=geometry,features=info,
        own_zero_source=True,fresh_networks=True,no_reference_output=True,
        no_direct_targets=True,no_damage_correction=True,physical_state_promoted=False))
    (root/'code_snapshot').mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name),root/'code_snapshot'/name)
    atomic_fields(root/'own_fields.npz',q=q,d=d,H_used=H)
    H,load = jnp.asarray(H),jnp.asarray(c.head.load,jnp.float64)
    reference_grad = None
    reports = []
    for head in c.head.heads:
        started = time.perf_counter()
        folder = root/head
        folder.mkdir()
        columns,gain = initial_scales(matrix,operator.to_scipy(),np.asarray(H),head=head,
            initial_damage=c.head.initial_damage,gc0=gc0,gc1=gc1)
        net = fresh_network(c,info['input_dimension'],matrix,head=head)
        image,metrics,objectives,latent = make_problem(c,s,columns,gain,head=head)
        predict,measure,get_latent = map(eqx.filter_jit,(image,metrics,latent))
        np.testing.assert_allclose(predict(net,H,load),c.head.initial_damage,rtol=1e-12,atol=1e-18)
        gradient = eqx.filter_grad(objectives['residual_l32'])(net,H,load)
        flat_gradient = np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(gradient)])
        if reference_grad is None:
            reference_grad = flat_gradient
        else:
            np.testing.assert_allclose(flat_gradient,reference_grad,rtol=1e-9,atol=1e-10)
        np.savez_compressed(folder/'parameterization.npz',column_scale=columns,trunk_gain=gain,
            initial_gradient=flat_gradient)
        eqx.tree_serialise_leaves(folder/'initial_network.eqx',net)
        stages = []
        for method in c.head.methods:
            fitter,_,_ = make_cached_lbfgs(net,objectives[method],metrics)
            def progress(row):
                print(json.dumps(dict(event='inner',head=head,method=method,**row)),flush=True)
            net,optimization = fitter(net,H,load,c.head.lbfgs.to_dict(),progress=progress)
            stages.append(dict(method=method,**optimization))
            if optimization['status'] == 'residual_reached':
                break
        prediction,z = np.asarray(predict(net,H,load)),np.asarray(get_latent(net,H,load))
        if head == 'sigmoid':
            derivative = prediction*(1-prediction)
        elif head == 'exp_square':
            derivative = -2*z*prediction
        else:
            derivative = np.ones_like(prediction)
        quantiles = [.0,.01,.1,.5,.9,.99,1.]
        derivative_stats = dict(quantiles=quantiles,native=np.quantile(abs(derivative),quantiles).tolist(),
            effective_trunk=np.quantile(abs(gain*derivative),quantiles).tolist())
        atomic_fields(folder/'fields.npz',damage=prediction,latent=z,H_used=np.asarray(H))
        eqx.tree_serialise_leaves(folder/'network.eqx',net)
        report = dict(head=head,ready=optimization['status']=='residual_reached',
            metrics={key:float(value) for key,value in measure(net,H,load).items()},stages=stages,
            initial_latent=initialization(head,c.head.initial_damage)[0],trunk_gain=gain,
            initial_gradient_parity_inf=float(np.max(abs(flat_gradient-reference_grad))),
            derivatives=derivative_stats,seconds=time.perf_counter()-started,
            fields_sha256=sha(folder/'fields.npz'),network_sha256=sha(folder/'network.eqx'),
            physical_state_promoted=False)
        atomic_json(folder/'result.json',report)
        reports.append(report)
        atomic_json(root/'result.json',dict(complete=False,reports=reports,physical_state_promoted=False))
        print(json.dumps(dict(event='head_complete',head=head,ready=report['ready'],**report['metrics'])),flush=True)
    if (protected != {path:sha(path) for path in protected}
            or code != {name:sha(Path(__file__).with_name(name)) for name in names}):
        raise RuntimeError('head comparison protected source/code changed')
    atomic_json(root/'result.json',dict(complete=True,reports=reports,physical_state_promoted=False))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    c = runpy.run_path(parser.parse_args().config)['get_config']()
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    run(c)
