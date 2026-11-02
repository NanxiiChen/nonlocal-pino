"""Saved-source noncausal arithmetic/gradient parity before the causal pilot."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.causal import comparison_config
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_causal_online import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network


def main(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    control=Path(a.control);saved=json.loads((control/'config.json').read_text())
    c=get_config();c.train.seed=saved['train']['seed'];c.reference.root=saved['reference']['root']
    comparison_config(c,control)
    if jax.default_backend()!='cpu':raise ValueError('This small saved-source check requires CPU only')
    previous=control/'source/examples/channel_nsch2d/losses.py'
    spec=importlib.util.spec_from_file_location('saved_uniform_channel_losses',previous)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c.sample.Nx=16;c.sample.Ny=8;c.sample.boundary_points=6;c.nl.degree=2;c.nl.neighbors=20
    c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
    c.arch.spatial_features=2;c.arch.time_features=4;c.train.compute_dtype='float64'
    with jax.enable_x64(True):
        raw=[geometry_data(amp,c)[0] for amp in ((.12,.08),(.06,.13))]
        batch={k:jnp.asarray(np.stack([v[k] for v in raw])) for k in raw[0]}
        model=build_network(c,jax.random.PRNGKey(9))
        times=jnp.asarray([.025,.15,.5,1.,2.,3.,4.5,6.])
        old=eqx.filter_jit(eqx.filter_value_and_grad(module.make_loss_fn(c),has_aux=True))(model,batch,times)
        new=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))(model,batch,times)
        gaps=[]
        for x,y in zip(jax.tree.leaves(old),jax.tree.leaves(new),strict=True):
            np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
            gaps.append(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
    report=dict(status='saved_source_noncausal_loss_parts_all_parameter_gradients_passed',
        control=str(control),saved_loss_source=str(previous),maximum_absolute_gap=max(gaps),
        compared_arrays=len(gaps),production_config_only_adds_causal=True,
        scope='Independent saved-source parity on a small actual coupled network, not a production GPU bitwise/trajectory guarantee',
        reference_solves=0,training_updates=0)
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'saved_uniform_losses.py').write_bytes(previous.read_bytes())
    root=Path(__file__).resolve().parents[2]
    for name in ('docs/tasks/channel_causal_preflight.py','examples/channel_nsch2d/losses.py',
                 'examples/channel_nsch2d/causal.py'):
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
