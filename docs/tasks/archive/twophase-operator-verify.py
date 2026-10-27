"""Frozen GPU replay, exact initialization and actual trained periodicity audit."""
from pathlib import Path
import json
import argparse
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.fields import predict_fields
from examples.twophase2d.problem import load_reference, validate_config
from nonlocal_pinn.training.runtime import configure_gpu_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',action='append',choices=['preconditioning','refinement','resolution','conserved','initial-time','confirmation','short-confirmation'])
    args=parser.parse_args()
    stages=args.stage if args.stage else ['preconditioning','refinement','resolution','conserved','initial-time','confirmation','short-confirmation']
    configure_gpu_policy('ordinary')
    if jax.default_backend()!='gpu':raise ValueError('Use JAX_PLATFORMS=cuda')
    records=[]
    for stage in stages:
        out=ROOT/('reports/twophase-operator-'+stage)
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['status']!='completed':raise ValueError('Incomplete '+stage)
        for item in manifest['runs']:
            run=Path(item['run']);c=ConfigDict(json.loads((run/'config.json').read_text()))
            validate_config(c);reference=load_reference(c)
            jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
            jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
            with jax.enable_x64(False):
                template=PhaseNetwork(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
            template=jax.tree.map(lambda a:a.astype(c.train.compute_dtype) if eqx.is_inexact_array(a) else a,template)
            initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
            for a,b in zip(jax.tree.leaves(initial),jax.tree.leaves(template),strict=True):
                np.testing.assert_array_equal(a,b)
            model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
            points=jnp.asarray(reference['points'],dtype=c.train.compute_dtype)
            times=jnp.asarray(reference['time'],dtype=c.train.compute_dtype)
            predict=eqx.filter_jit(lambda net:jax.vmap(lambda t:predict_fields(net,points,t,c))(times))
            fields=np.asarray(predict(model)).reshape(21,c.sample.Ny,c.sample.Nx,2)
            with np.load(run/'prediction.npz') as f:saved=np.stack((f['phi'],f['q']),axis=-1)
            tolerance=5e-6 if c.train.compute_dtype=='float32' else 1e-10
            np.testing.assert_allclose(fields,saved,atol=tolerance,rtol=0)
            fixed=stage!='preconditioning'
            if fixed:np.testing.assert_array_equal(model.modes,template.modes)
            axis=jnp.linspace(0.,1.,9,dtype=points.dtype)
            edge=jnp.concatenate((jnp.stack((jnp.zeros_like(axis),axis),axis=-1),
                                  jnp.stack((axis,jnp.zeros_like(axis)),axis=-1)))
            opposite=edge+jnp.concatenate((jnp.tile(jnp.asarray([[1.,0.]],dtype=points.dtype),(9,1)),
                                            jnp.tile(jnp.asarray([[0.,1.]],dtype=points.dtype),(9,1))))
            edges=eqx.filter_jit(lambda net:jax.vmap(lambda t:jax.vmap(net,in_axes=(0,None))(edge,t)-jax.vmap(net,in_axes=(0,None))(opposite,t))(times))
            gap=float(np.max(np.abs(np.asarray(edges(model)))))
            if fixed and gap>1e-10:raise ValueError('Fixed trained model violates periodic values')
            result=dict(stage=stage,arm=item['arm'],run=str(run),
                initial_arrays_exact=True,gpu_replay_max_absolute=float(np.max(np.abs(fields-saved))),
                replay_absolute_tolerance=tolerance,modes=np.asarray(model.modes).tolist(),
                periodic_frequencies_fixed=fixed,trained_boundary_value_gap=gap,
                physical_identity_validated=True, compute_dtype=c.train.compute_dtype)
            records.append(result);print(result,flush=True)
    directory=stages[0] if len(stages)==1 else 'conserved'
    output=ROOT/('reports/twophase-operator-'+directory+'/gpu-verification.json')
    output.write_text(json.dumps(dict(status='passed',runs=records),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
