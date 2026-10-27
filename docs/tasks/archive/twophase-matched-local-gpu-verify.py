"""Training-backend exact initial reconstruction and FP64 prediction replay."""
from pathlib import Path
import json
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.fields import predict_fields
from examples.twophase2d.problem import load_reference,validate_config
from nonlocal_pinn.training.runtime import configure_gpu_policy


def main():
    configure_gpu_policy('ordinary')
    if jax.default_backend()!='gpu':raise ValueError('Use training GPU backend')
    out=ROOT/'reports/twophase-matched-local'
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Both runs must complete')
    results=[]
    for item in manifest['runs']:
        run=Path(item['run']);c=ConfigDict(json.loads((run/'config.json').read_text()))
        validate_config(c);reference=load_reference(c)
        jax.config.update('jax_enable_x64',True)
        jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
        with jax.enable_x64(False):template=PhaseNetwork(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
        template=jax.tree.map(lambda a:a.astype(np.float64) if eqx.is_inexact_array(a) else a,template)
        initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
        for actual,expected in zip(jax.tree.leaves(initial),jax.tree.leaves(template),strict=True):np.testing.assert_array_equal(actual,expected)
        model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
        points=jnp.asarray(reference['points']);times=jnp.asarray(reference['time'])
        predict=eqx.filter_jit(lambda net:jax.vmap(lambda t:predict_fields(net,points,t,c))(times))
        actual=np.asarray(predict(model)).reshape(21,64,64,2)
        with np.load(run/'prediction.npz') as f:saved=np.stack((f['phi'],f['q']),axis=-1)
        # The established FP64 checkpoint replay tolerance is 1e-10.
        # Ordinary GPU projected-mean reductions can differ at roundoff.
        np.testing.assert_allclose(actual,saved,atol=1e-10,rtol=0)
        results.append(dict(method=item['method'],initial_arrays_exact=True,
                            final_gpu_prediction_max_absolute=float(np.max(np.abs(actual-saved))),
                            prediction_absolute_tolerance=1e-10))
    (out/'gpu-verification.json').write_text(json.dumps(dict(status='passed',runs=results),indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
