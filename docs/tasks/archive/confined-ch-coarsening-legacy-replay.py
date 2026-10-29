"""Read-only old-checkpoint compatibility check after adding static IC metadata."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.problem import load_reference,initial_field
from examples.confined_ch2d.network import predict_fields


def main():
    if jax.default_backend()!='cpu':raise ValueError('Use JAX_PLATFORMS=cpu')
    jax.config.update('jax_enable_x64',True)
    base=Path('/root/autodl-tmp/tf-logs/confined_ch2d');results={}
    for name in ('nonlocal','local-ad'):
        run=base/('20261027-confined-'+name+'-01')
        c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
        points=jnp.asarray(ref['points']);v=jnp.asarray(ref['volumes'])
        mean=jnp.dot(v,initial_field(points,c,xp=jnp))/v.sum()
        key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
        net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(c,key))
        @eqx.filter_jit
        def all_times(net):return jax.vmap(lambda t:predict_fields(net,points,t,mean,v))(jnp.asarray(ref['time']))
        actual=np.asarray(all_times(net));row={}
        with np.load(run/'prediction.npz') as f:
            for i,k in enumerate(('phi','mu')):
                np.testing.assert_allclose(actual[...,i],f[k],atol=1e-8,rtol=1e-8)
                row[k]=float(np.max(abs(actual[...,i]-f[k])))
        results[name]=row
    report=dict(status='passed',scope='No old run file written; all saved times and both physical fields replayed',max_absolute=results)
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
