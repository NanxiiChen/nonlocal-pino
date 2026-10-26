"""Independent topology audit of fixed2000-update checkpoints, CPU only."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from examples.corrosion2d.residual_experiment import build_model, load_reference


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU only')
    root = Path('/root/autodl-tmp/tf-logs/electropolish2d')
    out = {'scope': 'Fixed2000-update interim checkpoints; not final model selection', 'runs': {}}
    for form, mode in (('phase','physical'), ('phase','s_bc'), ('latent','s_bc')):
        run = root/f'20261025-data-{mode}-icmatched_graph-bcmatched_graph-{form}-stagger0-seed42'
        c = ConfigDict(json.loads((run/'config.json').read_text()))
        ref = load_reference(c)
        x = jnp.asarray(np.stack([ref['X'].ravel(),ref['Y'].ravel()],-1), dtype=jnp.float32)
        ts = jnp.asarray(ref['time'], dtype=jnp.float32)
        model = eqx.tree_deserialise_leaves(run/'ckpt_002000.eqx', build_model(c,jax.random.split(jax.random.PRNGKey(42))[0]))

        @eqx.filter_jit
        def predict(net):
            return jax.vmap(lambda t:jax.vmap(net,in_axes=(0,None))(x,t))(ts)

        fields = np.asarray(predict(model)).reshape(21,*ref['X'].shape,2).astype(float)
        p = fields[...,0]
        assert not np.any(p==.5)
        counts = ((p[:,:-1]-.5)*(p[:,1:]-.5)<0).sum(axis=1)
        hist = [{str(int(k)):int(v) for k,v in zip(*np.unique(a,return_counts=True))} for a in counts]
        data = {'path':str(run/'ckpt_002000.eqx'),'physical_times_seconds':ref['physical_time'].tolist(),
                'crossings_by_time':hist}
        for channel,name in enumerate(('phi','c')):
            data['terminal_rel_l2_'+name] = float(np.linalg.norm(fields[-1,...,channel]-ref[name][-1])/np.linalg.norm(ref[name][-1]))
        out['runs'][form+'/'+mode] = data
        print(form,mode,'terminal L2',data['terminal_rel_l2_phi'],data['terminal_rel_l2_c'],'all times one interface',bool(np.all(counts==1)), 'terminal counts',hist[-1],flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
