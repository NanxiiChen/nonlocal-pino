"""Read-only evolution of saved training checkpoints, with fixed evaluation."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from scipy import ndimage

from examples.corrosion2d import residual_experiment as r

OUT = Path(__file__).with_suffix('.json')
ROOT = Path('/root/autodl-tmp/tf-logs/corrosion2d')
RUNS = {'bounded42': ROOT/'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed42',
        'bounded2026': ROOT/'20261025-independent-seed2026-default-strategy'}


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU-only saved-checkpoint probe')
    jax.config.update('jax_default_matmul_precision', 'highest')
    c = ConfigDict(json.loads((RUNS['bounded42']/'config.json').read_text()))
    reference = r.load_reference(c)
    points = np.stack([reference['X'].ravel(),reference['Y'].ravel()],-1)
    x, times = jnp.asarray(points,dtype=jnp.float32),jnp.asarray(reference['time'],dtype=jnp.float32)
    zero,one,_ = r.boundary_values(c,points)
    template = r.build_model(c,jax.random.PRNGKey(0))

    @eqx.filter_jit
    def evaluate(net):
        return jax.vmap(lambda t:jax.vmap(net,in_axes=(0,None))(x,t))(times)

    output = {'scope':'Fixed full-grid/time evaluation of training1000..5000 checkpoints; no model selection','runs':{}}
    for label,run in RUNS.items():
        history={v['step']:v for v in map(json.loads,(run/'metrics.jsonl').read_text().splitlines())}
        records=[]
        for step in (1000,2000,3000,4000,5000):
            model=eqx.tree_deserialise_leaves(run/f'ckpt_{step:06d}.eqx',template)
            f=np.asarray(evaluate(model)).reshape(len(times),*reference['X'].shape,2).astype(float)
            p=f[...,0]
            initial=float(np.linalg.norm(p[0]-reference['phi'][0])/np.linalg.norm(reference['phi'][0]))
            terminal=float(np.linalg.norm(p[-1]-reference['phi'][-1])/np.linalg.norm(reference['phi'][-1]))
            np.testing.assert_allclose(terminal,history[step]['terminal_rel_l2_phi'],rtol=0,atol=3e-6)
            flat=f.reshape(len(times),-1,2)
            counts=[int(ndimage.label(a<.5)[1]) for a in p]
            records.append({'step':step,'initial_phi_relative_l2':initial,'terminal_phi_relative_l2':terminal,
                            'graph_BC_max_error':float(max(np.max(np.abs(flat[:,zero])),np.max(np.abs(flat[:,one]-1)))),
                            'liquid_components_by_time':counts,'late_extra_liquid_components':max(counts)>1})
        output['runs'][label]={'path':str(run),'checkpoints':records}
        print(label,json.dumps(records),flush=True)
    OUT.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
