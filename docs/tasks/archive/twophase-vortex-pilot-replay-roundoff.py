"""Independent checkpoint replay difference probe; does not alter stored arrays."""
from pathlib import Path
import json
import sys
import os
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.problem import load_reference
OUT=ROOT/'reports/twophase-vortex-pilot'
manifest=json.loads((OUT/'manifest.json').read_text())
backend=jax.default_backend()
result=dict(backend=backend,runs=[])
for item in manifest['runs']:
    run=Path(item['run']);c=ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    ref=load_reference(c)
    with jax.enable_x64(False):
        template=PhaseNetwork(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
    fields={}
    modes=['float32','float64'] if backend=='cpu' else ['float32']
    for dtype in modes:
        jax.config.update('jax_enable_x64',dtype=='float64')
        model=jax.tree.map(lambda a:a.astype(dtype) if eqx.is_inexact_array(a) else a,net)
        points=jnp.asarray(ref['points'],dtype=dtype);times=jnp.asarray(ref['time'],dtype=dtype)
        predict=eqx.filter_jit(lambda model:jax.vmap(lambda t:jax.vmap(model,in_axes=(0,None))(points,t))(times))
        fields[dtype]=np.asarray(predict(model)).reshape(21,64,64,2)
    with np.load(run/'prediction.npz') as data:
        saved=np.stack((data['phi'],data['q']),axis=-1)
    r=dict(method=item['method'],saved_vs_replay={})
    for dtype,a in fields.items():
        r['saved_vs_replay'][dtype]={name:dict(max_absolute=float(np.max(np.abs(a[...,k]-saved[...,k]))),
            relative_L2=float(np.linalg.norm(a[...,k]-saved[...,k])/np.linalg.norm(saved[...,k]))) for k,name in enumerate(['phi','q'])}
    result['runs'].append(r)
if backend=='gpu':
    if any(v['max_absolute']!=0 for r in result['runs'] for v in r['saved_vs_replay']['float32'].values()):
        raise ValueError('Same-backend frozen replay changed')
(OUT/f'replay-roundoff-{backend}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
