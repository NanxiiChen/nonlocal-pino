"""Fixed gradient streams; current source/precision and observed-perturbation fixtures."""
from pathlib import Path
import json,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
from jax.flatten_util import ravel_pytree
from ml_collections import ConfigDict
import numpy as np
from examples.twophase2d.train import _build_network

out=ROOT/'reports/soap-implementation-audit';out.mkdir(exist_ok=True)
rng=np.random.default_rng(402);cases=[];arrays={}
for mode in ['native','shared']:
 for name,shape,one,limit in [('vector',(7,),False,128),('square',(3,3),False,128),('tensor',(3,4,2),False,128),('partial',(6,3),False,4)]+([('vector_pre',(4,),True,128),('rectangular',(2,5),False,128)] if mode=='shared' else []):
  dtypes=['float32','float64'] if mode=='native' else ['float64']
  for dtype in dtypes:
   key=mode+'_'+name+'_'+dtype
   arrays[key+'_initial']=rng.normal(size=shape).astype(dtype)
   arrays[key+'_gradient']=rng.normal(size=(16,*shape)).astype(dtype)
   cases.append(dict(key=key,mode=mode,shape=shape,dtype=dtype,precondition_1d=one,maxdim=limit,weight_decay=0.,correct_bias=True,frequency=5))
# No-bias and inactive-in-current-project nonzero-decay controls avoid all basis ambiguity.
for name,correct,decay in [('no_bias',False,0.),('decay',True,.1)]:
 key=name;arrays[key+'_initial']=rng.normal(size=7);arrays[key+'_gradient']=rng.normal(size=(16,7))
 cases.append(dict(key=key,mode='native',shape=(7,),dtype='float64',precondition_1d=False,maxdim=128,weight_decay=decay,correct_bias=correct,frequency=5))
base=Path('/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-nonlocal')
c=ConfigDict(json.loads((base/'config.json').read_text()));jax.config.update('jax_enable_x64',True)
template=_build_network(c,jax.random.split(jax.random.PRNGKey(42))[0])
_,unravel=ravel_pytree(eqx.filter(template,eqx.is_inexact_array))
with np.load(ROOT/'docs/tasks/archive/twophase-repeatability-evidence/ordinary-autodiff-spectrum/frozen_arrays.npz') as f:
 arrays['measured_initial_gradients']=np.stack([np.asarray(unravel(a).net.output.weight) for a in f['gradient']])
arrays['common_second_gradient']=rng.normal(size=(2,64))*.01
np.savez_compressed(out/'fixtures.npz',**arrays)
(out/'fixtures.json').write_text(json.dumps(dict(cases=cases,lr=.001,b1=.9,b2=.99,eps=1e-8,synthetic_common_second_gradient=True,measured_first_gradients='actual saved FP64 initial gradient repeats, output-weight2x64 block'),indent=2)+'\n')
print('wrote',len(cases),'gradient-stream cases')
