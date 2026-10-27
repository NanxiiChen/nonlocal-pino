"""Independent scalar-coordinate recurrence audit of the actual project factory."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import jax,jax.numpy as jnp,optax
from ml_collections import ConfigDict
import numpy as np
from nonlocal_pinn.training.optimizers import make_optimizer

jax.config.update('jax_enable_x64',True)
if jax.default_backend()!='cpu':raise ValueError('CPU only')
out=ROOT/'reports/soap-implementation-audit'
c=ConfigDict(json.loads(Path('/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-nonlocal/config.json').read_text()))
optimizer=make_optimizer(c)
initial=np.array([.2,-.1,.5]);steps=1005
rng=np.random.default_rng(773);grads=rng.normal(size=(steps,3))*100
state=optimizer.init(jnp.asarray(initial))
@jax.jit
def solve(p,state,gradients):
 def step(carry,g):
  p,state=carry;delta,state=optimizer.update(g,state,p);p=optax.apply_updates(p,delta)
  return (p,state),p
 return jax.lax.scan(step,(p,state),gradients)[1]
actual=np.asarray(solve(jnp.asarray(initial),state,jnp.asarray(grads)))
p=initial.copy();m=np.zeros(3);v=np.zeros(3);gold=[]
for call,g in enumerate(grads,1):
 g=g*min(1.,c.train.grad_clip/np.linalg.norm(g))
 if call>1:
  k=call-1;m=c.train.soap_b1*m+(1-c.train.soap_b1)*g;v=c.train.soap_b2*v+(1-c.train.soap_b2)*g*g
  lr=max(c.train.lr_min,c.train.lr_init*c.train.lr_decay_rate**((call-1)/c.train.lr_decay_every))
  correction=np.sqrt(1-c.train.soap_b2**k)/(1-c.train.soap_b1**k)
  p=p-lr*correction*m/(np.sqrt(v)+1e-8)
 gold.append(p.copy())
gold=np.asarray(gold);difference=float(np.max(np.abs(actual-gold)))
# The production LR schedule is FP32 even for FP64 parameters; allow its
# accumulated scalar-rounding error, while detecting counter/clipping mistakes.
np.testing.assert_allclose(actual,gold,atol=2e-8,rtol=0)
np.testing.assert_array_equal(actual[0],initial)
(out/'wrapper.json').write_text(json.dumps(dict(status='passed',steps=steps,all_raw_gradients_above_clip=bool(np.all(np.linalg.norm(grads,axis=1)>c.train.grad_clip)),max_parameter_difference_from_numpy=difference,absolute_tolerance=2e-8,learning_rate_comparison='ideal real-valued continuous exponential; production scalar schedule FP32',effective_first_update_call=2,weight_decay_passed_by_factory=False),indent=2)+'\n')
print(difference,flush=True)
