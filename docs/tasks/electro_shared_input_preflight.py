"""Reproduce strict input qualification against retained Electro attempt02 draws."""
from pathlib import Path
import json,time
import jax,jax.numpy as jnp,numpy as np
from examples.electropolish2d.configs.train_operator_threeway import get_config
from examples.electropolish2d.configs.train_operator_threeway_local import get_config as local_config
from examples.electropolish2d.configs.train_operator_threeway_fem import get_config as fem_config
from examples.electropolish2d.comparison_sampling import make_comparison_sampler
from examples.electropolish2d.spatial_sampling import SpatialSampler
from examples.electropolish2d.fem_weak import RectangleWeakRows
from examples.corrosion2d.residual_experiment import grid,boundary_values
from nonlocal_pinn.operators import build_D
root=Path('/root/autodl-tmp/tf-logs/electropolish2d');records={};timings={}
steps=sorted({0,32,9999,*np.linspace(1,9998,61,dtype=int).tolist()});assert len(steps)==64
for method,factory in [('nonlocal',get_config),('local_ad',local_config),('fem_weak',fem_config)]:
 c=factory();_,_,points=grid(c);op=build_D(jnp.asarray(points,dtype=jnp.float32),c.nl.delta,method='sparse',kernel=c.nl.kernel)
 if method=='fem_weak':op=RectangleWeakRows(points,c.sample.Nx,c.sample.Ny)
 zero,one,_=boundary_values(c,points);ic=np.random.default_rng(42).choice(len(points),c.operator.ic_points,replace=False);bc=np.r_[np.flatnonzero(zero),np.flatnonzero(one)]
 spatial=SpatialSampler(op,points,~(zero|one),ic,bc,c);sample,_=make_comparison_sampler(c);key=jax.random.split(jax.random.PRNGKey(42),3)[2]
 run=root/('electro-threeway-20261114-02-'+method);old=np.load(run/'online_coefficients.npy');accepted=np.load(run/'online_accepted.npy');items=[];seconds=[]
 for step in steps:
  batch=spatial._build(step);ids,queries,_,mask,nearest=SpatialSampler._transfer(batch)
  start=time.perf_counter();image,local,coeff,count=sample(jax.random.fold_in(key,step),queries,nearest);jax.block_until_ready((image,local,coeff,count));seconds.append(time.perf_counter()-start)
  np.testing.assert_array_equal(coeff,old[step]);np.testing.assert_array_equal(count,accepted[step])
  first=len(mask);shared=np.asarray(local[:,first:first+len(ic)+len(bc)])
  items.append((np.asarray(image),shared,np.asarray(coeff)))
 records[method]=items;timings[method]={'first_seconds':seconds[0],'warm_mean_seconds':float(np.mean(seconds[1:])),'queries':len(queries)}
 print(method,timings[method],flush=True)
for method in ['local_ad','fem_weak']:
 for a,b in zip(records['nonlocal'],records[method],strict=True):
  for x,y in zip(a,b,strict=True):np.testing.assert_array_equal(x,y)
record={'status':'passed','platform':str(jax.devices()[0]),'updates':steps,'conditions_per_update':8,'methods':list(records),'encoder_and_common_ic_bc_max_gap':0.,'original_saved_coefficients_and_acceptance_exact':True,'known_input_computation':'NumPy FP64 on original FP32-rounded curve vertices, one FP32 output cast','sampler_seconds':timings,'source':str(Path('/tmp/electro_shared_input_preflight.py')),'cpu_tests':'Four comparison/legacy-online checks passed in6.417s','no_training':True}
Path('/root/nonlocal-pinn/artifacts/electro_shared_input_qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
