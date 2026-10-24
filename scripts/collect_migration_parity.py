"""Read-only fixed-input migration audit; creates only the requested output files."""
import contextlib,importlib,io,json,sys,tempfile
from pathlib import Path
import equinox as eqx
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
project,variant,output=sys.argv[1:]
def mod(original,current):return importlib.import_module(original if variant=='source' else current)
a={}
if project=='nonlocal':
 ops=mod('pinn.operators','nonlocal_pinn.operators')
 arch=mod('pinn.arch','nonlocal_pinn.nn.arch')
 q=np.linspace(0.,1.,7);xx,yy=np.meshgrid(q,q,indexing='xy');x=np.column_stack((xx.ravel(),yy.ravel()))
 D=ops.SparseD(x,.45,'cubic');v=jnp.sin(jnp.asarray(x[:,0])*2)+jnp.asarray(x[:,1])**2
 a['scalar_matrix']=D.to_scipy().toarray();a['scalar_matvec']=np.asarray(D@v)
 a['scalar_gradient']=np.asarray(jax.grad(lambda z:jnp.sum((D@z)**2))(v))
 elastic=ops.ElasticD(x,.45,dV=(1/6)**2,kernel='cubic')
 u=jnp.column_stack((jnp.asarray(x[:,0])**2,jnp.asarray(x[:,0])*jnp.asarray(x[:,1])))
 a['elastic_residual']=np.asarray(elastic.elastic_matvec(u,3.,2.))
 net=arch.make_pinn(2,1,hidden_dim=8,num_layers=2,trunk_type='mlp',activation=jnp.tanh,use_spatial_emb=True,use_temporal_emb=True,spatial_emb_dim=8,spatial_scale=2.,temporal_emb_dim=8,temporal_scale=1.,static=False,tensor_split=True,multi_head=False,film_hidden_dim=8,film_num_layers=1,key=jax.random.PRNGKey(9))
 a['network_output']=np.asarray(jax.vmap(lambda p:net(p,jnp.array([.2])))(jnp.asarray(x)))
 a['network_jacobian']=np.asarray(jax.jacfwd(lambda p:net(p,jnp.array([.2])))(jnp.asarray(x[10])))
 cases=['icemelting3d','electropolish2d','elasticity2d','corrosion1d','corrosion2d']
else:
 from tests.test_fracture2d_local_mechanics import model
 from tests.test_fracture2d_budget_learning_rate import controlled
 fem=mod('examples.fracture2d.fem_damage','fracture_pinn.damage.fem')
 mesh=mod('examples.fracture2d.mesh','fracture_pinn.mesh.preparation')
 history=mod('examples.fracture2d.history_training','fracture_pinn.training.history')
 m=model();op=fem.P1DamageOperator(m)
 d=jnp.array([.1,.2,.3,.4]);H=jnp.array([1.,2.,3.,4.]);q=np.arange(8)*1e-5
 a['mechanical_stiffness']=m.matrix(np.asarray(d)).toarray();a['tensile_driver']=np.asarray(m.driver(q))
 a['damage_stiffness']=op.stiffness.toarray()
 energy=fem.make_energy(op,m.volumes,gc0=3.,gc1=.7,floor=1e-15)
 a['energy']=np.asarray(energy(d,H));a['energy_gradient']=np.asarray(jax.grad(energy)(d,H))
 with tempfile.TemporaryDirectory() as t:
  c=controlled(Path(t));m,op,_=mesh.load_geometry(c);p=history.HistoryDamagePINN(c,m,op)
  a['history_initial_output']=np.asarray(p.image(p.net,p.context(H)))
  p.begin_load(1,.0001)
  with contextlib.redirect_stdout(io.StringIO()):p.solve(H,jnp.zeros(4),.0001)
  a['history_updated_output']=np.asarray(p.image(p.net,p.context(H)))
  for i,v in enumerate(jax.tree.leaves(eqx.filter(p.net,eqx.is_array))):a[f'network_leaf_{i}']=np.asarray(v)
  for i,v in enumerate(jax.tree.leaves(p.state)):a[f'optimizer_leaf_{i}']=np.asarray(v)
 cases=['fracture1d','fracture2d','fracture2d_shear','fracture2d_lshape']
configs={}
for case in cases:
 for f in sorted((Path.cwd()/'examples'/case/'configs').glob('*.py')):
  if f.name=='__init__.py':continue
  name=f'examples.{case}.configs.{f.stem}'
  config=importlib.import_module(name).get_config()
  configs[name]=config.to_dict()
p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(p,**a)
p.with_suffix('.configs.json').write_text(json.dumps(configs,sort_keys=True,indent=2,default=repr)+'\n')
print(json.dumps({'project':project,'variant':variant,'arrays':len(a),'configs':len(configs),'output':str(p)}))
