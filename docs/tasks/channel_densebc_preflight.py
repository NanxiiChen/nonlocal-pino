"""Input-only finite-prior dense-BC screen and production encoder parity."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.problem import boundary,boundary_loss_quadrature,quadrature,validate_config
from examples.channel_nsch2d.operators import first_derivative_stencils
from examples.channel_nsch2d.online import validate_construction_screen,build_input


def check(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    c=importlib.import_module(a.config).get_config();c.train.seed=a.seed
    old=json.loads((Path(a.control)/'config.json').read_text())
    c.reference.root=old['reference']['root'];c.log.run_name=old['log']['run_name']
    validate_config(c);before=json.loads(c.to_json());del before['boundary_sampling']
    if before!=old:raise ValueError('Uniform dense-BC control may change only boundary_sampling')
    prior=validate_construction_screen(c,a.construction_screen)
    cases=json.loads(Path(a.construction_screen).read_text())['cases']
    start=time.perf_counter();reports=[]
    for row in cases:
        amp=np.asarray(row['amplitudes']);p,_=quadrature(amp,c,c.sample.Nx,c.sample.Ny)
        b,w=boundary_loss_quadrature(amp,c)
        _,_,_,r=first_derivative_stencils(p,b[:,:2],c)
        if not np.all(w>0):raise ValueError('Boundary measures must be positive')
        reports.append(dict(kind=row['kind'],index=row['index'],amplitudes=amp.tolist(),gradient=r))
    # Actual schema/casting uses the same builder as background workers.
    fresh,construction=build_input(c.to_dict(),0)
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.network import build_network,encode,fields
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(a.seed));template=build_network(c,key)
    model=eqx.tree_deserialise_leaves(Path(a.control)/'model_initial.eqx',template)
    previous,_=geometry_data(fresh['amplitudes'],ConfigDict(old))
    new,_=geometry_data(fresh['amplitudes'],c)
    np.testing.assert_array_equal(new['encoder_boundary'],previous['boundary'])
    np.testing.assert_array_equal(new['encoder_boundary_weights'],previous['boundary_weights'])
    for name in previous:
        if name not in ('boundary','boundary_weights','boundary_neighbors','boundary_gradient_x','boundary_gradient_y'):
            np.testing.assert_array_equal(new[name],previous[name])
    convert=lambda d:{k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else c.train.compute_dtype) for k,v in d.items()}
    dn,do=convert(new),convert(previous)
    en,eo=encode(model,dn),encode(model,do)
    for x,y in zip(jax.tree.leaves(en),jax.tree.leaves(eo),strict=True):np.testing.assert_array_equal(x,y)
    f=eqx.filter_jit(lambda n,e,x:fields(n,e,x,jnp.array([0.,.5,3.,6.]),dn['amplitudes'],c))
    x=dn['points'][::977]
    actual,expected=np.asarray(f(model,en,x)),np.asarray(f(model,eo,x))
    np.testing.assert_array_equal(actual,expected)
    result=dict(status='passed',config=a.config,control=a.control,seed=a.seed,
        existing_laplacian_prior_screen=a.construction_screen,existing_screen=prior,
        new_boundary_gradient_cases=len(reports),
        maximum_boundary_condition=max(r['gradient']['maximum_condition'] for r in reports),
        maximum_boundary_polynomial_error=max(r['gradient']['polynomial_error'] for r in reports),
        production_input_arrays=len(fresh),production_nodes=len(fresh['points']),
        boundary_loss_points=len(fresh['boundary']),encoder_boundary_points=len(fresh['encoder_boundary']),
        physical_fields_maximum_gap=float(np.max(np.abs(actual-expected))),
        encoder_and_unchanged_domain_arrays_exact=True,seconds=time.perf_counter()-start,
        pde_solves=0,fem_solves=0,cases=reports,construction=construction)
    (output/'preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','existing_screen','construction')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','control','construction-screen','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',required=True,type=int);check(p.parse_args())
