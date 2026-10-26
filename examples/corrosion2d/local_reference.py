"""Independent centered-difference local PDE reference, with natural no flux."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from scipy import sparse

from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.configs.train_base import get_config


def local_spec(config):
    return json.loads(json.dumps({
        "case": "corrosion2d", "physical": config.physical.to_dict(),
        "nondim": config.nondim.to_dict(), "pits": config.pits.to_dict(),
        "domain": [[0.,1.],[0.,.5]], "operator": "local negative Laplacian",
        "dirichlet": "bottom open pit mouth zero; left/right/top one",
        "other_boundary": "bottom outside mouth: d_n phi=d_n q=0",
        "initial_condition": "analytic pit profile,c=c_se*h(phi); Dirichlet overrides",
    }))


def negative_laplacian(nx,ny):
    if nx<3 or ny<3:raise ValueError("Need at least3 nodes per direction")
    def axis(n,length):
        h=length/(n-1)
        a=sparse.diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='lil')/h**2
        a[0,1]=-2/h**2;a[-1,-2]=-2/h**2
        return a.tocsr()
    return (sparse.kron(sparse.eye(ny),axis(nx,1.))+
            sparse.kron(axis(ny,.5),sparse.eye(nx))).tocsr()


def solve_level(config,nx,rtol,atol,max_step):
    c=config.copy_and_resolve_references()
    c.sample.Nx=nx;c.sample.Ny=(nx+1)//2
    xx,yy,points=r.grid(c)
    matrix=negative_laplacian(nx,c.sample.Ny)
    times,fields,info=r.solve_reference(c,matrix,points,rtol,atol,max_step)
    return times,fields.reshape(len(times),*xx.shape,2),info


def generate(config,path,workdir):
    if jax.default_backend()!='cpu':raise RuntimeError('Local reference is CPU-only')
    jax.config.update('jax_enable_x64',True)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    workdir.mkdir(parents=True,exist_ok=False)
    levels={};records={}
    for name,nx,rtol,atol,step in (
        ('grid101',101,1e-7,1e-10,.01),
        ('grid201',201,1e-7,1e-10,.01),
        ('grid201_timefine',201,1e-8,1e-11,.005),
        ('grid401',401,1e-8,1e-11,.005)):
        print(f'LOCAL REFERENCE {name}: nx={nx}',flush=True)
        times,fields,info=solve_level(config,nx,rtol,atol,step)
        stride=(nx-1)//100
        levels[name]=fields[:,::stride,::stride]
        records[name]={'nx':nx,'ny':(nx+1)//2,**info}
        np.savez_compressed(workdir/f'{name}.npz',time=times,fields=fields)
        (workdir/'progress.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    def errors(a,b):
        return {key:{'terminal':float(np.linalg.norm(a[-1,...,i]-b[-1,...,i])/np.linalg.norm(b[-1,...,i])),
                     'pooled':float(np.linalg.norm(a[...,i]-b[...,i])/np.linalg.norm(b[...,i])),
                     'max_per_time':float(max(np.linalg.norm(u-v)/np.linalg.norm(v) for u,v in zip(a[...,i],b[...,i],strict=True)))}
                for i,key in enumerate(('phi','c'))}
    refinement={
        'time201':errors(levels['grid201'],levels['grid201_timefine']),
        'space101_201':errors(levels['grid101'],levels['grid201_timefine']),
        'space201_401':errors(levels['grid201_timefine'],levels['grid401'])}
    # Preserve measured uncertainty; do not silently label coarse data converged.
    metadata={'spec':local_spec(config),'method':'independent SciPy BDF + centered FD/Neumann mirror; analytic Jacobian',
              'levels':records,'refinement':refinement,'evaluation_grid':[101,51],
              'time_refinement_pass':all(v['max_per_time']<1e-4 for v in refinement['time201'].values()),
              'space_refinement_below_1e_3':all(v['max_per_time']<1e-3 for v in refinement['space201_401'].values()),
              'source_grid':[401,201]}
    if not metadata['time_refinement_pass']:raise ValueError(f'Local time refinement failed: {refinement}')
    c=config.copy_and_resolve_references();c.sample.Nx=101;c.sample.Ny=51
    xx,yy,_=r.grid(c);fields=levels['grid401']
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream,X=xx,Y=yy,time=times,physical_time=times*config.nondim.T_c,
                            phi=fields[...,0],c=fields[...,1],metadata_json=np.array(json.dumps(metadata)))
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


def load(config):
    path=Path(config.experiment.local_reference_path)
    with np.load(path) as data:ref={k:data[k] for k in data.files}
    metadata=json.loads(str(ref['metadata_json']))
    if metadata['spec']!=local_spec(config):raise ValueError('Local reference physical/BC/IC specification mismatch')
    if not metadata['time_refinement_pass']:raise ValueError('Local reference time refinement failed')
    for key in ('phi','c'):
        if ref[key].shape!=(21,51,101) or not np.isfinite(ref[key]).all():raise ValueError(f'Invalid local {key}')
    if not np.array_equal(ref['time'],np.linspace(0,1,21)):raise ValueError('Local snapshot times mismatch')
    xx,yy,points=r.grid(config)
    if not np.array_equal(xx,ref['X']) or not np.array_equal(yy,ref['Y']):raise ValueError('Local evaluation grid mismatch')
    zero,one,_=r.boundary_values(config,points)
    if r.boundary_error(np.stack([ref['phi'],ref['c']],-1),zero,one)>1e-10:raise ValueError('Local reference Dirichlet mismatch')
    return ref,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workdir',required=True,type=Path)
    args=parser.parse_args()
    generate(get_config(),args.output,args.workdir)


if __name__=='__main__':main()
