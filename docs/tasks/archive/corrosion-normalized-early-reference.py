"""Independent short-horizon probe; no changes to the training reference.

All nondimensional rate coefficients are linear in Tc; shorter Tc samples
exactly the same physical autonomous ODE on an earlier time window.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from examples.corrosion2d.configs.residual_pilot import get_config
from examples.corrosion2d.residual_experiment import boundary_values,load_reference,nondim,solve_reference


def main():
    c=get_config('corrosion2d','phase',42);ref=load_reference(c)
    points=np.stack([ref['X'].ravel(),ref['Y'].ravel()],axis=-1)
    row,col,w=(ref[k] for k in ('graph_row','graph_col','graph_weight'))
    n=len(points);matrix=sparse.diags(np.bincount(row,weights=w,minlength=n))-sparse.csr_matrix((w,(row,col)),shape=(n,n))
    z,o,_=boundary_values(c,points);free=~(z|o);cle=c.physical.c_le;dc=1-cle
    base=np.stack([ref['phi'][0].ravel(),ref['c'][0].ravel()],axis=-1)
    oldnd=np.array(nondim(c));records=[]
    for horizon in (5e-6,.05):
        c.nondim.T_c=horizon
        np.testing.assert_allclose(np.asarray(nondim(c))[:4]/oldnd[:4],horizon/50,rtol=1e-14)
        times,fields,info=solve_reference(c,matrix,points,1e-8,1e-11,.01)
        np.testing.assert_allclose(fields[0],base,atol=3e-7,rtol=0)
        q=fields[...,1]-dc*(3*fields[...,0]**2-2*fields[...,0]**3)-cle
        cl=q+cle
        rows=[]
        for i,t in enumerate(np.linspace(0,horizon,c.experiment.reference_snapshots)):
            rows.append({'physical_time_seconds':float(t),'phi_change_rms_free':float(np.sqrt(np.mean((fields[i,free,0]-base[free,0])**2))),
                         'c_change_rms_free':float(np.sqrt(np.mean((fields[i,free,1]-base[free,1])**2))),
                         'cL_min_free':float(cl[i,free].min()),'cL_max_free':float(cl[i,free].max()),
                         'fraction_cL_outside_current_map_by_1e-6':float(np.mean((cl[i,free]<-1e-6)|(cl[i,free]>cle+1e-6)))})
        records.append({'horizon_seconds':horizon,'solver_info':info,'rows':rows})
    output={'scope':'Same physical graph ODE rescaled only in time; independent early-time diagnostic, no training/reference-file replacement','records':records}
    Path(__file__).with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)

if __name__=='__main__':main()
