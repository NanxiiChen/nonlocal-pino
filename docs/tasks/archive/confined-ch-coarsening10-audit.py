"""Selected richer IC/operator audit; CPU only, no reference or run is rewritten."""
from pathlib import Path
import json,runpy
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import jax
from examples.confined_ch2d.configs.train_coarsening import get_config
from examples.confined_ch2d.problem import initial_field,initial_laplacian,relative
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.geometry import quadrature


def main():
    out=Path(__file__).parent;c=get_config();root=Path(c.reference.root)
    helpers=runpy.run_path(str(out/'confined-ch-coarsening-morphology.py'))
    d=helpers['read'](root/'preflight.npz');x,v=d['points'],d['volumes']
    phi=initial_field(x,c,xp=np)
    exact=dict(d);exact['phi']=np.array([phi]);exact['time']=np.array([0.])
    initial=helpers['morphology'](exact,'phi')[0]
    A=native_matrix(x,v,c)
    with jax.enable_x64(True):
        D=RegionalLaplacian(x,v,c)
        index=np.asarray(D.neighbors);weights=np.asarray(D.coefficients)
    off=sparse.coo_matrix((weights.ravel(),(np.repeat(np.arange(len(x)),index.shape[1]),index.ravel())),shape=A.shape).tocsr()
    B=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
    diff=A-B;sym=sparse.diags(v)@A;sym=sym-sym.T
    components,_=connected_components(A,directed=False)
    result=dict(config=c.to_dict(),points=len(x),analytic_initial=initial,
        preflight_initial_interpolation_l2=relative(d['phi'][0],phi,v),
        preflight_mass_drift=float(np.max(abs(d['mass']-d['mass'][0]))),
        preflight_energy_initial=float(d['energy'][0]),preflight_energy_terminal=float(d['energy'][-1]),
        preflight_energy_max_increase=max(0.,float(np.diff(d['energy']).max())),
        operator=dict(independent_matrix_max_absolute=float(np.max(abs(diff.data))) if diff.nnz else 0.,
            weighted_asymmetry_max_absolute=float(np.max(abs(sym.data))) if sym.nnz else 0.,
            constant_annihilation_max_absolute=float(np.max(abs(A@np.ones(len(x))))),
            connected_components=int(components),neighbors_min=int(np.min(np.sum(weights>0,axis=1))),
            neighbors_max=int(np.max(np.sum(weights>0,axis=1)))))
    # Independent integral quadrature at fixed queries/horizon, plus local-limit
    # IC diagnostic. This is not a proof of native evolution convergence.
    fine=helpers['read'](root/'fem_refined_dt-fields.npz')
    y,w=quadrature(fine['mesh_points'],fine['mesh_cells'])
    integral=runpy.run_path(str(out/'confined-ch-boundary-operator.py'))['integral_at']
    result['operator_consistency_scope']='Analytic initial field only; fixed query points, visible physical-volume integral. Not native evolution spatial convergence.'
    result['fine_quadrature_points']=len(y)
    result['fine_quadrature_mesh_size']=c.reference.mesh_sizes[1]
    result['horizons']={}
    for delta in (.04,.03,.02):
        coarse=integral(x,x,v,c,delta);refined=integral(x,y,w,c,delta)
        result['horizons'][str(delta)]=dict(coarse_fine_quadrature_relative=relative(coarse,refined,v),
            fine_integral_local_relative=relative(refined,-initial_laplacian(x,c),v))
    assert result['operator']['independent_matrix_max_absolute']<1e-9
    assert result['operator']['weighted_asymmetry_max_absolute']<1e-12
    assert result['operator']['constant_annihilation_max_absolute']<1e-9
    assert components==1
    (out/'confined-ch-coarsening10-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:val for k,val in result.items() if k not in ('config','analytic_initial')},indent=2))


if __name__=='__main__':main()
