"""Independent regional integral refinement on an analytic Neumann-compatible field."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from examples.confined_ch2d.configs.train_base import get_config
from examples.confined_ch2d.geometry import quadrature
from examples.confined_ch2d.problem import initial_field,initial_laplacian,relative


def integral_at(x,y,volumes,c,delta):
    rows=[];cols=[]
    for i,neighbors in enumerate(cKDTree(y).query_ball_point(x,delta)):
        rows.extend([i]*len(neighbors));cols.extend(neighbors)
    rows,cols=np.asarray(rows),np.asarray(cols)
    start=x[rows];end=y[cols];direction=end-start;r2=np.sum(direction*direction,axis=1)
    keep=np.ones(len(rows),dtype=bool)
    for cx,cy,r in c.geometry.holes:
        projection=np.zeros(len(rows));np.divide(np.sum((np.array([cx,cy])-start)*direction,axis=1),r2,out=projection,where=r2>0)
        closest=start+np.clip(projection,0,1)[:,None]*direction
        keep &= np.sum((closest-np.array([cx,cy]))**2,axis=1)>=r*r
    weights=120/(np.pi*delta**4)*(1-np.sqrt(r2)/delta)**2*volumes[cols]*keep
    difference=initial_field(x,c,xp=np)[rows]-initial_field(y,c,xp=np)[cols]
    return np.bincount(rows,weights=weights*difference,minlength=len(x))


def main():
    c=get_config();root=Path(c.reference.root)
    with np.load(root/'geometry.npz') as f:x=f['points'];v=f['volumes']
    with np.load(root/'fem_fine-fields.npz') as f:y,w=quadrature(f['mesh_points'],f['mesh_cells'])
    target=-initial_laplacian(x,c)
    distance=.5*min(c.geometry.axes)*(1-np.sum(((x-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes))**2,axis=1))
    for cx,cy,r in c.geometry.holes:distance=np.minimum(distance,np.linalg.norm(x-np.array([cx,cy]),axis=1)-r)
    result=dict(scope='Manufactured IC at shared query points; integral vs analytic negative Laplacian. Not a native evolution spatial-convergence proof.',coarse_points=len(x),fine_points=len(y),horizons={})
    for delta in (.09,.07,.05):
        coarse=integral_at(x,x,v,c,delta);fine=integral_at(x,y,w,c,delta)
        boundary=distance<delta;interior=~boundary
        result['horizons'][str(delta)]=dict(coarse_fine_quadrature_relative=relative(coarse,fine,v),
            fine_integral_local_relative=relative(fine,target,v),
            boundary_relative=relative(fine[boundary],target[boundary],v[boundary]),
            interior_relative=relative(fine[interior],target[interior],v[interior]))
    result['scope']+=' Boundary truncation/visibility, physical CAD polygon quadrature and horizon remain combined effects.'
    out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
