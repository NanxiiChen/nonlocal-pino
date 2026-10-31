"""Gmsh geometry/precomputed physical quadrature; no training dependencies."""
from pathlib import Path
import json,sys,subprocess,argparse
import numpy as np


def _generate(c,h,path,*,arrays=False):
    """Keep Gmsh MPI/PETSc shared libraries out of the FEniCS process."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('confined_ch')
        x,y=c.geometry.center;a,b=c.geometry.axes
        outer=gmsh.model.occ.addDisk(x,y,0,a,b)
        holes=[(2,gmsh.model.occ.addDisk(x,y,0,r,r)) for x,y,r in c.geometry.holes]
        surfaces,_=gmsh.model.occ.cut([(2,outer)],holes)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2,[v for dim,v in surfaces],1)
        gmsh.option.setNumber('Mesh.MeshSizeMin',h)
        gmsh.option.setNumber('Mesh.MeshSizeMax',h)
        gmsh.model.mesh.generate(2)
        msh=Path(path).with_suffix('.msh');gmsh.write(str(msh))
        if arrays:
            tags,coordinates,_=gmsh.model.mesh.getNodes()
            types,_,nodes=gmsh.model.mesh.getElements(2)
            if list(types)!=[2]:raise ValueError('Expected linear triangular Gmsh cells')
            tags=np.asarray(tags);order=np.argsort(tags)
            connectivity=np.asarray(nodes[0]).reshape(-1,3)
            positions=np.searchsorted(tags[order],connectivity)
            if np.any(positions>=len(tags)) or not np.array_equal(tags[order][positions],connectivity):
                raise ValueError('Gmsh element references an unknown node')
            return np.asarray(coordinates).reshape(-1,3)[:,:2],order[positions]
    finally:gmsh.finalize()


def create_mesh(c,h,path):
    """Linear CAD triangles, with Gmsh isolated from DOLFIN MPI libraries."""
    import meshio
    path=Path(path)
    request=path.with_suffix('.geometry.json')
    request.write_text(json.dumps(dict(config=c.to_dict(),h=h,path=str(path))))
    subprocess.run([sys.executable,'-m','examples.confined_ch2d.geometry','--input',str(request)],check=True,capture_output=True,text=True,cwd=Path(__file__).resolve().parents[2])
    m=meshio.read(path.with_suffix('.msh'))
    cells=np.concatenate([v.data for v in m.cells if v.type=='triangle'])
    points=m.points[:,:2]
    meshio.write(path,meshio.Mesh(points,[('triangle',cells)]))
    return points,cells


def quadrature(points,cells):
    tri=points[cells]
    volumes=.5*np.abs((tri[:,1,0]-tri[:,0,0])*(tri[:,2,1]-tri[:,0,1])-
        (tri[:,1,1]-tri[:,0,1])*(tri[:,2,0]-tri[:,0,0]))
    if np.any(volumes<=0):raise ValueError('Degenerate triangles')
    return tri.mean(axis=1),volumes


def visible_pairs(points,pairs,holes):
    """Symmetric segment visibility; ellipse is convex, holes are circular."""
    p,q=points[pairs[:,0]],points[pairs[:,1]]
    segment=q-p;length2=np.sum(segment*segment,axis=1)
    keep=np.ones(len(pairs),dtype=bool)
    for x,y,r in holes:
        t=np.clip(np.sum((np.array([x,y])-p)*segment,axis=1)/length2,0,1)
        closest=p+t[:,None]*segment
        keep &= np.sum((closest-np.array([x,y]))**2,axis=1)>=r*r
    return keep


if __name__=='__main__':
    from ml_collections import ConfigDict
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();r=json.loads(args.input.read_text())
    if 'output_arrays' in r:
        points,cells=_generate(ConfigDict(r['config']),r['h'],r['path'],arrays=True)
        np.savez(r['output_arrays'],vertices=points,cells=cells)
    else:
        _generate(ConfigDict(r['config']),r['h'],r['path'])
