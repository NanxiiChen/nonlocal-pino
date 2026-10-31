"""Continuous display reconstruction on the ORIGINAL perforated mesh.

Area-weight cell values onto incident vertices, then interpolate linearly inside
the original triangles. This is display-only; no metric/reference is replaced.
The positive reconstruction preserves constants and cell-value bounds. Original
mesh topology and analytic circle masks prevent interpolation through holes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,diags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle


class SmoothMeshDisplay:
    def __init__(self,vertices,cells,areas,holes,*,extent,resolution):
        vertices,cells,areas,holes=map(np.asarray,(vertices,cells,areas,holes))
        if (vertices.ndim!=2 or vertices.shape[1]!=2 or cells.ndim!=2 or cells.shape[1]!=3
                or areas.shape!=(len(cells),) or holes.ndim!=2 or holes.shape[1]!=3
                or not np.issubdtype(cells.dtype,np.integer)):
            raise ValueError('Display requires XY vertices, integer triangles, positive cell areas and circle rows')
        if (not all(np.isfinite(a).all() for a in (vertices,areas,holes)) or np.any(areas<=0)
                or np.any(cells<0) or np.any(cells>=len(vertices)) or np.any(holes[:,2]<=0)):
            raise ValueError('Invalid display mesh/measure/circle values')
        if (len(extent)!=4 or not np.isfinite(extent).all() or extent[0]>=extent[1] or extent[2]>=extent[3]
                or len(resolution)!=2 or any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in resolution)):
            raise ValueError('Display requires ordered finite extent and two integer resolutions>=2')
        self.tri=mtri.Triangulation(vertices[:,0],vertices[:,1],cells)
        measure=csr_matrix((np.repeat(areas,3),(cells.ravel(),np.repeat(np.arange(len(cells)),3))),
                           shape=(len(vertices),len(cells)))
        sums=np.asarray(measure.sum(axis=1)).ravel()
        if np.any(sums<=0):raise ValueError('Every display mesh vertex must have incident physical cells')
        self.projection=diags(1/sums)@measure
        self.xx,self.yy=np.meshgrid(np.linspace(*extent[:2],resolution[0]),
                                    np.linspace(*extent[2:],resolution[1]),indexing='xy')
        self.mask=self.tri.get_trifinder()(self.xx,self.yy)<0
        for cx,cy,r in holes:
            self.mask|=(self.xx-cx)**2+(self.yy-cy)**2<=r*r
        self.extent=tuple(extent)

    def __call__(self,cell_values):
        values=np.asarray(cell_values)
        if values.ndim!=2 or values.shape[1]!=self.projection.shape[1] or not np.isfinite(values).all():
            raise ValueError('Display fields must be finite(time,physical cells) arrays')
        vertices=np.asarray(self.projection@values.T).T
        fields=np.ma.stack([mtri.LinearTriInterpolator(self.tri,row)(self.xx,self.yy) for row in vertices])
        return np.ma.array(fields,mask=np.ma.getmaskarray(fields)|self.mask[None,:,:])


def reference_gallery(dataset,output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((dataset/'manifest.json').read_text())
    config=json.loads((dataset/'config.json').read_text())
    if manifest['status']!='passed':raise ValueError('A validated reference bank is required')
    ids=manifest['test_indices'];chosen=list(dict.fromkeys((ids[0],ids[len(ids)//2],ids[-1])))
    data=[]
    for index in chosen:
        with np.load(dataset/manifest['conditions'][index]['file'],allow_pickle=False) as f:
            data.append({key:f[key] for key in f.files})
    bound=max(float(abs(d['phi']).max()) for d in data)
    fig,axes=plt.subplots(len(chosen),len(data[0]['time']),figsize=(14,2.9*len(chosen)),
                          layout='constrained',squeeze=False)
    for row,(index,d) in enumerate(zip(chosen,data,strict=True)):
        display=SmoothMeshDisplay(d['mesh_points'],d['mesh_cells'],d['volumes'],d['holes'],
                                  extent=(0.,1.,.07,.93),resolution=(400,320))
        values=display(d['phi'])
        for column,t in enumerate(d['time']):
            ax=axes[row,column]
            im=ax.imshow(values[column],origin='lower',extent=display.extent,
                cmap='RdBu_r',vmin=-bound,vmax=bound,interpolation='nearest')
            for cx,cy,r in d['holes']:ax.add_patch(Circle((cx,cy),r,fill=False,edgecolor='black',linewidth=.6))
            ax.set(aspect='equal',xticks=[],yticks=[])
            if row==0:ax.set_title(f't = {t:.4g}')
            if column==0:ax.set_ylabel(f'Geometry {index}')
    fig.colorbar(im,ax=list(axes.ravel()),shrink=.8,pad=.01,label='Phase phi')
    fig.suptitle(f"{len(config['initial']['waves'])}-mode fixed ambient IC: native reference on different geometries\nContinuous mesh interpolation for display only")
    fig.savefig(output/'reference_fields.png',dpi=200);plt.close(fig)
    record=dict(dataset=str(dataset),indices=chosen,phase_bound=bound,
                nodes=[len(d['points']) for d in data],display='Area-weighted incident-cell projection and original-triangle linear interpolation; raw metrics unchanged')
    (output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(reference_gallery(args.dataset,args.output),indent=2))
