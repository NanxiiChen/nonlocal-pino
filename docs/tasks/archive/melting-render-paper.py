"""Publication-style 3D interfaces from the saved actual phi=0 meshes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
import numpy as np
import pyvista as pv

OUT=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/data/icemelting3d/operator_render3d_v1')
DATA=Path('/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1')
manifest=json.loads((DATA/'manifest.json').read_text())
indices=(5,9,2)
frames=(0,15,30,45,60)
seconds=(0,1.25,2.5,3.75,5.)
colors=('#0072B2','#56B4E9','#009E73','#E69F00','#D55E00')
ELEV,AZIM=23,-57
view=np.array([np.cos(np.deg2rad(ELEV))*np.cos(np.deg2rad(AZIM)),
               np.cos(np.deg2rad(ELEV))*np.sin(np.deg2rad(AZIM)),np.sin(np.deg2rad(ELEV))])
light=np.array([-1.,-1.,2.]);light/=np.linalg.norm(light)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':9,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','axes.linewidth':.5})
meshes={}
for index in indices:
    for frame in frames:
        with np.load(ROOT/'meshes'/f'case{index:03d}-frame{frame:03d}.npz') as source:
            v,f=source['vertices'].astype(float),source['faces']
        assert v.shape[1:]==(3,) and f.shape[1:]==(3,) and np.isfinite(v).all()
        meshes[index,frame]=(v,f)


def setup(ax):
    ax.patch.set_alpha(0)
    ax.set_proj_type('ortho')
    ax.view_init(elev=ELEV,azim=AZIM)
    for key in ('x','y','z'):
        getattr(ax,'set_'+key+'lim')(.08,.92)
        getattr(ax,'set_'+key+'ticks')([.2,.5,.8])
        getattr(ax,'set_'+key+'label')(rf'${key}^*$',labelpad=-3)
        axis=getattr(ax,key+'axis')
        axis.set_rotate_label(False)
        axis.label.set_rotation(0)
        axis.pane.fill=False
        axis.pane.set_edgecolor('#ffffff')
        axis.line.set_color('#8b969f')
        axis._axinfo['grid'].update(color=(.86,.88,.90,1),linewidth=.35,linestyle=':')
    ax.set_box_aspect((1,1,1),zoom=1.12)
    ax.tick_params(pad=-2,width=.4,length=2)


def silhouette(v,f):
    normal=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    front=(normal@view)>0
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    owners=np.tile(np.arange(len(f)),3)
    order=np.lexsort((edges[:,1],edges[:,0]))
    edges,owners=edges[order],owners[order]
    assert len(edges)%2==0
    np.testing.assert_array_equal(edges[::2],edges[1::2])
    keep=front[owners[::2]]!=front[owners[1::2]]
    return v[edges[::2][keep]]


def surface(ax,v,f,color,outline=True):
    if not len(f):return
    triangles=v[f]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    intensity=.60+.40*np.maximum(normals@light,0)
    facecolors=np.clip(np.array(to_rgb(color))[None,:]*intensity[:,None],0,1)
    collection=Poly3DCollection(triangles,facecolors=facecolors,edgecolors='none',
        linewidth=0,antialiased=False,zsort='average',rasterized=True)
    ax.add_collection3d(collection,autolim=False)
    if outline:
        ax.add_collection3d(Line3DCollection(silhouette(v,f),colors='#17384b',linewidths=.4),autolim=False)


def save(fig,stem):
    for extension,dpi in (('png',450),('pdf',600),('svg',450)):
        fig.savefig(OUT/(stem+'.'+extension),dpi=dpi,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

# A: the same physical volume at five times, matte surface plus thin silhouette.
fig,axes=plt.subplots(3,5,figsize=(8.0,5.9),subplot_kw={'projection':'3d'})
fig.subplots_adjust(left=.035,right=.995,bottom=.035,top=.94,wspace=.16,hspace=.18)
for row,index in enumerate(indices):
    for col,frame in enumerate(frames):
        ax=axes[row,col];setup(ax)
        surface(ax,*meshes[index,frame],color='#6CA6C7')
        if row==0:ax.set_title(f't = {seconds[col]:g} s',pad=1)
        if col==0:ax.text2D(-.12,.92,f'Sample {index}',transform=ax.transAxes,fontsize=8,fontweight='bold')
save(fig,'melting-paper-3d-snapshots')

# Main-paper sized strip for the more elongated of the existing three samples.
fig,axes=plt.subplots(1,5,figsize=(7.2,1.9),subplot_kw={'projection':'3d'})
fig.subplots_adjust(left=.02,right=.995,bottom=.08,top=.85,wspace=.18)
for col,frame in enumerate(frames):
    setup(axes[col]);surface(axes[col],*meshes[2,frame],color='#6CA6C7')
    axes[col].set_title(f'({chr(97+col)})  t = {seconds[col]:g} s',pad=0)
save(fig,'melting-paper-3d-single')

# B: time-colored interface contours in one fixed 3D coordinate frame.
fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),subplot_kw={'projection':'3d'})
fig.subplots_adjust(left=.02,right=.995,bottom=.19,top=.92,wspace=.1)
for ax,index in zip(axes,indices):
    setup(ax);ax.set_title(f'Sample {index}',pad=0)
    center=manifest['parameters'][index][:3]
    for frame,color in zip(frames,colors):
        v,f=meshes[index,frame]
        if not len(f):continue
        cells=np.column_stack([np.full(len(f),3),f]).ravel()
        mesh=pv.PolyData(v,cells)
        ax.add_collection3d(Line3DCollection(silhouette(v,f),colors=color,linewidths=.8),autolim=False)
        for normal in np.eye(3):
            cut=mesh.slice(normal=normal,origin=center)
            lines=cut.lines
            segments=[];offset=0
            while offset<len(lines):
                n=int(lines[offset]);ids=lines[offset+1:offset+n+1]
                segments.extend(cut.points[np.column_stack([ids[:-1],ids[1:]])])
                offset+=n+1
            if segments:
                ax.add_collection3d(Line3DCollection(segments,colors=color,linewidths=.4,alpha=.65),autolim=False)
handles=[Line2D([0],[0],color=color,lw=1.3,label=f'{t:g} s') for color,t in zip(colors,seconds)]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=5,frameon=False,fontsize=8,handlelength=1.6)
save(fig,'melting-paper-3d-overlay')

# C: one shared collection sorts every nested triangle by depth correctly.
fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),subplot_kw={'projection':'3d'})
fig.subplots_adjust(left=.02,right=.995,bottom=.19,top=.92,wspace=.16)
for ax,index in zip(axes,indices):
    setup(ax);ax.set_title(f'Sample {index}',pad=0)
    center=manifest['parameters'][index][:3]
    all_triangles=[];all_colors=[]
    for frame,color in zip(frames,colors):
        v,f=meshes[index,frame]
        if not len(f):continue
        if frame!=frames[-1]:
            cells=np.column_stack([np.full(len(f),3),f]).ravel()
            mesh=pv.PolyData(v,cells)
            left=mesh.clip(normal=(1,0,0),origin=center,invert=True)
            back=mesh.clip(normal=(1,0,0),origin=center,invert=False).clip(normal=(0,1,0),origin=center,invert=False)
            parts=[part.triangulate() for part in (left,back)]
            geometry=[(part.points,part.faces.reshape(-1,4)[:,1:]) for part in parts if part.n_cells]
        else:
            geometry=[(v,f)]
        for points,faces in geometry:
            triangles=points[faces]
            normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            length=np.linalg.norm(normals,axis=1)
            assert (length>0).all()
            normals/=length[:,None]
            intensity=.60+.40*np.maximum(normals@light,0)
            all_triangles.append(triangles)
            all_colors.append(np.clip(np.array(to_rgb(color))[None,:]*intensity[:,None],0,1))
    collection=Poly3DCollection(np.concatenate(all_triangles),facecolors=np.concatenate(all_colors),
        edgecolors='none',linewidth=0,antialiased=False,zsort='average',rasterized=True)
    ax.add_collection3d(collection,autolim=False)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=5,frameon=False,fontsize=8,handlelength=1.6)
save(fig,'melting-paper-3d-cutaway')

report=dict(samples=list(indices),frames=list(frames),physical_times=list(seconds),camera=dict(elevation=ELEV,azimuth=AZIM,projection='orthographic'),
    bounds=[.08,.92],axes='Dimensionless x*,y*,z*, equal aspect, same scale at every time',
    sources='Saved actual phi=0 meshes in '+str(ROOT/'meshes'),
    outputs=['melting-paper-3d-'+s+'.'+e for s in ('snapshots','single','overlay','cutaway') for e in ('png','pdf','svg')],
    surface='Matte diffuse shading only; no specular highlights or geometric smoothing.',
    overlay='True mesh silhouettes and three orthogonal intersection curves, time colors.',
    cutaway='Display-only front quadrant removal on outer four times exposes nesting; not a physical hole or changed field.',
    pdf_svg='Vector labels/axes/curves; dense filled surface layers rasterized at high dpi. Overlay curves fully vector.',
    scope='Publication visualization only, original dense prediction/mesh values unchanged; no new accuracy claim.')
(OUT/'melting-paper-3d-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
