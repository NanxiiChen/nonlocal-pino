"""Extract actual predicted interfaces and render fixed-scale 3D media."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reuse-meshes',action='store_true',help='Explicitly reuse already verified saved isosurfaces')
args=parser.parse_args()
OUT=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/data/icemelting3d/operator_render3d_v1')
FRAMES=ROOT/'frames'
MESHES=ROOT/'meshes'
FRAMES.mkdir(exist_ok=False)
if not args.reuse_meshes:
    MESHES.mkdir(exist_ok=False)
indices=(5,9,2)
selected=(0,15,30,45,60)
records=[]
meshes={}
start=time.perf_counter()
if args.reuse_meshes:
    previous=json.loads((ROOT/'render-provenance.json').read_text())
    assert previous['input_indices']==list(indices)
    records=previous['mesh_records']
    times=np.array(previous['times'],dtype=np.float32)
    physical=np.array(previous['physical_times'],dtype=np.float32)
    for case in records:
        for frame in case['frames']:
            with np.load(frame['path']) as saved:
                vertices,faces=saved['vertices'],saved['faces']
                assert saved['time']==times[frame['frame']]
            if len(faces):
                cells=np.column_stack([np.full(len(faces),3),faces]).ravel()
                mesh=pv.PolyData(vertices,cells).compute_normals(cell_normals=False,
                    point_normals=True,consistent_normals=True,auto_orient_normals=True)
                np.testing.assert_array_equal(mesh.points,vertices)
                assert mesh.n_open_edges==0
            else:
                mesh=None
            meshes[case['index'],frame['frame']]=mesh
else:
    for index in indices:
        with np.load(ROOT/f'case{index:03d}-phi.npz') as source:
            phi=source['phi']
            times=source['time']
            physical=source['physical_time']
            axis=source['axis']
        assert phi.shape==(61,96,96,96)
        np.testing.assert_array_equal(times,np.linspace(0,1,61,dtype=np.float32))
        h=1/(len(axis)-1)
        case=[]
        for k,volume in enumerate(phi):
            boundary=np.concatenate([volume[0].ravel(),volume[-1].ravel(),volume[:,0].ravel(),
                volume[:,-1].ravel(),volume[:,:,0].ravel(),volume[:,:,-1].ravel()])
            assert np.isfinite(volume).all() and boundary.max()<0
            if volume.max()<=0:
                vertices=np.empty((0,3),np.float32)
                faces=np.empty((0,3),np.int32)
                mesh=None
                surface_volume=0.
                surface_gap=0.
            else:
                vertices,faces,_,_=marching_cubes(volume,level=0,spacing=(h,h,h),
                    gradient_direction='descent',allow_degenerate=False)
                surface_gap=float(abs(map_coordinates(volume,(vertices/h).T,order=1,mode='nearest')).max())
                assert surface_gap<1e-4,(index,k,surface_gap)
                edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
                _,counts=np.unique(edges,axis=0,return_counts=True)
                assert (counts==2).all(),(index,k,'non-closed surface')
                signed=np.einsum('ij,ij->i',vertices[faces[:,0]],
                    np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6
                if signed<0:
                    faces=faces[:,::-1].copy()
                surface_volume=float(abs(signed))
                assert surface_volume>0
                cells=np.column_stack([np.full(len(faces),3),faces]).ravel()
                mesh=pv.PolyData(vertices,cells).compute_normals(cell_normals=False,
                    point_normals=True,consistent_normals=True,auto_orient_normals=True)
                np.testing.assert_array_equal(mesh.points,vertices)
                assert mesh.n_open_edges==0
            path=MESHES/f'case{index:03d}-frame{k:03d}.npz'
            np.savez_compressed(path,vertices=vertices,faces=faces,time=times[k],physical_time=physical[k])
            if k in selected and mesh is not None:
                mesh.save(MESHES/f'case{index:03d}-frame{k:03d}.vtp')
            meshes[index,k]=mesh
            case.append(dict(frame=k,time=float(times[k]),physical_time=float(physical[k]),
                vertices=len(vertices),triangles=len(faces),surface_volume=surface_volume,
                iso_interpolation_gap=surface_gap,path=str(path)))
        records.append(dict(index=index,frames=case,all_surfaces_closed=True,
            max_iso_interpolation_gap=max(r['iso_interpolation_gap'] for r in case),
            surface_volume_monotone=bool((np.diff([r['surface_volume'] for r in case])<=0).all())))
        print('MESHES',index,case[0]['triangles'],case[-1]['triangles'],flush=True)

font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
bold_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
font=lambda n:ImageFont.truetype(font_path,n)
bold=lambda n:ImageFont.truetype(bold_path,n)
NAVY='#18324d'
BG='#f2f7fc'
CAMERA=[(1.8,-1.5,1.35),(.5,.5,.5),(0,0,1)]
SCALE=.47
p=pv.Plotter(shape=(1,3),off_screen=True,window_size=(1440,480),lighting='none',border=False)
for col,index in enumerate(indices):
    p.subplot(0,col)
    p.set_background(BG)
    for pos,intensity in [((1,-1,2),.8),((-1,1,.7),.45),((1,2,2),.6),((.5,-.5,-1),.2)]:
        p.add_light(pv.Light(position=pos,focal_point=(.5,.5,.5),color='white',intensity=intensity))
    p.camera_position=CAMERA
    p.camera.parallel_projection=True
    p.camera.parallel_scale=SCALE
    p.add_axes(viewport=(.82,.01,.99,.18),line_width=2,color=NAVY)
p.enable_anti_aliasing('ssaa')
rendered=[]
raw_selected={}
foreground=[]
raw_first=None
raw_last=None
for k,t in enumerate(physical):
    for col,index in enumerate(indices):
        p.subplot(0,col)
        mesh=meshes[index,k]
        if mesh is None:
            p.remove_actor('ice',reset_camera=False,render=False)
        else:
            p.add_mesh(mesh,name='ice',color='#8ed2ef',smooth_shading=True,
                pbr=False,ambient=.28,diffuse=.65,specular=.85,specular_power=65,
                reset_camera=False,render=False)
        assert p.camera.parallel_projection and abs(p.camera.parallel_scale-SCALE)<1e-12
    p.render()
    image=p.screenshot(return_img=True,transparent_background=False)
    if k==0:
        renderer_class=p.render_window.GetClassName()
        capabilities=p.render_window.ReportCapabilities()
    assert image.shape==(480,1440,3)
    bg=np.array([242,247,252],dtype=np.int16)
    mask=abs(image.astype(np.int16)-bg).max(axis=2)>18
    foreground.append([int(mask[:,j*480:(j+1)*480].sum()) for j in range(3)])
    if k==0: raw_first=image.copy()
    if k==60: raw_last=image.copy()
    body=Image.fromarray(image)
    draw=ImageDraw.Draw(body)
    ruler=.2/(2*SCALE)*480
    for col,index in enumerate(indices):
        x=col*480+22
        draw.line((x,450,x+ruler,450),fill=NAVY,width=2)
        draw.line((x,446,x,454),fill=NAVY,width=2)
        draw.line((x+ruler,446,x+ruler,454),fill=NAVY,width=2)
        draw.text((x+ruler/2,439),'0.2 Lc',font=font(11),fill=NAVY,anchor='ms')
    if k in selected:
        raw_selected[k]=body.copy()
    canvas=Image.new('RGB',(1440,576),BG)
    canvas.paste(body,(0,64))
    draw=ImageDraw.Draw(canvas)
    draw.text((26,10),'3D ice melting | neural-operator prediction',font=bold(24),fill=NAVY)
    draw.text((26,42),'Ice-water interface: phi = 0   |   Fixed camera and physical scale',font=font(14),fill=NAVY)
    draw.text((1410,20),f't = {float(t):.2f} s',font=bold(25),fill=NAVY,anchor='ra')
    for col,index in enumerate(indices):
        draw.text((col*480+240,559),f'Held-out sample {index}',font=bold(17),fill=NAVY,anchor='mm')
    canvas.save(FRAMES/f'frame{k:03d}.png')
    rendered.append(canvas)
    if k%15==0:
        print('RENDER',k,float(t),flush=True)
p.close()
assert not np.array_equal(raw_first,raw_last)
assert (np.array(foreground[-1])/np.array(foreground[0])<.2).all(),foreground

# Five-time static gallery with the identical views used by the animation.
panel=384
left=150
gap=12
margin=24
W=left+5*panel+4*gap+2*margin
H=105+3*panel+2*gap+55
sheet=Image.new('RGB',(W,H),BG)
draw=ImageDraw.Draw(sheet)
draw.text((24,15),'Three-dimensional ice-water interfaces | Adam PINO',font=bold(27),fill=NAVY)
draw.text((24,52),'Actual checkpoint isosurfaces; 0-5 s; identical camera and scale across all times',font=font(16),fill=NAVY)
for col,k in enumerate(selected):
    x=left+margin+col*(panel+gap)
    draw.text((x+panel/2,91),f't = {float(physical[k]):g} s',font=bold(20),fill=NAVY,anchor='mm')
    for row,index in enumerate(indices):
        y=105+row*(panel+gap)
        crop=raw_selected[k].crop((row*480,0,(row+1)*480,480)).resize((panel,panel),Image.Resampling.LANCZOS)
        sheet.paste(crop,(x,y))
        if col==0:
            draw.text((24,y+panel/2),f'Sample {index}',font=bold(19),fill=NAVY,anchor='lm')
draw.text((24,H-25),'Surface = phi 0.  Render grid: 96^3.  Surface shading only; no geometric smoothing or prescribed shrinkage.',font=font(15),fill=NAVY)
sheet.save(OUT/'melting-render-3d-gallery.png')

# PNG poster and short H.264/GIF animations; endpoint holds are intentional.
rendered[0].save(OUT/'melting-render-3d-poster.png')
sequence=[rendered[0]]*10+rendered+[rendered[-1]]*10
video=OUT/'melting-render-3d.mp4'
with imageio.get_writer(video,fps=10,codec='libx264',quality=9,macro_block_size=16,
                         pixelformat='yuv420p',ffmpeg_log_level='error') as writer:
    for frame in sequence:
        writer.append_data(np.asarray(frame))
thumbs=[rendered[i].resize((720,288)) for i in (0,15,30,45,60)]
palette_sheet=Image.new('RGB',(720,288*len(thumbs)))
for j,thumb in enumerate(thumbs):palette_sheet.paste(thumb,(0,j*288))
palette=palette_sheet.quantize(colors=256)
gif_frames=[frame.resize((1200,480),Image.Resampling.LANCZOS).quantize(palette=palette,dither=Image.Dither.NONE) for frame in sequence]
gif_frames[0].save(OUT/'melting-render-3d.gif',save_all=True,append_images=gif_frames[1:],
    duration=100,loop=0,disposal=2,optimize=False)
with Image.open(OUT/'melting-render-3d.gif') as animation:
    durations=[]
    for k in range(animation.n_frames):
        animation.seek(k)
        durations.append(animation.info['duration'])
    assert sum(durations)==8100
    gif_count=animation.n_frames
with imageio.get_reader(video) as reader:
    video_count=reader.count_frames()
    assert video_count==81
    assert reader.get_data(0).shape==(576,1440,3)
report=dict(input_indices=list(indices),camera=CAMERA,parallel_scale=SCALE,
    iso_value=0,axis_order='time,x,y,z',grid=[96,96,96],times=times.tolist(),physical_times=physical.tolist(),
    renderer=renderer_class,capabilities=capabilities,mesh_records=records,
    foreground_pixel_counts=foreground,first_last_raw_render_equal=bool(np.array_equal(raw_first,raw_last)),
    geometric_smoothing=False,material=dict(color='#8ed2ef',ambient=.28,diffuse=.65,specular=.85,specular_power=65),
    gif_frames=gif_count,gif_duration_ms=sum(durations),video_frames=video_count,video_fps=10,
    seconds=time.perf_counter()-start,
    versions={key:importlib.metadata.version(key) for key in ('pyvista','vtk','scikit-image','imageio','imageio-ffmpeg','numpy','pillow')},
    scope='Actual phi=0 interface geometry from the unchanged checkpoint, no new training/reference or error metric. Render grid is visualization only; coordinates fixed across time. Display normals/light do not change mesh vertices.')
(OUT/'melting-render-surfaces.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'render-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print('RENDER_COMPLETE',report['seconds'],gif_count,video_count,flush=True)
