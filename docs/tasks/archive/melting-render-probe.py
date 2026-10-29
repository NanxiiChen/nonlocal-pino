"""Renderer setup check only; the sphere is not a scientific result."""
import pyvista as pv
p=pv.Plotter(off_screen=True,window_size=(640,640),lighting='none')
p.set_background('#f2f7fc')
p.add_mesh(pv.Sphere(radius=.3,center=(.5,.5,.5),theta_resolution=80,phi_resolution=80),color='#b9e7fc',smooth_shading=True,pbr=True,metallic=.06,roughness=.18)
for pos,intensity in [((1,-1,2),1.0),((-1,1,.7),.5),((1,2,2),.7)]:
 p.add_light(pv.Light(position=pos,focal_point=(.5,.5,.5),color='white',intensity=intensity))
p.camera_position=[(1.8,-1.5,1.35),(.5,.5,.5),(0,0,1)]
p.camera.parallel_projection=True
p.camera.parallel_scale=.55
p.enable_anti_aliasing('ssaa')
p.screenshot('docs/tasks/melting-render-probe.png')
print('window',p.render_window.GetClassName())
print(p.render_window.ReportCapabilities())
p.close()
