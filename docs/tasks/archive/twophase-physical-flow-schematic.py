"""Illustrative prescribed periodic vortex pairs, not CH/NO predictions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
N = 160
x = np.arange(N)/N
xx, yy = np.meshgrid(x, x, indexing='xy')
k = 2*np.pi*np.fft.fftfreq(N, d=1/N)
kx, ky = np.meshgrid(k, k, indexing='xy')
k2 = kx*kx+ky*ky
nonzero = k2 > 0
cases = [
    dict(name='Weaker, centered pair', circulation=.06, core=.07, separation=.40, offset=0.),
    dict(name='Stronger, centered pair', circulation=.12, core=.07, separation=.40, offset=0.),
    dict(name='Stronger, offset pair', circulation=.12, core=.07, separation=.40, offset=.13),
]
fields = []
checks = []
for case in cases:
    g, a, d, b = [case[key] for key in ('circulation','core','separation','offset')]
    centers = [(0.5+b-d/2,.43),(0.5+b+d/2,.43)]
    modes = [np.exp(-1j*(kx*cx+ky*cy)) for cx, cy in centers]
    wh = N*N*g*np.exp(-a*a*k2/4)*(modes[0]-modes[1])
    wh[0,0] = 0.
    ph = np.zeros_like(wh)
    ph[nonzero] = wh[nonzero]/k2[nonzero]
    uh, vh = 1j*ky*ph, -1j*kx*ph
    u, v, w = [np.fft.ifft2(value).real for value in (uh,vh,wh)]
    divergence = np.fft.ifft2(1j*kx*uh+1j*ky*vh).real
    error = float(np.max(np.abs(divergence)))
    assert error < 1e-12
    assert abs(w.mean()) < 1e-12
    checks.append(dict(**case,divergence_max=error,mean_vorticity=float(w.mean()),
                       rms_speed=float(np.sqrt(np.mean(u*u+v*v)))))
    fields.append((u,v,w,centers))

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,3,figsize=(12.5,4.5),layout='constrained')
limit = max(np.max(np.abs(field[2])) for field in fields)
for ax, case, (u,v,w,centers) in zip(axes,cases,fields):
    im = ax.pcolormesh(xx,yy,w,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto',rasterized=True)
    ax.streamplot(x,x,u,v,density=.8,color='#455264',linewidth=.65,arrowsize=.85)
    ax.add_patch(Circle((.5,.67),.11,facecolor='#fff8d9',edgecolor='#111827',linewidth=1.8,zorder=5))
    ax.text(.5,.67,'Initial\ndroplet',ha='center',va='center',fontsize=9,zorder=6)
    for sign, (cx,cy) in zip(('+','−'),centers):
        ax.text(cx,cy,sign,ha='center',va='center',fontsize=18,fontweight='bold',color='white')
    ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='x',ylabel='y',title=case['name'])
    ax.text(.02,.02,rf"$\Gamma={case['circulation']:.2f},\ a={case['core']:.2f},\ d={case['separation']:.2f},\ b={case['offset']:.2f}$",transform=ax.transAxes,fontsize=9,
            bbox=dict(facecolor='white',alpha=.9,edgecolor='none'),zorder=7)
fig.colorbar(im,ax=list(axes),shrink=.68,label='Prescribed vorticity')
fig.suptitle('A physical input family: a droplet exposed to counter-rotating stirring vortices\nIllustrative flow fields only; no predicted interface evolution',fontsize=13)
fig.savefig(HERE/'twophase-physical-flow-schematic.png',dpi=180)
(HERE/'twophase-physical-flow-schematic.json').write_text(json.dumps(dict(
    status='illustration_only_not_training_or_reference',domain='unit periodic square',
    periodized_gaussian='Fourier coefficient exp(-a^2 |k|^2/4), unit integral',
    construction='equal opposite circulation; -Delta psi=omega; v=(psi_y,-psi_x)',
    initial_droplet='circle shown for schematic only, not a modified training IC',cases=checks),indent=2)+'\n')
print(json.dumps(checks,indent=2))
