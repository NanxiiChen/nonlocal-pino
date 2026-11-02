"""Geometry-only design sketch; these curves are not qualified training inputs."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.interpolate import BSpline

from examples.channel_nsch2d.configs.train_default import get_config
from examples.channel_nsch2d.problem import walls


def design(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    c=get_config();x=np.linspace(0,c.geometry.length,2001)
    start,end,taper=1.15,2.75,.35
    initial_extent=c.initial.center[0]+np.sqrt(c.initial.radius**2+2*c.initial.radius*c.initial.transition)
    if initial_extent>=start:raise ValueError('Random wall support overlaps the initial diffuse droplet')
    inside=(x>=start)&(x<=end)
    # Eight positive coefficients per wall; the convex B-spline basis avoids
    # interpolation overshoot. Smooth tapers preserve straight end sections.
    degree=3;controls=8
    knots=np.r_[np.full(degree+1,start),np.linspace(start,end,controls-degree+1)[1:-1],
                np.full(degree+1,end)]
    basis=np.zeros((len(x),controls))
    basis[inside]=BSpline(knots,np.eye(controls),degree,extrapolate=False)(x[inside])
    def gate(z):
        z=np.clip(z,0,1)
        return z**3*(10-15*z+6*z*z)
    envelope=gate((x-start)/taper)*gate((end-x)/taper)
    basis*=envelope[:,None]
    rng=np.random.default_rng(20261102)
    samples=[];attempts=0
    while len(samples)<5:
        attempts+=1
        if attempts>10000:raise RuntimeError('Illustrative geometry rejection budget exceeded')
        coefficients=rng.uniform(.01,.45,(2,controls))
        depth=coefficients@basis.T
        lower=depth[0];upper=c.geometry.height-depth[1];gap=upper-lower
        slope=max(np.max(np.abs(np.gradient(lower,x))),np.max(np.abs(np.gradient(upper,x))))
        if not .55<=gap.min()<=.70 or slope>1.25:continue
        if not (np.isfinite(depth).all()and np.all(gap>0)):
            raise ValueError('Invalid illustrative wall geometry')
        if np.any(lower[x<start]!=0)or np.any(upper[x<start]!=c.geometry.height):
            raise ValueError('Straight upstream region changed')
        i=int(np.argmin(gap))
        samples.append(dict(lower=lower,upper=upper,coefficients=coefficients,
            minimum_gap=float(gap[i]),throat_x=float(x[i]),maximum_slope=float(slope)))
    b,t=walls(x,(c.geometry.upper,c.geometry.lower),c,xp=np)
    cases=[dict(lower=b,upper=t,minimum_gap=float(np.min(t-b)),throat_x=2.)]+samples
    fig,axes=plt.subplots(2,3,figsize=(14,5.2),constrained_layout=True)
    for i,(ax,row)in enumerate(zip(axes.ravel(),cases,strict=True)):
        lower,upper=row['lower'],row['upper']
        ax.fill_between(x,lower,upper,color='#edf5fa')
        ax.fill_between(x,-.08,lower,color='#d8dadd')
        ax.fill_between(x,upper,1.08,color='#d8dadd')
        ax.plot(x,lower,color='#243e51',lw=1.5);ax.plot(x,upper,color='#243e51',lw=1.5)
        ax.add_patch(Circle(c.initial.center,c.initial.radius,facecolor='#f4ba7d',edgecolor='#b55e19',lw=1.4))
        label='Current single constriction'if i==0 else f'Random smooth walls {i}'
        ax.set(title=f'{label}\nmin gap = {row["minimum_gap"]:.2f}, throat x = {row["throat_x"]:.2f}',
            xlim=(0,4),ylim=(-.05,1.05),xlabel='x',aspect='equal')
        if i%3==0:ax.set_ylabel('y')
        ax.set_yticks([0,.5,1]);ax.tick_params(labelsize=8)
    fig.suptitle('Geometry preview: identical initial droplet; stationary walls',fontsize=14)
    fig.savefig(output/'geometry-preview.png',dpi=180);fig.savefig(output/'geometry-preview.pdf')
    plt.close(fig)
    report=dict(scope='Illustrative geometry only; no PDE solution, neural prediction, trained prior or solver-accuracy claim',
        initial_droplet=c.initial.to_dict(),length=4.,height=1.,wall_support=[start,end],
        controls_per_wall=controls,basis='Positive cubic B-spline coefficients and C2 end tapers',
        proposal_only_constraints=dict(minimum_gap_range=[.55,.70],maximum_slope=1.25),
        accepted_samples=5,attempted_samples=attempts,
        samples=[{k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in row.items()if k not in ('lower','upper')}for row in samples])
    (output/'geometry-preview.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'preview-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items()if k!='samples'},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    design(p.parse_args().output)
