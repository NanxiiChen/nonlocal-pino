"""Reference-only morphology preflight; no network fitting or future-shape IC."""
from pathlib import Path
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import json
import numpy as np
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.precipitate2d.configs.train_multiple import get_config
from examples.precipitate2d.solver import solve_reference, relative
from examples.precipitate2d.problem import validate_config

OUT = Path(__file__).parent


def components(phi):
    mask = phi > 0
    labels,count = label(mask)
    parent = list(range(count+1))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a,b):
        if a and b:
            parent[root(a)] = root(b)
    for a,b in zip(labels[0,:],labels[-1,:],strict=True):join(a,b)
    for a,b in zip(labels[:,0],labels[:,-1],strict=True):join(a,b)
    sizes = {}
    for i in range(1,count+1):
        r = root(i)
        sizes[r] = sizes.setdefault(r,0)+int(np.sum(labels==i))
    return sorted((v/phi.size for v in sizes.values()),reverse=True)


def main():
    cases = {}
    for name,scale,method in [('elastic4_native',4,'nonlocal'),
                              ('elastic1_native',1,'nonlocal'),
                              ('no_elastic_native',0,'nonlocal'),
                              ('elastic4_local',4,'local_spectral')]:
        c = get_config()
        if scale == 0:c.physical.eigenstrain = (0.,0.,0.)
        else:
            c.physical.lame_lambda = 2450.*scale
            c.physical.shear_modulus = 1225.*scale
        validate_config(c)
        print(f'Preflight {name}',flush=True)
        data = solve_reference(c,c.sample.Nx,method,.001)
        cases[name] = data
        areas = [components(p) for p in data['phi']]
        record = dict(configuration=c.to_dict(), components=areas,
            initial_terminal_phi_change=relative(data['phi'][-1],data['phi'][0]),
            energy_initial=float(data['energy'][0]),energy_terminal=float(data['energy'][-1]),
            phase_min=float(data['phi'].min()),phase_max=float(data['phi'].max()),
            mass_drift=float(np.max(np.abs(data['phi'].mean(axis=(1,2))-data['phi'][0].mean()))))
        (OUT/f'precipitate-multiple-preflight-{name}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(name, 'initial areas',areas[0],'final areas',areas[-1],flush=True)
    indices = (0,5,10,15,20)
    fig,axes = plt.subplots(len(cases),5,figsize=(13,9),layout='constrained')
    for row,(name,data) in enumerate(cases.items()):
        for col,i in enumerate(indices):
            axes[row,col].imshow(data['phi'][i],origin='lower',extent=(0,1,0,1),vmin=-1,vmax=1,cmap='coolwarm')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(f't={c.physical.terminal_time*i/20:.1f}')
            if col==0:axes[row,col].set_ylabel(name,fontsize=9)
    fig.savefig(OUT/'precipitate-multiple-preflight.png',dpi=160);plt.close(fig)
    np.savez_compressed(OUT/'precipitate-multiple-preflight.npz',
        **{name:data['phi'] for name,data in cases.items()})
    print('Elastic effect',relative(cases['elastic4_native']['phi'][-1],cases['no_elastic_native']['phi'][-1]),flush=True)


if __name__=='__main__':main()
