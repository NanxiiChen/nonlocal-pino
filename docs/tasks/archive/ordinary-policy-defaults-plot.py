"""Representative final fields and melting full/sampled convergence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/ordinary-policy-defaults-20261026'


def main():
    audit=json.loads((OUT/'kks-audit.json').read_text())
    with np.load(OUT/'kks-audited-fields.npz') as saved:
        for case,policy,dtype in [('corrosion2d','ordinary','float64'),('electropolish2d','ordinary','float32')]:
            key=f'{case}-{policy}-{dtype}-42'
            p=saved[key+'_phi']
            with np.load(f'/root/autodl-tmp/data/{case}/results_matched_graph_20261024.npz') as reference:
                X,Y=reference['X'],reference['Y'];truth=reference['phi'];times=reference['time']
            p=p.reshape(truth.shape);ids=[0,5,10,15,20];err=np.abs(p-truth)
            fig,axes=plt.subplots(3,5,figsize=(14,6),layout='constrained')
            for row,values in enumerate((truth,p,err)):
                for col,i in enumerate(ids):
                    ax=axes[row,col];im=ax.pcolormesh(X,Y,values[i],shading='nearest',vmin=0,vmax=1 if row<2 else err[ids].max(),cmap='viridis' if row<2 else 'magma');ax.set(aspect='equal')
                    if row==0:ax.set_title(f't*={times[i]:.2f}')
                    if col==0:ax.set_ylabel(('Reference','PINN final5000','Absolute error')[row]+'\ny*')
                    if row==2:ax.set_xlabel('x*')
                    if row==1:
                        phi_image=im;value=audit['runs'][key]['per_time_phi'][i]
                        ax.text(.02,.03,f'L2={value:.2e}',transform=ax.transAxes,fontsize=8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
            fig.colorbar(phi_image,ax=axes[:2,:].ravel().tolist(),shrink=.8,label='phi')
            fig.colorbar(im,ax=axes[2,:].ravel().tolist(),shrink=.8,label='Absolute error')
            fig.suptitle(case+f': selected ordinary {dtype} default, representative untuned seed42')
            fig.savefig(OUT/(case+'-phi-fields.png'),dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    runs=[('Explicit ordinary FP32 seed42',Path('/root/autodl-tmp/tf-logs/icemelting3d/20261026-defaultcheck-ordinary-float32-seed42')),
          ('Earlier FP32 seed2026 (external policy not recorded)',Path('/root/autodl-tmp/tf-logs/icemelting3d/20261025-independent-seed2026-default-strategy'))]
    for label,run in runs:
        h=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
        ax.semilogy([v['step'] for v in h],[v['graph_terminal_rel_l2'] for v in h],label=label+', sampled')
        full=[v for v in h if 'graph_terminal_full_rel_l2' in v]
        ax.scatter([v['step'] for v in full],[v['graph_terminal_full_rel_l2'] for v in full],s=24,label=label+', full grid')
    ax.set(xlabel='Completed updates',ylabel='Terminal graph phi relative L2,5s',title='Melting: raw sampled evaluations and full-grid checkpoints')
    ax.grid(alpha=.2);ax.legend(fontsize=7);fig.savefig(OUT/'icemelting3d-convergence.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
