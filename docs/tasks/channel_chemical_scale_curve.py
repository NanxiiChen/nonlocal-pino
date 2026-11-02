"""Physical error histories and de-normalized chemical residual for a scale control."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot(control,trial,output):
    control,trial,output=map(Path,(control,trial,output))
    if output.exists():raise FileExistsError(output)
    configs=[json.loads((p/'config.json').read_text())for p in (control,trial)]
    before=json.loads(json.dumps(configs[1]));normalizer=before['loss'].pop('chemical_residual_scale')
    before['log']['run_name']=configs[0]['log']['run_name']
    if (before!=configs[0]or any('causal'in c for c in configs)
            or 'chemical_residual_scale'in configs[0]['loss']):
        raise ValueError('Require a uniform loss-only single-normalizer control with unchanged physical output')
    scales=(configs[0]['loss']['chemical_scale'],normalizer)
    histories=[];losses=[]
    for p in (control,trial):
        read=lambda name:[json.loads(s)for s in (p/name).read_bytes().splitlines(keepends=True)if s.endswith(b'\n')]
        rows=read('metrics.jsonl');parts=read('loss.jsonl')
        if (not rows or rows[0]['step']!=0 or any(b['step']<=a['step']for a,b in zip(rows,rows[1:]))
                or not parts or any(b['step']<=a['step']for a,b in zip(parts,parts[1:]))):
            raise ValueError('Finite strictly ordered actual histories required')
        histories.append(rows);losses.append(parts)
    if min(r[-1]['step']for r in histories)<500:raise ValueError('Wait for an actual500step physical monitor')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    names=('phi','terminal_phi',None,'velocity','pressure','chemical')
    titles=('Phase: all snapshots','Phase: terminal t=6','Raw chemical residual: sampled batches',
        'Velocity: positive times','Pressure: positive times','Chemical potential: positive times')
    for ax,key,title in zip(axes.ravel(),names,titles,strict=True):
        for rows,parts,scale,color in zip(histories,losses,scales,('#0072B2','#D55E00'),strict=True):
            if key is None:
                x=[r['step']for r in parts];y=scale*np.sqrt([r['parts']['chemical']for r in parts])
            else:x=[r['step']for r in rows];y=100*np.array([r[key]for r in rows])
            if np.any(~np.isfinite(y))or np.any(y<=0):raise ValueError('Positive finite measured values required')
            label=f'chemical row scale {scale:g}; output {configs[0]["loss"]["chemical_scale"]:g}'
            ax.semilogy(x,y,color=color,lw=1.8,label=label)
        ax.set(title=title,xlabel='Optimizer updates',ylabel='Raw RMS'if key is None else'Relative L2 (%)',xlim=(0,configs[0]['train']['iterations']))
        ax.grid(alpha=.2,which='both')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Same physical model / cold state / input protocol; chemical-row normalization only\n'
        'Actual monitors only; raw sampled chemical RMS is not a full-time accuracy certificate',fontsize=10)
    output.mkdir(parents=True);fig.savefig(output/'convergence.png',dpi=200)
    fig.savefig(output/'convergence.pdf');plt.close(fig)
    report=dict(status='physical_errors_and_raw_chemical_residual_plotted',runs=[str(control),str(trial)],
        last_steps=[r[-1]['step']for r in histories],chemical_row_scales=scales,
        mu_output_amplitude=configs[0]['loss']['chemical_scale'],histories=histories,loss_histories=losses,
        scope='One developmentgeometry/initialization; no smoothing/extrapolation/predictions/labels. Raw chemicalRMS equals rowScale*sqrt(loggedchemicalMSE), since both objectives are uniform. Actualfinalsampler proofs are separate.')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output/'convergence.png'),last_steps=report['last_steps'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','trial','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();plot(a.control,a.trial,a.output)
