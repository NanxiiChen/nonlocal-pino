"""Freeze five actual histories; compare common steps and retain stage origins."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


RUNS=[
    ('192 / fixed512','20261031-confined-geometry-b16-t16-width192-fp32-30k-01','#3478b8'),
    ('192 / fixed1024','20261031-confined-geometry-b16-t16-width192-bank1024-fp32-50k-01','#279267'),
    ('256 / fixed512','20261031-confined-geometry-b16-t16-width256-fp32-50k-01','#8d64b6'),
    ('192 / rolling from25k','20261031-confined-width192-online25000-rolling-01-train','#bc852b'),
    ('256 / rolling from0 (current)','20261031-confined-width256-online-cold50k-01-train','#d04b40'),
]


def scores_at_step(histories,labels,step):
    """Distinguish an unavailable run from a missing interior assessment."""
    if isinstance(step,bool) or not isinstance(step,int) or step<0:
        raise ValueError('Nonnegative integer comparison step required')
    common=[];unavailable=[]
    keys=['step','train_phi_pooled','train_phi_terminal','test_phi_pooled','test_phi_terminal']
    for label,history in zip(labels,histories,strict=True):
        first,last=history[0]['step'],history[-1]['step']
        if step<first or step>last:
            unavailable.append(dict(label=label,first_step=first,last_step=last,
                reason='not_started' if step<first else 'recorded_run_ended'))
            continue
        matching=[row for row in history if row['step']==step]
        if len(matching)!=1:raise ValueError(f'Missing or duplicate interior evaluation: {label} at{step}')
        common.append(dict(label=label,**{key:matching[0][key] for key in keys}))
    return common,unavailable


def compare(output):
    if not output.is_absolute():raise ValueError('Explicit absolute output path required')
    base=Path('/root/autodl-tmp/tf-logs/confined_ch2d');histories=[];configs=[];manifests=[]
    for label,name,color in RUNS:
        run=base/name;c=json.loads((run/'config.json').read_text())
        rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines(keepends=True) if s.endswith('\n')]
        if not rows or any(b['step']<=a['step'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Strictly increasing actual evaluation steps required')
        for row in rows:
            by={r['index']:r for r in row['conditions']}
            if len(by)!=len(row['conditions']):raise ValueError('Duplicate geometry score')
            if len(row['test_indices'])!=8 or len(row['train_monitor_indices'])!=16:
                raise ValueError('Unexpected monitored cohort')
            for split,ids in [('train',row['train_monitor_indices']),('test',row['test_indices'])]:
                for key in ['phi_pooled','phi_terminal','mu_pooled','mu_terminal']:
                    values=[by[i][key] for i in ids]
                    if not np.isfinite(values).all():raise ValueError('Nonfinite scores')
                    np.testing.assert_allclose(np.mean(values),row[split+'_'+key],rtol=1e-12,atol=1e-14)
        if 'continuation' in c['train']:
            marker=json.loads(Path(c['train']['continuation']['checkpoint']).read_text())
            if rows[0]['step']!=marker['step']:raise ValueError('Continuation origin mismatch')
        elif rows[0]['step']!=0:raise ValueError('Cold history must start0')
        histories.append(rows);configs.append(c)
        manifests.append(json.loads((Path(c['operator']['dataset_path'])/'manifest.json').read_text()))
    for c in configs[1:]:
        for key in ['physical','initial','geometry','nl','sample','loss','reference']:
            if c[key]!=configs[0][key]:raise ValueError(f'Physical comparison differs: {key}')
    # Verify the relabeled512..519 and1024..1031 references by arrays, not indices or hashes.
    for config,manifest in zip(configs[1:],manifests[1:],strict=True):
        for ia,ib in zip(manifests[0]['test_indices'],manifest['test_indices'],strict=True):
            left=Path(configs[0]['operator']['dataset_path'])/manifests[0]['conditions'][ia]['file']
            right=Path(config['operator']['dataset_path'])/manifest['conditions'][ib]['file']
            with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
                for key in ['points','volumes','holes','time','phi','mu']:
                    np.testing.assert_array_equal(a[key],b[key])
    step=histories[-1][-1]['step'];latest=[]
    common,unavailable=scores_at_step(histories,[label for label,name,color in RUNS],step)
    if not common or common[-1]['label']!=RUNS[-1][0]:raise ValueError('Current endpoint must have an actual assessment')
    for (label,name,color),history in zip(RUNS,histories,strict=True):
        keys=['step','train_phi_pooled','train_phi_terminal','test_phi_pooled','test_phi_terminal']
        last={key:history[-1][key] for key in keys};latest.append(dict(label=label,run=str(base/name),**last))
    output.mkdir(parents=True,exist_ok=False)
    for (label,name,color),history in zip(RUNS,histories,strict=True):
        (output/(name+'-metrics.jsonl')).write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in history))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5),layout='constrained')
    for (label,name,color),history in zip(RUNS,histories,strict=True):
        x=np.array([r['step'] for r in history]);y=100*np.array([r['test_phi_pooled'] for r in history])
        width=2.6 if name==RUNS[-1][1] else 1.7
        early=(x>=500)&(x<=step)
        if early.any():axes[0].plot(x[early]/1000,y[early],color=color,lw=width,marker='o',ms=3,label=label)
        axes[1].plot(x/1000,y,color=color,lw=width,marker='o',ms=2,label=label)
        axes[1].scatter(x[-1]/1000,y[-1],color=color,s=24,zorder=4)
    axes[0].set(title=f'Common early budget: 500--{step} updates',xlabel='Training updates (thousands)',ylabel='Validation relative L2 of phase (%)')
    axes[0].legend(fontsize=8)
    axes[1].set(title='Full recorded histories (no extrapolation)',xlabel='Total training updates (thousands)',ylabel='Validation relative L2 of phase (%)',yscale='log')
    axes[1].set_yticks([5,10,20,40,100],labels=['5','10','20','40','100'])
    axes[1].axvline(25,color='#bc852b',lw=.8,ls=':',alpha=.7)
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Confined CH: same 8 native validation geometries; raw evaluation points')
    for suffix in ['png','pdf']:fig.savefig(output/('comparison.'+suffix),dpi=190)
    plt.close(fig)
    def plot_budget(start,filename,title):
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.4),layout='constrained')
        for (label,name,color),history in zip(RUNS,histories,strict=True):
            selected=[row for row in history if start<=row['step']<=step]
            if not selected:continue
            shown_label=label
            if history[-1]['step']<step:shown_label+=f" (ends{history[-1]['step']/1000:g}k)"
            x=np.array([row['step'] for row in selected])/1000
            width=2.6 if name==RUNS[-1][1] else 1.7
            for ax,key in zip(axes,['test_phi_pooled','test_phi_terminal'],strict=True):
                y=100*np.array([row[key] for row in selected])
                ax.plot(x,y,color=color,lw=width,marker='o',ms=3,label=shown_label)
        for ax,field in zip(axes,['Whole space-time field','Final-time field'],strict=True):
            ax.set(title=field,xlabel='Training updates (thousands)',ylabel='Validation relative L2 of phase (%)')
            ax.grid(alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
        fig.suptitle(title+'; 8 identical native validation geometries')
        for suffix in ['png','pdf']:fig.savefig(output/(filename+'.'+suffix),dpi=190)
        plt.close(fig)
    plot_budget(500,'matched_budget',f'Recorded curves clipped at {step} updates')
    tail_start=max(500,step-8000)
    plot_budget(tail_start,'matched_tail',f'Recent training: {tail_start}--{step} updates')
    result=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),output=str(output),common_step=step,
        common_step_scores=common,unavailable_at_common_step=unavailable,last_observed_scores=latest,reference_arrays_exact_across_all_runs=True,
        aggregate_scores_reconstructed_all_histories=True,
        matched_curves=dict(plots=['matched_budget.png','matched_budget.pdf'],from_step=500,to_step=step,
            fields=['test_phi_pooled','test_phi_terminal'],smoothing='none',all_lines_clipped_at_same_step=True,
            each_line_retains_observed_stage_start_end=True),
        recent_curves=dict(plots=['matched_tail.png','matched_tail.pdf'],from_step=tail_start,to_step=step),
        initial_geometry_library='512or1024known inputs; rolling trajectories are PDE-only',
        scope='Same8reused native development geometries and runtime seed42; not localFEM/fresh test. DifferentGPU/padding/architecture/library/rolling origins. No causal isolated sampler/capacity claim. 192online stage starts25k; its parent is the192fixed1024curve before25k. Last observed endpoints have unequal budgets. No smoothing/extrapolation or new training.')
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    compare(parser.parse_args().output)
