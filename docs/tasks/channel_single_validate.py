"""Audit a completed single-geometry run, plot fields and actual convergence."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time


def plot_convergence(run,baseline,output):
    """Use actual monitoring histories and point counts, without rerunning models."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);baseline=Path(baseline);output=Path(output)
    actual=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    previous=[json.loads(s) for s in (Path(baseline)/'metrics.jsonl').read_text().splitlines()]
    baseline_config=json.loads((Path(baseline)/'config.json').read_text())
    config=json.loads((run/'config.json').read_text())
    counts=[]
    for folder in (baseline,run):
        with np.load(folder/'prediction.npz',allow_pickle=False) as f:counts.append(len(f['points']))
    def label(c,count):
        time='BDF2' if 'time_discretization' in c else 'continuous time'
        transport=', SBP phase flux' if 'phase_transport' in c else ''
        embedding=', linear/log time' if 'time_embedding' in c else ', linear time'
        origin=(', full-time L-BFGS' if 'refinement' in c['train'] else
                ', continued' if 'continuation' in c['train'] else ', cold')
        return (f'W{c["arch"]["width"]}, {c["operator"]["train_conditions"]} geometries, '
                f'{c["sample"]["Nx"]}x{c["sample"]["Ny"]}, {count} reporting points\n'
                f'{time}{transport}{embedding}{origin}')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,name in zip(axes.ravel(),('phi','terminal_phi','velocity','pressure'),strict=True):
        for rows,text in ((previous,label(baseline_config,counts[0])),(actual,label(config,counts[1]))):
            ax.semilogy([r['step'] for r in rows],[100*r[name] for r in rows],label=text)
        ax.axhline(10,color='k',linestyle=':',linewidth=.8)
        ax.set(title=name,xlabel='Updates',ylabel='Relative L2 (%)');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=1,fontsize=8)
    fig.suptitle('Actual monitoring histories; final acceptance uses the common dense FEM assessment')
    fig.savefig(output.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    return dict(baseline=counts[0],run=counts[1])


def main(pid,run,baseline):
    root=Path(__file__).resolve().parents[2];python=root/'.venv/bin/python';run=Path(run)
    print(f'Waiting for single-geometry driver {pid}',flush=True)
    while Path(f'/proc/{pid}').exists():time.sleep(2)
    summary=json.loads((run/'summary.json').read_text())
    config=json.loads((run/'config.json').read_text())
    if summary['status']!='completed' or config['operator']['train_conditions']!=1:
        raise ValueError('A completed actual single-geometry run is required')
    for module in ('validate_run','diagnose_phase'):
        with (run/f'{module}.log').open('x') as stream:
            subprocess.run([str(python),'-m',f'examples.channel_nsch2d.{module}',
                '--run',str(run)],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
    environment=os.environ.copy();environment['JAX_PLATFORMS']='cpu'
    for file,folder,label in (('prediction.npz','figures',None),('diagnostic_fields.npz','dense-prediction','Prediction')):
        command=[str(python),'-m','examples.channel_nsch2d.evaluate','--fields',str(run/file),'--output',str(run/folder)]
        if label is not None:command+=['--field-label',label]
        subprocess.run(command,cwd=root,env=environment,check=True)
    import numpy as np
    query_counts=plot_convergence(run,baseline,run/'convergence')
    with np.load(run/'sampling.npz',allow_pickle=False) as data:
        n=int(round(config['physical']['terminal_time']/config['time_discretization']['dt']))
        counts=np.bincount(data['intervals'].ravel(),minlength=n)
        if counts.shape!=(n,) or counts.min()==0:raise ValueError('Training did not visit every temporal interval')
    report=dict(status='independent_replay_and_diagnostics_completed',run=str(run),
        physical_validation=json.loads((run/'validation.json').read_text()),
        continuous_time_phase_balance=json.loads((run/'phase-balance.json').read_text()),
        interval_visit_counts=counts.tolist(),summary=summary,
        baseline=str(baseline),monitoring_query_counts=query_counts,
        comparison_scope='Actual histories label width, geometry/grid/time/transport/origin and reporting point counts; final acceptance uses the separate common dense FEM assessment',
        single_geometry_target_met=all(summary['metrics'][k]<.1 for k in ('phi','terminal_phi','velocity','pressure')),
        scope='One development FEM geometry and one initialization; not online geometry generalization')
    (run/'single-validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Single-geometry replay, full-grid fields, time draws and continuous balance audited',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--driver-pid',type=int,required=True)
    p.add_argument('--run',required=True);p.add_argument('--baseline',required=True)
    a=p.parse_args();main(a.driver_pid,a.run,a.baseline)
