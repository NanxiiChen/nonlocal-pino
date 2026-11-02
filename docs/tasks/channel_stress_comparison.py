"""Same-parent mixed-stress initialization controls: actual physical L2 curves."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compare(parent,runs,output):
    parent=Path(parent);runs=[Path(p) for p in runs];output=Path(output)
    if len(runs)!=2:raise ValueError('Exactly two completed mixed-stress controls required')
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,output/'comparison_source.py')
    configs=[];curves=[];results=[]
    for run in runs:
        c=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
        if summary['status']!='completed' or c['train']['refinement']['iterations']!=250:
            raise ValueError('Both actual prescribed250step controls must be complete')
        if Path(c['train']['refinement']['parent_run'])!=parent:
            raise ValueError('Both controls must name the same audited SOAP parent')
        for name in ('model_parent.eqx','optimizer_soap.eqx','known_inputs.npz','sampling.npz'):
            if (run/name).read_bytes()!=(runs[0]/name).read_bytes():raise ValueError(f'Paired provenance differs:{name}')
        rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
        if [r['refinement_step'] for r in rows]!=list(range(0,251,25)):
            raise ValueError('Incomplete actual L2 monitoring support')
        with np.load(run/'prediction.npz',allow_pickle=False) as f:
            shape=f['fields'].shape;reference=f['reference']
        if len(configs):np.testing.assert_array_equal(reference,first_reference)
        else:first_reference=reference
        result=dict(run=str(run),final_monitor=rows[-1],query_shape=list(shape),
                    scope='One development geometry, saved reporting queries, no seed-stability or population claim')
        if (run/'dense-fem-evaluation.json').is_file():
            result['dense_evaluation']=json.loads((run/'dense-fem-evaluation.json').read_text())
        else:result['dense_evaluation_status']='pending independent dense assessment'
        results.append(result);curves.append(rows)
        del c['log']
        if 'stress_initializer' in c['train']['refinement']:del c['train']['refinement']['stress_initializer']
        configs.append(c)
    if configs[0]!=configs[1]:raise ValueError('Control recipes differ beyond the declared initializer/log metadata')
    for name in ('losses.py','network.py','stress.py','temporal.py','transport.py','operators.py','data.py'):
        relative=Path('source/examples/channel_nsch2d')/name
        if (runs[0]/relative).read_bytes()!=(runs[1]/relative).read_bytes():
            raise ValueError(f'Actual numerical source snapshots differ:{name}')
    labels=('Isotropic stress initialization','Physics block equilibration')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key in zip(axes.ravel(),('phi','terminal_phi','velocity','pressure'),strict=True):
        for rows,label in zip(curves,labels,strict=True):
            ax.plot([r['refinement_step'] for r in rows],[100*r[key] for r in rows],marker='.',label=label)
        ax.axhline(100*curves[0][0][key],color='.35',ls='--',lw=1,label='Parent physical field')
        ax.axhline(10,color='k',ls=':',lw=.8,label='10% target')
        ax.set(title=key,xlabel='Accepted L-BFGS updates',ylabel='Relative L2 (%)',xlim=(0,250))
        ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=2,fontsize=9)
    fig.suptitle(f'Mixed-stress controls, same parent and prescribed budget\n{shape[1]} reporting queries, one geometry')
    fig.savefig(output/'convergence.png',dpi=180,bbox_inches='tight');fig.savefig(output/'convergence.pdf',bbox_inches='tight');plt.close(fig)
    report=dict(status='paired_initialization_physical_L2_comparison_completed',parent=str(parent),
        paired_numerical_source_and_input_bytes_exact=True,controls=results,curves=curves,
        scope='Same formulation/physical parameters, different auxiliary initial values; no accuracy promotion or overall-goal completion')
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True)
    p.add_argument('--run',required=True,action='append');p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.parent,a.run,a.output)
