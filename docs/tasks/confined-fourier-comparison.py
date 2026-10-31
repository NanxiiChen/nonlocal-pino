"""Compare common eight-geometry native L2 curves; never select checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.confined_ch2d.operator_history import training_records


def compare(runs, labels, output, allow_bank_growth=False):
    if len(runs)!=len(labels) or len(runs)<2:
        raise ValueError('Provide aligned run paths and labels for at least two runs')
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    records=[];partition=None;benchmark=None;reference_arrays=None
    for run,label in zip(runs,labels,strict=True):
        c=json.loads((run/'config.json').read_text())
        rows=training_records(run,'metrics.jsonl')
        if not rows or np.any(np.diff([r['step'] for r in rows])<=0):
            raise ValueError('Require nonempty ordered evaluation history')
        signature=(c['physical'],c['geometry'],c['initial'],c['operator']['dataset_path'])
        if allow_bank_growth:
            from ml_collections import ConfigDict
            from examples.confined_ch2d.operator_problem import specification
            signature=specification(ConfigDict(c))
            signature['prior'].pop('train_conditions')
            root=Path(c['operator']['dataset_path'])
            manifest=json.loads((root/'manifest.json').read_text())
            assert manifest['status']=='passed'
            own_partition=manifest['test_indices'];current=[]
            for index in own_partition:
                with np.load(root/manifest['conditions'][index]['file'],allow_pickle=False) as f:
                    current.append({key:f[key] for key in ('points','volumes','holes','phi','mu','time')})
            if reference_arrays is None:reference_arrays=current
            else:
                for first,second in zip(reference_arrays,current,strict=True):
                    for key in first:np.testing.assert_array_equal(first[key],second[key])
        if partition is None:partition=rows[0]['test_indices'];benchmark=signature
        expected_partition=own_partition if allow_bank_growth else partition
        if any(r['test_indices']!=expected_partition for r in rows) or signature!=benchmark:
            raise ValueError('Comparison requires identical physical/native validation arrays, geometry family and IC; bank growth must be explicitly allowed')
        for column,(key,title) in enumerate(zip(('test_phi_pooled','test_phi_terminal'),
                                ('All five saved times','Terminal time t = 3'),strict=True)):
            values=np.asarray([r[key] for r in rows])
            if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Positive finite L2 required')
            for row,x,xlabel in ((0,[r['step'] for r in rows],'Optimizer updates'),
                    (1,[r['elapsed_seconds']/60 for r in rows],'Cumulative elapsed time (minutes)')):
                ax=axes[row,column]
                ax.semilogy(x,100*values,label=label,linewidth=1.7)
                ax.set(xlabel=xlabel,ylabel='Phase relative L2 (%)',title=title)
        final=rows[-1]
        records.append(dict(run=str(run),label=label,step=final['step'],
            status='complete' if (run/'summary.json').is_file() and final['step']==c['train']['iterations'] else 'running',
            native_phi_pooled=final['test_phi_pooled'],native_phi_terminal=final['test_phi_terminal'],
            elapsed_seconds=final['elapsed_seconds'],architecture=c['arch'],sample=c['sample'],condition_batch=c['operator']['condition_batch']))
        records[-1]['training_input_geometries']=c['operator']['train_conditions']
        records[-1]['learning_rate']={k:c['train'][k] for k in ('lr_init','lr_min','lr_decay_every','lr_decay_rate')}
        records[-1]['continuation']=c['train']['continuation'] if 'continuation' in c['train'] else None
    for ax in axes.flat:
        ax.axhline(10.,color='black',linestyle=':',linewidth=.8)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fixed small-scale 10-mode CH; common 8 reused validation geometries')
    for suffix in ('png','pdf'):fig.savefig(output/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    result=dict(reference='Independent native graph/BDF; local FEM deferred',
        metric='Mean of area-weighted phase relative L2 over the same eight development geometries',
        scope='Architecture/optimization development; not a fresh final test or held-out checkpoint selection. Different B/Nt consume different numbers of physics samples per update; elapsed time includes setup, compilation and evaluation.',runs=records)
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('label','step','status','native_phi_pooled','native_phi_terminal')} for r in records],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--label',action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-bank-growth',action='store_true',help='Verify exact native validation arrays across explicitly different training banks')
    a=p.parse_args();compare(a.run,a.label,a.output,a.allow_bank_growth)
