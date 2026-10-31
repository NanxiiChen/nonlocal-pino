"""Compare16versus512training domains on identical eight validation domains.

Run from repository root in .venv. Uses saved physical-mesh metrics, no solver.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
runs=[root/'20261029-confined-geometry-richer-01',root/'20261029-confined-geometry-richer512-01']
output=runs[1]/'comparison';output.mkdir(exist_ok=False)
metrics=[json.loads((run/'evaluation/metrics.json').read_text()) for run in runs]
histories=[[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()] for run in runs]
summaries=[json.loads((run/'summary.json').read_text()) for run in runs]
effects=[json.loads((run/'geometry_effect/metrics.json').read_text()) for run in runs]
assert effects[0]['common_points']==effects[1]['common_points']
assert effects[0]['reference']==effects[1]['reference']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for index,(history,label) in enumerate(zip(histories,('16 training domains','512 training domains'),strict=True)):
    for ax,key in zip(axes,('phi_pooled','phi_terminal'),strict=True):
        for split,style in (('train','--'),('test','-')):
            ax.semilogy([r['step'] for r in history],[100*r[split+'_'+key] for r in history],
                linestyle=style,color=('C1','C0')[index],marker='o',markersize=3,
                label=label+(': same 16 train monitors' if split=='train' else ': same 8 validation domains'))
        ax.set(xlabel='Adam updates',ylabel='Phase relative L2 (%)',
            title='All saved times' if key=='phi_pooled' else 'Terminal time t = 0.75')
        ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(output/'convergence.png',dpi=180);fig.savefig(output/'convergence.pdf');plt.close(fig)
paired=[]
for old_id,new_id in zip(range(16,24),range(512,520),strict=True):
    rows=[next(r for r in m['conditions'] if r['index']==i) for m,i in zip(metrics,(old_id,new_id),strict=True)]
    paired.append(dict(original_validation_id=old_id,new_validation_id=new_id,
        old_phi_pooled=rows[0]['phi_pooled'],new_phi_pooled=rows[1]['phi_pooled'],
        old_phi_terminal=rows[0]['phi_terminal'],new_phi_terminal=rows[1]['phi_terminal']))
report=dict(runs=[str(r) for r in runs],same_development_validation_geometries=True,
    phase_metric='Mean across8domains of each physical-volume-weighted relative L2; pooled uses all saved times',
    old16=metrics[0]['test'],new512=metrics[1]['test'],paired_validation=paired,
    local_FEM=[m['local_FEM'] for m in metrics],
    geometry_response_error=[e['geometry_response_relative_error'] for e in effects],
    geometry_response_amplitude=[e['geometry_response_amplitude_ratio'] for e in effects],
    cold_trainer_seconds=[s['wall_seconds'] for s in summaries],
    warm_update_seconds=[s['mean_update_seconds_after_100'] for s in summaries])
(output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
