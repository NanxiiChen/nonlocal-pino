"""Validate paired inputs, HLO/numerics/timing and summarize other-case policies."""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/phasefield-gpu-policy-20261026'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='completed':raise RuntimeError('Finish four serial workers first')
    result={'status':'passed','scope':'FP32 current electropolish and melting configs,125 production-function updates; no long convergence claim',
            'static_entries':{
                'corrosion2d':{'entry':'examples/corrosion2d/train.py','policy':'train_matched enforces deterministic default; explicit ordinary research option','operator':'SparseD/segment_sum','timing':'existing measured controls reused'},
                'electropolish2d':{'entry':'examples/electropolish2d/train.py','policy':'shared train_matched adapter enforces deterministic default; explicit gpu_policy=ordinary supported','operator':'SparseD/physical_difference segment_sum'},
                'icemelting3d':{'entry':'examples/icemelting3d/train.py','policy':'inherits XLA_FLAGS; no entry/config forced deterministic setting; inspected parent environment unset','operator':'SparseD stable latent edge ratio + segment_sum'},
                'elasticity2d':{'entry':'examples/elasticity2d/train.py','policy':'inherits XLA_FLAGS; no forced flag found','operator':'ElasticD multiple gradient/residual segment_sum; potential forced-determinism cost, not timed here'},
                'corrosion1d':{'entry':'examples/corrosion1d/train.py','policy':'inherits XLA_FLAGS; no forced flag found','operator':'explicit activation/diffusion presets both SparseD','limitation':'default references missing configs/train_pinn.py; no automatic regime replacement; no complete-update timing here'}},
            'cases':{}}
    for case in ('electropolish2d','icemelting3d'):
        rows={policy:json.loads((OUT/f'{case}-{policy}/result.json').read_text()) for policy in ('ordinary','deterministic')}
        for filename in ('initial-arrays.npz','paired-inputs.npz'):
            with np.load(OUT/f'{case}-ordinary'/filename) as a,np.load(OUT/f'{case}-deterministic'/filename) as b:
                if a.files!=b.files:raise ValueError('Input structure differs')
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=f'{case}/{filename}/{key}')
        ordinary,deterministic=(rows[k] for k in ('ordinary','deterministic'))
        co,cd=(v['config'].copy() for v in (ordinary,deterministic))
        if case=='electropolish2d':
            co['train']=dict(co['train']);cd['train']=dict(cd['train'])
            co['train'].pop('gpu_policy');cd['train'].pop('gpu_policy')
        if co!=cd:raise ValueError('Configuration changed beyond policy')
        case_result={'paired_initial_and_all_pool_data_times_exact':True,'policies':rows,'speed_ratios_deterministic_over_ordinary':{}}
        for policy,row in rows.items():
            if row['status']!='completed' or len(row['samples'])!=75:raise ValueError('Incomplete workload')
            if row['protocol']['warmup_updates']!=50:raise ValueError('Warm-up differs')
            if [v['iteration'] for v in row['samples']]!=list(range(50,125)):raise ValueError('Update window differs')
            if sum(v['soap_refresh'] for v in row['samples'])!=15:raise ValueError('SOAP cadence differs')
            expected_gradnorm=3 if case=='electropolish2d' else 0
            if sum(v['gradnorm_refresh'] for v in row['samples'])!=expected_gradnorm:raise ValueError('GradNorm cadence differs')
            if row['parameters_final_dtype']['dtype']!='float32' or row['optimizer_final_dtype']['dtype']!='float32':raise ValueError('Wrong dtype')
            for key in ('forward','input_gradient'):
                kernel=row['kernels'][key];ops=kernel['optimized_hlo_control_ops']
                trip_counts=[int(v) for op in ops for v in re.findall(r'"known_trip_count":\{"n":"(\d+)"\}',op)]
                kernel['while_trip_counts']=trip_counts
                if policy=='deterministic' and not trip_counts:raise ValueError('Expected lowered scatter loop missing')
                if policy=='ordinary' and trip_counts:raise ValueError('Unexpected ordinary loop')
                if trip_counts and any(v!=row['batch']['padded_edges'] for v in trip_counts):raise ValueError('Unexpected loop count')
                if kernel['relative_error_vs_independent_numpy_scipy']>=(2e-6 if case=='electropolish2d' else 2e-5):raise ValueError('Numerical control failed')
        for key in ('steady','pde_only'):
            case_result['speed_ratios_deterministic_over_ordinary'][key]=deterministic[key]['mean_seconds']/ordinary[key]['mean_seconds']
        for key in ('forward','input_gradient'):
            case_result['speed_ratios_deterministic_over_ordinary'][key]=deterministic['kernels'][key]['steady']['mean_seconds']/ordinary['kernels'][key]['steady']['mean_seconds']
        a=np.asarray(ordinary['initial_losses']);b=np.asarray(deterministic['initial_losses'])
        case_result['initial_loss_vector_relative_difference']=float(np.linalg.norm(a-b)/np.linalg.norm(b))
        if case_result['initial_loss_vector_relative_difference']>=2e-5:raise ValueError('Initial loss mismatch too large')
        result['cases'][case]=case_result
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    x=np.arange(2);labels=['Electropolish\n(default: deterministic)','Melting\n(default: inherits env)']
    for ax,key,title in zip(axes,('steady','forward','input_gradient'),('Complete synchronized update','Frozen graph forward','Frozen graph input gradient'),strict=True):
        for j,policy in enumerate(('ordinary','deterministic')):
            values=[]
            for case in ('electropolish2d','icemelting3d'):
                row=result['cases'][case]['policies'][policy]
                values.append(1000*(row['steady']['mean_seconds'] if key=='steady' else row['kernels'][key]['steady']['mean_seconds']))
            bars=ax.bar(x+(j-.5)*.35,values,width=.35,label=policy)
            for bar,value in zip(bars,values,strict=True):ax.annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(x,labels);ax.set_yscale('log');ax.set(title=title,ylabel='Milliseconds (log scale)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Actual FP32 configurations: identical inputs, two GPU compilation policies\nFull updates: 50 warm-up + 75 measured, SOAP/GradNorm included; no L2 convergence comparison',fontsize=11)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    for case,v in result['cases'].items():print(case,json.dumps(v['speed_ratios_deterministic_over_ordinary']),flush=True)


if __name__=='__main__':main()
