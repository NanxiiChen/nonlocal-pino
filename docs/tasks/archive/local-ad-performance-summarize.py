"""Validate raw serial benchmark records and export comparison plots/tables."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
RAW=ROOT/'reports/local-ad-performance-20261026'
OUT=Path(__file__).with_name('local-ad-performance-summary')
KINDS=('graph','local_tensor','local_joint')
LABELS=('Nonlocal + time JVP','Local + tensor/time JVP','Local + joint/time grad')


def main():
    manifest=json.loads((RAW/'manifest.json').read_text())
    assert manifest['status']=='completed' and len(manifest['runs'])==6
    assert manifest['maximum_GPU_workers']==1
    assert all(v['returncode']==0 for v in manifest['runs'])
    rows=[];records={}
    for dtype in ('float32','float64'):
        records[dtype]={}
        for kind in KINDS:
            r=json.loads((RAW/f'{kind}-{dtype}.json').read_text())
            assert r['status']=='completed' and r['paired_initial_exact']
            assert r['reference_accuracy_evaluated'] is False
            assert r['warmup_updates']==50 and r['measured_updates']==75
            samples=r['update_samples']
            assert len(samples)==75 and [v['iteration'] for v in samples]==list(range(50,125))
            assert sum(v['soap_refresh'] for v in samples)==15
            assert sum(v['gradnorm_refresh'] for v in samples)==3
            for key in ('parameter_dtype','optimizer_initial_dtype','parameters_final_dtype','optimizer_final_dtype'):
                assert r[key]['dtype']==dtype
            np.testing.assert_allclose(r['steady']['mean_seconds'],np.mean([v['seconds'] for v in samples]),rtol=0,atol=1e-14)
            assert r['active_PDE_pairs_per_batch_min_max']==[3760,4032]
            assert r['batch_shapes']['IC']==5151 and r['batch_shapes']['BC']==[9,201]
            records[dtype][kind]=r
        baseline=records[dtype]['graph']
        for kind in KINDS:
            r=records[dtype][kind]
            np.testing.assert_allclose(r['first_losses'][2:],baseline['first_losses'][2:],
                                       rtol=0,atol=1e-6 if dtype=='float32' else 1e-12)
            rows.append({'dtype':dtype,'kind':kind,'mean_update_ms':1e3*r['steady']['mean_seconds'],
                         'ordinary_update_ms':1e3*r['ordinary']['mean_seconds'],
                         'gradnorm_soap_update_ms':1e3*r['gradnorm_soap']['mean_seconds'],
                         'pde_only_ms':1e3*r['pde_only']['mean_seconds'],
                         'speedup_over_graph':baseline['steady']['mean_seconds']/r['steady']['mean_seconds'],
                         'compile_seconds':r['training_compile_seconds'],
                         'peak_active_GiB':r['training_memory']['peak_bytes_in_use']/2**30,
                         'support_points':r['network_PDE_points'],
                         'block_means_ms':[1e3*v for v in r['block_means_seconds']]})
        previous=ROOT/f'docs/tasks/archive/fp64-performance-train-{dtype}.json'
        prior=json.loads(previous.read_text())
        np.testing.assert_allclose(baseline['first_losses'],prior['first_losses'],rtol=0,atol=1e-5 if dtype=='float32' else 1e-12)
    result={'status':'passed','scope':'Throughput at matched active PDE counts; no convergence/accuracy comparison',
            'manifest':manifest,'rows':rows,'raw_records':records}
    operator_manifest=json.loads((RAW/'operator-manifest.json').read_text())
    ordinary_manifest=json.loads((RAW/'ordinary-manifest.json').read_text())
    for item,count in ((operator_manifest,4),(ordinary_manifest,2)):
        assert item['status']=='completed' and len(item['runs'])==count
        assert all(v['returncode']==0 for v in item['runs'])
    result['operator_manifest']=operator_manifest
    result['ordinary_manifest']=ordinary_manifest
    result['operator_records']={}
    result['ordinary_graph_records']={}
    result['ordinary_graph_rows']=[]
    for dtype in ('float32','float64'):
        for policy in ('deterministic','ordinary'):
            item=json.loads((RAW/f'operator-{policy}-{dtype}.json').read_text())
            assert item['status']=='completed' and item['shapes']=={'values':[2559,16],'edges':3032}
            result['operator_records'][f'{policy}-{dtype}']=item
        r=json.loads((RAW/f'graph-ordinary-{dtype}.json').read_text())
        assert r['status']=='completed' and r['policy']=='ordinary' and r['xla_flags']==''
        assert r['warmup_updates']==50 and r['measured_updates']==75
        assert sum(v['gradnorm_refresh'] for v in r['update_samples'])==3
        assert sum(v['soap_refresh'] for v in r['update_samples'])==15
        assert r['active_PDE_pairs_per_batch_min_max']==[3760,4032]
        for key in ('parameter_dtype','optimizer_initial_dtype','parameters_final_dtype','optimizer_final_dtype'):
            assert r[key]['dtype']==dtype
        np.testing.assert_allclose(r['first_losses'],records[dtype]['graph']['first_losses'],
                                   rtol=1e-5 if dtype=='float32' else 1e-11,atol=1e-6 if dtype=='float32' else 1e-12)
        result['ordinary_graph_records'][dtype]=r
        result['ordinary_graph_rows'].append({
            'dtype':dtype,'mean_update_ms':r['steady']['mean_seconds']*1000,
            'pde_only_ms':r['pde_only']['mean_seconds']*1000,
            'ordinary_update_ms':r['ordinary']['mean_seconds']*1000,
            'gradnorm_soap_update_ms':r['gradnorm_soap']['mean_seconds']*1000,
            'compile_seconds':r['training_compile_seconds'],
            'peak_active_GiB':r['training_memory']['peak_bytes_in_use']/2**30,
            'speedup_over_deterministic_graph':records[dtype]['graph']['steady']['mean_seconds']/r['steady']['mean_seconds']})
    OUT.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    x=np.arange(3);width=.36
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    metrics=(('mean_update_ms','Complete SOAP/GradNorm update (ms)'),
             ('pde_only_ms','PDE-only loss evaluation (ms)'),
             ('peak_active_GiB','Peak training active GPU memory (GiB)'),
             ('compile_seconds','Training JIT compile time (s)'))
    for ax,(key,title) in zip(axes.ravel(),metrics,strict=True):
        for index,dtype in enumerate(('float32','float64')):
            data=[next(v for v in rows if v['dtype']==dtype and v['kind']==k)[key] for k in KINDS]
            bars=ax.bar(x+(index-.5)*width,data,width,label=dtype.upper())
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        ax.set_xticks(x,['Nonlocal','Local tensor','Local joint'])
        ax.set_title(title);ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Corrosion throughput: same network, IC/BC, active PDE count and SOAP/GradNorm\nSampling included; 50 warm-up + 75 measured updates; one GPU worker at a time')
    fig.savefig(OUT.with_suffix('.png'),dpi=180)
    fig.savefig(OUT.with_suffix('.svg'))
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,key,title in zip(axes,('mean_update_ms','pde_only_ms'),('Complete update (ms)','PDE-only loss evaluation (ms)'),strict=True):
        xx=np.arange(4)
        for index,dtype in enumerate(('float32','float64')):
            values=[next(v for v in rows if v['kind']==k and v['dtype']==dtype)[key] for k in KINDS]
            values.append(next(v for v in result['ordinary_graph_rows'] if v['dtype']==dtype)[key])
            bars=ax.bar(xx+(index-.5)*width,values,width,label=dtype.upper())
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        ax.set_xticks(xx,['Nonlocal\ndeterministic','Local\ntensor','Local\njoint','Nonlocal\nordinary control'])
        ax.set_title(title);ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Compiler policy control: ordinary graph timing is supplementary, not a new stable default')
    fig.savefig(OUT.with_name(OUT.name+'-policy.png'),dpi=180)
    fig.savefig(OUT.with_name(OUT.name+'-policy.svg'))
    plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)
    print('Ordinary graph rows',json.dumps(result['ordinary_graph_rows'],indent=2),flush=True)
    for dtype in ('float32','float64'):
        print(dtype,'joint time-only control',json.dumps(records[dtype]['local_joint']['joint_time_only_control'],indent=2),flush=True)


if __name__=='__main__':main()
