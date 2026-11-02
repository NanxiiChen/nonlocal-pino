"""Compare actual histories and audit the recorded matched-control streams."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def load(path):return json.loads(path.read_text())


def main(tag,phase_tag,coordinate_tag,incomplete_map_only):
    root=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
    output=Path('/root/autodl-tmp/data/channel_nsch2d')/(f'gradnorm-comparison-{tag}' if phase_tag is None else f'phase-comparison-{phase_tag}')
    if coordinate_tag is not None:
        output=Path('/root/autodl-tmp/data/channel_nsch2d')/f'coordinate-comparison-{coordinate_tag}'
    output.mkdir(parents=True,exist_ok=True)
    paths=dict(uniform_stream=root/f'{tag}-channel-stream-uniform-01',
        gradnorm_direct=root/f'{tag}-channel-direct-gradnorm-01',
        gradnorm_stream=root/f'{tag}-channel-stream-gradnorm-01')
    if phase_tag is not None:
        paths.update(stagger_stream=root/f'{phase_tag}-channel-stream-stagger-01',
            regions_stream=root/f'{phase_tag}-channel-stream-regions-01')
    if coordinate_tag is not None:
        paths.update(tanh_preconditioned_stream=root/f'{coordinate_tag}-channel-stream-tanh-preconditioned-01')
        if not incomplete_map_only:paths['tanh_stream']=root/f'{coordinate_tag}-channel-stream-tanh-01'
    elif incomplete_map_only:raise ValueError('Incomplete map-only scope requires coordinate-tag')
    actual={};results={}
    for name,path in paths.items():
        summary=load(path/'summary.json');audit=load(path/'validation.json')
        if summary['status']!='completed':raise ValueError(f'Incomplete control: {path}')
        with np.load(path/'sampling.npz',allow_pickle=False) as f:actual[name]={k:f[k] for k in f.files}
        cfg=load(path/'config.json');history=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
        results[name]=dict(run=str(path),summary=summary,validation=audit,history=history)
        if 'balance' in cfg:
            b=cfg['balance'];rows=[json.loads(s) for s in (path/'balance.jsonl').read_text().splitlines()]
            expected_steps=list(range(1,cfg['train']['iterations']+1,b['update_every']))
            assert [r['step'] for r in rows]==expected_steps
            previous=np.ones(len(rows[0]['weights']));maximum=0.
            for row in rows:
                norms=np.clip(np.asarray(row['gradient_l1_norms']),b['epsilon'],1/b['epsilon'])
                target=np.clip(norms.mean()/norms,b['epsilon'],b['cap'])
                expected=b['new_weight_fraction']*target+(1-b['new_weight_fraction'])*previous
                measured=np.asarray(row['weights'])
                np.testing.assert_allclose(measured,expected,rtol=2e-6,atol=2e-6)
                maximum=max(maximum,float(np.max(np.abs(measured-expected))));previous=measured
            with np.load(path/'balance_final.npz',allow_pickle=False) as f:
                np.testing.assert_array_equal(f['weights'],previous)
            results[name]['balance_audit']=dict(refreshes=len(rows),maximum_ema_gap=maximum,
                final_weights=previous.tolist(),groups=rows[-1]['groups'])
            for line in (path/'loss.jsonl').read_text().splitlines():
                row=json.loads(line);p=row['parts'];w=row['weights']
                terms=dict(phase=p['phase'],chemical=p['chemical'],momentum=p['momentum_x']+p['momentum_y'],
                    boundary=sum(p[k] for k in ['wall_phase','wall_mu','outlet_traction','outlet_phase','outlet_mu']))
                if 'continuity' in p:terms['continuity']=p['continuity']
                expected=sum(w[k]*terms[k] for k in w)
                np.testing.assert_allclose(row['total'],expected,rtol=3e-6,atol=2e-7)
        if 'stagger' in cfg:
            from examples.channel_nsch2d.partition import stage_at
            settings=type('Settings',(),cfg['stagger'])()
            expected=[stage_at(i,settings)[0] for i in range(cfg['train']['iterations'])]
            counts={k:expected.count(k) for k in ('phase','flow','joint')}
            if summary['stagger']['update_counts']!=counts:raise ValueError('Stagger counts disagree')
            switches=[json.loads(line) for line in (path/'stagger.jsonl').read_text().splitlines()]
            expected_steps=list(range(1,cfg['train']['iterations']+1,settings.switch_every))+[cfg['train']['iterations']]
            if [r['step'] for r in switches]!=expected_steps:raise ValueError('Stagger switch trace is incomplete')
            for row in switches:
                stage,active=stage_at(row['step']-1,settings)
                if stage!=row['stage'] or list(active)!=row['block_active']:raise ValueError('Invalid actual stagger trace')
            phase_names=('phase','chemical','wall_phase','wall_mu','outlet_phase','outlet_mu')
            for line in (path/'loss.jsonl').read_text().splitlines():
                row=json.loads(line);stage,active=stage_at(row['step']-1,settings)
                if row['stage']!=stage or row['block_active']!=list(active):raise ValueError('Loss stage trace disagrees')
                expected_total=sum(v*active[0 if k in phase_names else 1] for k,v in row['parts'].items())
                np.testing.assert_allclose(row['total'],expected_total,rtol=3e-6,atol=2e-7)
            results[name]['stagger_audit']=dict(update_counts=counts,actual_switches_passed=True)
        elif 'balance' not in cfg:
            for line in (path/'loss.jsonl').read_text().splitlines():
                row=json.loads(line)
                np.testing.assert_allclose(row['total'],sum(row['parts'].values()),rtol=3e-6,atol=2e-7)
    baseline=actual['uniform_stream']
    for name,d in actual.items():
        for k in baseline:np.testing.assert_array_equal(d[k],baseline[k])
    with np.load(paths['uniform_stream']/'known_inputs.npz',allow_pickle=False) as base:
        for name,path in paths.items():
            with np.load(path/'known_inputs.npz',allow_pickle=False) as data:
                if data.files!=base.files:raise ValueError('Input schemas differ')
                for k in base.files:np.testing.assert_array_equal(data[k],base[k])
    results['actual_draws_and_known_inputs_identical']=True
    # CPU is selected explicitly by the serial validator for this leaf-only
    # initialization audit; no CPU physical-field equivalence is assumed.
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.network import build_network
    templates={}
    for name,path in paths.items():
        c=ConfigDict(load(path/'config.json'));key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
        net=eqx.tree_deserialise_leaves(path/'model_initial.eqx',build_network(c,key))
        trunk=eqx.tree_at(lambda n:n.output,net,None)
        templates[name]=[np.asarray(a) for a in jax.tree.leaves(trunk) if eqx.is_array(a)]
    for name,leaves in templates.items():
        if len(leaves)!=len(templates['uniform_stream']):raise ValueError('Trunk signatures differ')
        for a,b in zip(leaves,templates['uniform_stream'],strict=True):np.testing.assert_array_equal(a,b)
    for name,path in paths.items():
        if name!='gradnorm_direct':
            assert (paths['uniform_stream']/'model_initial.eqx').read_bytes()==(path/'model_initial.eqx').read_bytes()
            assert (paths['uniform_stream']/'optimizer_initial.eqx').read_bytes()==(path/'optimizer_initial.eqx').read_bytes()
    results['initialization_audit']=dict(common_trunk_exact=True,stream_full_initial_model_bytes_exact=True,
        stream_initial_optimizer_bytes_exact=True,
        common_trunk_arrays=len(templates['uniform_stream']),
        note='Direct representation needs one additional output channel; full direct head equality is not claimed')
    if coordinate_tag is not None:
        mapped_path=root/f'{coordinate_tag}-channel-stream-tanh-01'
        with np.load(mapped_path/'prediction_initial.npz',allow_pickle=False) as a:
            with np.load(paths['tanh_preconditioned_stream']/'prediction_initial.npz',allow_pickle=False) as b:
                np.testing.assert_array_equal(a['fields'],b['fields'])
        results['initialization_audit']['mapped_initial_physical_fields_exact']=True
        if incomplete_map_only:
            assert (mapped_path/'model_initial.eqx').read_bytes()==(paths['tanh_preconditioned_stream']/'model_initial.eqx').read_bytes()
            assert (mapped_path/'optimizer_initial.eqx').read_bytes()==(paths['tanh_preconditioned_stream']/'optimizer_initial.eqx').read_bytes()
            partial=[json.loads(s) for s in (mapped_path/'metrics.jsonl').read_text().splitlines()]
            results['incomplete_map_only']=dict(run=str(mapped_path),history=partial,
                status='Stopped by FP32 inverse-J diagnostic overflow; not a completed5000control',
                last_logged_evaluation=partial[-1]['step'],recoverable_checkpoint=str(mapped_path/'ckpt_002500.eqx'),
                actual_draw_parity_claimed=False,
                note='No sampling.npz or final optimizer was published. Initial model/optimizer/physical fields still match exactly.')
    results['completed_control_names']=list(paths)
    results['scope']='One development FEM geometry/seed; representation and velocity lifting differ. Not a regional-kernel or population comparison.'
    columns=['phi','velocity','pressure'] if phase_tag is None and coordinate_tag is None else ['phi','dispersed_phase','velocity','pressure']
    if phase_tag is None and coordinate_tag is None:
        fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    else:
        fig,axes=plt.subplots(2,2,figsize=(12,7.4),layout='constrained');axes=axes.ravel()
    labels=dict(uniform_stream='Streamfunction, uniform',gradnorm_direct='Direct NS–CH + GradNorm',
        gradnorm_stream='Streamfunction + GradNorm',stagger_stream='Streamfunction + stagger',
        regions_stream='Streamfunction + phase regions',tanh_stream='Scaled tanh, original residual',
        tanh_preconditioned_stream='Scaled tanh + inverse Jacobian')
    for ax,field in zip(axes,columns,strict=True):
        for name in paths:
            rows=results[name]['history']
            ax.semilogy([r['step'] for r in rows],[100*r[field] for r in rows],label=labels[name])
        if incomplete_map_only:
            ax.semilogy([r['step'] for r in partial],[100*r[field] for r in partial],
                linestyle=':',color='0.45',label='Scaled tanh only (incomplete)')
        ax.set(title=field,xlabel='Updates',ylabel='Relative L2 (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output/'convergence.png',dpi=200);fig.savefig(output/'convergence.pdf');plt.close(fig)
    # Independent physical diagnostics, rather than weighted-loss comparisons.
    # FEM area is its assembled integral; network area uses its3977point grid.
    reference=Path(results['uniform_stream']['summary']['metrics']['reference'])
    fem=load(reference.with_suffix('.json'))['diagnostics']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for name in paths:
        audit=results[name]['validation']
        axes[0].plot(audit['section_x'],audit['flux'][-1],label=labels[name])
        axes[1].plot(audit['phase_area_times'],audit['phase_area'],label=labels[name])
    axes[0].axhline(fem[-1]['outlet_flux'],color='k',linestyle='--',label='FEM flow')
    axes[1].plot([d['time'] for d in fem],[d['solid_area'] for d in fem],
        color='k',linestyle='--',label='FEM dispersed phase')
    axes[0].set(xlabel='Channel x',ylabel='Cross-section flow',title='Final-time flow')
    axes[1].set(xlabel='Physical time',ylabel='Dispersed-phase area',
        title='Area diagnostic (separate quadratures)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/'physics.png',dpi=200);fig.savefig(output/'physics.pdf');plt.close(fig)
    results['reference_area_diagnostic']=dict(source=str(reference.with_suffix('.json')),
        stored_legacy_name='solid_area',meaning='dispersed-fluid area, not a solid phase',
        time=[d['time'] for d in fem],area=[d['solid_area'] for d in fem],
        scope='FEM assembled integral versus network mapped-grid integral; open-domain flux balance, not a constant-mass constraint')
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--phase-tag');p.add_argument('--coordinate-tag')
    p.add_argument('--incomplete-map-only',action='store_true')
    args=p.parse_args();main(args.tag,args.phase_tag,args.coordinate_tag,args.incomplete_map_only)
