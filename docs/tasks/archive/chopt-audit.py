"""Read-only replay/metrics of optimization controls; supervised probes excluded."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.window_training import field_metrics

BASE=Path('/root/autodl-tmp/tf-logs/confined_ch2d');OUT=Path(__file__).parent
POINT_RUNS={name:BASE/('20261027-chopt-'+name+'-01') for name in
    ('blend015','causal-additive','logtime-causal','metric','eliminated','elim-blend8')}
POINT_RUNS['original']=BASE/'20261027-coarsening10-nonlocal-01'
LEGACY_RUNS={name:BASE/('20261027-confined-'+name+'-01') for name in ('nonlocal','local-ad')}
WINDOW_RUNS={'windows30-42':BASE/'20261027-chopt-windows-01',
    'windows30-2026':BASE/'20261027-chopt-windows-2026-01',
    'windows15-42':BASE/'20261027-chopt-windows15-01'}


def structure(phi,mu,ref):
    v=ref['volumes'];out={}
    for key,a in [('phi',phi),('mu',mu)]:
        for target in ('local','nonlocal'):
            b=ref[key+'_'+target]
            out[key+'_'+target+'_positive']=relative(a[1:],b[1:],v)
            out[key+'_'+target+'_terminal_error_rms']=float(np.sqrt(np.sum(v*(a[-1]-b[-1])**2)/v.sum()))
            out[key+'_'+target+'_terminal_reference_rms']=float(np.sqrt(np.sum(v*b[-1]**2)/v.sum()))
    out['terminal_pure_fraction']=float(v[abs(phi[-1])>.9].sum()/v.sum())
    out['phase_range']=[float(phi.min()),float(phi.max())]
    return out


def main():
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('CPU replay required')
    @eqx.filter_jit
    def evaluate(net,x,v,mean,times,D):
        return jax.vmap(lambda t:predict_fields(net,x,t,mean,v,D))(times)
    result=dict(status='passed',scope='Physics-trained models only. Supervised representation probes are separate diagnostics, not optimized PINNs.',runs={},legacy_replays={})
    operators={}
    for name,run in {**POINT_RUNS,**{'legacy-'+k:v for k,v in LEGACY_RUNS.items()}}.items():
        c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
        x=jnp.asarray(ref['points']);v=jnp.asarray(ref['volumes']);mean=jnp.dot(v,initial_field(x,c,xp=jnp))/v.sum()
        key,_=jax.random.split(jax.random.PRNGKey(c.train.seed));net=eqx.tree_deserialise_leaves(run/'model_final.eqx',_build_network(c,key))
        if net.chemical_mode=='eliminated':
            if c.reference.path not in operators:operators[c.reference.path]=RegionalLaplacian(ref['points'],ref['volumes'],c)
            D=operators[c.reference.path]
        else:D=None
        actual=np.asarray(evaluate(net,x,v,mean,jnp.asarray(ref['time']),D));errors={}
        with np.load(run/'prediction.npz',allow_pickle=False) as saved:
            for j,k in enumerate(('phi','mu')):
                np.testing.assert_allclose(actual[...,j],saved[k],atol=1e-8,rtol=1e-8)
                errors[k]=float(np.max(abs(actual[...,j]-saved[k])))
        if name.startswith('legacy-'):
            result['legacy_replays'][name]=dict(run=str(run),max_absolute=errors);continue
        row=dict(run=str(run),role='physics_pinn',cpu_replay_max_absolute=errors,
            final_metrics=field_metrics(actual[...,0],actual[...,1],ref),structure=structure(actual[...,0],actual[...,1],ref),
            summary=json.loads((run/'summary.json').read_text()),best=json.loads((run/'best_metrics.json').read_text()))
        result['runs'][name]=row;print(name,row['final_metrics']['pooled_rel_l2_phi_local'],errors,flush=True)
    for name,run in WINDOW_RUNS.items():
        c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
        replay=json.loads((run/'replay.json').read_text())
        if Path(replay['checkpoint'])!=run/'trajectory.json':raise ValueError('Need explicit complete-window CPU replay first')
        with np.load(run/'replayed_prediction.npz') as d:phi=d['phi'];mu=d['mu']
        result['runs'][name]=dict(run=str(run),role='physics_pinn',replay=replay,
            final_metrics=field_metrics(phi,mu,ref),structure=structure(phi,mu,ref),summary=json.loads((run/'summary.json').read_text()))
    paired=[WINDOW_RUNS[k] for k in ('windows30-42','windows30-2026')]
    configs=[json.loads((r/'config.json').read_text()) for r in paired]
    for c in configs:del c['train']['seed'],c['log']['run_name']
    assert configs[0]==configs[1]
    sources=[{str(p.relative_to(r/'source')):p.read_bytes() for p in (r/'source').rglob('*') if p.is_file()} for r in paired]
    assert sources[0]==sources[1]
    result['windows30_matching_sources']=len(sources[0]);result['windows30_only_config_differences']=['train.seed','log.run_name']
    result['supervised_diagnostics']=[json.loads((OUT/('chopt-representation-'+name+'.json')).read_text()) for name in ('additive','blend')]
    (OUT/'chopt-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('All CPU controls and window source/config parity pass',flush=True)


if __name__=='__main__':main()
