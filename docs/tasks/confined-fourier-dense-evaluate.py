"""CPU validation of a fixed checkpoint against61native times; no optimizer."""
import argparse
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,fields
from examples.confined_ch2d.operator_problem import fixed_initial
from examples.confined_ch2d.problem import relative
from examples.confined_ch2d.operator_conditioning import initial_rhs_enabled,initial_operator_fields
from examples.confined_ch2d.operator_precision import compute_arrays
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate(run,dense,step,output):
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    manifest=json.loads((Path(c.operator.dataset_path)/'manifest.json').read_text())
    validation=json.loads((dense/'validation.json').read_text())
    if validation['status']!='passed' or validation['indices']!=manifest['test_indices']:
        raise ValueError('Passed matching dense validation required')
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    checkpoint=run/f'ckpt_{step:06d}.eqx'
    model=eqx.tree_deserialise_leaves(checkpoint,build_network(c,jax.random.key(0)))
    inputs=[];references=[]
    for i in manifest['test_indices']:
        with np.load(Path(c.operator.dataset_path)/manifest['conditions'][i]['file'],allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
        with np.load(dense/f'native_{i:03d}.npz',allow_pickle=False) as f:r={k:f[k] for k in f.files}
        for k in ('points','volumes','holes'):np.testing.assert_array_equal(d[k],r[k])
        np.testing.assert_allclose(r['phi'][::15],d['phi'],atol=1e-10,rtol=0)
        inputs.append(d);references.append(r)
    nmax=max(len(d['points']) for d in inputs);kmax=max(d['neighbors'].shape[1] for d in inputs)
    times=references[0]['time'];predict=eqx.filter_jit(lambda net,data:fields(net,data,jnp.asarray(times),c))
    horizon=c.physical.terminal_time
    if 'temporal' in c.sample and c.sample.temporal.curriculum:
        s=c.sample.temporal;progress=min((step-1)/(s.warmup_iterations-1),1.)
        horizon=s.initial_horizon*(c.physical.terminal_time/s.initial_horizon)**progress
    prefix=times<=horizon;rows=[];output.mkdir(parents=True,exist_ok=False)
    for index,d,r in zip(manifest['test_indices'],inputs,references,strict=True):
        n,k=d['neighbors'].shape;nb=np.repeat(np.arange(nmax)[:,None],kmax,axis=1)
        co=np.zeros((nmax,kmax));nb[:n,:k]=d['neighbors'];co[:n,:k]=d['coefficients']
        data=dict(points=np.pad(d['points'],((0,nmax-n),(0,0))),volumes=np.pad(d['volumes'],(0,nmax-n)),
            initial=np.pad(fixed_initial(d['points'],c,xp=np),(0,nmax-n)),neighbors=nb,coefficients=co,
            condition=d['condition'],condition_weights=d['condition_weights'])
        if initial_rhs_enabled(c):data['domain_fields']=initial_operator_fields(data,c)
        pred=np.asarray(predict(model,compute_arrays(jax.tree.map(jnp.asarray,data),c)))[:,:n]
        phi,mu=pred[...,0],pred[...,1];v=r['volumes']
        if c.train.compute_dtype=='float32':
            np.testing.assert_array_equal(phi[0],r['phi'][0].astype(np.float32))
        else:np.testing.assert_allclose(phi[0],r['phi'][0],atol=1e-13,rtol=0)
        action=np.sum(d['coefficients'][None,:,:]*(phi[:,:,None]-phi[:,d['neighbors']]),axis=-1)
        energy=np.sum(v*(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*phi*action),axis=1)
        row=dict(index=index,phi_pooled=relative(phi,r['phi'],v),phi_terminal=relative(phi[-1],r['phi'][-1],v),
            phi_by_time=[relative(a,b,v) for a,b in zip(phi,r['phi'],strict=True)],
            mu_pooled=relative(mu,r['mu'],v),trained_prefix_phi_pooled=relative(phi[prefix],r['phi'][prefix],v),
            mass_max_drift=float(abs(phi@v/v.sum()-r['phi'][0]@v/v.sum()).max()),
            energy_max_increase=float(max(0,np.diff(energy).max())))
        rows.append(row)
    result=dict(run=str(run),checkpoint=str(checkpoint),step=step,reference=str(dense),time=times.tolist(),
        training_horizon=horizon,conditions=rows,scope='Eight reused validation geometries; no field supervision or checkpoint selection')
    for key in ('phi_pooled','phi_terminal','mu_pooled','trained_prefix_phi_pooled'):
        result[key]=float(np.mean([r[key] for r in rows]))
    result['phi_by_time']=np.mean([r['phi_by_time'] for r in rows],axis=0).tolist()
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.plot(times,100*np.asarray(result['phi_by_time']));ax.axhline(10,color='black',ls=':',lw=.8)
    ax.axvline(horizon,color='grey',ls='--',label='Training horizon')
    ax.set(xlabel='Physical time',ylabel='Phase relative L2 (%)',title=f'8 geometry native validation, step {step}')
    ax.grid(alpha=.2);ax.legend();fig.savefig(output/'error_by_time.png',dpi=180);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('step','training_horizon','phi_pooled','phi_terminal','trained_prefix_phi_pooled')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dense','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    a=p.parse_args();evaluate(a.run,a.dense,a.step,a.output)
