"""One-geometry checkpoint diagnostic; labels used only to measure field error."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse,linalg
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,prepare,phase,apply_operator
from examples.confined_ch2d.operator_problem import fixed_initial


def probe(run,step,index,output):
    jax.config.update('jax_enable_x64',True)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if c.operator.metric!='biharmonic_resolvent':raise ValueError('This historical probe supports only the biharmonic metric; use the norm-specific defect diagnostic')
    root=Path(c.operator.dataset_path);manifest=json.loads((root/'manifest.json').read_text())
    with np.load(root/manifest['conditions'][index]['file'],allow_pickle=False) as f:
        reference={key:f[key] for key in f.files}
    keys=('points','volumes','neighbors','coefficients','metric','condition','condition_weights')
    data={k:jnp.asarray(reference[k]) for k in keys}
    data['initial']=jnp.asarray(fixed_initial(reference['points'],c,xp=np))
    model=eqx.tree_deserialise_leaves(run/f'ckpt_{step:06d}.eqx',build_network(c,jax.random.key(0)))
    @eqx.filter_jit
    def evaluate(net,d,times):
        h=prepare(net,d)
        def at(t):
            phi,dt=jax.jvp(lambda tt:phase(net,h,tt,d,c),(t,),(jnp.ones_like(t),))
            mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*apply_operator(phi,d)
            r=dt+c.physical.mobility*apply_operator(mu,d)
            return phi,r,c.physical.terminal_time*(d['metric']@r)
        return jax.vmap(at)(times)
    times=np.linspace(0,c.physical.terminal_time,257)
    padded=np.pad(times,(0,(-len(times))%32),mode='edge');parts=[]
    for start in range(0,len(padded),32):
        parts.append(tuple(np.asarray(a) for a in evaluate(model,data,jnp.asarray(padded[start:start+32]))))
    phi,r,transformed=[np.concatenate([p[k] for p in parts])[:len(times)] for k in range(3)]
    v=reference['volumes'];n=len(v);nb=reference['neighbors'];co=reference['coefficients']
    off=sparse.coo_matrix((co.ravel(),(np.repeat(np.arange(n),nb.shape[1]),nb.ravel())),shape=(n,n)).tocsr()
    A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
    rv=np.sqrt(v);S=(sparse.diags(rv)@A@sparse.diags(1/rv)).toarray()
    np.testing.assert_allclose(S,S.T,atol=1e-12)
    values,U=linalg.eigh(S)
    growing=(values>1e-8)&(values<c.physical.beta/c.physical.epsilon**2)
    growth=c.physical.mobility*values*(c.physical.beta-c.physical.epsilon**2*values)
    weights=1/(1+c.physical.terminal_time*c.physical.mobility*c.physical.epsilon**2*values**2)
    projection=phi@((rv[:,None]*U)[:,growing])
    rr=r@((rv[:,None]*U)[:,growing])
    truth=reference['phi']@((rv[:,None]*U)[:,growing])
    selected=np.arange(0,257,64)
    for i,j in enumerate(selected):np.testing.assert_allclose(times[j],reference['time'][i])
    norm=lambda x:np.sqrt(np.sum(v*x*x,axis=-1)/v.sum())
    top=np.argsort(growth)[-5:][::-1]
    result=dict(run=str(run),checkpoint_step=step,index=index,
        reference='Same native trajectory; diagnostic only, no optimizer update',
        time=times.tolist(),phase_rms=norm(phi).tolist(),raw_residual_rms=norm(r).tolist(),
        transformed_residual_rms=norm(transformed).tolist(),
        growing_subspace_residual_norm=np.linalg.norm(rr,axis=1).tolist(),
        reference_phase_rms=norm(reference['phi']).tolist(),
        phase_relative_error_by_saved_time=(norm(phi[selected]-reference['phi'])/norm(reference['phi'])).tolist(),
        growing_subspace_relative_error_by_saved_time=(np.linalg.norm(projection[selected]-truth,axis=1)/np.linalg.norm(truth,axis=1)).tolist(),
        growing_modes=int(growing.sum()),
        fastest_linear_modes=[dict(laplacian_eigenvalue=float(values[i]),growth=float(growth[i]),
            T_amplification=float(np.exp(c.physical.terminal_time*growth[i])),current_residual_weight=float(weights[i])) for i in top],
        scope='Frozen checkpoint and exact graph linearization around phi=0; nonlinear late flow is not this linear model')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('checkpoint_step','index','phase_relative_error_by_saved_time','growing_subspace_relative_error_by_saved_time','fastest_linear_modes')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();probe(a.run,a.step,a.index,a.output)
