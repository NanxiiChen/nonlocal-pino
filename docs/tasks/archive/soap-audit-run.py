"""Run actual SOAP classes with fixed streams and an optional COMMON basis tape."""
from pathlib import Path
import argparse,ast,importlib,importlib.util,json,sys,math
from itertools import chain
from abc import ABC,abstractmethod
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/soap-implementation-audit';SOURCE=OUT/'sources'


def basis(n):return np.linalg.qr(np.random.default_rng(900+n).normal(size=(n,n)))[0]
def rotation(n):return np.linalg.qr(np.random.default_rng(1900+n).normal(size=(n,n)))[0]


def extract_class(path,name,namespace):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[name]


def torch_setup(backend):
    import torch
    torch.set_num_threads(1)
    if backend=='author':
        spec=importlib.util.spec_from_file_location('audit_author_soap',SOURCE/'author-soap.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return torch,module.SOAP
    # Exact source classes; unrelated package imports/foreach paths are omitted.
    # Valid dense real tensors, merge_dims=False; no helper algorithm is mocked.
    namespace=dict(torch=torch,math=math,chain=chain,ABC=ABC,abstractmethod=abstractmethod,
                   Optimizer=torch.optim.Optimizer,__name__='audit_independent_soap')
    exec((SOURCE/'independent-exception.py').read_text(),namespace)
    extract_class(SOURCE/'independent-base.py','BaseOptimizer',namespace)
    klass=extract_class(SOURCE/'independent-soap.py','SOAP',namespace)
    return torch,klass


def torch_shared(optimizer,backend,torch):
    def initial(matrices):
        return [torch.tensor(basis(len(m)),dtype=m.dtype,device=m.device) if len(m)>0 else [] for m in matrices]
    def refresh(state):
        v=state['exp_avg_sq'];answer=[]
        for axis,q in enumerate(state['Q']):
            if len(q)==0:answer.append([]);continue
            n=len(q);perm=torch.arange(n-1,-1,-1,device=q.device)
            v=v.index_select(axis,perm)
            answer.append(q[:,perm]@torch.tensor(rotation(n),dtype=q.dtype,device=q.device))
        state['exp_avg_sq']=v
        return answer
    optimizer.get_orthogonal_matrix=initial
    if backend=='author':optimizer.get_orthogonal_matrix_QR=lambda state,maxdim,merge:refresh(state)
    else:optimizer.get_orthogonal_matrix_qr=lambda state,maxdim,merge:refresh(state)


def jax_shared(module):
    import jax.numpy as jnp
    old=(module.get_orthogonal_matrix,module.get_orthogonal_matrix_QR)
    def initial(m,qr_dtype):
        return None if m is None else jnp.asarray(basis(m.shape[0]),dtype=m.dtype)
    def refresh(GG,Q,v,precision,qr_dtype):
        answer=[]
        for axis,q in enumerate(Q.matrices):
            if q is None:answer.append(None);continue
            n=q.shape[0];perm=jnp.arange(n-1,-1,-1)
            v=jnp.take(v,perm,axis=axis)
            answer.append(q[:,perm]@jnp.asarray(rotation(n),dtype=q.dtype))
        return module.Preconditioner(answer),v
    module.get_orthogonal_matrix=initial;module.get_orthogonal_matrix_QR=refresh
    return old


def run_torch_case(backend,torch,klass,case,initial,gradients):
    dtype=torch.float64 if case['dtype']=='float64' else torch.float32
    # EXPLICIT precision setup: author allocations use torch's default dtype;
    # its eig/QR methods still intentionally force FP32 as in actual source.
    torch.set_default_dtype(dtype)
    p=torch.nn.Parameter(torch.tensor(initial,dtype=dtype))
    settings=dict(lr=.001,betas=(.9,.99),weight_decay=case['weight_decay'],
        precondition_frequency=case['frequency'],precondition_1d=case['precondition_1d'],
        merge_dims=False,correct_bias=case['correct_bias'],eps=1e-8)
    settings['max_precond_dim' if backend=='author' else 'max_precondition_dim']=case['maxdim']
    optimizer=klass([p],**settings)
    if case['mode']=='shared':torch_shared(optimizer,backend,torch)
    records=[]
    for g in gradients:
        p.grad=torch.tensor(g,dtype=dtype);optimizer.step();state=optimizer.state[p]
        if backend=='author':physical=optimizer.project_back(state['exp_avg'],state)
        else:physical=state['exp_avg']
        item=dict(parameter=p.detach().numpy().copy(),m_physical=physical.detach().numpy().copy(),v=state['exp_avg_sq'].detach().numpy().copy())
        for axis,m in enumerate(state['GG']):
            if len(m)>0:item['GG_'+str(axis)]=m.detach().numpy().copy()
        records.append(item)
    return {key:np.stack([r[key] for r in records]) for key in records[0]}


def run_jax_case(module,case,initial,gradients):
    import jax,jax.numpy as jnp,optax
    jax.config.update('jax_enable_x64',case['dtype']=='float64')
    old=jax_shared(module) if case['mode']=='shared' else None
    try:
        dtype=np.dtype(case['dtype']);p=jnp.asarray(initial,dtype=dtype)
        optimizer=module.soap(learning_rate=.001,b1=.9,b2=.99,eps=1e-8,
             precondition_frequency=case['frequency'],precondition_1d=case['precondition_1d'],
             max_precond_dim=case['maxdim'],weight_decay=case['weight_decay'],correct_bias=case['correct_bias'])
        state=optimizer.init(p)
        @jax.jit
        def update(parameter,state,gradient):
            delta,state=optimizer.update(gradient,state,parameter)
            return optax.apply_updates(parameter,delta),state
        records=[]
        for g in gradients:
            p,state=update(p,state,jnp.asarray(g,dtype=dtype));s=state[0]
            item=dict(parameter=np.asarray(p),m_physical=np.asarray(module.project_back(s.exp_avg,s.Q)),v=np.asarray(s.exp_avg_sq))
            for axis,m in enumerate(s.GG.matrices):
                if m is not None:item['GG_'+str(axis)]=np.asarray(m)
            records.append(item)
        return {key:np.stack([r[key] for r in records]) for key in records[0]}
    finally:
        if old is not None:module.get_orthogonal_matrix,module.get_orthogonal_matrix_QR=old


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['jax','author','independent'],required=True)
    args=parser.parse_args();meta=json.loads((OUT/'fixtures.json').read_text())
    inputs=np.load(OUT/'fixtures.npz');outputs={};descriptions=[]
    if args.backend=='jax':
        import jax
        if jax.default_backend()!='cpu':raise ValueError('CPU audit')
        module=importlib.import_module('soap_jax.soap')
        run=lambda c,p,g:run_jax_case(module,c,p,g)
        version=jax.__version__
    else:
        torch,klass=torch_setup(args.backend)
        run=lambda c,p,g:run_torch_case(args.backend,torch,klass,c,p,g)
        version=torch.__version__
    for case in meta['cases']:
        key=case['key'];initial=inputs[key+'_initial'];gradients=inputs[key+'_gradient']
        result=run(case,initial,gradients)
        np.testing.assert_array_equal(result['parameter'][0],initial)
        if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError(key)
        for field,array in result.items():outputs[key+'__'+field]=array
        descriptions.append(dict(case=key,status='completed',steps=len(gradients)))
        print(args.backend,key,'completed',flush=True)
    # Sensitivity tests use actual measured FP64 first-gradient perturbations;
    # common second gradient is synthetic, identical for all repeats/backends.
    sensitivities=[]
    measured=inputs['measured_initial_gradients'];common=inputs['common_second_gradient']
    streams=[('measured_fp64',measured,'float64')]
    g=measured[0].astype(np.float32);perturbations=[g.copy()]
    for i in [1,5,17,63]:
        candidate=g.copy();candidate.ravel()[i]=np.nextafter(candidate.ravel()[i],np.float32(np.inf));perturbations.append(candidate)
    streams.append(('one_ulp_fp32',np.stack(perturbations),'float32'))
    for label,variants,dtype in streams:
        case=dict(mode='native',dtype=dtype,precondition_1d=False,maxdim=128,weight_decay=0.,correct_bias=True,frequency=5)
        initial=np.zeros((2,64),dtype=dtype);deltas=[]
        for first in variants:
            result=run(case,initial,np.stack((first,common.astype(dtype))))
            deltas.append(result['parameter'][-1])
        baseline=deltas[0];norm=np.linalg.norm(baseline)
        sensitivity=[dict(max_absolute=float(np.max(np.abs(x-baseline))),relative_L2=float(np.linalg.norm(x-baseline)/norm)) for x in deltas[1:]]
        sensitivities.append(dict(label=label,changes=sensitivity))
        outputs['sensitivity_'+label]=np.stack(deltas)
    np.savez_compressed(OUT/(args.backend+'-streams.npz'),**outputs)
    (OUT/(args.backend+'-summary.json')).write_text(json.dumps(dict(backend=args.backend,runtime_version=version,cases=descriptions,sensitivities=sensitivities,precision_note='Torch default dtype matched to case; author eig/QR and independent QR retain source casts. Shared-basis cases override basis selection ONLY.'),indent=2)+'\n')


if __name__=='__main__':main()
