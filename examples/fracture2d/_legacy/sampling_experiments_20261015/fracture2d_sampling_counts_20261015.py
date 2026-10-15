"""Count exact sampled supports on the real fracture2d graph; no training."""
import csv
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from scipy.sparse import load_npz
from examples.fracture2d.configs.train_base import get_config
from examples.fracture2d.sampling import (make_sampling_rows,sample_core_halo,
    crack_sampling_probabilities,build_sampling_pool)

c=get_config();a=c.sample
out=Path('/tmp/fracture2d_sampling_counts_20261015');out.mkdir(exist_ok=True)
with np.load(Path(c.mesh.prepared_dir)/'mesh.npz',allow_pickle=False) as data:
    n=int(data['n_material']);xy=data['coordinates'][:n].copy()
D=load_npz(Path(c.mesh.prepared_dir)/'damage_operator.npz').tocsr()
D.sum_duplicates();D.eliminate_zeros()
assert D.shape==(n,n) and D.dtype==np.float64
width=int(np.max(np.diff(D.indptr)))
rows=make_sampling_rows(D,row_width=width)
p=np.asarray(crack_sampling_probabilities(jnp.asarray(xy),start=a.start,end=a.end,
    width=a.width,boost=a.boost,power=a.power))
assert p.dtype==np.float64 and p.min()>0 and np.isclose(p.sum(),1.)
A=D.copy();A.data[:]=1.;A.setdiag(1.);A.eliminate_zeros()
q=np.asarray(A.T@p).ravel()

def summary(values):
    values=np.asarray(values)
    return dict(mean=float(values.mean()),std=float(values.std()),min=int(values.min()),
        p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),
        p99=float(np.quantile(values,.99)),max=int(values.max()))

def expected_union(probability,count):
    coverage=np.asarray(A.T@probability).ravel()
    return float(np.sum(-np.expm1(count*np.log1p(-coverage))))

records=[]
for block in range(4):
    rng=np.random.default_rng(np.random.SeedSequence([a.seed,block]))
    cores=rng.choice(n,size=(a.pool_size,a.core_count),replace=True,p=p)
    for slot,core in enumerate(cores):
        pack=sample_core_halo(rows,core,node_capacity=a.halo_capacity)
        unique_core=np.unique(core).size
        union=int(pack['node_mask'].sum())
        # Independently gather original CSR support,not packed zero padding.
        direct=np.unique(np.concatenate((core,D[core].indices)))
        assert np.array_equal(pack['global_ids'][:union],direct)
        records.append(dict(block=block,slot=slot,drawn_core=a.core_count,unique_core=int(unique_core),
            halo_only=union-int(unique_core),unique_union=union,padded_forward=a.halo_capacity,
            padding=a.halo_capacity-union))
    if block==0:
        # Exercise the exact pool-builder path for a subset without retaining huge pools.
        _,counts=build_sampling_pool(rows,cores[:8],p,node_capacity=a.halo_capacity)
        np.testing.assert_array_equal(counts,[r['unique_union'] for r in records[:8]])

alternatives=[]
for count in [256,512,1024,2048,4096]:
    rng=np.random.default_rng(np.random.SeedSequence([a.seed,0]))
    draws=rng.choice(n,size=(256,count),replace=True,p=p)
    unions=[];unique_cores=[]
    for core in draws:
        union=np.unique(np.concatenate((core,D[core].indices))).size
        unions.append(union);unique_cores.append(np.unique(core).size)
    alternatives.append(dict(core_count=count,batches=256,unique_core=summary(unique_cores),
        union=summary(unions),exact_expected_union=expected_union(p,count),
        configured_padded_forward=a.halo_capacity,capacity_sufficient_for_tested_batches=max(unions)<=a.halo_capacity))

uniform=np.full(n,1/n,dtype=np.float64)
rng=np.random.default_rng(np.random.SeedSequence([a.seed,0]))
uniform_unions=[np.unique(np.concatenate((core,D[core].indices))).size
    for core in rng.choice(n,size=(256,a.core_count),replace=True,p=uniform)]
result=dict(created_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    config=a.to_dict(),material_nodes=n,operator_nonzeros=int(D.nnz),row_width=width,
    nonzeros_per_row=summary(np.diff(D.indptr)),sampled_batches=len(records),
    statistics={key:summary([r[key] for r in records]) for key in
        ['unique_core','halo_only','unique_union','padding']},
    exact_expected_unique_core=float(np.sum(-np.expm1(a.core_count*np.log1p(-p)))),
    exact_expected_unique_union=expected_union(p,a.core_count),
    configured_MLP_rows_per_sampled_forward=a.halo_capacity,
    configured_MLP_row_fraction=a.halo_capacity/n,
    configured_MLP_row_reduction=1-a.halo_capacity/n,
    full_to_sampled_MLP_rows=n/a.halo_capacity,
    effective_union_fraction=float(np.mean([r['unique_union'] for r in records])/n),
    alternative_core_counts=alternatives,
    uniform_control=dict(batches=256,union=summary(uniform_unions),exact_expected_union=expected_union(uniform,a.core_count)),
    inner_solve=dict(updates=c.train.inner_steps,global_forward_checks=2,
        full_total_rows=n*(c.train.inner_steps+2),
        random_total_rows=a.halo_capacity*c.train.inner_steps+2*n,
        note='image thenfull loss both callthe network at endofinner;featurescomputedonce globally'),
    notes=['Supportunion includescore;halo_only excludesuniquecore.',
           'One batched vmap invocation evaluates32768point rows;not32768serialPythoncalls.',
           'No network/optimizer training performed;row counts are not wall-time benchmarks.',
           'Sampling probability remains static geometry-based and detached;pool_size is not forward-batch size.'])
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'batches.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps(result,indent=2))
