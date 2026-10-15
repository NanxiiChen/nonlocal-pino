"""Finite exact-support pool resource probe; no model/optimizer training."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.sparse import load_npz
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from examples.fracture2d.configs.train_base import get_config
from examples.fracture2d.sampling import make_sampling_rows,sample_core_halo,build_sampling_pool,crack_sampling_probabilities

parser=argparse.ArgumentParser();parser.add_argument('--entries',type=int,required=True);parser.add_argument('--gpu',action='store_true')
a=parser.parse_args();c=get_config();settings=c.sample

def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])/1024
    raise RuntimeError('VmRSS is unavailable')

if a.gpu:
    assert jax.devices()[0].platform=='gpu'
else:assert jax.devices()[0].platform=='cpu'
D=load_npz(Path(c.mesh.prepared_dir)/'damage_operator.npz').tocsr()
D.sum_duplicates();D.eliminate_zeros()
with np.load(Path(c.mesh.prepared_dir)/'mesh.npz',allow_pickle=False) as m:
    xy=m['coordinates'][:int(m['n_material'])].copy()
rows=make_sampling_rows(D,row_width=int(np.max(np.diff(D.indptr))))
p=np.asarray(crack_sampling_probabilities(jnp.asarray(xy),start=settings.start,end=settings.end,
    width=settings.width,boost=settings.boost,power=settings.power))
rng=np.random.default_rng(np.random.SeedSequence([settings.seed,0]))
start=time.perf_counter();cores=rng.choice(len(xy),size=(a.entries,settings.core_count),replace=True,p=p)
draw_seconds=time.perf_counter()-start
base_rss=rss_mib()
start=time.perf_counter()
counts=np.array([sample_core_halo(rows,core,node_capacity=settings.halo_capacity)['node_mask'].sum()
    for core in cores],dtype=np.int32)
count_seconds=time.perf_counter()-start
capacity=int(counts.max())
start=time.perf_counter();pool,actual=build_sampling_pool(rows,cores,p,node_capacity=capacity)
build_seconds=time.perf_counter()-start
np.testing.assert_array_equal(actual,counts)
assert pool['global_ids'].shape==(a.entries,capacity)
fields={name:dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes) for name,value in pool.items()}
total=sum(value.nbytes for value in pool.values())
result=dict(created_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),entries=a.entries,
    core_count=settings.core_count,material_nodes=len(xy),tight_capacity=capacity,
    unique_union_mean=float(counts.mean()),unique_union_std=float(counts.std()),
    unique_union_variance=float(counts.var()),unique_union_min=int(counts.min()),
    unique_union_max=capacity,mean_padding=capacity-float(counts.mean()),
    sampling_seconds=draw_seconds,count_support_seconds=count_seconds,
    pack_pool_seconds=build_seconds,two_pass_total_seconds=draw_seconds+count_seconds+build_seconds,
    pool_bytes=total,pool_MiB=total/2**20,fields=fields,
    rss_before_pool_MiB=base_rss,rss_after_pool_MiB=rss_mib(),
    peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    backend=jax.devices()[0].platform,
    notes=['Measuredhostbuilder iscurrentcode;two-pass determinesmaxthenrebuilds support.',
           'PeakRSSincludesPython/JAXlibraries andtemporaryarrays;notallpoolbytes.',
           'Current builder useslistofpackedentries thenstacks;both coexisttransiently.',
           'No networkforward/backwardoroptimizer training;GPU storageonly.'])
if a.gpu:
    jax.block_until_ready(jax.device_put(jnp.array(0.,dtype=jnp.float64)))
    before=jax.devices()[0].memory_stats()
    start=time.perf_counter();device_pool=jax.device_put(pool);jax.block_until_ready(device_pool)
    transfer=time.perf_counter()-start
    after=jax.devices()[0].memory_stats()
    assert sum(value.nbytes for value in jax.tree.leaves(device_pool))==total
    result['gpu_upload_seconds']=transfer
    result['gpu_bytes_in_use_before']=int(before['bytes_in_use'])
    result['gpu_bytes_in_use_after']=int(after['bytes_in_use'])
    result['gpu_live_buffer_delta_MiB']=(after['bytes_in_use']-before['bytes_in_use'])/2**20
    result['gpu_device']=str(jax.devices()[0])
out=Path('/tmp/fracture2d_finite_pool_resources_20261015');out.mkdir(exist_ok=True)
path=out/f"{'gpu' if a.gpu else 'cpu'}_{a.entries}.json"
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
