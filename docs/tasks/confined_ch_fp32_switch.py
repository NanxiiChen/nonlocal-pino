"""Preserve a closed FP64 pair and stop only its verified owned processes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network,fields
from examples.confined_ch2d.operator_problem import fixed_initial
from nonlocal_pinn.training.optimizers import make_optimizer


def stop(journal,output):
    if output.exists():raise FileExistsError(output)
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise RuntimeError('Preservation audit requires CPU')
    before=json.loads(journal.read_text())
    if before['status']!='training':raise RuntimeError('Expected the still-running FP64 training stage')
    run=Path(before['run']);c=ConfigDict(json.loads((run/'config.json').read_text()))
    assert c.train.compute_dtype=='float64' and c.train.soap_b2==.9
    paths=sorted(run.glob('checkpoint_*.json'))
    if not paths:raise FileNotFoundError('No closed paired checkpoint')
    marker=paths[-1];pair=json.loads(marker.read_text());step=pair['step']
    assert pair['kind']=='intermediate_full_state' and step>=19000
    template=build_network(c,jax.random.key(0))
    model=eqx.tree_deserialise_leaves(run/pair['model'],template)
    state=eqx.tree_deserialise_leaves(run/pair['optimizer'],make_optimizer(c).init(eqx.filter(template,eqx.is_array)))
    for tree in (model,state):
        if not all(np.isfinite(a).all() for a in jax.tree.leaves(tree) if eqx.is_array(a)):
            raise FloatingPointError('Preserved FP64 pair contains nonfinite arrays')
    counters=[dict(path=jax.tree_util.keystr(p),value=int(a)) for p,a in jax.tree_util.tree_leaves_with_path(state)
              if eqx.is_array(a) and a.shape==() and np.issubdtype(a.dtype,np.integer)]
    assert len(counters)==2 and all(r['value']==step for r in counters),counters
    # Replay old/new FP64 paths on an actual geometry before interrupting it.
    root=Path(c.operator.dataset_path);manifest=json.loads((root/'manifest.json').read_text())
    with np.load(root/manifest['conditions'][512]['file'],allow_pickle=False) as f:
        data={k:jnp.asarray(f[k]) for k in ['points','volumes','neighbors','coefficients','condition','condition_weights']}
    data['initial']=fixed_initial(data['points'],c,xp=jnp)
    spec=importlib.util.spec_from_file_location('preserved_fp64_training',run/'source/examples/confined_ch2d/operator_training.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    t=jnp.asarray([0.,.75,1.5,2.25,3.])
    a=np.asarray(eqx.filter_jit(lambda n,d:fields(n,d,t,c))(model,data))
    b=np.asarray(eqx.filter_jit(lambda n,d:old.fields(n,d,t,c))(model,data))
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    identities=[]
    for pid,needle in [(before['runner_pid'],'docs.tasks.confined_ch_beta09_runner'),
                       (before['active_child_pid'],str(run.name))]:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if needle not in command or os.getsid(pid)!=pid:raise RuntimeError((pid,command))
        identities.append(dict(pid=pid,session=os.getsid(pid),command=command))
    snapshot=journal.with_name(journal.stem+'-before-fp32-stop.json')
    with snapshot.open('x') as f:json.dump(before,f,indent=2)
    for identity in identities:os.kill(identity['pid'],signal.SIGTERM)
    for identity in identities:
        for _ in range(30):
            status=Path(f'/proc/{identity["pid"]}/stat')
            if not status.exists() or status.read_text().split(') ',1)[1].startswith('Z'):break
            time.sleep(.2)
        else:raise RuntimeError(f'Owned process did not terminate: {identity}')
    rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines(keepends=True) if s.endswith('\n')]
    result=dict(status='intentionally_stopped_for_fp32',run=str(run),preserved_marker=str(marker),
        preserved_step=step,optimizer_counters=counters,all_payloads_finite=True,
        fp64_old_new_field_maximum_difference=float(abs(a-b).max()),owned_processes=identities,
        journal_snapshot=str(snapshot),latest_evaluation={k:v for k,v in rows[-1].items() if k!='conditions'},
        scope='User-directed precision trial on same GPU. Closed partial state preserved, not a completed30k result. No old config/checkpoint overwritten.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    before.update(status='intentionally_stopped_for_fp32',active_child_pid=None,stop_receipt=str(output),
                  preserved_checkpoint=str(marker),reason='User requests faster FP32 on existing GPU; retain recoverable partial FP64 control')
    temporary=journal.with_name(journal.name+'.fp32-stop.partial');temporary.write_text(json.dumps(before,indent=2)+'\n');os.replace(temporary,journal)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();stop(a.journal.resolve(),a.output.resolve())
