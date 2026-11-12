"""Compare eager/lazy pools across actual geometry, graph and epoch replacement."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np


def run(inputs, eager_source, output):
    import jax
    from examples.confined_ch2d.configs.train_operator_potential_spatial import get_config
    from examples.confined_ch2d.operator_core_halo import RollingSpatialPool
    if any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('Replacement validation must use CPU only')
    spec=importlib.util.spec_from_file_location('frozen_eager_spatial',eager_source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c=get_config();c.train.seed=42;c.operator.train_conditions=2;c.operator.condition_batch=2
    with np.load(inputs/'inputs.npz') as saved:
        old={k:saved[k][:2].copy() for k in saved.files if k!='metric'}
        replacement={k:saved[k][2:4].copy() for k in saved.files if k!='metric'}
    for key in ('points','condition','neighbors','coefficients'):
        if np.array_equal(old[key],replacement[key]):
            raise ValueError('This check requires different actual geometry and operators: '+key)
    output.mkdir(parents=True,exist_ok=False)
    for label in ('eager','lazy'):(output/label).mkdir()
    eager=module.RollingSpatialPool(c,old,output/'eager')
    lazy=RollingSpatialPool(c,old,output/'lazy')
    count=0;records=[];active=old
    try:
        # Duplicated slots exercise different chosen sets in one batch.
        for step,ids in ((1,[0,1]),(49,[1,0]),(50,[0,0]),(51,[0,1]),(52,[1,1]),(101,[1,0])):
            ids=np.asarray(ids)
            if step==50:
                eager.prefetch(ids,step);lazy.prefetch(ids,step)
                eager.install(np.array([0,1]),replacement)
                lazy.install(np.array([0,1]),replacement)
                active=replacement
            a,ca=eager.batch(ids,step);b,cb=lazy.batch(ids,step)
            np.testing.assert_array_equal(ca,cb)
            if set(a)!=set(b):raise ValueError('Payload schemas differ')
            for key in a:
                np.testing.assert_array_equal(a[key],b[key]);count+=1
            for key in ('points','volumes','initial','condition','condition_weights'):
                np.testing.assert_array_equal(b[key],active[key][ids])
            records.append(dict(step=step,slots=ids.tolist(),versions=lazy.versions.tolist(),
                physical_nodes=np.count_nonzero(active['volumes'][ids],axis=1).tolist()))
    finally:
        eager.close();lazy.close()
    current=Path('examples/confined_ch2d/operator_core_halo.py').resolve()
    shutil.copyfile(current,output/'lazy_source.py')
    shutil.copyfile(eager_source,output/'eager_source.py')
    report=dict(status='passed',inputs=str(inputs),original_eager_source=str(eager_source),
        current_source=str(current),exact_payload_arrays=count,records=records,
        scope='Changed physical points, boundaries, operator graph and initial field; prefetched slots invalidated before use; epoch rollover')
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--eager-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.inputs,args.eager_source,args.output)
