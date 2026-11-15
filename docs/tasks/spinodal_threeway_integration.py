"""CPU-only small-network integration on three unmodified production meshes."""
import argparse
from pathlib import Path
import json
import shutil
import time

import equinox as eqx
import jax
import numpy as np

from docs.tasks.electro_threeway_queue import write
from examples.confined_ch2d.operator_problem import specification
from examples.confined_ch2d.operator_training import load_bank,train_operator


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if jax.default_backend()!='cpu':raise RuntimeError('This integration check is CPU-only')
    jax.config.update('jax_enable_x64',True)
    from examples.confined_ch2d.configs.train_operator_compare_nonlocal import get_config as nl
    from examples.confined_ch2d.configs.train_operator_compare_local_ad import get_config as ad
    from examples.confined_ch2d.configs.train_operator_compare_fem import get_config as fem
    methods=(('nonlocal',nl),('fem',fem),('local_ad',ad));base=None;reports=[]
    for label,make in methods:
        c=make();source=Path(c.operator.dataset_path)
        original=json.loads((source/'manifest.json').read_text());root=args.output/('bank-'+label);root.mkdir()
        del c.operator.precompute;del c.operator.comparison_reference
        c.operator.dataset_path=str(root);c.operator.train_conditions=2;c.operator.test_conditions=1
        c.operator.local_validation_indices=();c.operator.condition_batch=1
        c.operator.online.workers=1;c.operator.online.refresh_size=2;c.operator.online.prefetch_batches=1
        c.operator.online.refresh_every=2;c.operator.metric_workers=1
        c.train.iterations=2;c.train.seed=42;c.sample.Nt=1
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=2
        c.log.log_dir=str(args.output);c.log.run_name=label;c.log.eval_every=2;c.log.checkpoint_every=2
        records=[]
        for target,index in enumerate((0,1,1024)):
            row=original['conditions'][index];file=f'condition_{target}.npz'
            shutil.copyfile(source/row['file'],root/file)
            records.append(dict(index=target,file=file,holes=row['holes']))
        write(root/'manifest.json',dict(status='passed',specification=specification(c),conditions=records,
            train_indices=[0,1],test_indices=[2],scope='Integration subset of already qualified native records; not new references'))
        before=time.perf_counter();_,_,bank=load_bank(c)
        common={key:np.asarray(bank[key]) for key in ('points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights')}
        if base is None:base=common
        else:
            for key in common:np.testing.assert_array_equal(common[key],base[key],err_msg=key)
        if label=='fem':
            if set(bank)-set(common)!={'fe_points','fe_mass','fe_initial','fe_neighbors','fe_coefficients','fe_metric','fe_cells'}:
                raise ValueError('Unexpected FEM training payload')
        del bank
        run=train_operator(c);summary=json.loads((run/'summary.json').read_text())
        if summary['new_updates']!=2:raise ValueError('Canonical loop did not complete')
        for index in (0,1,2):
            with np.load(run/f'prediction_{index:03d}.npz') as p:
                np.testing.assert_array_equal(p['phi'][0],p['phi_reference'][0].astype(np.float32))
                if not np.isfinite(p['phi']).all() or not np.isfinite(p['mu']).all():raise ValueError('Nonfinite predictions')
        reports.append(dict(method=label,seconds=time.perf_counter()-before,run=str(run),updates=2))
    first=args.output/'nonlocal'
    for label,_ in methods:
        run=args.output/label
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (first/name).read_bytes()!=(run/name).read_bytes():raise ValueError('Cold states differ')
        for name in ('training_draws.npz','known_encoder_inputs.npz','first_online_encoder_inputs.npz'):
            with np.load(first/name) as a,np.load(run/name) as b:
                if set(a.files)!=set(b.files):raise ValueError('Input schema mismatch')
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    write(args.output/'validation.json',dict(status='passed',cases=reports,
        common_initial_states_and_actual_inputs=True,shared_bank_arrays=8,
        scope='Three production meshes, width4/B1/Nt1/twoSOAPupdates; CPU integration only, not training accuracy/resources'))


if __name__=='__main__':main()
