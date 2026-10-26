"""Repeat the physical graph-IC control through the same __main__ entry point."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
NAME='20261025-data-physical-icmatched_graph-bcmatched_graph-phase-stagger0-seed42-replay'
WORKER=r'''
import runpy,sys
from examples.corrosion2d.configs import residual_pilot as preset
original=preset.get_config
name=sys.argv[1]
def configuration(case,formulation,seed):
    config=original(case,formulation,seed)
    config.experiment.initial_sampling='matched_graph'
    config.experiment.boundary_sampling='matched_graph'
    config.log.run_name=name
    return config
preset.get_config=configuration
sys.argv=['residual_experiment','train','--case','corrosion2d','--formulation','phase','--seed','42']
runpy.run_module('examples.corrosion2d.residual_experiment',run_name='__main__')
'''

def main():
    run=ROOT/NAME
    if run.exists():raise FileExistsError(run)
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true',
               JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-normalized-data-cache')
    log=Path('/tmp')/(NAME+'.log')
    with log.open('x') as stream:
        completed=subprocess.run([sys.executable,'-u','-c',WORKER,NAME],env=env,stdout=stream,stderr=subprocess.STDOUT)
    if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
    if completed.returncode:raise RuntimeError(log.read_text())
    baseline=ROOT/NAME.removesuffix('-replay')
    matches={p:(baseline/p).read_bytes()==(run/p).read_bytes() for p in
             ('model_initial.eqx','ckpt_001000.eqx','ckpt_002000.eqx','ckpt_003000.eqx','ckpt_004000.eqx','ckpt_005000.eqx','model_final.eqx','prediction.npz')}
    # NPZ container metadata is not required to match; compare its arrays below.
    import numpy as np
    with np.load(baseline/'prediction.npz') as a,np.load(run/'prediction.npz') as b:
        array_matches={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
    a=json.loads((baseline/'config.json').read_text());b=json.loads((run/'config.json').read_text())
    a['log']['run_name']=b['log']['run_name'];assert a==b
    success=all(v for k,v in matches.items() if k!='prediction.npz') and all(array_matches.values())
    losses_equal=(baseline/'loss_history.jsonl').read_bytes()==(run/'loss_history.jsonl').read_bytes()
    evaluations=[]
    for directory in (baseline,run):
        rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        for row in rows:row.pop('elapsed_seconds')
        evaluations.append(rows)
    evaluations_equal=evaluations[0]==evaluations[1]
    success=success and losses_equal and evaluations_equal
    record={'baseline':str(baseline),'replay':str(run),'updates':5000,'xla_flags':env['XLA_FLAGS'],
            'same_configuration_except_run_name':True,'checkpoint_bytes_equal':matches,
            'physical_prediction_arrays_equal':array_matches,'trajectory_replay_exact':success,
            'logged_loss_weights_exact':losses_equal,'all_evaluations_equal_excluding_elapsed_time':evaluations_equal,
            'scope':'1000-update saved checkpoints and final fields/parameters; optimizer state not serialized'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if not success:raise RuntimeError('Full-trajectory replay differs; preserve records and investigate')

if __name__=='__main__':main()
