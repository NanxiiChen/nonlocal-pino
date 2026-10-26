"""Verify the complete legacy baseline against prior s_bc by direct paths."""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')

def main():
    a=ROOT/'20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed42'
    b=ROOT/'20261025-transport-physical_difference-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42'
    files=['model_initial.eqx',*[f'ckpt_{i:06d}.eqx' for i in range(1000,5001,1000)],'model_final.eqx']
    checks={name:(a/name).read_bytes()==(b/name).read_bytes() for name in files}
    with np.load(a/'prediction.npz') as x,np.load(b/'prediction.npz') as y:
        predictions={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
    losses=(a/'loss_history.jsonl').read_bytes()==(b/'loss_history.jsonl').read_bytes()
    evaluations=[]
    for run in (a,b):
        rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        for row in rows:row.pop('elapsed_seconds')
        evaluations.append(rows)
    ca=json.loads((a/'config.json').read_text());cb=json.loads((b/'config.json').read_text())
    cb['log']['run_name']=ca['log']['run_name']
    for key,value in [('transport_evaluation','physical_difference'),('ch_norm','l1'),('ch_rms_floor',1e-12)]:
        assert cb['experiment'].pop(key)==value
    assert ca==cb
    out={'prior':str(a),'new_baseline':str(b),'checkpoint_bytes_equal':checks,'prediction_arrays_equal':predictions,
        'logged_loss_weights_equal':losses,'all_evaluations_equal_excluding_elapsed_time':evaluations[0]==evaluations[1],
        'config_equal_after_explicit_new_fields_and_run_name':True,'updates':5000,
        'scope':'Complete saved1000-update parameter checkpoints, loss/weight/evaluation trajectory; optimizer internal state not serialized'}
    out['trajectory_replay_exact']=all(checks.values()) and all(predictions.values()) and losses and evaluations[0]==evaluations[1]
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2),flush=True)
    assert out['trajectory_replay_exact']

if __name__=='__main__':main()
