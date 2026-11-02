"""Exercise the real spawned input-only channel producer without training."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(output):
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.configs.train_operator import get_config
    from examples.channel_nsch2d.online import RollingGeometryPool
    c=get_config();c.train.seed=42;c.operator.train_conditions=4
    c.operator.online=ConfigDict(dict(workers=2,refresh_size=2,refresh_every=1,prefetch_batches=2))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter();pool=RollingGeometryPool(c,output,np.empty((0,2))).start()
    try:
        for step in range(5):
            result=pool.before_step(step)
            if result is not None:
                slots,updates=result
                if updates['points'].shape!=(2,3977,2):raise ValueError('Production-grid shape changed')
        state=pool.save_state(5);summary=pool.summary()
    finally:pool.close()
    summary.update(status='real_CPU_producer_passed_no_training',wall_seconds=time.perf_counter()-begin,
        pool_state=str(state),single_geometry_target_met=False,online_training_started=False)
    (output/'receipt.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
