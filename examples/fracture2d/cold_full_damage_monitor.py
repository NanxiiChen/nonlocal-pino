"""Read-only CPU field logging across a fresh prefix and its same-run tail."""
import argparse
import json
import os
from pathlib import Path
import runpy
import time

from ml_collections import ConfigDict
import numpy as np
from matplotlib.tri import Triangulation
from tensorboardX import SummaryWriter

from pinn.utils import log_config_text
from examples.fracture2d.tensorboard_monitor import (
    atomic_json, publish_endpoint, scalar_values, sha)


def selected_root(prefix, continuation):
    """Switch only once the continuation has restored and verified its prefix."""
    ready = (continuation/'preflight.json').exists() and (continuation/'result.json').exists()
    return continuation if ready else prefix


def check_zero(prefix):
    identity = json.loads((prefix/'identity.json').read_text())
    if identity['own_zero'] is not True or identity['optimizer_persistent'] is not True:
        raise ValueError('field logger requires a freshly initialized own-zero trajectory')
    zero = prefix/'state_0000'
    receipt = json.loads((zero/'result.json').read_text())
    if receipt['optimizer_count'] != 0 or receipt['load'] != 0. or receipt['reaction'] != 0.:
        raise ValueError('fresh run has nonzero initial physical load or optimizer count')
    if sha(zero/'fields.npz') != receipt['fields_sha256']:
        raise ValueError('fresh zero field hash differs')
    with np.load(zero/'fields.npz', allow_pickle=False) as data:
        for key in ('q', 'd', 'H'):
            if not np.isfinite(data[key]).all() or np.any(data[key] != 0.):
                raise ValueError(f'fresh initial physical {key} is not zero')
    return identity


def main(a, config_path):
    if (any(type(v) is not int or v < 1 for v in
            (a.scalar_every,a.flush_seconds,a.figure_dpi,a.cpu_threads))
            or not np.isfinite(a.poll_seconds) or a.poll_seconds <= 0):
        raise ValueError('invalid cold-trajectory monitoring controls')
    if not Path(a.event_dir).resolve().is_relative_to(Path(a.log_root).resolve()):
        raise ValueError('events must be inside the explicit canonical log root')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-a.cpu_threads:])
    suite, prefix, tail = map(Path, (a.run_root,a.prefix_solution_root,a.solution_root))
    while not (prefix/'preflight.json').exists() or not (prefix/'result.json').exists():
        if (suite/'exit.json').exists():
            raise RuntimeError('fresh trajectory exits before monitor preflight')
        time.sleep(a.poll_seconds)
    identity = check_zero(prefix)
    c = identity['config']; n = c['full_loading']; refroot = Path(n['reference_root'])
    refs = json.loads((refroot/'result.json').read_text())['curve']
    with np.load(Path(c['trial']['geometry_dir'])/'mesh.npz') as mesh:
        nodes, cells = int(mesh['n_material']), int(mesh['n_material_cells'])
        xy = mesh['coordinates'][:nodes].copy()
        tri = Triangulation(xy[:,0],xy[:,1],mesh['triangles'][:cells])
        volumes, dofs = mesh['volumes'].copy(), len(mesh['load_direction'])
    events = Path(a.event_dir); events.mkdir(parents=True,exist_ok=True)
    if any(events.glob('events.out.tfevents.*')):
        raise ValueError('refusing existing fresh-trajectory event files')
    writer = SummaryWriter(str(events),flush_secs=a.flush_seconds)
    log_config_text(writer,ConfigDict(dict(prefix=c,monitor=a.to_dict()),allow_dotted_keys=True))
    writer.add_text('provenance','Fresh zero q/d/H, new network and Adam count0; all prefix loads recomputed. Reference is evaluation-only; positive endpoints remain provisional.',0)
    writer.add_scalar('train/reaction',0.,0)
    writer.add_scalar('train/max_damage',0.,0)
    writer.add_scalar('train/optimizer_count',0.,0)
    done = set(); scalar_done = set(); published = []; seen = 0
    out = suite/'tensorboard_figures'; out.mkdir(exist_ok=True)
    try:
        while True:
            root = selected_root(prefix,tail)
            result = json.loads((root/'result.json').read_text()); curve = result['curve']
            # Drain the prefix journals even after the tail becomes ready, so no
            # last-prefix scalar steps disappear during the phase transition.
            journal_roots = [prefix] if root == prefix else [prefix, tail]
            for journal_root in journal_roots:
                for folder in sorted(journal_root.glob('state_[0-9][0-9][0-9][0-9]')):
                    if int(folder.name[-4:]) == 0: continue
                    for path in sorted(folder.glob('sweep_[0-9][0-9][0-9][0-9].json')):
                        if str(path) in scalar_done: continue
                        row = json.loads(path.read_text()); scalar_done.add(str(path))
                        if row['sweep'] != 1 and row['sweep'] % a.scalar_every: continue
                        count = row['optimizer_count']
                        if count % n['inner_updates']:
                            raise ValueError('optimizer count is not an integral outer step')
                        step = count//n['inner_updates']
                        if step <= seen:
                            raise ValueError('fresh optimizer-derived outer steps are not increasing')
                        seen = step
                        for tag,value in scalar_values(row).items(): writer.add_scalar(tag,value,step)
            for row in curve:
                if row['index'] in done: continue
                published.append(publish_endpoint(writer,root,out,tri,xy,row,curve,refroot,refs,
                    dpi=a.figure_dpi,dofs=dofs,volumes=volumes))
                done.add(row['index'])
                print(json.dumps(dict(event='fresh_load_published',**published[-1])),flush=True)
            writer.flush()
            complete = root == tail and result['complete']
            status = dict(complete=complete,own_zero=True,event_dir=str(events),log_root=a.log_root,
                active_solution_root=str(root),published_loads=published,last_outer_step=seen,
                monitor_config_sha256=sha(config_path),monitor_source_sha256=sha(__file__))
            atomic_json(suite/'tensorboard_status.json',status)
            if complete: break
            if (suite/'exit.json').exists():
                raise RuntimeError('fresh trajectory exits before final monitoring')
            time.sleep(a.poll_seconds)
    finally:
        writer.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    main(runpy.run_path(args.config)['get_config'](),args.config)
