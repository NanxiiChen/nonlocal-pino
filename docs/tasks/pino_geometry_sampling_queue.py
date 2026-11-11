"""Own the serial GPU queue for confirmed channel and ceramic sampling controls."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from docs.tasks.pino_geometry_sampling_results import paired, read, write

REPO = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs')
DATA = Path('/root/autodl-tmp/data/particle_corrosion2d')
PARTICLE_BASE = DATA/'20261107-spaced-six-01'
CHANNEL_COHORT = Path('/root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/unseen3/manifest.json')
CHANNEL_MODES = ('full', 'scatter', 'tiles')


def gpu_info():
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    uuid, name, total, used = (v.strip() for v in query.split(','))
    return dict(uuid=uuid, name=name, memory_MiB=int(total), used_MiB=int(used))


def queue(output, after, particle_inputs):
    output.mkdir(parents=True, exist_ok=False)
    journal = output/'journal.json'
    state = dict(status='waiting_for_regular_queue', pid=os.getpid(), gpu=gpu_info(), stages=[],
                 started_utc=datetime.now(timezone.utc).isoformat())

    def save():
        write(journal, state)

    save()
    base_env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', MPLBACKEND='Agg')

    def run(label, command, *, cpu=False, fraction='0.92', allow_oom=False):
        env = dict(base_env, JAX_PLATFORMS='cpu' if cpu else 'cuda')
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            card = gpu_info()
            if card['uuid'] != state['gpu']['uuid'] or card['used_MiB'] > 1024:
                raise RuntimeError(f'Exclusive timing requires original idle GPU: {card}')
            env.update(XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION=fraction)
        log = output/f'{label}.log'
        item = dict(label=label, command=command, log=str(log), status='running',
                    platform='cpu' if cpu else 'cuda', gpu_pool_fraction=None if cpu else float(fraction))
        state['stages'].append(item)
        tick = time.perf_counter()
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, *command], cwd=REPO, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            item['pid'] = child.pid
            save()
            code = child.wait()
        item.update(pid=None, seconds=time.perf_counter()-tick, exit_code=code,
                    status='passed' if code == 0 else 'failed')
        if code:
            failure = log.read_text().lower()
            if allow_oom and any(marker in failure for marker in
                                 ('resource_exhausted', 'out of memory', 'failed to allocate')):
                item['status'] = 'oom'
                save()
                return False
        save()
        if code:
            raise RuntimeError(f'{label} exited {code}; see {log}')
        return True

    def channel(mode, block, steps):
        label = f'channel-{steps}-b{block}-{mode}'
        name = f'{output.name}-{label}'
        path = LOGS/'channel_nsch2d'/name
        capacity = 8192 if mode == 'tiles' else 15633
        command = ['-m','examples.channel_nsch2d.train',
                   '--config=examples/channel_nsch2d/configs/train_spatial_control.py',
                   '--seed=42', f'--config.log.run_name={name}', f'--config.train.iterations={steps}',
                   f'--config.train.microbatch.size={block}', f'--config.sample.spatial.mode={mode}',
                   f'--config.sample.spatial.node_capacity={capacity}']
        return path, run(label, command, allow_oom=steps == 60)

    try:
        gate = read(particle_inputs.parent/'input-audit.json')
        if gate['status'] != 'passed':
            raise ValueError('Ceramic input parity must pass before GPU runs')
        while read(after)['status'] == 'running':
            time.sleep(30)
        while gpu_info()['used_MiB'] > 1024:
            time.sleep(30)
        state['status'] = 'running'
        save()

        for block in (6, 3, 2):
            probes = []
            for mode in CHANNEL_MODES:
                path, ok = channel(mode, block, 60)
                if not ok:
                    break
                probes.append(read(path/'summary.json')['spatial'])
            if len(probes) == len(CHANNEL_MODES):
                write(output/'channel-probes.json', dict(block=block, steps=60, arms=probes))
                break
        else:
            raise RuntimeError('No shared memory-feasible channel block size')

        ceramic = []
        for mode in ('area','wall'):
            label = f'ceramic-{mode}'
            suffix = '' if mode == 'area' else '_wall_cores'
            inputs = PARTICLE_BASE/'inputs64' if mode == 'area' else particle_inputs
            path = LOGS/'particle_corrosion2d'/f'{output.name}-{label}'
            run(label, ['-m','examples.particle_corrosion2d.train',
                '--config',f'examples/particle_corrosion2d/configs/train_physical_spaced_w256{suffix}.py',
                '--inputs',str(inputs),'--reference',str(PARTICLE_BASE/'local_dev4'),'--output',str(path),
                '--seed','17617494'])
            if read(path/'summary.json')['steps'] != 20000:
                raise ValueError('Ceramic full budget missing')
            run(label+'-eval', ['-m','docs.tasks.pino_geometry_sampling_results','ceramic-eval',
                               str(path),str(PARTICLE_BASE/'native_dev4')])
            ceramic.append(path)
        paired(ceramic,'ceramic',output/'ceramic-paired.json')

        channel_paths = []
        for mode in CHANNEL_MODES:
            path, _ = channel(mode, block, 2000)
            run(f'channel-{mode}-online-audit',
                ['-m','docs.tasks.pino_geometry_sampling_results','channel-audit',str(path)], cpu=True)
            run(f'channel-{mode}-reserved', ['-m','examples.channel_nsch2d.cohort_evaluate',
                '--run',str(path),'--manifest',str(CHANNEL_COHORT),'--output',str(path/'reserved')])
            channel_paths.append(path)
        paired(channel_paths,'channel',output/'channel-paired.json')
    except Exception as error:
        state.update(status='failed', failure=str(error))
        raise
    else:
        state['status'] = 'complete'
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--particle-inputs', type=Path, required=True)
    args = parser.parse_args()
    queue(args.output, args.after, args.particle_inputs)
