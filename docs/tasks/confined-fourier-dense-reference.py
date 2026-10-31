"""Dense native-only validation on existing meshes; no JAX or FEM imports."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.native_reference import native_matrix,native
from examples.confined_ch2d.operator_problem import case_config
from examples.confined_ch2d.problem import relative


def one(request):
    config,record,root,output=request
    c=ConfigDict(config);index=record['index']
    with np.load(Path(root)/record['file'],allow_pickle=False) as file:d={k:file[k] for k in file.files}
    cc=case_config(c,d['holes'],index);cc.reference.snapshots=61
    A=native_matrix(d['points'],d['volumes'],cc)
    coarse=native(cc,d['points'],d['volumes'],A,c.reference.dt,1e-7)
    fine=native(cc,d['points'],d['volumes'],A,c.reference.refined_dt,1e-9)
    errors={name:max(relative(a,b,d['volumes']) for a,b in zip(coarse[name],fine[name],strict=True)) for name in ('phi','mu')}
    if errors['phi']>c.reference.phase_tolerance or errors['mu']>c.reference.chemical_tolerance:
        raise ValueError(f'Dense native time refinement failed: {index}, {errors}')
    anchor={name:float(abs(fine[name][::15]-d[name]).max()) for name in ('phi','mu')}
    if max(anchor.values())>1e-10:raise ValueError(f'Existing reference mismatch: {index}, {anchor}')
    np.savez_compressed(Path(output)/f'native_{index:03d}.npz',points=d['points'],volumes=d['volumes'],
        holes=d['holes'],time=np.linspace(0,c.physical.terminal_time,61),phi=fine['phi'],mu=fine['mu'])
    return dict(index=index,time_refinement=errors,existing_anchor_difference=anchor,seconds=coarse['seconds']+fine['seconds'])


def main(dataset,output,workers):
    c=json.loads((dataset/'config.json').read_text());manifest=json.loads((dataset/'manifest.json').read_text())
    if workers<1 or manifest['status']!='passed':raise ValueError('Positive workers and validated source required')
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();rows=[]
    ids=manifest['test_indices']
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(one,(c,manifest['conditions'][i],str(dataset),str(output))) for i in ids]
        for future in as_completed(futures):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    result=dict(status='passed',dataset=str(dataset),indices=ids,snapshots=61,conditions=sorted(rows,key=lambda r:r['index']),
        wall_seconds=time.perf_counter()-start,purpose='Dense validation only; no training payload or future-field supervision',
        reference='Independent ordered graph and SciPy BDF on the existing physical meshes; no FEM/JAX')
    (output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,required=True)
    a=p.parse_args();main(a.dataset,a.output,a.workers)
