"""Two-arm Spinodal audit after the user explicitly omitted Local AD.

The final-field audit is retained from the frozen three-way helper with only
its module-local method list narrowed. Do not edit the reference snapshot.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from docs.tasks.electro_threeway_queue import read, write
from docs.tasks.spinodal_threeway_support import paired, reference_checks, lines

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs/confined_ch2d')
METHODS = ('nonlocal', 'fem')


def final_metrics(runs,steps):
    from examples.confined_ch2d.operator_problem import specification
    result={}
    for method,run in zip(METHODS,runs,strict=True):
        c=ConfigDict(read(run/'config.json'));native=Path(c.operator.dataset_path)
        bank=read(native/'manifest.json');local=Path(c.operator.comparison_reference.path)
        manifest=read(local/'manifest.json')
        if (manifest['status']!='passed' or manifest['specification']!=specification(c)
                or manifest['indices']!=bank['test_indices']):raise ValueError('Unqualified or mismatched common local reference')
        for name in manifest['sources']:
            if (local/'source'/name).read_bytes()!=(ROOT/name).read_bytes():
                raise ValueError('Reference construction source changed '+name)
        monitor=lines(run/'local_metrics.jsonl')
        expected=list(range(0,steps+1,c.log.eval_every))
        if expected[-1]!=steps:expected.append(steps)
        if [row['step'] for row in monitor]!=expected:raise ValueError('Missing actual local evaluation records')
        for row in monitor:
            if [entry['index'] for entry in row['conditions']]!=bank['test_indices']:
                raise ValueError('Convergence cohort changed')
            for channel,key in (('phi','phase_mean'),('chemical','chemical_mean'),('channel_mean','channel_mean')):
                np.testing.assert_allclose(row[key],np.mean([r[channel] for r in row['conditions']]),rtol=2e-14)
        by_index={row['index']:row for row in monitor[-1]['conditions']};rows=[]
        for item in manifest['cases']:
            index=item['index']
            with np.load(run/f'prediction_{index:03d}.npz') as prediction,np.load(local/item['file']) as ref:
                with np.load(native/bank['conditions'][index]['file']) as source:
                    for key in ('points','volumes','time','holes','mesh_points','mesh_cells'):
                        np.testing.assert_array_equal(prediction[key],source[key])
                    for field in ('phi','mu'):
                        np.testing.assert_array_equal(prediction[field+'_reference'],source[field])
                for key in ('points','volumes','time','holes'):np.testing.assert_array_equal(prediction[key],ref[key])
                p=np.stack((prediction['phi'],prediction['mu']),axis=-1).astype(float)
                r=np.stack((ref['phi'],ref['mu']),axis=-1).astype(float)
                if not np.isfinite(p).all() or not np.isfinite(r).all():raise ValueError('Nonfinite final fields')
                norm=np.linalg.norm(r.reshape(-1,2),axis=0)
                if np.any(norm<=0):raise ValueError('Zero channel norm')
                errors=np.linalg.norm((p-r).reshape(-1,2),axis=0)/norm
                later=np.linalg.norm((p[1:]-r[1:]).reshape(-1,2),axis=0)/np.linalg.norm(r[1:].reshape(-1,2),axis=0)
                row=dict(index=index,phi=float(errors[0]),chemical=float(errors[1]),
                    channel_mean=float(errors.mean()),chemical_positive_times=float(later[1]))
                for key in ('phi','chemical','channel_mean','chemical_positive_times'):
                    np.testing.assert_allclose(row[key],by_index[index][key],rtol=2e-13,atol=2e-14)
                rows.append(row)
        result[method]=dict(run=str(run),checkpoint=str(run/'model_final.eqx'),cases=rows,
            means={key:float(np.mean([r[key] for r in rows])) for key in ('phi','chemical','channel_mean','chemical_positive_times')},
            monitor=str(run/'local_metrics.jsonl'),native_monitor=str(run/'metrics.jsonl'))
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    runs = [LOGS / (args.prefix + '-' + method) for method in METHODS]
    result = dict(status='passed', methods=list(METHODS),
                  omitted_method='local_ad: omitted by user after resource measurement',
                  pairing=paired(runs, 50000),
                  reference=reference_checks(read(runs[0] / 'config.json')),
                  results=final_metrics(runs, 50000),
                  scope='One initialization; all8reserved local-FEM geometries every500updates; final50k weights')
    write(args.bundle / 'final.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
