"""Split saved physical errors into frozen growing/stable CH spectral subspaces.

Diagnostic only: the homogeneous initial-mean Jacobian is not the nonlinear
evolution operator. No model fitting, reference labels in training or PDE change.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh


def diagnose(run, output):
    config=json.loads((run/'config.json').read_text())
    root=Path(config['operator']['dataset_path'])
    manifest=json.loads((root/'manifest.json').read_text())
    physical=config['physical'];rows=[]
    for index in manifest['test_indices']:
        with np.load(root/manifest['conditions'][index]['file'],allow_pickle=False) as f:
            v=f['volumes'];neighbors=f['neighbors'];coefficients=f['coefficients']
            reference=f['phi'];initial=reference[0]
        with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as f:
            predicted=f['phi']
        n=len(v);A=np.diag(coefficients.sum(axis=1))
        np.add.at(A,(np.arange(n)[:,None],neighbors),-coefficients)
        rootv=np.sqrt(v);S=rootv[:,None]*A/rootv[None,:]
        np.testing.assert_allclose(S,S.T,rtol=2e-12,atol=2e-10)
        eigenvalues,U=eigh((S+S.T)/2,check_finite=True)
        mean=v@initial/v.sum()
        rates=physical['mobility']*eigenvalues*(physical['epsilon']**2*eigenvalues+
            physical['beta']*(3*mean**2-1))
        growing=rates < -1e-9
        error=((predicted-reference)*rootv)@U
        target=(reference*rootv)@U
        total=np.sum(error**2);terminal=np.sum(error[-1]**2)
        denominator=np.sum(target**2);terminal_denominator=np.sum(target[-1]**2)
        rows.append(dict(index=index,growing_modes=int(growing.sum()),
            pooled_error_energy_growing_fraction=float(np.sum(error[:,growing]**2)/total),
            terminal_error_energy_growing_fraction=float(np.sum(error[-1,growing]**2)/terminal),
            pooled_growing_component_relative_L2=float(np.sqrt(np.sum(error[:,growing]**2)/denominator)),
            pooled_other_component_relative_L2=float(np.sqrt(np.sum(error[:,~growing]**2)/denominator)),
            terminal_growing_component_relative_L2=float(np.sqrt(np.sum(error[-1,growing]**2)/terminal_denominator)),
            terminal_other_component_relative_L2=float(np.sqrt(np.sum(error[-1,~growing]**2)/terminal_denominator))))
    result=dict(run=str(run),scope=__doc__,conditions=rows)
    for key in ('pooled_error_energy_growing_fraction','terminal_error_energy_growing_fraction'):
        result['mean_'+key]=float(np.mean([r[key] for r in rows]))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();diagnose(args.run,args.output)
