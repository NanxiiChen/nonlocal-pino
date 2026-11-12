"""Measure prescribed initial-interface coverage; never adapt training samples."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from ml_collections import ConfigDict
from examples.particle_corrosion2d.problem import initial_fields


def summarize(hit,weights):
    if hit.shape!=weights.shape or hit.ndim!=2 or hit.dtype!=bool:
        raise ValueError('Boolean pool/core indicators and aligned weights required')
    count=hit.sum(axis=-1)
    mass=(hit*weights).sum(axis=-1)/weights.sum(axis=-1)
    return dict(sets=len(hit),mean_core_hits=float(count.mean()),minimum_core_hits=int(count.min()),
        maximum_core_hits=int(count.max()),zero_hit_fraction=float(np.mean(count==0)),
        weighted_indicator_mean=float(mass.mean()),weighted_indicator_std=float(mass.std(ddof=1)))


def run(paths,output):
    if len(paths)<2:raise ValueError('At least two completed controls required')
    reports=[];first_draws=None;first_shapes=None;first_geometry=None
    for path in paths:
        c=ConfigDict(json.loads((path/'config.json').read_text()))
        if first_geometry is None:first_geometry=c.geometry.to_dict()
        elif c.geometry.to_dict()!=first_geometry:raise ValueError('Prescribed initial geometry differs')
        if json.loads((path/'summary.json').read_text())['steps']!=20000:
            raise ValueError('Completed20kcontrols required')
        first_event=json.loads((path/'online_refresh.jsonl').read_text().splitlines()[0])
        if first_event['completed_updates']!=50:
            raise ValueError('Only the original50update initial bank interval is qualified')
        with np.load(path/'sampling.npz') as samples:
            ids,pools=samples['condition_ids'][:50],samples['pool_ids'][:50]
        if first_draws is None:first_draws=(ids,pools)
        else:
            np.testing.assert_array_equal(ids,first_draws[0])
            np.testing.assert_array_equal(pools,first_draws[1])
        with np.load(path/'online_initial_inputs.npz') as bank:
            xy=np.take_along_axis(bank['graph_points'],bank['core_indices'][...,None],axis=2)
            weights=bank['pde_weights'].astype(float)
        if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Positive finite PDE weights required')
        np.testing.assert_allclose(weights.sum(axis=-1),c.sample.K_core,rtol=1e-7)
        if first_shapes is None:first_shapes=xy.shape
        elif xy.shape!=first_shapes:raise ValueError('Pool/core array shapes differ')
        phi=initial_fields(c,xy.reshape(-1,2))[:,0].reshape(weights.shape)
        bands={}
        for lower,upper in ((.05,.95),(.1,.9),(.25,.75)):
            hit=(phi>lower)&(phi<upper)
            all_sets=summarize(hit.reshape(-1,hit.shape[-1]),weights.reshape(-1,weights.shape[-1]))
            consumed=summarize(hit[ids,pools].reshape(-1,hit.shape[-1]),weights[ids,pools].reshape(-1,weights.shape[-1]))
            bands[f'{lower:g}<{chr(966)}0<{upper:g}']=dict(all_initial_sets=all_sets,first50_consumed_sets=consumed)
        reports.append(dict(run=str(path),core_count=c.sample.K_core,bands=bands))
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'source.py')
    report=dict(status='passed',records=reports,fields='Only prescribed analytical initial phase',
        reference_fields_loaded=False,training_performed=False,samples_modified=False,
        limitation='Initial pools and actual first50draws only; repeated pool sets are not independent replicates; no evolved-interface or causal claim')
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.runs,args.output)
