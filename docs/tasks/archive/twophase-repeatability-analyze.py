"""Summarize bounded repeatability probes, retaining cross-process limitations."""
from pathlib import Path
import json
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict
from examples.twophase2d.train import _build_network


def main():
    if jax.default_backend()!='cpu':raise ValueError('CPU analysis only')
    root=ROOT/'reports/twophase-repeatability'
    names=['ordinary-autodiff','ordinary-symmetric','deterministic-autodiff',
           'ordinary-autodiff-spectrum','ordinary-symmetric-independent',
           'ordinary-symmetric-independent-repeat']
    results={name:json.loads((root/name/'result.json').read_text()) for name in names}
    assert all(r['status']=='completed' for r in results.values())
    common=np.load(root/'ordinary-autodiff/times.npy')
    for name,r in results.items():np.testing.assert_array_equal(np.load(root/name/'times.npy'),common[:r['steps']])
    first=results['ordinary-autodiff']
    assert all(r['changed_entries']==0 for r in results['ordinary-symmetric']['frozen_gradient_repeat'])
    assert all(r['parameters']['changed_entries']==0 for name in ['ordinary-symmetric','ordinary-symmetric-independent','ordinary-symmetric-independent-repeat','deterministic-autodiff'] for r in results[name]['path_repeat'])
    assert first['path_repeat'][0]['parameters']['changed_entries']==0
    assert first['path_repeat'][1]['parameters']['changed_entries']>0
    assert all(r['Q']['changed_entries']==0 for r in first['optimizer_identical_gradient_repeat'])
    jax.config.update('jax_enable_x64',True)
    base=Path(first['source_run']);c=ConfigDict(json.loads((base/'config.json').read_text()))
    template=_build_network(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
    a=root/'ordinary-symmetric-independent';b=root/'ordinary-symmetric-independent-repeat'
    aa=np.load(a/'frozen_arrays.npz');bb=np.load(b/'frozen_arrays.npz')
    ga=aa['gradient'][0];gb=bb['gradient'][0]
    ma=eqx.tree_deserialise_leaves(a/'model_0.eqx',template)
    mb=eqx.tree_deserialise_leaves(b/'model_0.eqx',template)
    flatten=lambda tree:np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(eqx.filter(tree,eqx.is_inexact_array))])
    x,y=flatten(ma),flatten(mb)
    cross=dict(identical_time_arrays=True,identical_numerical_probe_source='v2 for both; only output name differs',
        frozen_loss_equal=results[a.name]['frozen_loss']==results[b.name]['frozen_loss'],
        frozen_gradient_max_absolute=float(np.max(np.abs(ga-gb))),
        frozen_gradient_relative_L2=float(np.linalg.norm(ga-gb)/np.linalg.norm(ga)),
        parameters100_max_absolute=float(np.max(np.abs(x-y))),
        parameters100_relative_L2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
        model_bytes_equal=(a/'model_0.eqx').read_bytes()==(b/'model_0.eqx').read_bytes(),
        covariance_max_absolute=max(float(np.max(np.abs(aa[key]-bb[key]))) for key in aa.files if key.startswith('GG_')),
        limitation='Equivalent pullback removes tested in-process noise; independent processes still differ. Exact compiler/kernel source not isolated.')
    original=np.load(root/'ordinary-autodiff-spectrum/frozen_arrays.npz')['gradient'][0]
    agreement=dict(max_absolute=float(np.max(np.abs(original-ga))),relative_L2=float(np.linalg.norm(original-ga)/np.linalg.norm(original)))
    assert agreement['relative_L2']<1e-12
    block=next(r for r in results['ordinary-autodiff-spectrum']['covariance_spectra'] if 'output.weight' in r['path'] and r['size']==64)
    assert block['numerical_rank']==2
    summary=dict(status='passed',purpose='Locate same-seed early differences; no full L2/production-fix claim',
        process_count=len(names),short_trajectories=2*len(names),total_updates=sum(2*r['steps'] for r in results.values()),
        fixed_seed=42,exact_training_initial_arrays=True,identical_time_arrays=True,
        ordinary_frozen_loss_equal=len(set(first['frozen_loss']))==1,
        ordinary_frozen_gradient_max_absolute=max(r['max_absolute'] for r in first['frozen_gradient_repeat']),
        ordinary_operator_transpose_max_relative_variation=max(r['relative_L2'] for r in first['operator_transpose_repeat']),
        common_second_gradient_update_relative_range=[min(r['second_update']['relative_L2'] for r in first['gradient_to_soap_amplification']),max(r['second_update']['relative_L2'] for r in first['gradient_to_soap_amplification'])],
        ordinary_100step_parameter_relative_difference=first['path_repeat'][-1]['parameters']['relative_L2'],
        first_parameter_difference_step=next(r['step'] for r in first['path_repeat'] if r['parameters']['changed_entries']>0),
        frozen_gradient_optimizer_repeats_exact=True,symmetric_inprocess_100step_repeats_exact=True,
        deterministic_inprocess_5step_repeats_exact=True,
        original_symmetric_gradient_agreement=agreement,rank_deficient_example=block,
        independent_process_symmetric=cross,
        interpretation=['Standard operator reverse scatter accumulation produces measured within-process gradient noise.',
          'Near-null initial SOAP covariance amplifies tiny gradient differences into materially different actual updates.',
          'Equal frozen-gradient optimizer repetitions are exact for tested inputs; not evidence of random eigensolver noise.',
          'Symmetric adjoint and deterministic execution isolate within-process trigger but do not establish full cross-process/full-training reproducibility.',
          'Final L2 disparity is compatible with this measured pathway; its precise size and all later-stage sources are not uniquely attributed.'],
        next_protocol='Stabilize/validate SOAP weak-direction initialization or a numerically robust equivalent before long repeated runs. First use two identical-seed full runs; then a predeclared small paired-seed local/nonlocal cohort. Retain all outcomes; report mean/std/range and equal-wall accuracy.',
        defaults_changed=False,production_code_changed=False,reference_changed=False,git_history_changed=False)
    (root/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
