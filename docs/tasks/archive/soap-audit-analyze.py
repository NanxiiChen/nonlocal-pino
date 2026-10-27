"""Quantitative source parity classification; no silent tolerance changes."""
from pathlib import Path
import json,numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/soap-implementation-audit'
meta=json.loads((OUT/'fixtures.json').read_text());fixture=np.load(OUT/'fixtures.npz')
streams={name:np.load(OUT/(name+'-streams.npz')) for name in ['jax','author','independent']}
records=[]
for c in meta['cases']:
 key=c['key'];row=dict(case=key,mode=c['mode'],dtype=c['dtype'],parameter_difference={},physical_moment_difference={},second_moment_difference={})
 for name in ['author','independent']:
  for field,target in [('parameter','parameter_difference'),('m_physical','physical_moment_difference'),('v','second_moment_difference')]:
   difference=float(np.max(np.abs(streams['jax'][key+'__'+field]-streams[name][key+'__'+field])))
   row[target][name]=difference
  if c['mode']=='shared':
   for field in ['parameter','m_physical','v']:
    np.testing.assert_allclose(streams['jax'][key+'__'+field],streams[name][key+'__'+field],atol=1e-12,rtol=0)
  elif c['weight_decay']==0:
   # Actual basis-selection pipelines differ in precision/backend. Use bounded
   # FP32-scale agreement only, not bitwise parity of native eigenbases.
   np.testing.assert_allclose(streams['jax'][key+'__parameter'],streams[name][key+'__parameter'],atol=2e-6,rtol=0)
 records.append(row)
# Analytic post-versus-pre-update decay difference at the first real update.
key='decay';m=streams['jax'][key+'__m_physical'][1];v=streams['jax'][key+'__v'][1]
lr=meta['lr'];b1=meta['b1'];b2=meta['b2'];eps=meta['eps'];a=lr*.1
u=-lr*np.sqrt(1-b2)/(1-b1)*m/(np.sqrt(v)+eps)
p=fixture[key+'_initial'];actual=streams['jax'][key+'__parameter'][1];author=streams['author'][key+'__parameter'][1]
np.testing.assert_allclose(actual,p*(1-a)+u,atol=1e-15,rtol=0)
np.testing.assert_allclose(author,(p+u)*(1-a),atol=1e-15,rtol=0)
np.testing.assert_allclose(actual-author,a*u,atol=1e-15,rtol=0)
wrapper=json.loads((OUT/'wrapper.json').read_text());assert wrapper['status']=='passed'
source=(OUT/'sources/jax-soap.py').read_text().rstrip('\n')
installed=(ROOT/'.venv/lib/python3.12/site-packages/soap_jax/soap.py').read_text().rstrip('\n')
assert source==installed
summary=dict(status='passed',installed_version='0.2.2',installed_source_matches_pinned_remote=True,
  sources=dict(author='https://github.com/nikhilvyas/SOAP/blob/a1e553530fde97d0e6b307d7c82ac6d38b072340/soap.py',
    jax='https://github.com/haydn-jones/SOAP_JAX/blob/a7c92f174c49b13136294a94ae0d46266b4f671f/src/soap_jax/soap.py',
    independent='https://github.com/kozistr/pytorch_optimizer/blob/8281be7c1dc39669328fe5b462eff10817b19175/pytorch_optimizer/optimizer/soap.py'),
  stream_cases_per_implementation=len(records),implementation_count=3,steps_per_case=16,refresh_calls=[6,11,16],
  shared_basis_parameter_max=max(max(r['parameter_difference'].values()) for r in records if r['mode']=='shared'),
  shared_basis_physical_moment_max=max(max(r['physical_moment_difference'].values()) for r in records if r['mode']=='shared'),
  native_zero_decay_parameter_max=max(max(r['parameter_difference'].values()) for r in records if r['mode']=='native' and r['case']!='decay'),
  core_tested_formulas='Initialization skip, bias correction, projection/backprojection, refresh timing, first-moment basis transport, second-moment permutation agree.',
  nonzero_decay_difference=dict(classification='Real order difference versus author, inactive in current project factory',
      author='(old_parameter + normalized_update)*(1-lr*weight_decay)',jax='old_parameter*(1-lr*weight_decay)+normalized_update',
      max_gap_first_real_update=float(np.max(np.abs(actual-author))),max_gap_after16calls=float(np.max(np.abs(streams['jax'][key+'__parameter']-streams['author'][key+'__parameter'])))),
  wrapper=wrapper,sensitivities={name:json.loads((OUT/(name+'-summary.json')).read_text())['sensitivities'] for name in streams},
  reference_precision_notes=['Torch reference stream runs explicitly set default dtype to case dtype; author otherwise fails for FP64 matrix parameters under FP32 default.',
      'Author native eig/QR force FP32; JAX defaults follow parameter dtype; independent eig follows parameter dtype but QR forces FP32.',
      'Shared-basis tape isolates state equations from backend eigenbasis/precision differences; it is a diagnostic, not an optimizer modification.',
      'Sensitivity compares one output-weight2x64 block; measured FP64 first-gradient variants and a synthetic/common second gradient; FP32 tests perturb one entry by one ULP. Not physical-field L2.'],
  confirmed_current_training_implementation_bug=False,
  conclusion='No tested core translation/wrapper error explains current instability; rank-deficient basis sensitivity appears in author/independent implementations at their arithmetic precision. Not a proof of all-path correctness or a full-training fix.',
  production_files_changed=False,dependencies_changed=False,full_training_runs=0,git_history_operations=False,records=records)
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ['status','shared_basis_parameter_max','shared_basis_physical_moment_max','native_zero_decay_parameter_max','nonzero_decay_difference','conclusion']},indent=2))
