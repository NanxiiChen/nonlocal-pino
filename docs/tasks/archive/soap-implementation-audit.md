# SOAP implementation audit — completed

## Request and conclusion

User asks whether the project's SOAP implementation is incorrect, noting
many open-source alternatives. Audit the actual installed optimizer, not just
an algorithm label. In the CURRENT tested configuration/paths, no core SOAP
translation or factory-call defect explaining the observed variability is
found. Shared-basis source parity passes; actual reference implementations
also exhibit low-rank initialization sensitivity at their arithmetic precision.
This is bounded numerical evidence, not proof that every possible option/path
is bug-free or that final PINN L2 is fixed.

## Actual sources and provenance

- Installed/pinned SOAP_JAX0.2.2:
  https://github.com/haydn-jones/SOAP_JAX/blob/a7c92f174c49b13136294a94ae0d46266b4f671f/src/soap_jax/soap.py
- Author PyTorch:
  https://github.com/nikhilvyas/SOAP/blob/a1e553530fde97d0e6b307d7c82ac6d38b072340/soap.py
- Independent pytorch_optimizer:
  https://github.com/kozistr/pytorch_optimizer/blob/8281be7c1dc39669328fe5b462eff10817b19175/pytorch_optimizer/optimizer/soap.py
- Upstream JAX history/known fix:
  https://github.com/haydn-jones/SOAP_JAX/blob/a7c92f174c49b13136294a94ae0d46266b4f671f/README.md

Installed metadata identifies the pinned revision in pyproject.toml. The
installed file matches the immutable upstream source and prior repeatability
probe's archived upstream file. Author/JAX branch heads checked against GitHub
metadata coincide with those revisions at this audit; independent source is
pinned to its latest SOAP-file commit. Raw sources/licenses retained. Direct
paths and byte comparisons only; no file hash/checksum loading gates.
The project factory src/nonlocal_pinn/training/optimizers.py imports soap_jax.soap,
wraps global norm clipping and an Optax LR schedule, and passes no weight_decay
argument (library default0). The project has no separate handwritten SOAP core.

## Source-equation audit

| Item | Current JAX versus author | Evidence |
|---|---|---|
| First call | builds Gram/basis, zero parameter update | all streams first parameters exact initial |
| Effective Adam step | call count minus1 | bias/no-bias/vector recurrence |
| Projection/backprojection | tensor axes/contractions agree |1D/2D/3D and partial-axis streams |
| Gram EMA | updated every call | full state streams including off-refresh calls |
| Basis refresh | after parameter update, calls6/11/16 at frequency5 |16-call stream controls |
| First moment | re-express old-basis moment in new basis | common-basis full physical moment |
| Second moment | permute with column ordering, not full covariance rotation | shared-basis variance arrays |
| Epsilon/bias correction | same placement as author | exact unpreconditioned FP64 controls |

Author code roundtrips first moment through physical coordinates even on
non-refresh steps; JAX keeps its already-current coordinates until a refresh.
These are equivalent in exact arithmetic, with roundoff differences measured.
Independent code stores the first moment physically; that is also equivalent.
A different moment storage convention alone is not a defect.

The JAX README documents an OLDER stale-EMA bug on non-refresh calls. Current
pinned0.2.2 includes its fix; source inspection and off-refresh stream comparisons
confirm it is not the current tested failure. Do not infer that the installed
version has an old issue solely because an upstream README mentions it.

## Executed numerical controls

Three actual implementations,16fixed gradient-stream cases each,16calls per
case:1D unpreconditioned, square2D,3D tensor, partial-axis preconditioning,
FP32/FP64,1D preconditioned and rectangular rank-deficient controls, correct_bias
on/off, plus nonzero decay. Shapes include3x3,3x4x2,6x3 with maxdim4,2x5 and
vectors. All tested paths use dense real gradients and merge_dims=False.
Streams share raw parameters/gradients and lr.001,betas(.9,.99),eps1e-8,
frequency5, limits/1D flags explicit in fixtures.json. No PDE reference fields.

Two complementary comparisons:

1. NATIVE: each implementation's actual eig/QR and precision pipeline. Compare
   within FP32-scale tolerance, not bitwise eigenbasis identity across backends.
   Maximum parameter gap among tested zero-decay native streams1.19209e-7.
2. SHARED BASIS: explicitly prescribe the SAME orthogonal initialization and
   refresh basis/column permutations in all three implementations, leaving
   moment/update equations unchanged. This intentionally isolates state/step
   bookkeeping from eigenbasis gauge/rounding. SixFP64cases cover three refreshes;
   maximum parameter gap2.22045e-16, physical first-moment gap5.88418e-15.
   All compared shared-basis moments/parameters pass absolute1e-12. This is a
   diagnostic override, not a changed production optimizer.

A separate1005-step NumPy recurrence checks the ACTUAL shared project factory:
all raw gradient norms exceed clip10; first update skipped; bias/eps/continuous
exponential LR decay cross the1000-step point. Parameter max gap1.79178e-9
under declared2e-8 bound accounting for the existing FP32 scalar LR schedule.
This verifies wrapper counting/clipping/scheduling rather than mirroring its
Optax composition in another test. Full PINN training is not repeated.

Runtime: JAX0.10.1/Optax0.2.8 in .venv; system Torch2.8.0+cu128 for CPU reference
calls, exchanging fixed NPZ arrays. No new package install or environment
modification. No FEniCS activity or concurrent GPU training. All test workers
exit0; GPU idle at completion.

## Reference-loading and precision caveats

Author source is imported directly. Independent SOAP/BaseOptimizer/exception
class bodies are extracted EXACTLY with AST and future annotations into an
isolated namespace, omitting unrelated package/foreach imports. Tested dense,
real, merge_dims=False paths use their original class bodies and real Torch
optimizer/state helpers; no numerical helper is mocked. This is not a full
packaged-install/API integration test. Shared-basis controls replace basis
selection only, explicitly and symmetrically across implementations.

Torch stream tests set default dtype to the CASE dtype so author's dtype-less
Gram allocation is consistent with its parameter. Author native eig/QR still
explicitly downcast to FP32 and cast bases back. Independent initial eig follows
matrix precision, but QR downcasts to FP32; current JAX follows parameter dtype
unless mu_dtype/qr_dtype explicitly overridden. Thus nominallyFP64 parameters
do not imply all three implementations useFP64basis selection.

Separate test: author's unmodified matrix path withFP64parameters and normal
FP32Torch default raises 'expected dtype float for end but got dtype double'.
This limitation is reproduced and recorded, not bypassed silently; reference
stream precision setup above is explicit. It is not an error in current JAX.

## Reproduced sensitivity, distinct from a coding defect

Use the actual measured five near-identical FP64 initial-gradient repeats for
the network output-weight2x64 block. Apply a COMMON SYNTHETIC second gradient,
identical across variants/backends (not the prior full-network second-time
experiment). Compare second-update vectors, not physical fields/L2:

- Current JAX nativeFP64:69–84% relative update differences.
- Independent nativeFP64:74–84%.
- Author native FP32 eig erases this particularFP64scale perturbation and returns
  identical updates. This is not proof of robust initialization: perturb ONE
  first-gradient FP32 entry by ONE representable step (ULP), and author's
  second updates differ62–78%. Independent FP32 gives similar62–78%, current
  JAX FP32 gives79–93% under its native eigensolver rounding.

The rank-deficient-gradient example is shared by the author design: a2x64
single-gradient right Gram has rank at most2. Different implementations can
choose substantially different bases in62near-null directions; coordinatewise
Adam normalization need not be invariant to those rotations. These experiments
support a common numerical sensitivity, not a JAX-specific missing transpose,
wrong step index or corrupted moments. The earlier25–26% figure referred to
FULL-network actual second-time gradients; do not confuse it with these separate
single-block/synthetic-second-gradient controls. None of these percentages is
a final physical-field L2 or claimed accuracy improvement.

## Confirmed nonzero-weight-decay difference

Let u be the normalized negative gradient update and a=lr*weight_decay.
Author and independent: p_new=(p_old+u)*(1-a).
Current JAX: p_new=p_old*(1-a)+u.
Therefore JAX-author=a*u. First real-update vector control reproduces this
identity to1e-15, max gap~1e-7 at lr.001/decay.1; after16calls gap5.49e-7.
This is a REAL ordering difference versus author, not a basis-gauge artifact.
Current factory does not enable decay and uses the library defaultZERO, so
this branch cannot explain the investigated training variation. No unrelated
production/package repair is made for an inactive option.

## Scope, records and task-end

No confirmed core translation/call defect explaining CURRENT instability is
found in these tests. This is not a proof of every option/shape/backend. No
production/dependency/default/reference/old-checkpoint changes; no complete
training run or final-L2 gain is claimed. Preserve installed-version provenance;
do not silently change optimizer implementation or precision under old presets.

Companion analysis.json contains numerical results; raw fixtures, stream/state
arrays, native sensitivity arrays, wrappers, source/licenses and metadata are
archived in soap-implementation-audit-evidence/. reports/soap-implementation-audit
retains all outputs/logs. Scripts soap-audit-fixtures/run/wrapper/author-dtype/
analyze are archived alongside this task and discover repository root by
pyproject.toml. Use .venv/bin/python with JAX_PLATFORMS=cpu for fixture/wrapper/
JAX runs and system Torch Python for --backend=author|independent. Analyzer is
NumPy-only. Scripts use the reports directory; use fresh outputs when repeating
so archived results remain unchanged. All final schema/link/path/diff checks
are in the companion validation.json.

Updated example README, DEVELOPMENT, artifact index, PROGRESS and append-only
DECISIONS. Completed task archived; no blocker or worker remains. No Git history
operation is authorized/performed. A numerically stable weak-direction treatment
or a different implementation's full-run L2/time advantage remains future work.
