# Same-seed two-phase repeatability diagnosis — completed

## User request and conclusion

User explicitly chooses SAME-seed rerun diagnosis before deciding a larger
local/nonlocal complete-training cohort. This bounded investigation locates
an early trigger/amplifier, not a new field-accuracy result or completed fix.

Ordinary operator reverse accumulation produces measured FP64 gradient noise;
SOAP initial rank-deficient Gram eigendirections amplify tiny perturbations
into materially different updates. Equal initial values/time draws therefore
do not guarantee equal training paths. Equivalent symmetric pullback removes
the tested WITHIN-process noise, but independent processes still differ at
roundoff and diverge: do not promote it as a complete reproducibility remedy.
The precise compiler/kernel source of remaining cross-process differences is
not isolated. This limitation is part of the completed diagnosis.

## Context and immutable inputs

Original same-config nonlocal pooled local-reference errors2.84283% versus
3.77757%, with identical initial checkpoint/config excluding name and first
loss. Current strategy is FP64/ordinary, inverse cubic, original mixed residual,
exact IC/grid mass, spectral-flux advection, SOAP5000+L-BFGS1000. Earlier
local3.36431%/terminal3.22328% versus nonlocal3.77757%/3.26634% single pair
cannot establish a stable accuracy ordering. No new complete training runs.

Load EXACT saved GPU initial arrays from
/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-nonlocal/model_initial.eqx
and its config.json, verify every leaf against GPU-reconstructed seed42 initial
values. Canonical FP32 open-stratified time draws are promoted to FP64;
all arms' saved times match exactly at shared steps. Physical reference remains
/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz; its field values do
not enter the objective. Use the original canonical _make_step JIT for restarted
paths, original optimizer, exact same immutable model/state/times in each repeat.
No PDE/network/sampling/optimizer/default/reference/Git history change.

## Six bounded GPU processes

| Process/arm | Adjoint | Policy | Steps per restart | Restarts |
|---|---|---|---:|---:|
| ordinary-autodiff | original AD | ordinary |100|2|
| ordinary-symmetric | symmetric L | ordinary |100|2|
| deterministic-autodiff | original AD | deterministic |5|2|
| ordinary-autodiff-spectrum | original AD | ordinary |2|2|
| ordinary-symmetric-independent | symmetric L | ordinary |100|2|
| ordinary-symmetric-independent-repeat | symmetric L | ordinary |100|2|

Total12short trajectories/814updates. Every process also repeats the frozen
operator transpose5times, frozen full loss/gradient5times, identical frozen-
gradient SOAP initialization3times, and controlled different-initial-gradient
states with an IDENTICAL common second gradient. All processes exit0.
The first three use probe v1. V2 adds output-name selection, array/spectrum
recording and finite-difference validation; numerical stepping is unchanged.
The FINAL TWO independent-process controls use exactly the SAME v2 source,
arguments except output name, to avoid confusing source-recording changes with
process effects. No concurrent GPU workers or FEniCS work.

## Measured causal chain

1. Frozen ordinary scalar loss is exactly10.29859750798121 in all5evaluations.
   Full gradients differ by max6.94e-18..1.39e-17. Isolated operator AD-transpose
   repeats vary by up to1.435e-16 relative. Direct L action agrees with that
   transpose to floating roundoff; inputs and samples are unchanged.
2. IDENTICAL frozen gradient fed repeatedly to SOAP gives bitwise-identical
   GG/Q/updates. For these tested inputs the optimizer's eigensolver is not
   independently generating random state differences.
3. Different measured near-identical gradients yield GG differences~1e-21,
   Q basis differences O(1). After feeding a COMMON second gradient, actual
   second update vectors differ25.27–26.47% relative. This excludes merely
   harmless eigenvector sign changes as an explanation for all Q differences.
4. Canonical original-JIT restarts: first update changes no parameters (SOAP
   initialization); optimizer states already differ. Parameter divergence first
   appears at update2, despite identical first TWO scalar losses. By100,
   parameter relative difference32.13%; loss difference0.001786. These are
   parameter/optimizer norms, NOT physical-field L2 errors.

## Why SOAP amplifies it

Inspect the actual installed soap_jax/soap.py (archived source). The first
call forms per-axis gradient Gram matrices, initializes Q via eigh with
isotropic jitter1e-30, and returns zero parameter updates. Subsequent calls
project gradients into Q, apply coordinatewise Adam-like moment normalization,
then project back. Directional normalization is not invariant under arbitrary
rotations of a degenerate eigenspace.

Independent NumPy eigvalsh of an actual output-weight64x64Gram gives numerical
rank2 at threshold4.62e-18, max eigenvalue3.251693e-4, min about-8.47e-20
(roundoff around zero in a theoretical PSD matrix). A2x64gradient has right
Gram rank at most2 algebraically. Thus62directions are essentially unobserved
initially, and the selected eigenbasis is sensitive to roundoff. This output
block is an example of rank deficiency, not a proof that it alone causes the
entire update difference. Small scalar identity jitter does not uniquely select
a basis in an exactly degenerate subspace. This is demonstrated sensitivity
in this initialization/configuration, not a claim that SOAP always fails.

## Isolated intervention and remaining limit

For reciprocal uniform periodic weights, L_ij=-w_ij, w_ij=w_ji, and diagonal
row sum give L^T=L in the Euclidean grid inner product. A probe-only custom VJP
keeps forward L exactly and applies L to the cotangent, avoiding the gather
transpose's scatter accumulation. Dot-product symmetry checked; original and
custom full gradients agree within1.39e-17 (relative7.46e-17), and actual
parameter directional finite differences pass. This is an algebraically equal
adjoint, not new residual weighting or altered physics. It is NOT automatically
valid for row-normalized/nonuniform/truncated nonsymmetric operators, and a
production derivative API would require appropriate JVP/higher-order support.

Ordinary symmetric-adjoint repeats:5frozen gradients exact, and both100-step
restarts exact at every milestone in each of three tested processes. The
original deterministic-policy5-step control likewise matches exactly, at
substantially slower probe execution (not a new full benchmark/default).

BUT two identical-source independent symmetric processes retain identical
saved time arrays/frozen loss while first gradient differs max6.94e-18,
relative2.61e-17; initial GG differs2.54e-21 and100-step parameters29.68%
relative. The two independent model files differ, while each process's own
pair of model files is identical. Thus the scatter replacement addresses one
trigger but does not establish full-process reproducibility. Cross-process
compilation/execution roundoff remains observed; no particular CUDA kernel,
autotuning mechanism or sole final-L2 cause is established by these tests.

## Validation, artifacts and reproduction

- Archived companion analysis.json contains exact key measurements and limits.
- Archived per-arm result JSONs, saved time arrays, frozen gradient/Gram NPZs,
  short checkpoint leaves and loss paths preserve raw evidence. Probe v1,
  v2, confirmation launcher and CPU analyzer accompany this task.
- reports/twophase-repeatability retains all source snapshots, upstream SOAP
  source, raw outputs and logs. No checksum/hashing gate; direct paths only.
- Use .venv/bin/python; CUDA for the probe and CPU for analysis. V2 requires
  --policy=ordinary|deterministic --adjoint=autodiff|symmetric --steps=<n>
  --output-name=<fresh-name>. Existing outputs fail rather than overwrite.
  Set XLA_FLAGS='', JAX_ENABLE_COMPILATION_CACHE=false,
  XLA_PYTHON_CLIENT_PREALLOCATE=false, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=2.
- GPU initializer reconstruction, shared sample arrays, loss/gradient/adjoint
  agreement, parameter finite differences, spectrum/rank and repeated-state
  assertions all pass. Final archive/config/link/JSON/diff checks are recorded
  in validation.json. No unchanged expensive unit suite is repeated.

## Decision on next experimental scale

Do not immediately expand local/nonlocal mean/std claims. First investigate
and validate numerically stable SOAP initialization/weak-direction handling,
without introducing reference-dependent priors or changing the PDE. Equivalent
adjoint is a useful partial diagnostic, not sufficient alone. After a candidate
passes fixed-input perturbation tests, use TWO complete SAME-seed cold starts
with unchanged fixed budget to check reproducibility and final L2/time. Then
start a small predeclared paired-seed cohort (e.g. three seeds), retaining all
outcomes, means/spreads/worst case and accuracy-versus-wall curves. More seeds
must not be selected to obtain a favorable winner. This protocol is a future
recommendation; this task launches no long cohort and claims no accuracy gain.

## Task-end

Status: completed bounded same-seed diagnosis; no blocker/worker remains.
Production files, default strategy, reference and old checkpoints untouched.
Updated example README, artifact index, PROGRESS and DECISIONS; archived
scripts/evidence/task. No commit/merge/push or other Git history operation.
A full cross-process remedy, final-L2 benefit and local/nonlocal multi-seed
statistics remain explicitly separate future work.
