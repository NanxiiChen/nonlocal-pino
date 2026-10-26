# Ice melting without an embedded motion law — 2026-10-24

## Status and corrected conclusion

Completed response to the user's objection that the previous trial function was
nearly constructed from the reference. That objection is valid: earlier numbers
only establish correction of an already accurate nominal-motion prior, and did
not establish autonomous evolution learning. Its production recommendation was
suspended and its record qualified before these new experiments.

Seven new matched-budget pilots remove the explicit motion term. Predicting a
finite tanh variable and using the exact transformed nonlocal residual remains
promising with **only the prescribed initial condition**: three seeds give 5 s
full-grid relative L2=8.66e-5–1.53e-4, compared with 0.003804–0.004662 for the
original phase residual in the same IC-only representation. The largest
absolute radius error over all 18 seed/direction terminal checks is 0.0724% vs
3.0894%. Untrained fields are stationary with terminal L2=0.7718035.

[Evidence/provenance](icemelting-weak-prior.json),
[figure](icemelting-weak-prior.png),
[earlier conditional strong-prior experiment](icemelting-late-time-accuracy.md).
These results concern the same fixed graph and one prescribed problem; they
are not continuous-PDE spatial convergence or a general-solver guarantee.

The 0.003804–0.004662 phase-residual controls above also have a hard initial
condition and are not historical production baselines. The original no-hard-IC
SOTA checkpoint gives full-grid physical4 s L2=0.01412 vs analytic and 0.01466
vs the same-graph numerical reference. See the
[baseline metric audit](icemelting-baseline-metric-audit.md) for aligned 4 s
metrics, exact provenance and the unequal training-budget limitation.

## Trial functions and information allowed

Old strong-prior control:

$$
u_\theta=u_0-N_3t^*+t^*N_\theta(x,t^*),\qquad \phi_\theta=\tanh(u_\theta).
$$

Its zero network already produces the nominal linear-radius profile. Its
terminal untrained relative L2 was ~4.25e-4, so its excellent initial accuracy
must not be attributed to training.

New IC-only control:

$$
u_\theta=u_0(x)+t^*N_\theta(x,t^*),\qquad
\phi_\theta=\tanh(u_\theta),\qquad \tanh u_0=\phi_0.
$$

u0=(R0*-|x-xc|)/(sqrt(2)*eps*) comes from the given initial condition. The trial
function contains no explicit lambda*t or N3*t, evolved radius, width penalty,
Eikonal law or future profile. A zero final layer initially produces a
stationary sphere. Its initial time derivative is learnable through N(x,0),
not prescribed as zero throughout training. The network receives x,y,z,t and
can change subsequent position, width and shape.

At the unrestricted function-class level, any differentiable finite u with
given u0 can be written this way: N=(u-u0)/t for t>0, extended at zero by u_t.
Thus hard IC alone imposes no additional motion law. This does not assert a
finite 64x4 network can exactly represent every such function. tanh constrains
the range but not future thickness, since spatial gradients of u are learned.
No floating-point arctanh of saturated phase arrays is used: the known analytic
IC provides u0 directly.

Both objectives use the **unchanged graph equation**. For finite u:

$$
r_\phi=(1-\phi_i^2)r_u,
$$

$$
r_u=(u_i)_t+N_1\sum_jw_{ij}\sinh(u_i-u_j)
\frac{\cosh(u_i)}{\cosh(u_j)}-N_2\tanh(u_i)+N_3.
$$

The stable log-domain ratio and derivation are unchanged from the earlier audit.
N3 belongs in the PDE residual as the specified forcing; it is no longer
integrated explicitly into the trial function. The transformed MAE is a
changed optimisation objective with the same pointwise equation. Its scalar
loss cannot be compared directly to the original-phase MAE.

## Matched experiment and reproducibility

Code: examples/icemelting3d/accuracy.py. New default configuration:
examples/icemelting3d/configs/accuracy_weak_prior.py. Explicit methods:
ic_phase, ic_balanced, ic_curriculum. The former latent_* modes remain available
only as explicit strong-prior controls through accuracy_pilot.py. README now
uses the IC-only trial function and clearly distinguishes the two experiments.
Production train.py/losses.py and their default config are unchanged.

From the repository root, using its .venv:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m examples.icemelting3d.accuracy reference
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m examples.icemelting3d.accuracy check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m examples.icemelting3d.accuracy train
```

For the completed runs the unchanged verified reference was copied into a new
output root rather than regenerated. Identity checks and SHA256 ensure matched
physics and graph. The reference command above is for a fresh output directory;
existing files are protected against overwrite. Select a new directory in a
copied config and pass --config PATH to repeat. --method ic_balanced selects
one method. Explicitly pass accuracy_pilot.py for historical strong-prior modes.

Output roots, all separate local ignored directories:

- reports/icemelting-weak-prior-20261024
- reports/icemelting-weak-prior-20261024-seed7
- reports/icemelting-weak-prior-20261024-seed123

Each method preserves full config, source/package snapshot, initial/final
metrics, history, checkpoint and highest-precision six-direction geometry
checks. Reference labels are used only by evaluation; they do not enter the
loss, batching, updates or fixed stopping budget.

Parameters and budget match the previous pilots: Tc=5 s, t* in [0,1], physical
4 s at t*=0.8, M=0.1, lambda=5, R0=35, eps*=0.0225, 128³ grid, delta=0.01,
six-neighbour cubic graph. SOAP, 2000 updates, 64x4 MLP with Fourier embeddings,
LR=1e-3 and configured decay, float32 parameters, highest matmul precision,
256 uniform grid cores x16 time samples per step. Within each paired seed,
network initialisation and sample draws are matched. Both IC-only methods start
from exactly the same stationary field. Seven runs: paired methods at seeds
42/7/123 and one expanding-horizon curriculum at seed42.

Evaluate all 128³ nodes at six physical times 0–5 s. Original phase residual,
reference-interface MAE, max phase error, same-grid volume, and interpolated
near-axis radius/width accompany full-grid relative L2. Six-direction geometry
uses the same highest precision as training and matched reference coordinates.

## Verification and control validity

- Hard IC max error is exactly zero with a nonzero correction network.
- At fixed network weights, doubling physical lambda changes IC-only trial
  output by exactly zero at all tested nonzero times. It still changes the PDE.
- Zero correction produces exactly zero untrained motion in every IC-only mode.
- Exact residual identity: max abs error 3.49e-13 on independent finite states.
- Saturated-tail stable ratio: ~1e-15 error against 80-digit arithmetic.
- Reused fixed-graph float64 reference has dt-halving max field difference
  3.46e-9; no rerun was needed because physics/discretisation are unchanged.
- All seven runs completed full-grid acceptance evaluation and all six geometry
  directions. Original production suites were not rerun because untouched.
- The summary figure was inspected; final tracked/new-file whitespace and JSON
  consistency checks are part of task-end verification.

## Numerical results

Paired three-seed ranges:

| IC-only objective | 4 s relative L2 | 5 s relative L2 | Largest absolute terminal radius error over 18 direction/seed checks |
|---|---:|---:|---:|
| Original phase residual | 0.004222–0.005160 | 0.003804–0.004662 | 3.0894% |
| Exact transformed residual | 4.99e-5–1.34e-4 | 8.66e-5–1.53e-4 | 0.0724% |

Paired terminal L2 reductions are 52.81x / 28.50x / 43.92x for seeds 42/7/123.
The untrained stationary endpoint has radius ~0.349948 and L2=0.7718035; the
matched reference near-axis radius is 0.100049. Thus an approximate future
trajectory is not already embedded in this new initial state.

Seed42 terminal examples:

| Mode | Relative L2 | +x near-axis radius | Reference-interface MAE |
|---|---:|---:|---:|
| ic_phase | 0.00466244 | 0.10167259 | 0.0371652 |
| ic_balanced | 8.829e-5 | 0.10003867 | 0.00045568 |
| ic_curriculum | 1.196e-4 | 0.10002559 | 0.00076171 |

Transformed terminal max pointwise phase error is ~0.00258–0.00261. Its
affected inner-ice region is small, so report this alongside global relative
L2. Measured transformed volume errors are zero in positive-node counts;
this is a quantised same-grid diagnostic, not proof of exact continuous volume.
Original phase controls overestimate count-based volume by ~6.3–7.3%.

The single-seed curriculum gives no extra benefit at this budget. This is an
expanding-horizon experiment, not a validation or refutation of true temporal
slabs/causal training for general problems.

## Revised recommendation and limits

Remove explicit nominal motion from the preferred experiment. The evidence now
supports further validation of **IC-only tanh variables and the exact transformed
nonlocal residual** for this case. It does not require supplying a future radius
or phase profile. Earlier strong-prior results remain historical conditional
controls, not evidence for that claim.

A production integration should be opt-in and validated at the intended
sampling/budget with multiple seeds and geometry/pointwise metrics. No such
integration or production-default change is delivered here. Broader geometry,
parameter and IC tests remain separate work. Spatial quadrature inconsistency
(~0.686 second moment), continuous-PDE accuracy and FEM parameter/BC issues
are also not corrected by this optimisation experiment.

## Session diff, blocker and exact continuation

Extended isolated runner with IC-only methods, no-speed-dependence checks and
safe method validation. Added accuracy_weak_prior.py; switched only the research
runner default to it. Updated example README, qualified the older archived task,
updated docs/DECISIONS.md and docs/PROGRESS.md, and archived this report/JSON/PNG.
No old checkpoint or external artifact changed; no Git history operations.

No blocker for this control experiment. Next task, if implementing production:
port the IC-only branch and stable exact residual into an explicit optional
train/loss mode, retaining original settings as a control. Reuse reference and
snapshots; use a fresh output path. Do not resume historical runs or restore the
nominal-motion term to obtain low errors.
