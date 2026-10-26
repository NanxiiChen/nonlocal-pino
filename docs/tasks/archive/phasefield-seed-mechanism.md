# Cross-seed failure mechanism diagnosis —2026-10-26

## Goal and outcome

Complete. User asks whether fixed IC/BC sampling removes helpful randomness
and causes the cross-seed failures, or whether other mechanisms are involved.
Read-only probes of7 final KKS checkpoints,10 intermediate corrosion checkpoints,
actual samplers/pool coverage and saved loss/weight histories provide the
following evidence. No retraining, physical/default change or unique-cause
claim is made. No blocker or required remaining action.

Main conclusion: loss of IC/BC randomness is not the strongest supported
explanation. Corrosion constrains every nodal IC/BC and fails despite very
small data errors; an existing same-seed residual-weighting control succeeds
with the original objective and fails with bounded weighting. The evidence
supports sensitivity of optimization paths in a stiff coupled loss. Fixed
PDE pools do leave coverage holes, but most error and high PDE residuals occur
on covered nodes too. Their causal influence cannot be eliminated from frozen
models alone. Electropolish's failed model additionally has a poor full-grid
initial fit, so its mechanism need not be identical to corrosion's.

## What is fixed versus random

Corrosion uses all5151 nodal ICs plus9 pit and201 solid BC nodes at every
update; IC targets come from the prescribed analytic field and boundary
values, not later reference snapshots. There are no omitted IC/BC graph
nodes. BC evaluation times vary with the PDE temporal batch. Geometry and
IC/BC coordinates/targets are identical for successful and failed seeds.
KKS initialization and temporal draws change through the run seed; the fixed
100-entry spatial PDE pool cycles50 times during5000 updates. Melting omits
an IC loss by exactly embedding the initial field, and its sparse-mode time
keys are fixed by iteration. Changing its seed mainly changes initialization.

Electropolish uses the same2000 fixed off-grid random IC points and128 random
BC points on each of top/bottom for all seeds. These are soft sampled data,
not the complete graph constraints. Distinguish their sample loss from
whole-grid IC error. The surface/SDF definition uses a fixed2000-segment
profile and fixed surface seed independent of evaluation batch shape; no
batch-dependent IC geometry mismatch was found.

## Strong IC/BC fit does not distinguish corrosion success/failure

The following values use common full-grid/time probes of the final models.
BC is maximum physical(phi,c) error over all declared graph nodes/all21 times.

| Corrosion | Initial phi L2 | Max BC physical error | Initial q RMSE/cLe,free | Initial Dq L2,free | Terminal phi L2 |
|---|---:|---:|---:|---:|---:|
| bounded42,successful | 5.72736e-4 | 2.44597e-3 | 0.0211174 | 0.420019 | 0.00143305 |
| bounded7,failed | 7.28248e-4 | 2.81668e-3 | 0.0146959 | 0.320124 | 0.422001 |
| bounded2026,failed | 7.97391e-4 | 6.27637e-5 | 0.0236309 | 0.663168 | 0.407016 |
| original7,successful | 4.55754e-4 | 3.27110e-4 | 0.00601366 | 0.197320 | 0.00422477 |

Failed2026 BC error is~39 times smaller than successful42's, yet terminal
phi error is~284 times larger. Failed7 even has smaller whole-grid initial
q/Dq errors than bounded42. This does not exclude sensitive localized initial
concentration gradients, but neither nodal BC accuracy nor these initial norm
metrics alone explains outcomes. Low initial phi error cannot certify the
coupled time evolution. All physical output bounds remain enforced by KKS.

## Existing same-seed residual intervention

The saved corrosion original7 and bounded7 configs differ only in
experiment.formulation and log.run_name; initial parameter bytes are exactly
equal. Geometry, IC/BC,spatial pool,seed/time stream,optimizer/scales and budget
are identical. Original phase/raw-KKS-edge/L1 gives terminal phi0.00422488;
bounded eta0.1 gives0.422001. The previous control/audits verified source and
schedule equivalence for the disabled continuation settings.

This is direct evidence that the bounded objective can worsen the trajectory
under the same fixed data. It is stronger evidence than attributing failure
to lost IC/BC randomness alone, but is not repeated-seed statistical proof or
a unique explanation of every failing run.

For finite phi, bounded weighting preserves the physical PDE zero set:

$$
w_\eta(\phi)=\frac{1+\eta}{4\phi(1-\phi)+\eta},\quad
L_{AC}\propto w_\eta(\phi)|r_{AC}|.
$$

The optimized parameter gradient changes, including

$$
\nabla_\theta\bigl(w_\eta|r_{AC}|\bigr)
=w_\eta\,\operatorname{sign}(r_{AC})\nabla_\theta r_{AC}
+|r_{AC}|\nabla_\theta w_\eta.
$$

The implementation differentiates this state-dependent weight, rather than
freezing it. A small AD/product-rule check passes; no new training with the
weight detached is claimed. Scaling/weighting a residual does not preserve
its optimization path or condition it uniformly across different states.

## Full residual and conditioning evidence

Probe residuals use the complete reference graph and common positive times
.05,.10,...,1; t=0 is excluded from PDE statistics. Native AC is computed with
the exact FP32 physical-difference formula of training. CH is independently
reconstructed with raw-KKS q and SciPy edge action. These are frozen common-
time diagnostics, not the exact random training-batch loss or its expectation
at arbitrary times. Never compare raw/latent/bounded loss magnitudes as if
identical objectives. Within the same bounded objective:

| Corrosion | Full-grid scaled bounded AC MAE | Fixed-pool-distribution AC MAE | Full-grid CH MAE |
|---|---:|---:|---:|
| bounded42 | 0.0660973 | 0.0657980 | 0.0923171 |
| bounded7 | 1.51380 | 1.41950 | 0.133128 |
| bounded2026 | 1.19405 | 1.13871 | 0.163243 |

The failed model has high PDE imbalance on training-covered points as well,
not merely small training residual and hidden large held-out residual. This
supports failure to optimize the PDE sufficiently within5000 updates. It does
not establish a strict local minimum, zero-gradient stationary point or that
longer training could never recover.

In bounded2026, unscaled AC term RMS over the positive-time free grid is:
time derivative1.30534, diffusion1.11902e8, reaction1.02005e8 and driving
5.76857e7. The successful bounded42 also has~1e8 diffusion/reaction versus
~0.738 time derivative. The equation requires large terms to cancel; applying
one overall AC scale1e-7 does not remove their internal scale disparity.
Changing concentration changes the phase drive, and KKS/one shared network
couples both physical outputs. This is a concrete conditioning concern,
not a measured Hessian spectrum or isolated gradient-causality result.
Both successful and failed models hit BC GradNorm caps10000, so cap activation
alone does not diagnose the failure. Final adaptive PDE weights are broadly
similar, while PDE loss values differ substantially. Corrosion cL derivative
saturation fractions are similarly~0.14–0.17%; total saturation is not the
large collapsed-concentration mechanism from earlier full-latent pilots.

The general mechanism of stiffness and imbalance in back-propagated PINN
loss gradients is discussed in Wang,Teng,Perdikaris,
[Understanding and mitigating gradient pathologies in PINNs](https://arxiv.org/abs/2001.04536).
That paper supplies context, not evidence proving the unique cause here.

## Intermediate checkpoint trajectories

Fixed evaluations of training checkpoints1000,2000,3000,4000,5000 show:

| Training updates | bounded42 terminal phi L2 | bounded2026 terminal phi L2 |
|---|---:|---:|
| 1000 | 0.332873 | 0.311042 |
| 2000 | 0.377753 | 0.328429 |
| 3000 | 0.120053 | 0.371919 |
| 4000 | 0.0145272 | 0.399760 |
| 5000 | 0.00143305 | 0.407016 |

Both have extra liquid components at1000/2000. The successful42 removes them
by3000 and improves later evolution;2026 retains them through5000 despite
improving IC/BC errors. The observed distinction is a different optimization
trajectory under the fixed budget, not an already-proven separate physical
solution branch or an IC that only the successful network can represent.

## Fixed spatial PDE pool: real gap, insufficient explanation on its own

Corrosion has4941 free PDE nodes;35(0.708%) never appear as a core in the100
fixed subgraphs, though they can still serve as halo inputs. The pool is
reused50 times. In failed2026 these nodes contain0.935% of terminal squared
phi error and0.114% of squared Dq error. Over99% of phase error lies elsewhere.
Whole-grid versus pool-distribution AC MAE differs only~4.9% and both are high.
Thus the observed failure is not simply all error concentrated at unseen
spatial nodes. Holes could still influence coupled optimization/dynamics;
a fixed-pool versus refreshed-pool training intervention is not yet tested.

Electropolish has37 never-core nodes among4949(0.748%). The failed42 model's
unseen nodes contain1.268% of phase squared error, and its full/pool AC MAE
are0.197507/0.195757. Again the failure extends to covered points.

## Electropolish differs from corrosion

Failed original42 full-grid initial phi L2=0.0792173; successful7/2026 have
0.00166628/0.00161465. Original42 initial q RMSE/cLe=0.114213 and initial Dq
L2=0.888872, versus7's0.0527522/0.568377 and2026's0.0524576/0.539636.
The failed model does not fit the initial field as accurately even though
all runs use identical fixed sample coordinates. Its same-budget full AC/CH
MAEs0.1975/0.1260 also exceed the successful~0.011/0.010 values. Finite fixed
IC coverage/soft constraint strength may contribute here, but different seeds
can fit the same data and full graph well; removing randomness is not proven
as the cause. Earlier nodal-data+s_bc controls succeeded in both observed7/42,
but also changed sample counts/loss coordinates and are not an isolated
resampling experiment.

## Controlled next experiments, not performed in this diagnostic turn

1. Hold spatial data/pool fixed and cross initialization seed with temporal
   sampling seed, since current KKS run.seed changes both together.
2. For initialization2026, compare original/bounded objectives under exactly
   the same temporal sequence/source/runtime; seed7 already supplies a useful
   such control. Report global phi/c/q/Dq and convergence, not best checkpoints.
3. Change one sampling mechanism at a time: fixed versus refreshed PDE pool,
   then full fixed nodal IC/BC versus unbiased random nodal minibatches. For
   corrosion, off-grid BC resampling changes the finite-graph training targets
   and should not be conflated with merely injecting optimizer noise. Preserve
   fixed JIT shapes/total sample budgets and the declared reference model.

Extra sampling noise might help or hurt optimization. No resampling cure,
IC-hard-constraint remedy, pre-scale removal or new default is claimed.

## Artifacts, validation and task end

- [Common-grid probe source](phasefield-seed-mechanism-probe.py),
  [complete metrics](phasefield-seed-mechanism-probe.json),
  [diagnostic plot](phasefield-seed-mechanism-probe.png).
- [Checkpoint probe source](phasefield-seed-mechanism-checkpoints.py),
  [10-checkpoint results](phasefield-seed-mechanism-checkpoints.json).

All7 final phi norms and10 intermediate norms reproduce saved GPU logs within
3e-6 absolute tolerance. Exact reference graph edges/weights, core distribution
normalization, masks/time definitions and positive reference norms verified.
State-dependent gradient product rule checked by AD; seven-model plot visually
inspected. Strict JSON, source syntax, links/image and git diff --check pass.
No production numerical code changed, so no unchanged code suite or full
training was repeated. All probes exit. Complete; no blocker/remaining action.
Memory changed: docs/PROGRESS.md,docs/DECISIONS.md and this archived task.
