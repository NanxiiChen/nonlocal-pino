# Corrosion latent-residual mechanism diagnosis — 2026-10-25

## Status and scope

Completed checkpoint diagnosis: bulk/interface objective redistribution, stiff
reactions and learned concentration drive are supported mechanisms. Algebraic
cancellation alone does not establish good optimization or physical relative L2.
The probes do not prove a unique cause or rule out successful transformation.
Hard IC is excluded from the theoretical requirements. No new training, PDE,
default, reference, checkpoint or Git-history change.

[Reference alignment/pilots](coupled-latent-residual-experiments.md);
[KKS/scale ablation](corrosion-kks-scale-ablation.md);
[probe source](corrosion-latent-mechanism.py);
[outputs and direct checkpoint paths](corrosion-latent-mechanism.json).

## Reproduction and validation

From repository root, JAX environment only:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u docs/tasks/archive/corrosion-latent-mechanism.py
```

Four checkpoints are evaluated at all graph nodes at physical10/25/50 s;
Dirichlet nodes are excluded from PDE statistics. Graph rows, columns, weights
and diagonal are array-exact against native SparseD for every run. Both
formulations use the research trainer's compute_pde; CH arrays are identical.
Actual AC training uses mean absolute residual, matching this diagnostic.
Scalar pre_scale/GradNorm weights do not change within-AC spatial fractions.
Sampled batches and other losses still affect the actual training trajectory.

CPU full-grid terminal phi relative L2 replays: original KKS1e-7=0.03483963,
latent KKS1e-7=0.56838573, latent KKS1e-9=0.43678762, latent direct-c1e-9=
0.53644675. These match prior results within FP32 execution differences.
No independent reference integration or training was repeated.

## Loss and gradient redistribution at fixed weights

Hold the accurate original KKS1e-7 checkpoint fixed; change only AC formulation.
Bulk is reference phi<0.01 or>0.99; interface is its complement. Of14823
free-node/time samples,75.82% are bulk.

| Statistic | Original AC | Transformed AC |
|---|---:|---:|
| Interface fraction of absolute residual |91.46%|0.529%|
| Bulk fraction of absolute residual |8.54%|99.471%|
| Bulk/interface parameter-gradient L2 norm ratio |0.02956|2.89035|

Bulk phase MAE is2.3578e-4 versus0.03435 at the interface. Predicted highly
saturated samples (phi<1e-5 or>1-1e-5) have phase MAE5.27e-7 but account for
87.85% of transformed AC absolute residual.

Region gradients share the same global denominator. Unscaled L2 norms are
9.04e6 bulk/3.06e8 interface (original),2.76e9 bulk/9.54e8 interface (latent).
This establishes parameter-sensitivity redistribution, not just larger losses.
Gradient norms are not additive loss shares. Do not infer that transformed
updates necessarily increase physical error: the full AC gradient cosine with
diagnostic squared-relative-L2 gradient is positive for both original0.00480
and transformed0.08255. These are unweighted Euclidean probes, not historical
GradNorm/SOAP update directions. Reference fields enter diagnostic masks/error
gradients only, never training operations.

## Coupling and near-solid limit

Let a=4phi(1-phi), h=3phi^2-2phi^3, dc=c_se-c_le, q=c-dc*h-c_le,
and phi=(1+tanh(z))/2. The normalized residual is r_hat=r_phi/a.

$$
\widehat r_{\rm reaction}=\tfrac12N_2(1-2\phi)-\tfrac32N_1\Delta c\,q.
$$

At fixed physical phase,

$$
\frac{\partial r_\phi}{\partial c}=-6N_1\Delta c\,\phi(1-\phi),\qquad
\frac{\partial\widehat r_\phi}{\partial c}=-\tfrac32N_1\Delta c
=-1.5477587413\times10^{10}.
$$

Independent centered finite differences have4.57e-11 relative error. Sensitivity
amplifications at phi0.001/0.01/0.5 are250.25/25.25/1. With AC scale1e-7,
a concentration perturbation1e-4 changes scaled latent residual by0.1548.
This is a partial derivative, not a causal decomposition of residual into
independent error sources. KKS makes q=c_L-c_le but retains the concentration
output Jacobian and the coupling of concentration transport with phase motion.

For uniform stationary phi=1-epsilon, c=c_se=1, Dphi=0, q is O(epsilon^2):

$$
r_\phi=-2N_2\varepsilon+O(\varepsilon^2),\qquad
\widehat r_\phi=-N_2/2+O(\varepsilon).
$$

Here N2=1.76e9. At epsilon1e-9, original/latent residuals are-3.52/-8.8e8;
with scale1e-7, magnitudes are3.52e-7/88. The script independently checks
this reaction-only limit. For the uniform evolution z_t tends to N2 even
when physical phase is almost solid. Real spatial solutions can instead
balance diffusion and drive against reaction; this test does not claim
otherwise. It demonstrates that accurate physical tails need not have small
latent loss. Finite-z pointwise zero equivalence does not give a uniform
physical-error estimate as phase approaches endpoints.

On the accurate checkpoint, latent diffusion/double-well absolute means are
7.45e8/8.15e8 while the time term is2.40. FP64 algebraic reassembly of the same
FP32 raw outputs/time derivatives changes latent residual by2.21e-6 relative
to residual MAE (at most1.11e-5 across four checkpoints). Simple residual
roundoff does not explain the large failure. This is not full FP64 training
and does not exclude precision effects during training.

## Melting comparison and next experiment

For current dimensionless horizons, melting has one phase equation, prescribed
drive, reaction coefficient0.0987654 and latent forcing7.85674. Corrosion
couples AC/CH with N2=1.76e9 over50 s; only AC is transformed. Scalar pre_scale
cannot change within-AC spatial weights or term ratios. Smaller AC scale
removed concentration saturation but not poor L2; removing KKS also failed.
KKS is not an algebraic incompatibility, and saturation is not a sufficient
explanation. Electropolish also failed without that saturation.

Matched melting controls show a real residual gain under a common IC-only
representation; corrosion controls share soft IC/BC losses. This experimental
difference is not a hard-IC prerequisite or a reason to discount the matched
melting result. No cross-model representation ablation has been performed.

The targeted next test is bounded/partial inverse-Jacobian preconditioning,
retaining PDE, KKS, soft IC/BC, reference and matched budgets. For example,

$$
r_\eta=\frac{1+\eta}{4\phi(1-\phi)+\eta}\,r_\phi,\qquad\eta>0
$$

has centre weight1, maximum(1+eta)/eta and the same physical PDE zeros. This
is a separate research objective, not the exact latent residual or a silent
fallback. No eta is selected, implemented or claimed to improve L2. A matched
intervention and coupled AC/CH gradient measurements are required to establish
whether limiting bulk amplification recovers physical accuracy.

## Session diff, validation and exact continuation

Added this report, CPU probe and JSON; updated DECISIONS and PROGRESS. No
production API/code or external user artifact changed. Probe execution, native
graph/CH identities, concentration finite differences and near-solid limit pass.
Final JSON/whitespace/diff checks are recorded at session end. No blocker or
running process. Diagnosis is complete; bounded preconditioning is a distinct
next experiment, not unfinished work in this task.
