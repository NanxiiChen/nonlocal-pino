# 1D Phase-Field Fracture with a Nonlocal Damage Operator

This example is the first controlled fracture test for the scalar nonlocal
operator `SparseD`. It deliberately nonlocalises only the AT2 damage-gradient
term. The one-dimensional mechanical equilibrium is eliminated analytically,
so errors in the damage operator can be separated from displacement-network
optimisation errors.

The intended first implementation is therefore:

```text
mechanics       exact 1D constant-stress reduction
damage          PINN on a fixed grid
damage operator SparseD, D_delta d approximately equals -d''
loading         incremental, displacement controlled
coupling        staggered mechanics/history/damage updates
```

This is not yet a fully nonlocal fracture model. A future two-dimensional
extension may use local AD, PDDO, or a damage-aware `ElasticD` for mechanics,
while retaining `SparseD` for the scalar damage regularisation.

The current local FEM reference is the fine-grid `nx=2000` classical
history-field solution. It uses fixed `du=1e-5`, DG0 history values, and the
direct linear staggered damage solve described below. Its archive is recorded
in `STATUS.md` and is the reference used by the PINN configuration.

Numerical FEM/reference and direct-baseline archives belong under
`/root/autodl-tmp/data/fracture1d/`. Every PINN run writes its TensorBoard events,
configuration, checkpoints, and result archive under
`/root/autodl-tmp/tf-logs/fracture1d/<run_name>/`.

## 1. Problem Background and Scope

This example studies quasi-static tensile fracture in a one-dimensional bar.
The bar is displacement controlled, and a scalar phase-field damage variable
represents a diffuse crack. The purpose is not to build a generic neural
elasticity solver. It is to isolate the nonlocal damage operator and test
whether a PINN can reproduce the same localized fracture response as a direct
SparseD reference.

The computational comparison has three levels:

1. local FEM versus direct nonlocal SparseD: model and finite-horizon effects;
2. direct nonlocal SparseD versus PINN: neural representation and optimization
   error;
3. PINN versus local FEM: combined model and neural error, interpreted only
   after the first two comparisons.

The current result is an initial engineering-acceptable PINN response in the
explicit `warning` gate mode. The damage profile and complete reaction curve
are close to the corrected direct nonlocal baseline. The configuration also
supports `train.gate_mode="strict"`; strict validation uses whatever explicit
gate values are present in the selected config.

## 2. Model Scope

The reference body is a bar on

$$
\Omega=(0,L_0),
\qquad
u(0)=0,
\qquad
u(L_0)=\bar u.
$$

The assumptions are:

- small strain;
- quasi-static monotonic tension;
- no body force;
- constant cross-sectional area;
- AT2 damage with a history-field irreversibility approximation;
- an optional small spatial variation of Young's modulus to trigger
  localisation without modifying the fracture energy;
- a small explicitly configured residual stiffness for numerical robustness.

The phase field is a diffuse internal variable. A developing crack is not a
new geometric boundary, and its formation does not remove `SparseD` edges or
require rebuilding the operator.

## 3. Local AT2 Formulation

Let

$$
\varepsilon=u',
\qquad
\psi_0^+(\varepsilon,x)
=\frac12E(x)\langle\varepsilon\rangle_+^2,
$$

where \(\langle z\rangle_+=\max(z,0)\). The one-dimensional tension-compression
split is explicit; compressive strain must not increase the history field.

Use the degradation function

$$
g(d)=(1-d)^2+k_{\mathrm{res}},
\qquad
0\le d\le1,
$$

where `k_res` is an explicitly configured small residual stiffness. With an
additive residual stiffness, the damage derivative of \(g\) is unchanged.

The physical interpretation is `0 <= d <= 1`, but the classical FEM
history-field reference below does not add a box constraint or a separate
pointwise damage-irreversibility constraint to the damage solve.

The local energy is

$$
\mathcal E_\ell(u,d)
=\int_\Omega
\left[g(d)\psi_0^+(\varepsilon,x)+\psi_0^-(\varepsilon,x)\right]dx
+G_c\int_\Omega
\left(\frac{d^2}{2\ell_0}
+\frac{\ell_0}{2}|d'|^2\right)dx.
$$

For monotonic tension, \(\psi_0^-=0\). The Euler equations are

$$
\frac{d}{dx}\left[g(d)E(x)u'\right]=0,
$$

and, using the common history-field approximation,

$$
\frac{G_c}{\ell_0}\left(d-\ell_0^2d''\right)
=2(1-d)\mathcal H.
$$

The natural external-boundary condition for the local damage equation is

$$
d'(0)=d'(L_0)=0.
$$

### 2.1 History field and its limitation

The history field is

$$
\mathcal H_n(x)
=\max\left(\mathcal H_{n-1}(x),
\psi_0^+(\varepsilon_n(x),x)\right).
$$

It is an external field, not a network output. This is the widely used Miehe
history-field approximation. It is not identical to the exact incremental
variational inequality

$$
d_n\ge d_{n-1}.
$$

The FEM reference uses this history-field approximation as its only
irreversibility mechanism. It does not enforce `d_n >= d_{n-1}` with a bound
solver, reject the step for that condition, or silently clip the damage field.
Any damage healing or excursion outside the physical range is therefore a
diagnostic of this approximation, not an additional constraint in the solve.

Within a staggered iteration, always compare the current tensile energy with
the history from the previous converged load step:

$$
\mathcal H^{(k)}
=\max\left(\mathcal H_{n-1},\psi_0^+(\varepsilon^{(k)})\right).
$$

Do not accumulate maxima from unconverged inner iterates. Otherwise a transient
strain overshoot can permanently overdrive damage.

## 4. Exact Elimination of the 1D Displacement Field

Mechanical equilibrium gives

$$
g(d(x))E(x)u'(x)=\sigma,
$$

where \(\sigma\) is constant throughout the bar. Therefore, for any fixed
damage field,

$$
u'(x)=\frac{\sigma}{E(x)g(d(x))}.
$$

The prescribed end displacement determines the stress:

$$
\bar u
=\int_0^{L_0}u'(x)dx
=\sigma\int_0^{L_0}\frac{dx}{E(x)g(d(x))},
$$

so that

$$
\boxed{
\sigma
=\frac{\bar u}
{\displaystyle\int_0^{L_0}\frac{dx}{E(x)g(d(x))}}
}.
$$

The strain and displacement then follow from

$$
\varepsilon(x)=\frac{\sigma}{E(x)g(d(x))},
$$

$$
u(x)=\int_0^x\varepsilon(s)ds.
$$

Thus, the displacement field is not ignored. It is exactly condensed out of
the one-dimensional problem. The first PINN should output only \(d(x)\); stress,
strain, displacement, and tensile energy are deterministic functions of that
output.

This reduction is not available in general two-dimensional elasticity, where
stress is not spatially constant and the displacement components are coupled.

## 5. Analytical Validation Results

### 4.1 Homogeneous monotonic response

For constant \(E\), homogeneous damage, monotonic tension, and
\(k_{\mathrm{res}}=0\), define

$$
a=\frac{G_c}{E\ell_0}.
$$

The homogeneous relation is

$$
d(\varepsilon)
=\frac{\varepsilon^2}{\varepsilon^2+a}.
$$

Standard AT2 has no finite elastic damage threshold: \(d>0\) for every
\(\varepsilon>0\). The quantity \(\sqrt a\) is the strain at which \(d=1/2\),
not a damage-onset strain.

The homogeneous stress is

$$
\sigma(\varepsilon)
=E\varepsilon
\left(\frac{a}{\varepsilon^2+a}\right)^2.
$$

Its peak occurs at

$$
\boxed{
\varepsilon_{\mathrm{peak}}
=\sqrt{\frac{G_c}{3E\ell_0}}
},
\qquad
d_{\mathrm{peak}}=\frac14,
$$

with

$$
\boxed{
\sigma_{\mathrm{peak}}
=\frac{3\sqrt3}{16}
\sqrt{\frac{EG_c}{\ell_0}}
}.
$$

These peak formulas are exact only for zero residual stiffness. A sufficiently
small configured `k_res` should make its effect negligible over the validation
load range, but the code must report the value used.

### 4.2 Stationary crack profile

The familiar profile

$$
d(x)=\exp\left(-\frac{|x-x_c|}{\ell_0}\right)
$$

is the infinite-domain AT2 surface-energy minimiser with \(d(x_c)=1\),
\(d\to0\) at infinity, and no elastic/history driving force. It is not the
general solution of a loaded finite bar.

On the finite symmetric interval \([-A,A]\), with \(d(0)=1\) and natural
conditions \(d'(\pm A)=0\), the corresponding local profile is

$$
d(x)
=\frac{\cosh((A-|x|)/\ell_0)}{\cosh(A/\ell_0)}.
$$

The exponential is recovered as \(A/\ell_0\to\infty\). For the exponential
profile, the full width at \(d=e^{-1}\) is \(2\ell_0\); the full width at
\(d=e^{-2}\) is \(4\ell_0\).

The stationary-profile test must be kept separate from the loaded coupled-bar
test.

On the computational interval \([0,1]\), the stationary diagnostic fixes
\(x_c=1/2\), imposes \(d(x_c)=1\), and uses zero loading and zero history. Its
finite-domain local profile is

$$
d(x)
=\frac{\cosh((1/2-|x-1/2|)/N_\ell)}{\cosh(1/(2N_\ell))}.
$$

## 6. Non-Dimensionalisation

Use

$$
x^*=\frac{x}{L_0},
\qquad
u^*=\frac{u}{L_0},
\qquad
\varepsilon^*=\varepsilon,
\qquad
\sigma^*=\frac{\sigma}{E_0},
\qquad
\mathcal H^*=\frac{\mathcal H}{E_0},
$$

where \(E_0\) is the reference Young's modulus. Define

$$
N_\ell=\frac{\ell_0}{L_0},
\qquad
N_G=\frac{G_c}{E_0L_0},
\qquad
e(x^*)=\frac{E(x)}{E_0}.
$$

Dropping stars from spatial arguments, the condensed mechanics is

$$
\sigma^*
=\frac{\bar u^*}
{\displaystyle\int_0^1
\frac{dx}{e(x)g(d(x))}},
$$

$$
\varepsilon(x)
=\frac{\sigma^*}{e(x)g(d(x))},
$$

and

$$
\mathcal H_n^*(x)
=\max\left(
\mathcal H_{n-1}^*(x),
\frac12e(x)\langle\varepsilon_n(x)\rangle_+^2
\right).
$$

The local damage equation is

$$
d-N_\ell^2d''
=\frac{2N_\ell}{N_G}(1-d)\mathcal H^*.
$$

For \(E_0=210\times10^3\) N/mm2, \(G_c=2.7\) N/mm,
\(\ell_0=0.024\) mm, and \(L_0=1\) mm,

$$
N_\ell=0.024,
\qquad
N_G=1.286\times10^{-5},
$$

$$
\sqrt{\frac{N_G}{N_\ell}}\approx0.02315
\quad(d=1/2),
$$

while the zero-residual-stiffness homogeneous peak strain is

$$
\varepsilon_{\mathrm{peak}}^*
=\sqrt{\frac{N_G}{3N_\ell}}
\approx0.01336.
$$

## 7. Nonlocal Damage Formulation

The repository operator is

$$
(D_\delta d)_i
=\sum_{j:\,|x_j-x_i|\le\delta}
J_\delta(|x_j-x_i|)(d_i-d_j)\,\Delta x.
$$

With the implemented second-moment normalisation,

$$
D_\delta d\to-d''
$$

at smooth interior points as the horizon and discretisation are refined
consistently. The nonlocal damage equation is therefore

$$
\boxed{
d+N_\ell^2D_\delta d
=\frac{2N_\ell}{N_G}(1-d)\mathcal H^*
}.
$$

The graph is built once on the fixed material grid. Diffuse damage does not
delete edges or alter the graph topology.

For the current uniform-grid `SparseD`, let \(h\) be the particle volume used
inside the operator. The operator-consistent non-dimensional fracture energy is

$$
\mathcal E_{\mathrm{frac},h}^*
=\frac{N_G}{2N_\ell}
\left[
h\sum_i d_i^2
+N_\ell^2 h\,\mathbf d^T D_\delta\mathbf d
\right],
$$

with

$$
h\,\mathbf d^T D_\delta\mathbf d
=\frac{h^2}{2}\sum_{i,j}J_{ij}(d_i-d_j)^2\ge0.
$$

This diagnostic deliberately uses the same uniform particle volume as the
current operator, including at endpoint particles. The compliance integral and
reconstructed displacement use explicit trapezoidal quadrature instead. Do not
mix these weights silently; report both choices and verify convergence under
refinement.

### 6.1 Horizon and grid requirements

To approximate local AT2, the required scale hierarchy is

$$
h\ll\delta\ll\ell_0.
$$

Refining \(h\) at fixed \(\delta\) converges only to the finite-horizon model.
If \(\delta/\ell_0=O(1)\), \(\delta\) is a second model length and the resulting
profile, peak response, and effective fracture energy need not equal local AT2.

Every study must report:

- \(h/\delta\);
- \(\delta/\ell_0\);
- the average and minimum neighbour counts;
- full-domain and boundary-excluded errors;
- the discrete damage energy;
- the measured crack-profile width.

### 6.2 External boundaries

Horizon truncation preserves constants and approaches a natural no-flux
condition in the local variational limit. It is not pointwise identical to
\(d'=0\) at finite \(\delta\). Boundary points have one-sided horizons and may
exhibit a boundary layer.

The initial implementation must retain all damage rows, report errors both
inside and outside a boundary collar of width \(\delta\), and must not silently
mask endpoint residuals merely to improve an error metric.

## 8. Localisation Imperfection

A perfectly homogeneous bar has a homogeneous solution and will not reliably
choose the centre as a localisation point. Symmetry alone does not create a
weak point.

For the coupled localisation benchmark, use an explicitly configured small
modulus imperfection on the non-dimensional coordinate, for example

$$
e(x^*)=\frac{E(x^*)}{E_0}
=1-a_E
\exp\left(-\frac{(x^*-x_c^*)^2}{2(w_E^*)^2}\right),
$$

with all parameters declared in the config. This keeps \(G_c\) and the damage
regularisation uniform. Under displacement control, the softer region has a
larger strain and tensile energy and therefore provides a reproducible
localisation site.

`imperfection.center` and `imperfection.width` are non-dimensional coordinates
on \([0,1]\). Do not reinterpret them as dimensional lengths.

Do not emulate spatially varying toughness by dividing only the history term by
\(G_c(x)\). A true spatially varying fracture energy also changes the gradient
term and requires a separate variational derivation.

The homogeneous validation case must set the imperfection amplitude exactly to
zero.

## 9. Incremental Staggered Algorithm

The problem is path dependent and must be solved incrementally. A terminal-load
one-shot solve cannot recover the history field or the selected metastable
branch.

At load step \(n\), retain the fixed previous-step history
\(\mathcal H_{n-1}\). Then iterate:

```text
given d^(k)
    compute g(d^(k)) and e(x)
    compute sigma^(k) from the exact compliance integral
    compute epsilon^(k) and psi_plus^(k)
    set H_used^(k) = max(H_(n-1), psi_plus^(k))
    solve/train the damage equation with H_used^(k) fixed
    obtain d^(k+1)
    recompute sigma^(k+1), epsilon^(k+1), and psi_plus^(k+1)
    set H_next^(k+1) = max(H_(n-1), psi_plus^(k+1))
    test damage, stress, and history fixed-point convergence

after convergence
    d_n = d^(k+1)
    H_n = H_next^(k+1)
    advance the applied displacement
```

Warm-start the damage network from the previous staggered iteration and load
step. `train.gate_mode` makes the load-step policy explicit. In `strict` mode,
do not advance the load if the inner staggered solve has not converged. In
`warning` mode, retain the same checks but log a gate warning and commit the
last finite candidate after `max_stagger_iters`; this mode is for inspecting
the completed PINN field and reaction curve, not for claiming fixed-point
convergence.
The declared convergence tests must include

$$
\|d^{(k+1)}-d^{(k)}\|_\infty,
\qquad
|\sigma^{(k+1)}-\sigma^{(k)}|,
\qquad
\|\mathcal H_{\mathrm{next}}^{(k+1)}
-\mathcal H_{\mathrm{used}}^{(k)}\|_\infty.
$$

The final damage field must therefore have been solved with a history field
consistent, to tolerance, with the history field that is committed.

The outer displacement controller supports adaptive increments. The base FEM
configuration sets `loading.du_min = loading.du = loading.du_max =
loading.du_softening = 1e-5`, so its reference path uses a fixed increment. An
override may enable adaptive growth up to `loading.du_max`; once the accepted
reaction begins to decrease, the increment is capped at `loading.du_softening`
and never grows on the softening branch. In `strict` mode, if the inner
staggered solve fails, the trial state is discarded, the last converged `d_n`,
`H_n`, and displacement are restored, and the trial is retried with
`du <- max(loading.du_min, cutback_factor * du)`. In `warning` mode, no
cutback is triggered by tolerance failure: the finite candidate is committed
and all gate values are saved for later inspection. `output.save_every` controls
snapshots only; it is not the accepted loading increment. The base FEM
reference sets it equal to the accepted increment so every load state is
available for PINN comparison.

## 10. Reference Solver and PINN Roles

The two computational paths have different purposes:

| Path | Mechanics | Damage | Purpose |
|---|---|---|---|
| FEniCS solver | local weak-form displacement solve | local CG1 weak form | independent local AT2 reference |
| PINN | exact 1D condensed mechanics | `SparseD` strong residual | test the nonlocal damage operator |

The FEM solver must not use `SparseD`; otherwise the reference would share the
same operator error as the PINN.

### 9.1 Direct nonlocal baseline

Before training a network, solve the fixed-history nonlocal equation directly.
For a fixed history vector, the documented damage equation is linear:

$$
\left[I+N_\ell^2D_\delta
 +\frac{2N_\ell}{N_G}\operatorname{diag}(\mathcal H)\right]d
=\frac{2N_\ell}{N_G}\mathcal H.
$$

`nonlocal_baseline.py` assembles this sparse system from `SparseD` and solves
the same condensed mechanics/history staggered path without a neural network.
The selected baseline uses `cell_mass_lumped` history: tensile energy is first
computed from element strain on DG0 cells, then projected to nodal damage rows
by the standard CG1 lumped-mass rule. The former nodal-history path remains a
diagnostic comparison. Differences should be reported in this order:

1. local FEM versus direct nonlocal baseline: finite-horizon/model error;
2. direct nonlocal baseline versus PINN: optimisation/representation error;
3. PINN versus local FEM: combined error only after the first two are known.

The PINN and transition paths must use this same history state. They retain
`H_prev_cell` on the DG0 cells, update it from cell strain reconstructed from
the condensed displacement, and project the resulting cell history to nodal
damage rows. History convergence is checked in the cell space; applying a
nodal `max` to a projected previous history is a different discretization and
is not an accepted baseline replay.

The current pilot uses `Nx=1601`, `delta=0.006`, and fixed `du=1e-5`.
Since `delta/N_l=0.25`, it is a finite-horizon pilot, not an asymptotic local
AT2 limit. The `(h, delta)` refinement study remains a separate diagnostic.

`SparseD.to_scipy()` is used only by this direct baseline to assemble one
SciPy CSR matrix for the fixed-history linear solve. It is not called by the
PINN training loop. The PINN creates its fixed-shape JIT executables once;
compile logging has shown no load-dependent recompilation. The dominant cost
near localization is repeated neural optimization of difficult fixed-history
states, not JAX recompilation.

### 9.2 Frozen-history localization diagnostic

`frozen_pilot.py` solves the direct-baseline history field at selected load
snapshots without continuation. It supports independent networks, network
warm-start with optimizer reset, optimizer-state carry, and MAE/MSE residual
optimization. The pilot is diagnostic only: it does not replace the incremental
PINN acceptance gate or add a supervised term to production training.

`transition_pilot.py` can instead split each outer staggered history update
into explicit fixed-shape substeps,

$$
  H(\lambda)=H_{\rm prev}+\lambda(H_{\rm used}-H_{\rm prev}),
  \qquad \lambda=1/K,\ldots,1,
$$

where every substep must satisfy the active fixed-history residual gate before
the next one is attempted. This is a transition diagnostic, not a replacement
for the physical history update or the production continuation path.

The transition diagnostic also supports target-free transfer experiments:
`--warm-start-steps` minimizes the residual at `H_previous` rather than fitting
the previous-load damage field. `--staged-output-iterations` optionally performs
output-layer-only residual updates once per outer stagger, followed by full
network residual updates on all history substeps. `--adaptive-history` bisects a
failed history interval up to `--max-history-continuation-steps`. These options
are default-off diagnostics and do not alter production training.
`--residual-lr` explicitly controls the residual optimizer's Adam learning rate
when `--residual-optimizer adam` is selected. The configured base learning rate
remains the energy-phase rate for a hybrid diagnostic. With
`--residual-optimizer rprop`, use `--rprop-step-size`; its step bounds and
sign-change factors come from the explicit `train.rprop_*` configuration.
`--continuation-optimizer residual` continues with the selected residual
optimizer, while `--continuation-optimizer lbfgs` selects L-BFGS only for the
full-network history substeps. Warm start and output-only staging always use
the selected residual optimizer.

For a plain MLP with linear output, `frozen_pilot.py --output-head-lstsq`
freezes the trained trunk and directly solves the Jacobi-scaled residual
least-squares problem over the final output weights and bias. At fixed history,
both the damage equation and final linear output are affine, so this is an exact
neural-subspace diagnostic. It uses the history and residual only; direct
damage is not part of this coefficient solve. The mode is restricted to
residual MSE with zero hard-point weight and is not a production update.

The frozen pilot also computes a target-free fixed-history defect. For a network
field with raw residual `r`, it assembles the same direct linear system

$$
A(H)=I+N_l^2D_\delta+\frac{2N_l}{N_G}\operatorname{diag}(H)
$$

and solves

$$
A(H)e=r.
$$

Here `e` is the exact discrete field error relative to the fixed-history linear
solution, up to the sign convention `d_corrected = d - e`. The defect is saved
as a diagnostic only; it is never used as a supervised target or optimizer
update.

The pilot also exposes the default-off `--full-network-lm` refinement. It uses
all trainable network parameters and a damped matrix-free Gauss-Newton step. The
`--lm-residual-mode scaled_residual` mode uses the configured Jacobi-scaled
residual. The `--lm-residual-mode defect` mode uses a fixed-history Cholesky
factor to optimize the exact field defect. When LM is enabled, every `--lm-*`
control is required explicitly; there are no hidden production damping values.
The refinement is target-free and restricted to the plain linear-output frozen
diagnostic. It does not alter `train.py` or the production optimizer.

The former nodal-history baseline contained a one-node center spike, but it is
now diagnostic-only. The selected `cell_mass_lumped` baseline has a smoother
multi-layer localization profile while retaining a concentrated history peak.
The tested finite-capacity network can follow the smooth pre-localization branch
and partially track the transition, but has not yet matched the pointwise
localized equation at the configured residual tolerance. This distinction must
be resolved before accepting a full PINN continuation; do not silently smooth or
clip the direct baseline.

### 9.3 Fixed-history pseudo-time diagnostic

`pseudo_time_pilot.py` applies the implicit pseudo-time residual from
arXiv:2604.23528v1 while keeping the direct-baseline `H` fixed:

$$
q_\tau(d_\theta;d_{\rm old},H)
 = \frac{d_\theta-\operatorname{stopgrad}(d_{\rm old})}{\tau}
   + r_d(d_\theta;H).
$$

The previous output is updated after each optimizer step. The pilot also
solves the corresponding direct vector step,

$$
(I+\tau A_H)d^{k+1}=d^k+\tau b_H,
$$

so pseudo-time progress can be separated from neural representation error.
Fixed `tau` is the first diagnostic; the pilot also exposes the paper's
finite-difference inverse-Jacobian step estimate. Neither mode changes the
production history update or acceptance gate.

`frozen_pilot.py` also compares the production unconstrained `linear` output
with the diagnostic `exp_square` output,
`d=exp(-raw**2)`, across the configured `tanh`, `gelu`, `silu`, and `swish`
activations. For `exp_square`, `d_init` is a final-damage initialization and
the raw output bias is set to `sqrt(-log(d_init))`; this pilot mode is not
enabled in the production configuration.

The fixed-history pilot may also enable the explicit diagnostic branch
`arch.use_local_hat_enrichment=True` with an odd `arch.local_hat_count` (the
base value is `33`). It adds trainable CG1 hat corrections centered on the
configured imperfection node. The center must lie on the fixed grid and the
patch must remain inside the domain. The branch is default-off and is not a
second irreversibility or box constraint; production acceptance still uses the
same residual gates. With `--hat-update=exact`, supported only for the linear
output, the pilot freezes the hat gradients during Adam updates and solves the
small weighted least-squares coefficient problem at each residual check. This
is an alternating optimization diagnostic, not a change to the damage
equation.

The residual objective also exposes the diagnostic fields
`loss.hard_point_count`, `loss.hard_point_weight`, and
`loss.hard_point_mode`. When the weight is positive, the loss adds the mean
pointwise residual loss over the current top-k scaled-residual rows. `dynamic`
selects the rows inside every optimizer step; `fixed` refreshes the rows at
residual checks and keeps them fixed between checks. Both modes use only the
current residual and do not use reference damage or fixed crack coordinates.
The base weight is zero; this experiment is not enabled in production.
The `smooth` mode uses a bounded, stop-gradient residual score to form a
continuous row weighting; `loss.hard_point_temperature` controls its score
sharpness. Freezing the Fourier feature kernel is controlled independently by
`arch.train_spatial_embedding=False`; the base configuration leaves it
trainable.

The optional `arch.local_equilibrium=True` branch decomposes the damage as

$$
d_\theta=d_{\rm loc}+c_\theta,\qquad
d_{\rm loc}=\frac{\beta H}{1+\beta H},\qquad
\beta=\frac{2N_\ell}{N_G},
$$

and supplies bounded `d_loc` together with the previous damage as immutable
context to the correction network. This branch is a residual-only
fixed-history diagnostic, requires linear output and no supervised warm start,
and is default-off; it is not a production constraint.

### 9.4 Legacy 2D comparison warning

The apparent convergence of the legacy 2D fracture script is not a direct
convergence reference for this 1D solver. That script uses a pre-existing
Dirichlet crack, a loose global L2 tolerance (`tol=1e-3`), and updates the
history from inner-iteration fields rather than retaining the previous
converged history as an immutable input. Its results must therefore not be
used to claim that the corrected 1D staggered solver should converge under the
same conditions.

The PINN initially outputs only `d(x)`. It has no displacement PDE loss and no
displacement boundary loss because the condensed mechanical solution satisfies
both end displacements exactly.

For fixed \(\mathcal H\), define

$$
r_d
=d+N_\ell^2D_\delta d
-\frac{2N_\ell}{N_G}(1-d)\mathcal H.
$$

The first implementation should use the full fixed grid. Subgraph sampling is
unnecessary for this small one-dimensional problem and would add another error
source before the full operator is validated.

## 11. Training Methods, Tricks, and Current Results

### 11.1 Enabled methods

The following methods are enabled in the current best curve run:

| Method | Role | Why it is used |
|---|---|---|
| Exact condensed 1D mechanics | Removes displacement from the neural output | Eliminates displacement-network optimization error and computes reaction stress deterministically |
| `cell_mass_lumped` history | Computes tensile energy on DG0 cells and projects only for nodal damage rows | Matches the direct nonlocal reference and avoids the former nodal-history spike |
| Full fixed grid | Uses all `Nx=1601` points at every damage solve | Prevents localized history peaks from being missed by sampling |
| Fixed-shape JAX execution | Builds `SparseD`, grid arrays, and network shapes once | Avoids load-dependent recompilation |
| Float32 default with highest matmul precision | Numerical precision for routine runs; float64 remains an explicit reference option | Preserves the measured curve while reducing memory and compute cost |
| Width-200 MLP with swish and spatial scale 16 | Main representation capacity | Best controlled fixed-history representation found so far |
| 33 trainable CG1 hat functions | Local trainable correction around `x=0.5` | Resolves the localized center and nearby shoulders without changing the PDE |
| Residual MSE with Jacobi scaling | Fixed-history PINN objective | Preconditions rows with different history magnitudes |
| Adam with optimizer reset per fixed history | Inner damage optimization | Retains network weights as a warm start but avoids stale moments after history changes |
| Incremental displacement loading | Advances by fixed `du=1e-5` | Preserves the history-dependent loading path and metastable branch |
| Warning gate mode | Continues with a finite candidate after failed tolerance checks | Allows a complete engineering response curve to be inspected; all gate values remain recorded |
| TensorBoard field and curve logging | Writes figures every `train.eval_every` steps | Shows training progress rather than only the final result |

The current production network remains damage-only. Mechanics, strain,
displacement, tensile energy, and reaction stress are deterministic functions of
the predicted damage field.

### 11.2 Methods tested but not enabled

These methods are retained as diagnostics because they either performed worse,
did not solve the localized pointwise defect, or changed the controlled
formulation.

| Method | Observed limitation | Current decision |
|---|---|---|
| Width 64 with tanh and no hats | Larger localized residual than width 200 | Not used for the best curve |
| Width 400 | Harder warm start and worse residual at comparable budgets | Not used |
| `exp_square` damage output | Keeps damage in `(0,1]`, but saturates near `d=1` and worsens the center gradient | Diagnostic only; linear output remains active |
| Exact alternating hat update | Worse than joint Adam; scaled residual L-infinity was about `5.35e-5` after 60000 updates | Not used |
| Dynamic hard-point weighting | Tested weights destabilized Adam and increased pointwise residuals | Weight remains zero |
| RPROP | Passed one frozen fixed-history gate, but the corrected cell-history transition did not close the outer coupling | Not used for the current full curve |
| LM refinement | Reduced a fixed-history defect only from about `1.93e-5` to `1.90e-5` | Diagnostic only |
| Pseudo-time damage loss | PINN fixed-history residual remained far above the strict gate | Not enabled |
| Local-equilibrium output decomposition | Did not remove the localized representation bottleneck | Default-off diagnostic |
| Gaussian enrichment and L-BFGS continuation | Useful exploratory branches but no accepted improvement over the current trainable hats | Not used in production training |
| Post-processing hat least-squares correction | Reduced residual, but the corrected field was not the neural forward path | Not accepted as PINN training |
| Joint displacement/damage network | Adds a second optimization problem that exact 1D condensation already removes | Not justified for this example |
| Target damage supervision | Would hide the actual residual/representation problem | Forbidden in production training; references are evaluation-only |
| Clipping, box constraints, active sets, under-relaxation, and Anderson acceleration | Would alter the declared history-field damage formulation or hide healing | Not used; healing remains a warning diagnostic |
| Fully nodal history | Valid as a separate discretization, but it does not match the selected cell-history reference and produced a stronger grid-scale history peak | Retained only as a separate comparison model |

### 11.3 Gate policy

Two policies are explicit:

| Policy | Behavior |
|---|---|
| `train.gate_mode="strict"` | Failed fixed-point gates trigger rollback/cutback and can stop the run |
| `train.gate_mode="warning"` | Failed gates are logged and saved; the last finite candidate is committed so the load path can reach `u_final` |

The warning mode does not make an unconverged field mathematically converged.
It separates two questions: whether the PINN can reproduce the engineering
response, and whether it satisfies a strict pointwise fixed-history solve.
Non-finite values, invalid endpoint displacement, and non-constant reconstructed
stress remain hard errors in both modes.

### 11.4 Representative full-load result

The current best full run used:

```text
run:       best_cellhistory_full_width200_hat33_adam80k_warning_curve_tb
grid:      Nx=1601, delta=0.006, cubic SparseD
history:   cell_mass_lumped
loading:   du=1e-5, unique load steps=2000, u_final=0.02
network:   width=200, swish, spatial scale=16, linear output, 33 hats
optimizer: Adam, residual MSE, Jacobi scaling
```

The run produced one gate warning at `u_bar=0.01175`, where the localized
transition is most difficult. All other load steps passed the relaxed
curve-diagnostic gates. The result is an initial acceptable engineering result,
not a strict fixed-point certification.

PINN and direct nonlocal peak response:

| Quantity | PINN | Direct nonlocal |
|---|---:|---:|
| Peak `u_bar` | `0.01158` | `0.01156` |
| Peak reaction stress | `7.30735e-3` | `7.30642e-3` |
| Maximum reaction difference over the full path | `1.06e-5` | reference |
| Reaction MAE over the full path | `1.03e-7` | reference |

Representative relative damage L2 errors against the corrected direct baseline:

| `u_bar` | Relative damage L2 |
|---:|---:|
| `0.0050` | `1.19e-3` |
| `0.0100` | `3.22e-4` |
| `0.0116` | `1.26e-3` |
| `0.0118` | `1.78e-2` |
| `0.0140` | `1.73e-2` |
| `0.0200` | `1.25e-2` |

The early relative error is less informative because the reference damage is
very small. The corresponding absolute damage L2 error is about `3.34e-3` at
the final load and reaches about `4.70e-3` just after localization.

The result archive stores `d_l2_u_bar`, `d_l2_abs_direct_nonlocal`, and
`d_l2_rel_direct_nonlocal`. TensorBoard tags are:

```text
train/damage_field
train/stress_strain_curve
eval/direct_nonlocal/d_l2_abs
eval/direct_nonlocal/d_l2_rel
```

Representative static figures:

| Figure | Path |
|---|---|
| Damage fields | `/root/autodl-tmp/data/fracture1d/d_fields_representative_best_cellhistory.png` |
| Full stress-strain curve | `/root/autodl-tmp/data/fracture1d/stress_strain_full_best_cellhistory_warning_tb.png` |
| Damage L2 error | `/root/autodl-tmp/data/fracture1d/d_l2_error_best_cellhistory_warning_tb.png` |

The complete TensorBoard run, including events, checkpoints, configuration,
stdout log, and archive, is under:

```text
/root/autodl-tmp/tf-logs/fracture1d/best_cellhistory_full_width200_hat33_adam80k_warning_curve_tb/
```

### 11.5 Float32 and Fourier-embedding ablation

Two serial float32 runs used the same best architecture and corrected
cell-history baseline, with `u_final=0.015` for a controlled precision and
embedding comparison:

| Configuration | Peak `u_bar` | Maximum full-path reaction error | Post-transition maximum error | Final absolute damage L2 | Warnings |
|---|---:|---:|---:|---:|---:|
| float32, `spatial_emb_dim=32`, `spatial_scale=16` | `0.01156` | `1.03e-5` | `1.22e-7` | `6.83e-4` | `2` |
| float32, `spatial_emb_dim=64`, `spatial_scale=4` | `0.01155` | `7.15e-3` | `1.13e-7` | `3.14e-3` | `2` |

The `64/4` run followed the pre-localisation branch, but delayed the reaction
drop by one load step near `u_bar=0.01175`. Its small post-transition error is
therefore measured only after the delayed jump and does not remove the large
full-path error. The `32/16` run tracks the direct reaction curve to the same
precision as the float64 reference and is now the default configuration. The
float64 run remains the high-precision reference.

The spatial embedding dimension counts Fourier feature pairs, so increasing it
from `32` to `64` doubles the embedded input width. The scale changes the
frequency range available to the trainable spatial representation; it does not
change the SparseD operator or the fracture formulation.

The shared architecture also declares `temporal_emb_dim=8` and
`temporal_scale=1.0`, but they are inactive in this example. Fracture1d uses
`use_temporal_emb=False`, the static MLP path, and a dummy constant time input.
The physical load coordinate `u_bar` is handled by explicit incremental loading
and history updates, not by a temporal embedding. These fields therefore have
no effect on the current parameter count or results.

Float32 artifacts:

```text
/root/autodl-tmp/tf-logs/fracture1d/float32_best_spatial32_scale16_u0015/
/root/autodl-tmp/tf-logs/fracture1d/float32_best_spatial64_scale4_u0015/
/root/autodl-tmp/data/fracture1d/float32_spatial_embedding_comparison_u0015.png
```

## 12. Implementation Plan for Luna

This section is the implementation contract. Complete the stages in order and
do not start a later stage until the preceding verification passes.

### Stage 0: inspect and preserve project conventions

Read before editing:

- `AGENTS.md`;
- `src/pinn/operators.py`, especially `SparseD`;
- `examples/elasticity2d/train.py` and `losses.py` for factory and JIT APIs;
- `examples/fracture1d/solver.py` and `configs/solver_base.py`;
- `tests/test_elasticd.py` for regression-test style.

Required conventions:

- remain on branch `dev`;
- use explicit `ml_collections.ConfigDict` fields, never `.get()` defaults;
- keep fixed JAX shapes across all training steps;
- make `loss_fn` accept `x_points` explicitly;
- build `SparseD` once outside JIT;
- do not commit or push without current-turn user permission.

### Stage 1: add scalar operator regression tests

Create `tests/test_sparsed_fracture.py` before implementing training.

Required tests:

1. `SparseD @ ones` is zero to numerical tolerance.
2. Full and subgraph scalar matvec agree on core rows.
3. A smooth quadratic has the correct interior sign and magnitude,
   \(D_\delta x^2\approx-2\).
4. Report convergence under fixed \(\delta\), \(h\to0\).
5. Report coupled refinement with decreasing \(h\) and \(\delta\).
6. Verify the discrete quadratic form is non-negative.
7. Separate full-domain error from points farther than \(\delta\) from the
   endpoints.

Do not tune tolerances until a failing result looks green. Derive tolerances
from an observed refinement table and document them in the test.

### Stage 2: correct `configs/solver_base.py`

Keep every required field explicit. Organise fields as:

```text
c.mesh
c.physical
c.geometry
c.imperfection
c.loading
c.solver
c.output
```

Required physical/config fields include:

```text
physical.E
physical.Gc
physical.l0
physical.k_res
geometry.L0
imperfection.amplitude
imperfection.width
imperfection.center
loading.du
loading.du_min
loading.du_max
loading.du_softening
loading.cutback_factor
loading.growth_factor
loading.max_cutbacks
loading.u_final
solver.atol_u
solver.atol_d
solver.atol_H
solver.atol_sigma
solver.atol_stress_constancy
solver.linear_solver
solver.max_stagger_iters
output.log_every
output.save_every
output.dir
output.filename
output.plot_dir
```

Remove the current toughness-driving-force imperfection. Replace it with the
documented modulus field. Validate at startup that all physical scales are
positive, `0 <= amplitude < 1`, `k_res >= 0`, and the imperfection centre lies
inside the bar. Declare in config comments that imperfection centre and width
are non-dimensional coordinates on `[0, 1]`.

### Stage 3: correct the FEniCS reference solver

Update `solver.py` as an independent local reference.

Required changes:

1. Use the non-dimensional computational domain `[0, 1]` consistently in
   docstrings, mesh construction, plots, and analytical functions; retain
   `geometry.L0` only as the dimensional scale.
2. Construct the non-dimensional modulus field
   `e(x_star) = E(x_star) / E0`. Use `e(x_star) * g(d)` in the non-dimensional
   elasticity weak form and include `k_res` in `g(d)`. Do not insert
   dimensional `E` into this weak form.
3. Update the non-dimensional history with
   `H_star = maximum(H_prev_star, 0.5 * e * positive(eps)**2)`. Do not mix
   dimensional history values with the non-dimensional damage equation.
4. At every staggered iteration, form `H_used` from the fixed previous
   converged history `H_prev`, not the history of the preceding inner iterate.
   After the damage solve, recompute `H_next` from the updated mechanics.
 5. Solve the linear damage equation with `H_used` fixed and commit the raw
    solution as the next staggered damage iterate. Do not add under-relaxation,
    box constraints, active-set iterations, clipping, or Anderson acceleration.
  6. In `strict` gate mode, require `d`, reaction stress, and
     `H_next - H_used` to satisfy their configured fixed-point tolerances.
     In `warning` mode, record the same values and commit the finite candidate
     after the stagger limit if they do not pass.
  7. Fail loudly in `strict` gate mode if a load step reaches
     `max_stagger_iters` without convergence.
 8. Replace the incorrect AT2 critical-strain labels with homogeneous peak
    strain/stress labels.
 9. Keep the exponential/cosh profile comparison in a separate stationary
    profile diagnostic; do not label the final loaded damage field as an exact
    exponential solution.
 10. Reconstruct
    `sigma_star = e * ((1 - d)**2 + k_res) * epsilon`, verify its spatial
   constancy, and report the reaction per unit area as
   `reaction_star = F / (E0 * A) = sigma_star`. Do not retain the current
   unweighted MPI mean of `(1 - d)**2 * epsilon`.
 11. Save at least `x`, `x_H`, `u_bar`, `snap_u_bar`, `u`, `d`, `reaction_star`,
     and `H`. A compatibility alias named `force` is permitted only if its
    documented meaning is this reaction per unit area.
 12. Save enough metadata to reconstruct all non-dimensional parameters and
     the modulus imperfection.
  13. Use rollback/cutback loading in `strict` gate mode: if a trial displacement
      does not converge, do not commit `d`, `H`, or the load. Restore the previous
      converged state, reduce `du` by the configured `cutback_factor`, and retry
      until convergence or the explicit minimum increment/retry limit is reached.
      In `warning` mode, commit the finite candidate and record the gate warning.
      Record the accepted increment and cutback count for every committed load
      step.

Verification:

- with zero imperfection and sufficiently small load increments, reproduce the
  homogeneous damage relation before localisation;
- recover the zero-`k_res` analytical peak within discretisation tolerance;
- show load-step refinement of force-displacement and maximum damage;
- report damage growth and any excursion outside the physical range as
  diagnostics; do not enforce them as a second irreversibility formulation.

### Stage 4: create `configs/train_base.py`

Follow repository naming conventions. Required groups are:

```text
c.arch
c.physical
c.geometry
c.imperfection
c.nl
c.sample
c.train
c.loss
c.output
c.reference
c.baseline
```

Required nonlocal fields:

```text
nl.delta
nl.kernel
nl.method
```

Required sampling fields:

```text
sample.Nx
```

Required direct-baseline field:

```text
baseline.max_stagger_iters
```

Required architecture fields include `arch.damage_output`, `arch.d_init`,
`arch.use_local_hat_enrichment`, and `arch.local_hat_count` in addition to the
network shape, activation, and embedding fields used by the selected repository
architecture. For the production `linear` output,
`arch.d_init` is the direct initial damage value. The diagnostic
`exp_square` output interprets it as the final damage value and initializes the
raw bias with `sqrt(-log(arch.d_init))`.

The first implementation is full-grid only. Do not add an unused minibatch or
subgraph config.

The production damage network uses one scalar unconstrained output. Diagnostic
pilots may select `exp_square`, which maps a raw output `q` to
`exp(-q**2)` and therefore enforces `0 < d <= 1`; this is an output
parameterization experiment, not an additional production irreversibility or
history constraint. Report the damage range for both modes.

Required loading/training fields must explicitly separate:

- number of load steps or displacement increment;
- minimum/maximum adaptive displacement increments;
- cutback/growth factors and maximum retry count;
- maximum staggered iterations per load step;
- optimiser iterations per fixed-history damage solve;
- damage and mechanics convergence tolerances;
- fixed-history damage residual MAE and L-infinity tolerances;
- fixed-history residual norm (`mae` or diagnostic `mse`);
- fixed-history objective (`residual` or full-grid nonlocal `energy`);
- residual scaling (`none` or Jacobi row scaling);
- raw and scaled residual acceptance tolerances;
- gate policy (`strict` or `warning`);
- whether optimizer state is reset at each fixed-history solve;
- stress-constancy tolerance for the condensed reconstruction;
- learning-rate schedule;
- logging, evaluation, and checkpoint intervals;
- damage-healing diagnostic tolerance.

The reference group must also declare the expected `mesh_nx`, `history_space`,
and `history_method`. The base PINN path uses the same fixed `du=1e-5` as the
accepted FEM reference. The baseline group declares its larger direct-solver
stagger limit separately from the neural optimizer limit.

Do not copy PINN hyperparameters from elasticity without recording that they
are initial numerical choices rather than physical constants.

### Stage 5: create `losses.py`

Implement small pure functions:

```text
_net_d(net, x_points)
compute_mechanics(d, x_points, e_field, u_bar, quadrature_weights, k_res)
damage_residual(net, x_points, D, N_l, N_G, H_points)
loss_damage(...)
make_loss_fn(...)
```

`compute_mechanics` must:

1. evaluate `g(d)`;
2. compute the compliance integral using fixed trapezoidal weights on the
   sorted full grid;
3. reconstruct displacement by cumulative trapezoidal integration of strain,
   using the same endpoint rule as the compliance denominator, so that
   `u[0] = 0` and `u[-1] = u_bar` hold to numerical precision;
4. compute constant stress, strain, displacement, and tensile energy.

At setup, validate the modulus field, sorted grid, trapezoidal weights,
physical scales, and residual-stiffness configuration. During training and
evaluation, check compliance, stress, strain, displacement, and loss for
finiteness outside the JIT boundary and fail loudly on invalid values.

`loss_damage` must evaluate the network on the full operator grid and form the
documented nonlocal residual or its full-grid quadratic nonlocal energy. The
current best base configuration uses Jacobi-scaled residual MSE; pilot scripts
may select MAE or the full-grid energy. Raw and scaled residuals are both recorded. Keep
`H_used` fixed during each damage optimisation call.

`make_loss_fn` must follow the repository API:

```python
loss_fn(net, w_ema, bias, update_w, x_points=None, H_points=None)
```

There is initially one trained loss term, so do not introduce GradNorm. Return
the same auxiliary tuple structure expected by the shared training pattern.
`H_points` is a fixed-shape traced JAX array because its values change between
staggered iterations. Do not capture a new history array in a newly constructed
loss closure at every iteration; that would create avoidable JIT recompilation.
Every call must pass the complete operator grid in its construction order, and
`H_points` must have the same length and ordering. Validate these invariants
outside JIT. `H_points` is required and must never silently default to zero.
The `x_points=None` default exists only for repository API compatibility; the
training code must always pass the full grid explicitly.

Add unit tests for:

- homogeneous damage residual;
- exact displacement endpoints after numerical integration;
- constant stress throughout the reconstructed bar;
- no compressive history growth;
- finite gradients through the compliance calculation;
- JIT execution with fixed shapes.

### Stage 6: create `train.py`

Use helpers:

```text
_build_grid
_build_network
_build_D
_build_optimizer
_make_step
_eval_reference
```

`_make_step` must forward both `x_points` and `H_points` as dynamic JAX array
arguments to the loss function.

Setup sequence:

1. load and validate all config fields;
2. build the fixed 1D grid and explicit quadrature weights;
3. build the modulus field;
4. build the damage-only network;
5. build `SparseD` once;
6. load the FEM reference file and fail with the expected path if absent;
7. initialise `d`, `H_prev`, optimiser state, and output directories.

Before the full PINN continuation, run `nonlocal_baseline.py` with the same
grid, horizon, material, and fixed load increment. The direct baseline must
converge first; otherwise a PINN run cannot be diagnosed cleanly.

Training sequence for every load step:

1. retain immutable `H_prev` from the previous converged step;
2. compute mechanics from the current damage prediction;
3. form `H_used = maximum(H_prev, psi_plus)`;
4. pass the fixed-shape `H_used` array into the compiled damage step and
   optimise the damage network;
5. recompute mechanics and `H_next`, then check stress, history, the maximum
   damage change, and both raw and Jacobi-scaled fixed-history residuals;
6. repeat until mechanics, history, damage-change, and configured scaled
   residual tolerances pass. The final residual gate must be evaluated again
   with the projected `H_next` that will be committed, not only with the
   `H_used` that was passed to the preceding damage solve;
7. in `strict` mode, if the staggered iteration limit is reached, restore the
   committed network, optimiser state, damage, and history, reduce `du`, and
   retry the same load;
8. in `warning` mode, if the staggered iteration limit is reached, record the
   gate warning and commit the last finite candidate;
9. in `strict` mode, fail if the configured minimum `du` or cutback limit is
   reached;
10. record `min(d_n-d_prev)` as a healing diagnostic without clipping or
    rejecting the step as a second irreversibility constraint;
11. commit the selected `H_n` and advance the load;
12. save checkpoint, fields, reaction stress, energy, gate warnings, and error
     metrics.

At every `train.eval_every` load steps, `train.py` also writes TensorBoard
figures under `train/damage_field` and `train/stress_strain_curve`. The damage
figure compares the current PINN field with the load-interpolated direct
nonlocal field and reports the relative damage L2 error. The curve figure
shows the complete direct reference path and only the PINN path reached so far.
The corresponding scalar histories are `eval/direct_nonlocal/d_l2_abs` and
`eval/direct_nonlocal/d_l2_rel`.

The JIT-compiled damage step must receive changing scalar flags and values as
JAX arrays. Grid, operator edge arrays, and collocation shapes must remain
fixed across all calls.

### Stage 7: diagnostics and acceptance criteria

Produce separate plots/tables for:

1. operator convergence on manufactured smooth fields;
2. finite-domain stationary crack profile;
3. homogeneous analytical damage and stress response;
4. local FEM versus direct nonlocal baseline force-displacement;
5. direct nonlocal baseline versus PINN force-displacement;
6. damage profiles at common load coordinates;
7. history profiles compared through `x_H`/nodal interpolation;
8. raw and scaled damage residual MAE and L-infinity norms;
9. healing diagnostics and damage range;
10. \(h/\delta\) and \(\delta/\ell_0\) sweeps;
11. full-domain versus boundary-excluded damage error;
12. local and nonlocal damage energy.

The implementation is accepted only when:

- all operator and loss unit tests pass;
- the direct SparseD baseline converges at fixed history and along the full
  configured load path;
- Python compilation succeeds;
- full training steps execute under JIT without shape retracing caused by
  changing collocation sizes;
- the homogeneous benchmark uses the correct AT2 peak formulas;
- the local FEM reference converges under load-step refinement;
- the nonlocal result shows a documented trend toward local AT2 as
  \(h,\delta\to0\) with \(h/\delta\to0\);
- in `strict` mode, no load step advances without staggered convergence;
- in `strict` mode, every accepted PINN fixed-history solve satisfies the
  configured damage-change and active residual tolerances; raw residuals remain
  mandatory diagnostics;
- in `warning` mode, every committed load step carries an explicit
  `gate_warning` flag when those tolerances are not met;
- no healing is hidden by clipping or exception handling.

## 13. Project Structure

For the current verified results, unresolved numerical issues, and next steps,
see [`STATUS.md`](STATUS.md).

```text
examples/fracture1d/
├── README.md
├── STATUS.md                 # verified progress and open numerical issues
├── train.py                  # damage-only nonlocal PINN, incremental loading
├── losses.py                 # condensed mechanics and damage residual
├── frozen_pilot.py           # fixed-history localization diagnostics
├── transition_pilot.py       # one-load staggered transition replay
├── plot_stress_strain.py     # reference/baseline/PINN curve comparison
├── nonlocal_baseline.py      # direct SparseD fixed-history baseline
├── solver.py                 # independent local FEniCS reference
└── configs/
    ├── train_base.py
    └── solver_base.py
```

The current implementation provides `solver.py`, `configs/solver_base.py`,
`train.py`, `losses.py`, `frozen_pilot.py`, `transition_pilot.py`,
`plot_stress_strain.py`,
`nonlocal_baseline.py`, and
`configs/train_base.py`. The regression tests in
`tests/test_sparsed_fracture.py`, `tests/test_fracture_losses.py`, and
`tests/test_nonlocal_baseline.py` cover the operator, condensed mechanics,
fixed-history loss, direct baseline, and fixed-shape JIT path.
The FEM solver also writes `analytical_comparison.png`, `stress_strain.png`,
and `stationary_profile.png` alongside its configured result archive.
The fine-grid local reference completes the full loading path without
cutbacks; further mesh and diagnostic sweeps are optional refinement studies,
not blockers for the current PINN comparison.

## 14. References

- Miehe, C., Hofacker, M., and Welschinger, F. (2010). A phase field
  model for rate-independent crack propagation: Robust algorithmic
  implementation based on operator splits. *Computer Methods in Applied
  Mechanics and Engineering*, 199, 2765-2778.
- Miehe, C., Welschinger, F., and Hofacker, M. (2010). Thermodynamically
  consistent phase-field models of fracture: Variational principles and
  multi-field FE implementations. *International Journal for Numerical Methods
  in Engineering*, 83, 1273-1311.
- Pham, K., Amor, H., Marigo, J.-J., and Maurini, C. (2011). Gradient damage
  models and their use to approximate brittle fracture. *International Journal
  of Damage Mechanics*, 20, 618-652.
- Du, Q., Gunzburger, M., Lehoucq, R. B., and Zhou, K. (2012). Analysis and
  approximation of nonlocal diffusion problems with volume constraints. *SIAM
  Review*, 54, 667-696.
