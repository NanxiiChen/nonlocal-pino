# Confined Cahn–Hilliard phase separation

Elliptical container with two impermeable circular holes.
All walls have neutral surface energy and zero mass flux. Standard mixed CH is
phi_t=M Delta mu, mu=beta(phi^3-phi)-epsilon^2 Delta phi, with homogeneous
normal derivatives of phi and mu in the local model. No fluid/mechanical solve.

Canonical entry `train.py` defaults to `configs/train_coarsening_windows.py`:
the ten-wave initial field, epsilon0.02, terminal time1.5 and30 logarithmic
time windows. Total training is SOAP5000 plus L-BFGS500, initial learning rate
0.001, ordinary FP64. It predicts phase, eliminates chemical potential exactly,
and preserves physical-volume mass. With the validated reference available:

    python examples/confined_ch2d/train.py

An empty run name is filled with a timestamp and `nonlocal-windows`; an explicit
`--config.log.run_name=my-run` is honored. `--seed` records runtime provenance,
not a strategy parameter. The reference is
`/root/autodl-tmp/data/confined_ch2d/coarsening_multimode10/reference.npz`.
See [default window formulation and validation](#default-strategy-sequential-pinn-windows).
Chemical-potential accuracy remains unresolved despite the better phase result.

For the random-geometry **neural operator**, `configs/train_operator.py` now
selects the accepted width256/online/cold50k SOTA. Its fixed10mode/T3problem is
separate from the single-IVP windowed case above. See the
[operator SOTA recipe and checkpoint](#geometry-operator-sota).

`configs/train_base.py` remains the explicit legacy simple-pilot foundation,
not the current CLI default. Its physical/reference/checkpoint records and
all global-time research presets remain available without changing their inputs.

The nonlocal arm uses the positive quadratic regional pair kernel, physical
triangle-centroid areas and symmetric segment visibility around holes. It
expresses native closed no-exchange dynamics without a separate Neumann loss.
Finite-horizon fields need not have zero pointwise local AD normal derivative.
Local AD uses both scalar Hessians plus two explicit Neumann losses. Both share
the analytic phase IC, weighted mass constraint and direct output map. Legacy pilot configs add no mass loss or residual transform; optional global-time
residual metrics are described below.

To regenerate the default complex-case reference, run independent mixed-P2
reference checks in fenicsenv before JAX training:

    python -m examples.confined_ch2d.solver --preflight
    python -m examples.confined_ch2d.solver

Do not run FEniCS concurrently with JAX GPU training. Explicit artifact paths
are under /root/autodl-tmp/data/confined_ch2d and tf-logs/confined_ch2d.

References: [DOLFIN CH mixed theta-method demo](https://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/demos/cahn-hilliard/python/demo_cahn-hilliard.py.html),
[complex porous-domain CH](https://doi.org/10.1016/j.commatsci.2017.08.025),
[regional Neumann models](https://doi.org/10.1186/s13662-025-03923-x).

## Formulation and boundary scope

The dimensionless local equations and boundary data are

$$
\phi_t=M\Delta\mu,\qquad
\mu=\beta(\phi^3-\phi)-\epsilon^2\Delta\phi,
\qquad \partial_n\phi=\partial_n\mu=0.
$$

The CAD outer ellipse has center(0.5,0.5), axes(0.5,0.4). Circular holes are
(cx,cy,r)=(0.35,0.54,0.075),(0.68,0.43,0.055). Default scales are
beta1,epsilon0.02,M0.02,T1.5; the explicit legacy simple pilot uses epsilon0.05,
T1. These are dimensionless test choices.
The phase energy has no wall term: walls are neutral, with no preferential
wetting. The closed container exchanges no mass with its surroundings.

For triangle centroid points with cell areas m_i, the positive pair operator is

$$
(A_\delta v)_i=\sum_j m_j K_\delta(x_i,x_j)(v_i-v_j),\qquad
K_\delta(x,y)=\frac{120}{\pi\delta^4}
\left(1-\frac{|x-y|}{\delta}\right)_+^2\chi_{[x,y]\subset\overline\Omega}.
$$

Segment visibility is symmetric. No pair passes through an impermeable hole;
the convex ellipse needs no additional segment clipping. The full-ball second
moment of the radial part is2I in2D. There is no individual row renormalization.
Natural finite-horizon truncation differs from pointwise local Neumann at
finite delta; local-limit/geometry/quadrature error must be measured separately.
Using triangle-centroid areas also avoids the bounding-box area/N estimate.

The native mixed dynamics is

$$
\dot\phi=-M A_\delta\mu,\qquad
\mu=\beta(\phi^3-\phi)+\epsilon^2 A_\delta\phi.
$$

With V=diag(m_i), L=VA is symmetric positive semidefinite and L1=0.
Consequently its exact ODE conserves physical weighted mass and dissipates
sum_i m_i F(phi_i)+(epsilon^2/2)phi^T L phi. A finite PINN residual does not
alone enforce exact conservation, so both arms share weighted phase projection.
No extra conservation loss is used. Scalar normal-derivative penalties belong
only to local AD; native nonlocal AD normals are diagnostic of the local limit.

## Legacy simple-pilot initial condition and training

The initial phase is a deterministic smooth mixture, flattened to fourth order
at each wall to avoid an initial-boundary incompatibility. Define normalized
coordinates z=(x-center)/axes, outer-wall function
s_0=(min(axes)/2)(1-|z|^2), hole functions
s_k=(|x-c_k|^2-r_k^2)/(2r_k), and H(s)=s^4/(s^4+w^4). Then

$$
\phi_0(x)=a\prod_{k\geq0}H(s_k(x))
\left[\cos(2\pi z_x)+0.6\sin(1.5\pi z_y)\right],\quad a=0.3,\ w=0.04.
$$

Both phi_0 and its local chemical potential have zero normal derivative on the
analytic walls. Only phi_0 is hard constrained in the PINN; future boundary
values and interface motion are not encoded. The nonperiodic network includes
raw coordinates plus fixed sine/cosine features and has no periodic wrap.

Training uses original mixed residuals, direct additive phase output, weighted
hard mass,FP64 ordinary compilation and SOAP5000 at initial learning rate1e-3,
then PDE-only L-BFGS500. Eight stratified times plus t0 enter each residual
assessment. Local AD additionally penalizes both normal derivatives on all
three boundary components; weights and arc-length quadrature are explicit.
The reference selects/logs validation metrics but supplies no training labels.

For this legacy simple case, explicitly select its configuration for either arm
(its reference can be regenerated with solver `--config examples.confined_ch2d.configs.train_base`):

    XLA_FLAGS='' python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_base.py --seed=42 --config.log.run_name=my-confined-nonlocal
    XLA_FLAGS='' python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_base.py --seed=42 --config.nl.method=local_ad --config.log.run_name=my-confined-local-ad

Seed is run provenance, not a strategy-config field. Every run saves its direct
config, checkpoints and source snapshot; model replay uses
that saved config and the explicit reference path. A result is not transferable
to a different geometry or finite-horizon target without validation.

Reference time stepping uses monolithic implicit BDF2 with a backward-Euler
first step; no unconditional energy-stability theorem is claimed for this
unstabilized scheme. Actual energy decay and time refinement are checked.
The initial continuum chemical potential uses an independent analytic NumPy
Laplacian, verified against automatic derivatives, rather than mass inversion
of an interpolated field. A CN startup reference failed its chemical-potential
refinement gate and is retained explicitly as failed evidence.
The monolithic mixed-BDF construction is also described in the
[lifex CH tutorial](https://lifex.gitlab.io/lifex-public/classlifex_1_1tutorials_1_1Tutorial07__AD.html).

## Legacy simple-pilot reference validation

The fixed quadrature has2369 triangle centroids, with areas summing to
0.6014558924; delta=0.07. Local reference uses quadratic mixed FEM on nominal
mesh sizes0.025/0.0175 (9768/19202 mixed DOFs), BDF2 steps0.005/0.0025,
and21 saved times. Predeclared maximum-time relative L2 gates are1% for phi
and4% for mu. Both field gates, mass and energy checks pass:

| Check | Maximum phi difference | Maximum mu difference |
|---|---:|---:|
| Local mesh refinement | 0.5810% | 0.2051% |
| Local time refinement | 0.03769% | 0.1449% |
| Native graph time refinement | 2.82e-6% | 1.64e-5% |

The largest local phase mesh difference is at t0 from FEM interpolation of
the same analytic IC. Native/local integral mass drifts are below5.38e-16 and
saved reference energies decrease. The local FEM mean sampled by coarse
centroid quadrature can have a small separate offset/drift; it is not the
exact FEM mass integral and is not used as the PINN's hard prescribed mean.

Native graph reference is independently assembled in SciPy and solved by BDF,
with rtol1e-7/1e-9 and the two maximum steps above. It imports no training/JAX
operators. The local/common target is independently solved FEM, not a second
implementation of the training graph. Pooled native/local phi gap is4.2077%
and mu gap32.3404% before training. This includes finite-horizon, geometry and
quadrature effects; it is not a lower bound on PINN error.

An independent manufactured-IC integral check at2369/4696 triangle points
finds1.06% quadrature change at fixed delta0.07. Decreasing delta0.09/0.07/0.05
reduces the fine-integral vs analytic-Laplacian IC discrepancy68.1%/58.3%/44.6%,
while coarse/fine quadrature difference grows0.56%/1.06%/3.04%. The initial wall
layer is sharp relative to delta. This is an operator consistency diagnostic,
not proof of full evolution convergence or pointwise Neumann satisfaction.
Joint mesh/horizon refinement is necessary for stronger continuum claims.

Direct CPU checkpoint replay, after a completed run:

    JAX_PLATFORMS=cpu python examples/confined_ch2d/evaluate.py --run=/absolute/run/path --checkpoint=/absolute/run/path/model_final.eqx

It validates saved fields and plots five times with holes explicitly retained.

## First complete matched comparison

Fresh seed42 runs share initialized parameters,18 frozen sources and all
settings except nl.method/run name. Both finish SOAP5000+L-BFGS500. Final
update5500 uses the same refined **local FEM reference**:

| Metric | Nonlocal | Local AD |
|---|---:|---:|
| Pooled phi relative L2 | 4.1935% | 5.6639% |
| Terminal phi relative L2 | 4.9998% | 3.8354% |
| Boundary-band pooled phi L2 | 5.2634% | 6.7320% |
| Trainer wall time | 146.47s | 529.03s |
| Mean compiled SOAP update | 14.78ms | 65.41ms |

Wall time includes compilation/evaluation/refinement, excludes reference,
operator setup and CPU replay. On this single serial-GPU pair, nonlocal is
3.61x faster with lower pooled error; local AD has lower terminal error.
No cross-seed ordering is established. Nonlocal own-target pooled/terminal
phi is0.8898%/0.6469%; do not mix that target with the common-local comparison.
Best common-local checkpoints are updates750/5500; the table deliberately uses
equal completed budgets. Direct final CPU replays agree to2.45e-15; weighted
mass and sampled native energy diagnostics pass.

**Native no-exchange does not imply pointwise local Neumann for the MLP.**
The measured boundary AD normals are appreciably nonzero for nonlocal, much
smaller for the explicitly constrained local control. Finite-horizon effects
and MLP boundary extension have not been separated. The demonstrated capability
is removing separate normal-derivative losses for the native regional model;
exact classical Neumann at finite horizon remains unverified. Weak FEM can
also incorporate homogeneous Neumann naturally.

Chemical potential remains weaker than phase: common pooled mu L2 is41.6%/
46.1%, falling to6.71%/10.11% when t0 is excluded. Initial mu fitting dominates
the fixed final loss audit. This prevents attributing the full convergence
speed difference solely to boundary penalties or claiming all-field accuracy.

The two runs live under /root/autodl-tmp/tf-logs/confined_ch2d as
20261027-confined-nonlocal-01 and20261027-confined-local-ad-01. The preset is a
validated first pilot, not a global-SOTA selection. Detailed per-wall normals,
best/final metrics, initial residual allocation, references and limitations
belong to the [complete record](../../docs/tasks/archive/confined-ch-neumann.md).

[Common physical fields](../../docs/tasks/archive/confined-ch-common-fields.png),
[convergence/time/energy](../../docs/tasks/archive/confined-ch-comparison.png),
[reference gap](../../docs/tasks/archive/confined-ch-reference-gap.png),
[full audit](../../docs/tasks/archive/confined-ch-analysis.json).

## Sharper multimode case

`configs/train_coarsening.py` defines the physical case used by the default
window strategy and its explicit global-time controls, retaining the
same ellipse, holes, CH equations and boundary conditions. It uses epsilon0.02,
terminal dimensionless time1.5, amplitude0.98 and ten prescribed spatial waves.
With the wall envelope E defined above, its initial field is

$$
\phi_0(x)=0.98\,E(x)\tanh\!\left(\frac{1}{0.55}
\sum_{k=1}^{10}a_k\cos(\pi\mathbf k_k\cdot z+\theta_k)\right).
$$

All wavevectors, amplitudes and phases are explicit in the config; no random
seed selects the initial structure. The wall width is0.03. The saturation is
part of the physical IC only: learned phase output and residuals remain direct
and original. Fourth-order wall vanishing retains initial phi/mu Neumann
compatibility. The equilibrium planar -0.8..0.8 transition width is about0.062
versus0.155 in the first pilot.
For either IC, each wall has a smooth defining function s with nonzero normal
gradient. Thus phi_0=O(s^4), Delta phi_0=O(s^2), and the normal derivative of
the initial local chemical potential is O(s); both initial Neumann conditions
hold analytically. This does not constrain future interface positions.

Run the independent morphology preflight and then the full reference checks:

    python -m examples.confined_ch2d.solver --config=examples.confined_ch2d.configs.train_coarsening --preflight
    python -m examples.confined_ch2d.solver --config=examples.confined_ch2d.configs.train_coarsening

The new data path is
`/root/autodl-tmp/data/confined_ch2d/coarsening_multimode10`.
Training quadrature uses mesh size0.02 and delta0.03. Independent local reference
mesh sizes are0.014/0.01 with time steps0.005/0.0025 and21 snapshots. The first
0.02/0.014 mesh check failed; it is retained rather than used as a training
target. The same1% phase/4% chemical maximum-time thresholds remain in force.
The legacy simple preset and its data/checkpoints remain available. Use this preset
through the canonical train.py entry only after reference gates pass:

    XLA_FLAGS='' python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening.py --seed=42 --config.log.run_name=my-coarsening-nonlocal

Add `--config.nl.method=local_ad` for the local AD control. Same original
two-PDE loss, hard IC/mass, FP64 ordinary compilation and5000+500 budget apply.
This is a richer physical test, not an accuracy improvement or SOTA claim.
The refined reference passes the same gates: maximum-time mesh differences
are0.552% for phi/3.014% for mu; time differences0.048%/0.180%. Integral mass
drift is below1.1e-15 and all saved energies decrease. The pure-phase area
fraction |phi|>0.9 is27.17% initially and46.14% at1.5; significant sign regions
(area>=0.5%) change12 ->9. Counts remain mesh/threshold sensitive. These fields
are FEM references; the matched PINN attempt below does not reproduce them well.
At delta0.03 the independent initial-field quadrature diagnostic changes5.66%
on finer integration; finite-horizon/local effects remain, as in the pilot.

Reference validation and morphology evidence are in
[the coarsening record](../../docs/tasks/archive/confined-ch-coarsening.md).
[Physical fields](../../docs/tasks/archive/confined-ch-coarsening10-selected-fields.png)
show times0,0.15,0.45,0.9,1.5 with a fixed color scale.

### Matched richer-case training attempt

Both seed42 cold starts complete SOAP5000+L-BFGS500 with identical initialized
parameters and19 source files. The shared target is the refined local FEM;
final-budget results are:

| Metric | Nonlocal | Local AD |
|---|---:|---:|
| Pooled phi relative L2 | 82.96% | 82.95% |
| Terminal phi relative L2 | 94.55% | 99.41% |
| Positive-time mu relative L2 | 119.09% | 123.25% |
| Trainer wall time | 174.65s | 696.63s |
| Mean compiled SOAP update | 19.55ms | 93.48ms |

**This budget does not learn the richer case successfully.** Both best
validation checkpoints are untrained step0 (pooled phi72.31%), so they are not
successful trained models. Nonlocal own-native pooled phi83.02% remains poor;
the pretraining3.779% native/local target gap cannot explain the fitting failure.
The harder physical case now trains with windows by default; the original
simple preset remains explicitly selectable.

The predicted terminal pure fractions are11.26%/8.38% vs reference46.14%.
Fields change substantially from the IC, but have incorrect morphology and
weak phase contrast. Initial chemical loss contributes52.66%/39.46% of the
fixed17-time weighted objective; positive-time chemical errors remain large,
so this is not solely an initial-time residual issue. Network capacity,
space-time lifting and optimizer/temporal sampling effects are not separated.
Do not infer a unique cause or benefit from longer training from this pair.

Both final and best CPU checkpoint replays agree within1.54e-14. Hard mass
drift stays below3.5e-16, but sampled native PINN energies have positive
increments7.30e-4/3.38e-4; artifact replay success is not physical accuracy.
Classical-normal diagnostics retain the finite-horizon limitation above.

[Phase L2 curves](../../docs/tasks/archive/confined-ch-coarsening10-l2-convergence.png),
[common physical fields](../../docs/tasks/archive/confined-ch-coarsening10-common-fields.png),
[full audit](../../docs/tasks/archive/confined-ch-coarsening10-analysis.json) and
[training record](../../docs/tasks/archive/confined-ch-coarsening-training.md)
retain the failed attempt. Runs are20261027-coarsening10-nonlocal-01 and
20261027-coarsening10-local-ad-01 under the existing confined_ch2d log root.

### Default strategy: sequential PINN windows

`configs/train_coarsening_windows.py` is the CLI default. The original
simple-case preset and the unsuccessful global-time controls remain available.
The physics, IC, geometry, native operator and refined FEM reference are unchanged.
The user selected time windows again after the global-time and intermediate-
difficulty studies. Global controls below remain available as research records.

    XLA_FLAGS='' python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening_windows.py --seed=42 --config.log.run_name=my-coarsening-windows

The initial PDE-derived evolution scale is about0.0068, whereas the old uniform
time stratum has width0.1875. At the failed final model, the two PDE gradient
directions have cosine-0.826, with comparable norms. Supervised diagnostics show
the same original network can fit native phase to4.24% (diagnostic only, not a
PINN result). These observations support a temporal optimization/representation
problem; they do not establish a unique cause. Exponential lifting alone fails,
and several global-time sampling/conditioning controls still leave45–55% error.

The recommended strategy trains30 consecutive log-spaced windows. Each starts
from the **previous PINN endpoint**, frozen as a quadrature field; no reference
values enter the objective or supply window initial states. For a window of
width \(\Delta t\), local time \(s=(t-t_j)/\Delta t\), and positive regional
operator \(A=-D_\delta\), use

$$
\phi_\theta=\Pi_m[\phi_j+sN_\theta(x,s)],\qquad
\mu_\theta=\beta(\phi_\theta^3-\phi_\theta)+\epsilon^2 A\phi_\theta,
$$

$$
r_s=\partial_s\phi_\theta+\Delta t M A\mu_\theta.
$$

Here \(\Pi_m\) projects to the fixed physical-volume mean. Initial state and
mass are hard constraints; the chemical equation is eliminated exactly, leaving
one evolution residual and no redundant chemical loss. Its zero set is unchanged.
For physical volumes \(V\), put \(S=V^{1/2}AV^{-1/2}\) and use the frozen metric

$$
P_{\Delta t}=V^{-1/2}
\left(I+\Delta t M\epsilon^2S^TS\right)^{-1}V^{1/2},\qquad
\mathcal L=\langle |P_{\Delta t}r_s|^2\rangle_{V,s}.
$$

The matrix is SPD; Cholesky factors are precomputed without numerical jitter.
This is residual conditioning, not an inverse-tanh map or a trajectory prior.
Time samples are uniform within each window. Boundaries are
\(t_j=0.005[\exp(j\log(1+1.5/0.005)/30)-1]\).
The64x4 GELU network uses8 axial spatial modes and time embedding dimension8.
Keep FP64, ordinary compilation, SOAP initial learning rate0.001 and the existing
schedule. Weights warm start, but optimizer state and learning-rate schedule reset
at each window:650+29x150=5000 SOAP updates and36+29x16=500 L-BFGS iterations.
All final endpoints propagate; reference L2 is monitoring only.

**Switching currently follows a fixed budget, not a validated accuracy gate.**
The first window takes650 SOAP updates and at most36 L-BFGS iterations;
each later window takes150 and at most16. L-BFGS can stop early using its
gradient/function tolerances or evaluation/line-search limits. Both recorded
30-window runs reached the iteration cap in every window. The trainer records
the optimizer status and proceeds; it does not require optimizer success, a
prescribed PDE residual, endpoint stability or reference L2 before advancing.
Finite-value checks reject nonfinite objectives/gradients, but are not accuracy
certificates. No residual-driven time-boundary adaptation or failed-gate retry
is implemented. Also, width-scaled/preconditioned losses differ between windows;
a common raw-loss threshold would not automatically mean a common field error.

The next window uses the previous final predicted phase at every quadrature
point as its frozen initial field. The increment vanishes at the left endpoint,
so field continuity is hard constrained; the same chemical law/operator gives
continuous mu too. Weights warm start, optimizer state resets, and old windows
are not jointly corrected later. Hence prior errors can propagate and be
amplified or damped by the dynamics, in addition to each window's own fitting
error. Hard mean conservation only prevents mean drift; it does not fix interface
or curvature errors. The full-trajectory/native terminal errors already include
this propagation, but are not a guarantee for longer horizons. Independent-time
residual checks and extra-training endpoint stability would be candidates for
a future acceptance rule, not features of the present implementation.

| Common refined FEM target | Previous global seed42 | Windows seed42 | Windows seed2026 |
|---|---:|---:|---:|
| Pooled phase relative L2 | 82.96% | 3.908% | 3.866% |
| Terminal phase relative L2 | 94.55% | 5.754% | 5.672% |
| Native phase pooled / terminal | 83.02% / 94.63% | 0.724% / 0.809% | 0.678% / 0.748% |
| Trainer wall time | 174.65s | 281.25s | 288.23s |

Both cold starts have identical code/configuration except seed and run name.
This is evidence from two seeds, not a broad stability guarantee. A same-budget
15-window control is slightly worse: common phase4.008%/5.936%, native chemical
pooled31.35% vs31.01% for30 windows. In that historical study,30 windows outperformed15.
The bundle changes trial functions, spatial modes, loss metric and optimizer
resets together; it does not isolate the gain to one component. Thirty stored
networks also mean more total parameter capacity than one global network,
despite the same total update counts. The new scheme has no matched local-AD
window control; earlier timing comparisons do not measure that comparison.

**Chemical potential remains inaccurate.** Native pooled mu errors are31.01%
and29.99%; excluding the exactly satisfied initial field they are52.98% and51.23%.
Terminal errors are102.11% and97.87%, with seed42 error RMS0.04474 vs reference
RMS0.04381. Small phase L2 and exact chemical algebra do not guarantee accurate
curvature or transport. All-field accuracy remains open. Sampled energies decrease
and mass drifts are below3e-16; these do not prove continuous energy dissipation.
Phase/mu are continuous across windows to roundoff, while time derivatives can
jump (seed42 weighted RMS jumps0.050–8.93). Window states are fixed quadrature
arrays; no off-grid physical extension or exact classical wall-normal claim is made.

Load the **whole trajectory**, using direct paths:

    JAX_PLATFORMS=cpu python examples/confined_ch2d/evaluate.py --run=/root/autodl-tmp/tf-logs/confined_ch2d/20261027-chopt-windows-01 --checkpoint=/root/autodl-tmp/tf-logs/confined_ch2d/20261027-chopt-windows-01/trajectory.json

`trajectory.json` lists every model and fixed initial-state file by relative
path, without hashes. One final-window model is not the whole solution.
Incomplete manifests are rejected. Complete CPU replays, state links, exact
chemical relation and saved GPU fields pass for both seeds and the15-window
control. Stage `metrics.jsonl` reports L2 over a growing available time prefix;
it is **not** a fixed-full-domain training convergence curve.

[Before/after fields](../../docs/tasks/archive/chopt-before-after.png),
[reference/prediction/error](../../docs/tasks/archive/chopt-optimized-fields.png),
[errors versus physical time](../../docs/tasks/archive/chopt-time-errors.png),
[seed comparison and energy](../../docs/tasks/archive/chopt-accuracy-energy.png),
[complete diagnostic audit](../../docs/tasks/archive/chopt-analysis.json) and
[optimization record](../../docs/tasks/archive/confined-ch-optimization.md)
retain successful and failed controls separately from supervised diagnostics.

### Single global-time network

The global research presets use **one network and one checkpoint** over
\(0\le t\le1.5\). No intermediate PINN field is propagated or frozen, and
reference values do not enter the objective. These experiments have not yet
matched the window strategy's accuracy; do not label them a solved SOTA.

`configs/train_coarsening_global_average.py` uses a single64x4 spatial MLP
that outputs24 coefficients \(a_k(x)\). Its continuous phase trial function is

$$
s=\frac{\log(1+t/0.005)}{\log(1+1.5/0.005)},\qquad
\phi_\theta(x,t)=\Pi_m\left[\phi_0(x)+s\sum_{k=0}^{23}
a_{k,\theta}(x)T_k(2s-1)\right].
$$

The Chebyshev functions \(T_k\) have global support. All coefficients are
trained together; they are not networks for separate time intervals. Predicting
a time requires a network call and basis evaluation, without marching from
earlier times. The known scalar IC is an input feature, spatial modes remain8,
and output weights initialize at0.01 times their usual scale. Hard initial
state/mass and exact native chemical elimination remain. The model has16,344
parameters, versus15,817 in the joint space-time MLP control.

To keep initial-state information explicit, use the integrated evolution law

$$
R(t)=\phi_\theta(t)-\phi_0+M A\int_0^t\mu_\theta(q)\,dq,
\qquad \mu_\theta=\beta(\phi_\theta^3-\phi_\theta)+\epsilon^2A\phi_\theta.
$$

For continuous functions, \(R(t)=0\) for all times is equivalent to the IVP.
Train on the time-average residual \(R(t)/t\), preconditioned by
\((I+tM\epsilon^2A^2)^{-1}\). Division by positive \(t\) removes the automatic
\(O(t)\) suppression of early residuals; there is no division at zero.
The loss integrates its squared physical-volume norm over physical time.
This is a changed least-squares metric, not an inverse-tanh output transform.

All80 positive log-Chebyshev nodes, including terminal1.5, enter every update.
The initial chemical field is computed directly from the prescribed IC. A
global integration matrix approximates only the chemical-time integral;
training uses no AD time derivative. AD time derivatives remain available and
are independently checked in diagnostics. Phase is degree24 and eliminated mu
degree72 in \(s\), so80 intervals avoid polynomial interpolation aliasing.
Gaussian integration handles the physical-time Jacobian. Independent native
IVP checks give maximum preconditioned quadrature defects1.46e-4/2.65e-6/3.71e-9
for32/64/128 nodes, without changing or training on that reference.

    XLA_FLAGS='' python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening_global_average.py --seed=42 --config.log.run_name=my-global-average

`configs/train_coarsening_global_newton.py` retains the optimizer-only
Newton-CG cost control: same5000 SOAP updates/objective, then up to200
matrix-free Newton iterations. It was interrupted after9m15s in inner CG Hessian
products, with only iteration1 logged;200 iterations did not finish. Its5000-step
`model_soap.eqx` is retained and replayed, but excluded from completed-run
comparisons. The helper passes a manufactured PDE test; it is not recommended
for this case. Each global preset uses
FP64 and ordinary compilation. Load a global model directly, for example:

    JAX_PLATFORMS=cpu python examples/confined_ch2d/evaluate.py --run=/root/autodl-tmp/tf-logs/confined_ch2d/20261028-chglobal-average-01 --checkpoint=/root/autodl-tmp/tf-logs/confined_ch2d/20261028-chglobal-average-01/model_final.eqx

The other global controls retain full-horizon differential residuals, a
space-time residual inverse, the unnormalized integral law, a Neumann inverse
norm, and an energy-balance term. The latter is derived from
\(E(t)-E(0)+M\int_0^t\langle\mu,A\mu\rangle_Vdq=0\); it improves chemical
errors but worsens phase here. No independent reference labels train any
control. Numerical details, all outcomes and continuous PDE diagnostics are
in [the global audit](../../docs/tasks/archive/chglobal-analysis.json) and
[task record](../../docs/tasks/confined-ch-global.md).

Ten complete seed42 controls use5000 SOAP+500 L-BFGS each. Common-FEM phase
pooled/terminal errors are51.01%/61.25% (spatial inverse),97.97%/99.85%
(space-time inverse),84.26%/73.40% (joint-MLP integral),55.84%/64.96%
(Neumann inverse),35.61%/63.40% (global basis),89.13%/90.09% (energy), and
**21.54%/43.18% (R/t)**. Additional direct residual/causal/width128 controls give
57.84%/65.55%,56.77%/64.46%,65.63%/73.93%, respectively. R/t has the lowest phase error among completed
global controls, not an acceptable all-field result: native mu pooled258.1%,
positive-time440.9%, terminal2716.0%; phase range-1.76..1.67 and saved energy
increase0.0671 remain. Its trainer time is197.77s, excluding preprocessing.
Every complete final/best checkpoint passes independent CPU replay; reference
and physical identities are unchanged. No single-network SOTA/default is
promoted and no multi-seed stability claim follows from exploratory runs.

`train_coarsening_global_strong.py` instead uses the instantaneous AD evolution
residual in the Neumann-inverse norm, equally weighted at80 positive nodes and
zero. Only the physical-volume constant null mode is lifted; disconnected
graphs are rejected explicitly. `train_coarsening_global_wide.py` changes only
hidden width64 to128 (16,344 to57,240 parameters). Neither fixes this case.
`train_coarsening_global_causal.py` uses stopped exponential prefix weights
during SOAP, following only Eq3.2 of
[Wang, Sankaran and Perdikaris](https://arxiv.org/pdf/2203.07404).
Our variant has strength1, floor1e-8 and normalization by the weight sum.
L-BFGS returns to uniform weights; all times and one network remain throughout.
Its terminal SOAP weight is only0.0196 after5000 updates, so it is not converged.

A separate **validation-only** fit with free coefficients at each grid node
shows that the24-degree time basis can represent native phase to0.057% and
mu to0.793% on dense time queries. This is not a trained PINN and supplies no
training labels. Its initial derivative residual remains large; it does not
rule out derivative-resolution issues or identify spatial capacity versus
optimization uniquely. Increasing neural width alone did not solve the problem.
The single-network high-accuracy goal remains open; retain the experimental
configs and [diagnostic projection](../../docs/tasks/archive/chglobal-representation-floor.json)
without promoting a failed model as the default.

[Full result table](../../docs/tasks/archive/chglobal-results.md),
[global convergence](../../docs/tasks/archive/chglobal-convergence.png),
[fields and absolute error](../../docs/tasks/archive/chglobal-fields.png), and
[error/energy diagnostics](../../docs/tasks/archive/chglobal-errors-energy.png)
show unresolved accuracy rather than hiding it behind residual loss.

## Global mixed fields and explicit scales (experimental)

The optional `train_coarsening_mixed_joint.py`,
`train_coarsening_mixed_gradnorm.py` and `train_coarsening_mixed_stagger.py`
presets compare loss training on the original ten-wave/T1.5 case. All use one
shared spatial trunk with two24-coefficient time-expansion heads (phi and mu),
one global checkpoint, the same known-IC chemical scale0.5 and the same two
instantaneous Neumann-inverse/chemical-unit losses described below. Physical
mu is0.5 times its raw expansion; both loss rows are normalized to chemical
units and divided by0.5. This scale is not inferred from future reference fields.
No chemical elimination, time windows or frozen intermediate states are used.

All three start cold with SOAP5000/LR0.001 plus joint L-BFGS500, ordinary FP64.
Select them through the canonical `train.py`, for example:

    python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening_mixed_gradnorm.py --seed=42

`train.loss_balance.kind=fixed` retains unit weights. `gradient_norm` reuses the
project's inverse-gradient-norm rule: every25 updates, compute L1 norms of both
loss gradients over all trainable parameters, bound the norms to[1e-6,1e6],
then set weights to their mean divided by each norm, bounded to[1e-6,10000].
The explicit mixing alpha is1. Weights are stopped optimizer state, not
differentiated as functions of model parameters. This is gradient-norm balancing,
not the original relative-training-rate GradNorm algorithm.

`kind=stagger` alternates the active phase/chemical loss every25 updates for
the first4000 updates, then uses both losses for1000 updates. All shared
parameters remain trainable with one optimizer state: this alternates objectives,
not frozen physical fields. It gives2000 phase-only,2000 chemical-only and1000
joint SOAP updates. The adaptive run freezes its final weights during joint
L-BFGS; fixed/stagger use unit joint weights. No changing weights or alternating
objectives occur inside line searches. Logs retain unmasked parts, adaptive and
effective weights, gradient norms, active masks and exact stage counts. These
are controlled experiments, not a default or an accuracy guarantee.

`train_coarsening_mixed_gradnorm_raw.py` is an additional control with the same
shared network and gradient balancing, but no Neumann inverse/spatial metric.
Its two losses are the squared physical residuals divided by explicit known-IC
scales: evolution100 (rounded initial native rate RMS99.655) and chemical0.5
(rounded initial mu RMS0.4886). Thus it tests raw PDE residuals with separate
scale normalization. These initial quantities require only the prescribed IC
and operator, not future reference data. It is a later separate control, not
part of the identical-source three-arm cohort above.

Four complete seed42 cold starts on the original complex case give the
following final-checkpoint errors; all use5000+500 updates. Phase uses the
common FEM target; mu uses its corresponding native nonlocal target.

| Two-head strategy | Phase pooled / terminal | Mu pooled / terminal | Trainer |
|---|---:|---:|---:|
| Fixed joint, Neumann inverse |55.88% /62.62%|67.73% /277.33%|265.60s|
| Gradient balance, Neumann inverse |63.63% /71.51%|147.12% /475.95%|293.13s|
| Alternating losses, Neumann inverse |68.83% /67.28%|68.14% /348.65%|282.42s|
| Raw normalized, gradient balance |55.73% /64.23%|104.09% /364.33%|113.81s|

None beats the retained phase-only global R/t or the selected window strategy.
Best-phase checkpoint replay does not change that conclusion. In the common
Neumann-inverse norm, adaptive weighting reduces chemical loss from0.03038 to
0.00002972 but increases evolution loss from0.02053 to0.20032; the fields remain
wrong. Matching constitutive algebra or parameter gradient norms is not an
accuracy certificate. First-three initial parameters,57 source files and time
arrays are identical; the later raw control preserves initial parameters/times
but explicitly changes the residual norm/scales and optional-loss source.
One cohort does not rule out every two-head, field-block or GradNorm variant.
[Complete validation/results](../../docs/tasks/archive/chbalance-results.md),
[four-control convergence](../../docs/tasks/archive/chbalance-convergence.png),
[energy](../../docs/tasks/archive/chbalance-energy.png) and
[raw-control fields](../../docs/tasks/archive/chbalance-gradnorm-raw-final-fields.png)
are retained. Failed controls are optional experiments, not defaults.

The original learned formulation uses one shared MLP with one final linear
layer and two output channels, not two independent heads/networks. The recent
`global_basis` experiments instead output phase time coefficients only and
eliminate chemical potential exactly; they have one PDE loss. Neither should
be confused with the new optional `arch.model=global_mixed` controls:

- `train_coarsening_global_mixed.py`: one shared spatial MLP, with24 global
  phase coefficients and24 independent chemical coefficients in its final layer.
- `train_coarsening_global_separate.py`: two independent spatial MLPs, one per
  field, jointly optimized on all times. This increases parameter count and
  removes shared trunk parameters, but does not decouple the physical equations.
- `train_coarsening_mixed_block.py`: the same independent field networks, with
  a frozen inverse of the linear mixed space-time residual block.

All enter through train.py, save one complete model checkpoint, and query
arbitrary times on[0,1.5]. There are no frozen intermediate fields or time-window
models. The variable phi is the phase field; mu is its chemical potential.
For log-time s and Chebyshev polynomials T_k, the new outputs are

$$
\phi=\phi_0+s\sum_{k=0}^{r-1}a_k(x)T_k(2s-1),\qquad
\mu=S_\mu\sum_{k=0}^{r-1}b_k(x)T_k(2s-1),\quad S_\mu=0.5.
$$

Weighted mean projection retains exact phase mass. Only phi has a hard initial
condition; mu is learned and checked against its constitutive relation, including
t0. The mu output scale0.5 is a rounded estimate from the known initial field
(native initial RMS0.48857). No future reference field determines a training
scale. Reference-only diagnostics show mu RMS decreases to0.043815 at T1.5;
this also makes late relative mu error more sensitive to absolute error.
Output scaling changes parameterization; residual normalization changes the loss.
They are distinct choices.

The two strong mixed residuals are

$$
r_\phi=\phi_t+M A_\delta\mu,\qquad
r_\mu=\mu-\beta(\phi^3-\phi)-\epsilon^2A_\delta\phi.
$$

In the first two configs, use

$$
\mathcal L_\phi=\left\|\frac{A_\delta^+r_\phi}{M S_\mu}\right\|_V^2,
\qquad \mathcal L_\mu=\left\|\frac{r_\mu}{S_\mu}\right\|_V^2,
\qquad\mathcal L=\mathcal L_\phi+\mathcal L_\mu.
$$

Here mass projection makes r_phi zero mean; the connected-graph Neumann inverse
has only its constant mode lifted. Both normalized residuals have chemical units
before division by S_mu. This is a weighted least-squares choice, not proof of
balanced parameter gradients or guaranteed accuracy.

The block control instead uses the global integration matrix W and residuals
R=phi-phi0+M A W mu, C=r_mu, including t0. It applies the inverse of

$$
\mathcal P=\begin{pmatrix}I&M A W\\-\epsilon^2A&I\end{pmatrix},\qquad
E_\phi=(I+M\epsilon^2 A^2W)^{-1}(R-M A W C),\quad
E_\mu=C+\epsilon^2A E_\phi.
$$

The loss is the mean over all fixed global time nodes of
||E_phi||_V^2+||E_mu/S_mu||_V^2. E_phi has phase units, E_mu chemical units.
Spatial eigendecomposition and temporal inverses are precomputed once; all
network fields and gradients remain jointly optimized. This preserves the zeros
of the fixed-quadrature mixed residual. Quadrature consistency with the continuous
PDE is a separate requirement;80 nodes resolve the cubic of the rank24 phase
expansion in log time. It is a global residual transformation, not marching
physical states or solving from reference snapshots.

Numerical residual/gradient checks and cold starts are recorded in
[the active task](../../docs/tasks/confined-ch-global.md) and
[mixed-field audit](../../docs/tasks/archive/chmixed-analysis.json).
These are research controls, not a new default or a SOTA/stability claim.

The three new physics-only seed42 controls all completed5000SOAP+500L-BFGS:

| Global model | Phase pooled / common FEM | Phase terminal / common FEM | Mu pooled / native |
|---|---:|---:|---:|
| Shared mixed |58.49%|63.50%|70.28%|
| Separate mixed |71.24%|71.30%|74.06%|
| Separate + mixed block |46.11%|38.60%|50.51%|
| Prior eliminated R/t |21.54%|43.18%|258.07%|

These are final checkpoints. Block conditioning improves terminal phase/mu and
has no energy increase at the saved times, but pooled phase remains worse than
the prior R/t control. Mu terminal error remains130.06%, and the high-accuracy
global problem is unresolved. This is one-seed exploration, not a stable ranking;
shared/separate also differ in parameter count. The block recipe changes both
integral formulation and residual norm, so gains are not an output-scale-only
or inverse-only ablation. No new SOTA/default promotion.

A separate explicitly supervised-only diagnostic fits the SAME independent-field
network from identical initial weights to native fields: phase0.43846%, mu2.7160%.
These are not PINN results, and its weights never initialize a physics run.
Its block PDE loss is8.985e-5 versus the physics-trained2.762e-3, demonstrating
a feasible better joint approximation in this architecture. Its strong evolution
loss remains.458, so field accuracy alone does not imply accurate time derivatives.
This distinguishes optimization-path difficulty from an absolute lack of field
capacity; it does not prove that temporal derivative representation is adequate
for every residual. The block run's boundary band(about21.1% of physical area)
accounts for57.4% of terminal native squared phase error; bulk error28.5% also
remains. This is diagnostic only, not a reference-guided training weight.

[Results and limits](../../docs/tasks/archive/chmixed-results.md),
[convergence](../../docs/tasks/archive/chmixed-convergence.png),
[block fields](../../docs/tasks/archive/chmixed-block-fields.png), and
[supervised diagnostic residual audit](../../docs/tasks/archive/chmixed-representation-audit.json).

## Optional phase-only tanh and Jacobian residual controls

The tanh option retains exact chemical elimination and one global-time phase
network. It is separate from the mixed-field controls above. Use:

    python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening_tanh.py --seed=42
    python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening_tanh_jacobian.py --seed=42
    python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening_tanh_lowfreq.py --seed=42

All use instantaneous PDE residuals, not the polynomial integral rule: tanh of
a polynomial no longer has the finite-degree chemical unaliasing property.
The prescribed IC is lifted into the latent coordinate and projected by one
implicitly differentiated scalar shift:

$$
u(x,t)=\operatorname{atanh}(\phi_0(x)/a)+s(t)\sum_k c_k(x)T_k(2s(t)-1),
\qquad\phi_i(t)=a\tanh(u_i(t)+\alpha(t)),
\qquad \sum_i v_i\phi_i=\bar\phi_0\sum_i v_i.
$$

The fixed default amplitude a=1.1 gives10% room beyond the unit potential wells;
it is an explicit representation choice, not a reference-trajectory fit.
The frozen FEM/native fields slightly exceed1 (about2.5--2.8%), so strict
`--config.phase.output_map.amplitude=1.0` excludes some reference values.
Initial values and target mass must lie strictly inside the selected amplitude.
Do not replace the nonlinear mass shift with an additive correction after tanh.

Define J_i=a*sech²(u_i+alpha). The scalar mass equation gives

$$
\dot\alpha=-\frac{\sum_i v_iJ_i\dot u_i}{\sum_i v_iJ_i},\qquad
r_i=J_i(\dot u_i+\dot\alpha)+M(A_\delta\mu)_i,\qquad
\mu=\beta(\phi^3-\phi)+\epsilon^2A_\delta\phi.
$$

Map-only training uses r. The Jacobian option instead uses

$$
\widetilde r_i=\frac{r_i}{J_i}
=\dot u_i+\dot\alpha+\frac{M(A_\delta\mu)_i}{J_i}.
$$

It applies the existing invertible spatial metric AFTER this division. No
Jacobian floor, clipping or stopped-gradient weights are used. Evaluate J via
its stable exponential form; nonfinite losses fail loudly. The transformed
residual need not have zero weighted mean, so the metric is the full lifted
(A+Pi0)^-1, not a pseudoinverse that discards its constant part. A does not
commute with multiplication by1/J: CH has no claimed global cancellation like
the local Allen--Cahn reaction term in the melting case. Equal residual roots
do not guarantee better relative L2 at finite training budgets.

Spatial features are explicitly sin/cos(pi*k*z), z=(x-center)/axes. The default
k=1..8 has shortest physical wavelengths.125 in x and.10 in y. The lower-frequency
control changes k=1..5, wavelengths.20/.16, and consequently changes input size/
initialization as well. These are fixed features, not trainable frequencies.
The initial basic wave vectors have coordinate components up to5; tanh sharpening
also contains higher harmonics. In these global-basis models the spatial MLP is
static and has NO temporal Fourier embedding; time uses24 log-Chebyshev modes.
Per-run source/configs and comparison outcomes belong in the active task.
Load new checkpoints with their saved output-map config even though parameter
leaf shapes can agree with an unmapped model.

Completed seed42 controls (each5000SOAP+500L-BFGS) give:

| Phase-only recipe | Max spatial mode | Phase pooled / FEM | Phase terminal / FEM |
|---|---:|---:|---:|
| Original instantaneous residual |8|59.01%|65.16%|
| Tanh map, original residual |8|58.56%|64.98%|
| Tanh map, residual r/J |8|73.07%|78.42%|
| Tanh map, residual r/J |5|69.27%|75.10%|
| Integral R/t, historical control |8|21.54%|43.18%|
| Integral R/t, new frequency-only control |5|17.99%|29.37%|

These are final checkpoints. The tanh/Jacobian result is specific to the
instantaneous-residual recipe; tanh on the best integral recipe has not been
validated here. In particular, this is not evidence that every tanh scaling or
CH residual transform fails. The historical integral control is not a fresh
matched rerun. Lower modes change feature dimension and initialization, so the
result is exploratory one-seed evidence, not a stable/parameter-matched ranking.

For the next phase-only integral comparison, select the explicit working preset:

    python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_coarsening_average_lowfreq.py --seed=42

It is equivalent to train_coarsening_global_average.py with
`--config.arch.spatial_modes=5`. This new lowest-phase experiment still has native
mu129.29% pooled/836.60% terminal, phase range[-1.34285,1.24822], and sampled
energy rise0.007078. It is not SOTA or an accurate final solver; the original
simple pilot remains an explicit option. The current CLI default is the richer
time-window strategy; tanh/Jacobian remains explicit opt-in.

Independent Brent/polynomial checks of actual trained time derivatives agree
within2.3e-12; exact chemical elimination/mass and every final/best CPU replay
pass. The two Jacobian controls have constant-mode loss shares only0.052%/0.020%
at final checkpoints and dense midpoint losses within1% of training-node losses;
neither gross constant-mode domination nor obvious between-node aliasing explains
all degradation. Map Jacobians do not underflow in these checks. Fourier mode8
has2.46--3.30 times the native A-squared feature gain of mode5 for the axis probes;
there is room to reduce stiffness even without an input-grid sampling violation.
[Results and limits](../../docs/tasks/archive/chtanh-results.md),
[convergence](../../docs/tasks/archive/chtanh-convergence.png),
[frequency audit](../../docs/tasks/archive/chtanh-frequency-audit.json), and
[new low-frequency phase fields](../../docs/tasks/archive/chtanh-integral-m5-fields.png).

## Intermediate-difficulty three-wave control

`configs/train_moderate.py` defines a separate physical initial-value problem:
three saturated low-frequency waves, coordinate wave vectors (1,1), (2,-1),
(1,-2), and terminal time0.75. It retains the richer case's ellipse/two holes,
epsilon0.02, mobility0.02, beta1, regional delta0.03, quadrature mesh size0.02,
IC amplitude0.98/scale0.55/wall width0.03. The initial coefficients/phases are
explicit in that config. This keeps visible two-phase interfaces and complex
walls while reducing the initial pattern count and evolution duration.

The training scheme is the same phase-only global M5/rank24 integral R/t
experiment: one network over the whole interval, exact chemical elimination,
hard IC/physical-volume mass,80 rebuilt global temporal nodes, ordinary JIT,
FP64, SOAP5000 at initial learning rate0.001 and L-BFGS500. No tanh output map,
inverse-Jacobian weighting, reference labels in the loss or time windows.
This optional case is separate from the richer time-window CLI default.

Generate its independent reference first, in `fenicsenv`:

    python examples/confined_ch2d/solver.py --config examples.confined_ch2d.configs.train_moderate

After that process finishes, train in `.venv`:

    python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_moderate.py --seed=42

Its reference is stored separately at
`/root/autodl-tmp/data/confined_ch2d/moderate_multimode3/reference.npz`.
FEM mesh sizes0.014/0.01 and time steps0.005/0.0025 produce11 snapshots.
The same independent FEM/native mesh/time, mass and energy gates apply; no
relaxation is implied by calling the case easier. Relative L2 values from this
case and the ten-wave/T1.5 case concern different initial-value problems and
different reference norms; a smaller value is not a same-problem method gain.

The first complete5000+500 cold start takes188s and reaches common-FEM phase
relative L2 of5.16% pooled/9.77% terminal (native4.29%/8.94%). Reference mesh/time
checks pass the original tolerances; the field undergoes substantial change.
Chemical errors remain45%/156% against the native reference, with late phase
overshoots and a small sampled energy increase. This intermediate benchmark is
easier than the ten-wave case but the global solver is still experimental.
See [result and validation limits](../../docs/tasks/archive/chmoderate-results.md),
[reference/PINN/error fields](../../docs/tasks/archive/chmoderate-final-fields.png),
and [convergence/energy](../../docs/tasks/archive/chmoderate-convergence-energy.png).


## Geometry-operator SOTA

The recommended `configs/train_operator.py` selects the measured
`train_operator_online_width256_cold50k.py` recipe: one global-time geometry
operator with fixed small-scale10mode initial data and continuously refreshed
random-hole inputs. Only hole centers/radii vary; the initial ambient function
is fixed. Boundary and domain-function point streams use the general
GNOT-inspired encoder below, without query-to-hole distance bias.

```bash
source .venv/bin/activate
python -m examples.confined_ch2d.train \
  --config=examples/confined_ch2d/configs/train_operator.py
```

| Setting | SOTA value |
|---|---|
| Physical problem | Fixed small-scale10modes, T3, beta1, epsilon0.06, mobility0.02 |
| Random holes | Both centers x:[0.22,0.78], y:[0.27,0.73]; radii:[0.045,0.12]; clearance0.025 |
| Quadrature / spatial operator | Gmsh h0.035, physical cell areas, visible regional pairs, delta0.11 |
| Network | Heterogeneous point streams; width256,4heads of64,4conditional blocks,3time-FiLM layers |
| Fourier features |128spatial directions /256sin-cos channels, scale2;32time pairs, maximum frequency4 |
| Initial training pool |1024validated known-input geometries, fixed padding1536nodes/128neighbors |
| Online production |6CPUworkers; replace32slots every100updates from update1;64prefetched inputs |
| Training batch |16geometries,16shared positive times plus t0; t0PDEweight0.2 |
| Optimizer | SOAP, beta1=beta2=0.9, refresh5, axis cap512, gradient clip10 |
| Learning rate |0.001 initially; multiply by0.95 every1000updates; floor0.00001 |
| Arithmetic / compilation | FP32 / ordinary GPU policy / highest matmul precision |
| Budget | Cold50000SOAPupdates, fresh model and optimizer; no time windows or state import |
| Outputs / constraints | Predict phi, eliminate mu through the graph; hard initial field and physical-volume mass |
| Training loss | Phase PDE only, frozen initial-mean linear stable-part resolvent; no duplicate IC/mass loss |
| Recovery | Paired model/SOAP/pool checkpoints every5000updates |

The initial pool is at
`/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1`.
Online workers construct geometry, graph, quadrature, known inputs and residual
metrics only. They solve no training PDE/FEM trajectories. This run installed
16000fresh inputs and actually sampled15994of them; evaluation fields are kept
outside training payloads. Every time within a geometry shares the spatial
preparation, and temporal derivatives use JVP.

The accepted checkpoint is the fixed-budget **50000-update final model**:

```text
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-online-cold50k-01-train/model_final.eqx
```

Its exact replay configuration and optimizer are the adjacent `config.json`
and `optimizer_final.eqx`. Load by direct file path with the saved config;
no hash gate or validation-selected earlier checkpoint is used. Canonical
configuration matches this saved recipe apart from runtime seed/run name.
Seed is run provenance, not a strategy-config field.

| Mean physical-cell weighted relative L2 | Pooled five saved times | Terminal t3 |
|---|---:|---:|
| Phase phi,8reserved development geometries | **4.8329%** | **5.1935%** |
| Chemical potential mu, same8geometries |10.7329% |14.2757% |
| Phase phi,16frozen historical training-domain monitors |5.0529% |5.4186% |

The8development geometries are reused across these studies, with one run
initialization; this is not a fresh blind test or cross-seed stability claim.
References are independently time-refined **native nonlocal** solutions.
Local FEM for this harder10mode case remains deferred. Worst geometry phi L2
is9.5686% pooled and10.5862% terminal. Thus the mean phase target is met;
not every terminal sample is below10%, and mu is less accurate than phi.

Measured cold training wall time is102.26min on NVIDIA RTX6000D; mean GPU
update after100steps is96.88ms. Full-run active JAX memory peak is20.43GiB
(distinct from reserved allocator memory). Independent CPU replay of24saved
trajectories passes, with maximum phi gap7.15e-7 and mass drift5.85e-8.
See the [completed training/audit record](../../docs/tasks/archive/confined-ch-width256-online-cold50k.md),
[SOTA promotion record](../../docs/tasks/archive/confined-ch-operator-sota.md),
and `artifacts/index.yaml` for configuration, metrics and direct artifact paths.
The [low/median/worst physical-field gallery](/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-online-cold50k-01-train/final_fields_02/gallery.json)
and its PNG/PDF files remain external artifacts.

The old mild pilot is available explicitly as `configs/train_operator_mild.py`;
historical presets retain their numerical settings. No-argument `train.py`
continues to select the separate30-window single-IVP case.

## Geometry-conditioned neural operator pilot

The historical `configs/train_operator_mild.py` learns geometry-to-phase evolution on
an ellipse with two random circular impermeable holes. Outer center/axes remain
(0.5,0.5)/(0.5,0.4); only hole centers and radii vary. This is a deliberately
shorter, milder problem, not the original ten-wave/T1.5case. Historical pilot command:

```bash
source .venv/bin/activate
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_mild.py
```

The first pilot uses a **finite** bank of16training/8held-out geometries drawn
from a continuous prior. This is not infinite online geometry sampling. Initial
and positive-time references of held-out domains do not enter a training step;
reference values are evaluation-only. Final weights use a fixed5000-update
budget, not checkpoint selection on test errors. Existing runs fail rather than
being overwritten. No-argument train.py keeps the established windowed PINN.

| Quantity | Pilot setting |
|---|---|
| Hole1 center bounds | x:[.28,.42], y:[.46,.61] |
| Hole2 center bounds | x:[.60,.74], y:[.36,.51] |
| Radii | hole1:[.05,.085], hole2:[.04,.075] |
| Validity | Conservative full-circle containment in ellipse; inter-hole and wall clearance>.025 |
| Physics | Same regional CH, beta1,M.02,epsilon.06,T.2 |
| Quadrature/operator | Per-geometry Gmsh h.045, physical triangle-centroid areas, delta.11 visible pair kernel |
| Geometry condition |96outer-wall points and48points/hole, coordinates, outward domain normals, inner/outer labels, arc lengths |
| Decoder | Shared spatial preparation; time FiLM and time JVP; no query-local initial features |
| Training | Adam1e-3,5000updates,B4,4shared stratified positive times plus t0; ordinaryFP64/highest |
| Physical constraints | Exact prescribed IC and per-domain volume-weighted mass; chemical potential eliminated |
| Loss | PDE-only phase residual in the existing biharmonic-resolvent metric; no extra IC/mass/chemical loss |

The same ambient function is restricted to every domain. With
z=(x-center)/axes, the prescribed fixed input is

$$
\phi_0(x)=.25\left[\cos(\pi(z_x+z_y))+
.4\cos(\pi(2z_x-z_y)+.7)\right].
$$

There is no moving-hole-dependent initial envelope. Different domains therefore
have different phase mass, fixed by their own physical volumes. Classical local
Neumann derivatives can be incompatible at t0; local reference uses the weak
natural-boundary evolution. No compatible classical initial chemical potential
is assumed. Native and local errors remain separate.

A point-measure encoder replaces image patchification. It supports compression
into a small latent array or explicit `arch.latent_tokens=0` to retain boundary
point tokens. Attention aggregation uses positive boundary arc-length measures.
Token labels distinguish inner/outer walls without assigning physical identities
to the two holes. Coordinate/time raw queries do not receive evolved fields.
Encoder/prepared features stay inside parameter differentiation every update;
no learned cache is reused after an optimizer update.

Each geometry has its own physical points/volumes/operator. Batches pad N/K to
fixed shapes with zero physical volume and zero edges, mask padded outputs, and
preserve true per-geometry means. Arbitrary geometry/mesh transfer is not claimed
from a fixed-two-circle training distribution. Encoding geometry does not itself
enforce Neumann conditions; those remain the regional closed-pair physics.

Prepare references in the separate FEniCS environment, before GPU training:

```bash
conda activate fenicsenv
OPENBLAS_NUM_THREADS=1 python -m examples.confined_ch2d.operator_reference --config=examples.confined_ch2d.configs.train_operator_mild
OPENBLAS_NUM_THREADS=1 python -m examples.confined_ch2d.operator_reference --config=examples.confined_ch2d.configs.train_operator_mild --local
conda deactivate
```

The historical mild bank root is
`/root/autodl-tmp/data/confined_ch2d/operator_geometry_v1`.
It contains direct-path per-geometry meshes, independent SciPy/BDF native
references with time refinement, and two held-out mixed-P2FEM checks. The local
reference checks validate phase refinement/mass/energy; chemical comparisons
exclude t0 because its wall compatibility is not imposed. No data hashes.

Use `python -m examples.confined_ch2d.operator_evaluate --run=<run-path>
--output=<new-output-directory>` for independent saved-field metrics and actual
physical-mesh figures. Local and native errors, frozen-initial baseline and
wrong-geometry encoder swaps diagnose fitting separately from geometry bias and
conditional dependence. Generated media/arrays remain local artifacts.
Numerical results and current pilot limitations are recorded in the
[completed experiment](../../docs/tasks/archive/confined-ch-geometry-operator.md).

The stronger `configs/train_operator_longer.py` control changes only the terminal
time to.75 and uses its separately validated bank. Geometry, ambient IC,
epsilon, mesh and kernel remain identical. Both representative local FEM checks
pass: phase mesh differences.160/.229%, time differences below.00034%.
Native/local phase differences are4.78/4.84% pooled and5.76/5.71% terminal;
keep these operator-discretization errors separate from neural fitting error.

An experimental **general heterogeneous point encoder** is available through
`configs/train_operator_heterogeneous.py`, on the same longer bank. It borrows
GNOT's separate input encoders and normalized attention, with two input streams:
boundary coordinates/normals/type with arc measures, and physical domain
coordinates/initial phase with cell-area measures. Raw decoder queries contain
coordinates/time, without query-local initial values. Domain initial values are
known prescribed inputs, never positive-time labels. Geometry dependence enters
through the sampled domain/boundary sets rather than fitted hole parameters.

Each spatial layer rereads both streams with independently normalized measures.
For softmax-over-channel query/key features, one stream computes

$$
\mathcal A(q)=
\frac{\widetilde q^{\mathsf T}\sum_i w_i\widetilde k_i v_i^{\mathsf T}}
     {\widetilde q^{\mathsf T}\sum_i w_i\widetilde k_i}.
$$

The numerator is interpreted as a row-vector contraction. Positive physical
weights integrate the input function, zero weights exclude padded domain nodes,
and the two stream outputs are averaged after separate normalization. Repeated
conditional reads then feed the existing separable time decoder. Attention has
linear input/query-point complexity for fixed feature width, without a dense
query-by-input matrix. There is no query-to-wall distance bias or manually
chosen attention neighborhood. Width64/heads4/four conditional layers replace
the boundary-only encoder as an explicit architecture bundle; no other training
or physical scheme changes in this control.

This is a GNOT-inspired variant, **not** a full reproduction: query-set
self-attention and geometric mixture-of-experts are omitted to preserve exact
independence of raw spatial queries. All times share learned spatial preparation
inside the parameter-gradient computation. [GNOT, sections3.3-3.4](https://proceedings.mlr.press/v202/hao23c/hao23c.pdf)
provides the multi-input encoding and normalized attention construction.
[Perceiver IO](https://arxiv.org/abs/2107.14795) is another general route using
latent read/process/write; the earlier small latent pilot does not rule it out.
[Transolver](https://arxiv.org/abs/2402.02366) learns soft slices from point
features rather than fixed geometric patches. None of these papers by itself
establishes convergence for our physics-only CH training objective.

Matched5000Adam/B4 cold starts on the strongerT.75bank give:

| Metric | Boundary-only control | Heterogeneous point streams |
|---|---:|---:|
| Native phase relative L2, pooled / terminal |5.293% /6.220% |2.380% /2.771% |
| Native chemical relative L2, pooled / terminal |10.763% /13.077% |5.204% /6.458% |
| Local FEM phase relative L2, pooled / terminal (2domains) |6.886% /7.907% |5.720% /6.704% |
| Common-query geometry-response relative error |84.42% |42.65% |
| Geometry-response amplitude / reference amplitude |43.54% |94.74% |
| Geometry-response correlation |.548 |.905 |
| Cold trainer wall time |314.31s |406.65s |
| Warm update after100 |21.46ms |36.29ms |
| Peak active JAX memory |.408GiB |.518GiB |

Both runs have identical20,000actual sampled geometry IDs, all16training inputs
visited and no held-out draws; nonarchitecture configs agree. Actual time
arrays were not saved: time parity follows the same seed/sampler implementation,
not a saved-array comparison. CPU reload of all24final phi/mu fields agrees
with GPU saves within1.9e-15. Exact IC/weighted mass remain; five stored times
have nonincreasing predicted energy. The default JAX allocator reserved35.56GiB
in each run; this pool is distinct from active memory. Times are single-process
observations including different compilations, not a statistical speed result.

Geometry response means the field minus its across-geometry mean at identical
coordinates. It is evaluated on7556common fluid probes, interpolating native
evolution increments through valid triangles; it is not a separate exact-grid
FEM result. Whole-field and response errors have different denominators.
The larger response amplitude and correlation support improved conditional
fidelity, but42.65%response error still leaves room for improvement. Input
streams, positional lifting, depth and width changed together; no single-change
causal attribution is justified. Eight held-out domains were reused for
architecture development; fresh test geometries and seed stability remain
unmeasured. The model accepts point sets without circle-specific attention,
but arbitrary topology/mesh/3D accuracy has not been demonstrated.

To reproduce the stronger general-model pilot:

```bash
source .venv/bin/activate
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_heterogeneous.py
```

Direct run directories are
`/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-longer-direct-01`
and
`/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-longer-heterogeneous-01`.
Each has saved configuration/source, `model_final.eqx`, evaluation fields/curves,
geometry-response diagnostics and actual geometry draws. This remains an opt-in
control; the no-argument windowed PINN and mild operator preset are preserved.

### Richer geometry response and smoother visualization

`configs/train_operator_richer.py` is a separate physical benchmark using the
same general point-stream architecture, Adam5000/B4 and T.75/epsilon.06/M.02.
It refines the actual Gmsh mesh from h.045 to.035 and fixes eight ambient waves
across all geometries. With z=(x-center)/axes,

$$
\phi_0(x)=.22\sum_{j=1}^{8}a_j\cos\!\left(\pi k_j\cdot z+\vartheta_j\right).
$$

| kx, ky | amplitude a | phase |
|---|---:|---:|
|1,0 |.25 |.2 |
|0,1 |.25 |1.1 |
|1,1 |.65 |.7 |
|1,-1 |.6 |2.2 |
|2,0 |.5 |1.4 |
|0,2 |.4 |.4 |
|2,-1 |.4 |2.8 |
|2,1 |.35 |1.9 |

The amplitude bound is.748; there is no geometry-dependent wall envelope.
The local linearized CH guide gives growth M|k_phys|^2(beta-epsilon^2|k_phys|^2)
with k_phys=pi(kx/ax,ky/ay), positive for all eight selected modes. This is a
small-perturbation local guide, not a nonlinear/nonlocal growth guarantee.

Both circles independently sample centers x:[.22,.78],y:[.27,.73] and radii
[.045,.12], conditioned on the existing sufficient ellipse containment and
clearance>.025. They no longer have fixed left/right placement boxes. Boundary
tokens still use only inner/outer types, without hole-specific identities.
Different domains restrict the exact same initial function; their phase masses
can differ. Classical Neumann compatibility at t0 is not imposed.

Prepare the new bank and representative local FEM checks in fenicsenv:

```bash
OPENBLAS_NUM_THREADS=1 python -m examples.confined_ch2d.operator_reference \
  --config=examples.confined_ch2d.configs.train_operator_richer
OPENBLAS_NUM_THREADS=1 python -m examples.confined_ch2d.operator_reference \
  --config=examples.confined_ch2d.configs.train_operator_richer --local
```

After leaving fenicsenv, train through the same canonical entry in .venv:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_richer.py
```

The bank is `/root/autodl-tmp/data/confined_ch2d/operator_geometry_richer_v1`.
This remains a finite16train/8validation pilot, with evolved fields used only for
evaluation. Initial field, geometry distribution and mesh differ from the earlier
benchmark, so cross-case L2 differences do not isolate a numerical-method gain.

`operator_evaluate` now defaults to `--plot-style=smooth`; explicit `flat` keeps
cell-color plots available. Smoothing projects incident physical cell values to
vertices using cell areas, then linearly interpolates only within the original
mesh triangles onto a400x320display grid. The original hole topology and circle
masks remain. Both reference and prediction use the same reconstruction; plotted
absolute error is their reconstructed difference. All L2/mass/energy metrics
still use unmodified centroid fields and physical areas. Display interpolation
does not increase solver resolution or preserve a separate interpolated mass.
Use `python -m examples.confined_ch2d.operator_plotting --dataset=<bank-path>
--output=<new-directory>` for a reference-only evolution gallery.

### Larger operator-only training bank

`configs/train_operator_richer512.py` keeps the richer physical problem, model
and5000Adam/B4budget, and expands training coverage to512random domains. The
bank remains finite. Six spawned CPU workers construct mesh/quadrature, the
same visible-pair graph and dense biharmonic residual metric. New training
inputs contain only these fixed quantities and the analytic IC: no evolution
integration, native trajectory or FEM solve is needed for them.

The original accepted draws16..23 remain validation domains, reindexed512..519.
Training uses accepted draws0..15,24..519. Existing16training references are
retained as labeled monitors only; they do not enter optimization. Training
L2 curves therefore refer to those16monitors, not to all512training domains.
The eight native references and two already refined local FEM references are
reused with checked physical/geometry identity. This is the same development
validation cohort, not a newly independent final test.

Prepare from `.venv`, without an activated fenicsenv; the explicit meshing
interpreter runs in separate CPU subprocesses:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_richer512.py \
  --config.operator.action=prepare
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_richer512.py
```

Preparation refuses an existing output directory. The source bank path,
reserved draws, monitor IDs and worker count are explicit in `operator.precompute`.
The resulting direct-path bank is
`/root/autodl-tmp/data/confined_ch2d/operator_geometry_richer512_v1`.
Its `operator_only_v1` manifest records reference availability per domain;
legacy fully labeled banks remain supported. Array inputs are padded to one
fixed shape before GPU training, and learned encoder values are recomputed
inside each optimizer step. Larger geometry coverage may improve generalization;
the unchanged finite training budget can also limit per-input convergence.

Measured parallel preparation takes124.70s with six workers; the bank occupies
about6.11GiB including auxiliary meshes/configs. Three new-domain operator probes
agree with the independent ordered assembler within1.82e-12. No new PDE/FEM
trajectory was solved. The original16reference cache remains unchanged.

One matched5000Adam/B4cold comparison, same model/initial parameters and the
same eight validation geometries, gives the following final-checkpoint errors
(percent; mean of per-domain physical-volume-weighted relative L2):

| Metric |16training domains |512training domains |
|---|---:|---:|
| Native phase, all saved times |9.184 |4.690 |
| Native phase, terminal t=.75 |11.200 |5.454 |
| Worst validation phase, all saved times |13.715 |6.725 |
| Native chemical potential, all saved times |20.811 |11.661 |
| Same16training monitors, phase all times |2.016 |4.314 |
| Local FEM phase, mean of2validation domains |12.846 |8.976 |
| Local FEM phase, terminal, mean of2domains |15.402 |10.431 |
| Cold trainer wall time (seconds) |431.30 |524.42 |
| Warm synchronized update (milliseconds) |43.02 |43.22 |

Expanded coverage improves validation while fitting the old16monitors less
closely; this supports reduced overfitting in this single comparison. The total
20,000condition draws visit all512training domains. Cache loading/validation and
initial inference compilation take120.70s in the larger run, followed by87.96s
for the first parameter-update compilation; these costs are included in cold
wall time, separate from124.70s preparation. Native-reference accuracy remains
different from local-PDE accuracy: the unchanged validated native/local reference
bias is about7-9% on the two local-check domains. No multi-seed or new final-test
claim is made. Both mass drift and sampled energy monotonicity pass on the24
labeled monitoring/validation trajectories.

The final run is
`/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-richer512-01`.
Its `model_final.eqx`, `config.json`, `training_draws.npz`, `evaluation/` and
`geometry_effect/` retain the direct checkpoint, actual draws, raw-field metrics
and plots. The canonical single-IVP window default and earlier operator controls
remain available; this is an explicit geometry-operator research configuration.

On identical4627common fluid probes, geometry-response relative error improves
77.027%->35.514%, amplitude recovery74.423%->88.478%, and correlation.6453->.9362.
This supplements whole-field L2: the larger model-training bank better resolves
geometry-dependent evolution. All24saved fields replay on CPU within1.64e-15;
initial model parameters and reused reference arrays are exactly unchanged.
The actual512condition IDs are each visited20..68times in20,000draws.
[Validation and preparation record](../../docs/tasks/archive/confined-ch-geometry-richer.md).

For a longer cold budget on the same512operator bank, use the explicit preset:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_richer512_long.py
```

This changes only `train.iterations=20000`. It creates a fresh model and Adam
state; it does not resume the5000checkpoint. The same exponential LR schedule
(1e-3 initial,.95 decay per1000steps,1e-5 floor),500step monitoring and1000step
checkpoint saves remain. The same16labeled training monitors and8validation
geometries support comparison; future reference fields remain evaluation-only.

A matched20000step cold run preserves the initial parameters exactly and
reproduces the5000step prefix (checkpoint parameter relative difference7.28e-15;
common evaluation maximum difference4.31e-16). Final-checkpoint errors are:

| Metric (%) |5000steps |20000steps |
|---|---:|---:|
| Validation phase, all saved times |4.690 |2.463 |
| Validation phase, terminal t=.75 |5.454 |2.837 |
| Worst validation phase, all saved times |6.725 |3.874 |
| Validation chemical potential, all saved times |11.661 |6.971 |
| Same16training monitors, phase all times |4.314 |2.245 |
| Local FEM phase, mean of2domains |8.976 |8.111 |
| Local FEM phase, terminal, mean of2domains |10.431 |9.473 |

Cold wall time is1401.54s (~23min22s), including118.65ssetup and93.89sfirst
update; warm synchronized updates average41.89ms. The same512geometries receive
80,000actual condition draws. The20000final model is selected by the fixed
budget, not the minimum validation error. During15000..20000steps, validation
phase L2 fluctuates2.333..2.791%; a linear fit falls0.0585percentage points per
1000steps. It is more fully optimized but not demonstrated fully converged.
Native/local reference bias remains about7-9% on the two checked domains;
smaller native fitting errors do not imply the same local-PDE accuracy.

The longer run is
`/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-richer512-long-01`.
Use its savedconfig/model_final.eqx and evaluation/geometry_effect/comparison
outputs by direct path. The5000step preset/run remains the matched short-budget
control; the20000preset is an explicit option for higher fitting accuracy.

Geometry-response diagnostics on the same4627common fluid probes improve from
35.514% to19.284% relative error; amplitude recovery88.478%->93.316% and
correlation.9362->.9825. Every training geometry is visited123..201times. All24
labeled predicted trajectories satisfy hard IC/mass and sampled nonincreasing
energy; CPU/GPU field replay agrees within1.45e-15. Actual condition draws in
the common5000step prefix are identical. Full convergence/physics-loss curves,
paired metrics and the updated smooth physical fields are under the long run's
comparison/ and evaluation/ directories.
[Long-budget validation record](../../docs/tasks/archive/confined-ch-operator-long-training.md).

### Retained mixed-band previews

The earlier T2/20mode and T3/24mode mixed-band reference previews were superseded
by the fixed small-scale10mode case below. Their configurations and external
outputs are retained for provenance; neither candidate has a trained neural
operator or a prepared512geometry bank. See the archived experiment record below for their
selection history. Do not use their evolved fields as10mode references.

### Fixed small-scale 10-mode geometry operator

Only the two-hole geometry varies. Every geometry restricts the same fixed
ambient initial function, without a geometry-dependent envelope:

$$
\phi_0(x)=0.036\sum_{j=1}^{10}\cos(\pi k_j\cdot z+\theta_j),
\qquad z=(x-\mathrm{center})/\mathrm{axes}.
$$

The directions are (3,0), (0,3), (3,1), (1,3), (3,-1), (1,-3), (3,2), (2,3),
(3,-2), (2,-3). Ordered phases are explicit in
`configs/train_operator_smallscale10.py`. These retain ten modes of the displayed
small-scale component, omitting (3,+/-3). The analytic amplitude bound is .36;
the shortest wavelength is .2353. No initial-condition parameters or additional
scale selector are introduced.

The physical case uses T=3, beta=1, epsilon=.06, mobility=.02, mesh size=.035 and
regional horizon=.11, with the same broad random two-hole prior. Fine initial
ripples relax before larger geometry-dependent phase regions develop. Independent
native references show 37.79% terminal relative spread on 4627 common fluid
queries. This is a different physical problem from the eight-mode/T.75 case;
cross-case errors do not isolate a method improvement.

#### Preparation and training

The source bank is
`/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_v1`.
Six CPU workers prepare 24 independent ordered-graph/BDF reference trajectories
with time refinement (340.25s; maximum phase/chemical differences about
1.08e-9/9.39e-10). Run in fenicsenv:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.confined_ch2d.operator_reference \
  --config=examples.confined_ch2d.configs.train_operator_smallscale10
```

The selected 512-domain preset explicitly sets
`operator.local_validation_indices=()`: local FEM is deferred at the user's
request. Native temporal gates remain mandatory. Existing nonempty local
requests still require passed matching FEM files. The source preset retains
indices (16,17) for an optional later `--local` study; no FEM job is queued.
Partial FEM outputs are retained without a validated local-reference claim.

After leaving fenicsenv, activate .venv. Prepare the operator-only training bank:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_512.py \
  --config.operator.action=prepare
```

Its path is
`/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1`.
Preparation takes 125.76s with six workers, about 6.08GiB. The 496 new training
inputs contain only geometry, quadrature, initial data, operator and frozen
residual metric; no trajectory or FEM solve is performed for them. Sixteen
fresh labeled training monitors and eight fresh validation trajectories are
used exclusively for evaluation. Old accepted draws 16..23 remain held out,
reindexed 512..519. Preparation requires new output directories; existing caches
can be reused for training without regenerating them.

Train either cold optimizer control through the same entry:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_512.py
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_512_soap.py
```

Both use 20000 optimizer calls, B4, ordinary FP64, initial LR .001 with .95 decay
per 1000 calls and 1e-5 floor, four time strata plus t0, and the same generic
point architecture. Exact analytic IC and physical-volume mass, chemical
elimination and the frozen biharmonic residual norm remain; only the PDE enters
the training objective. There are no time windows or evolved training labels.
SOAP uses b1=.9, b2=.99, preconditioning every five calls, max dimension 128,
no one-dimensional preconditioning and global gradient clipping at 10.

#### Matched cold optimizer result

Validation numbers average per-domain relative L2 on the same eight development
geometries, using raw centroid fields and physical cell areas. Pooled values
combine the five saved times 0, .75, 1.5, 2.25, 3; they are not a continuous-time
quadrature or a fresh independent test set.

| Metric | Adam | SOAP |
|---|---:|---:|
| Validation phase L2, five times (%) |51.260 |39.561 |
| Validation phase L2, terminal T3 (%) |50.784 |39.170 |
| Training-monitor phase L2, five times (%) |50.959 |37.994 |
| Validation chemical-potential L2, five times (%) |47.528 |37.708 |
| Cold trainer wall time (s) |1435.614 |2030.369 |
| First compiled update (s) |94.309 |201.786 |
| Warm synchronized update after call 100 (ms) |42.785 |69.817 |
| Geometry-response error on common queries (%) |91.711 |80.071 |
| Geometry-response amplitude recovery (%) |28.423 |41.953 |
| Geometry-response correlation |.422 |.637 |

Geometry-response rows use centered trajectories on 4627 common queries,
separate from physical-area field L2.

SOAP lowers the fixed-budget pooled phase error by 22.82% relative to Adam.
It first matches both Adam final phase metrics at 8500 calls/1045.75s, about
17.4min versus Adam's 23.9min. SOAP is slower per call but reaches this accuracy
sooner. These are observations from one cold run per optimizer, not a stable
optimizer ranking.

Both curves continue falling. During the last 5000 calls, Adam phase L2 changes
55.07->51.26% and SOAP 44.00->39.56%; fitted slopes are -.729/-.824 percentage
points per 1000 calls. The fixed final checkpoint is used in both runs. This
supports slow, incomplete fitting; it does not establish a stationary trap or
an insufficient model capacity. Significant internal phase-region errors and
weak geometry response remain. Exact IC/mass and nonincreasing energy at five
saved times do not certify field accuracy. Local FEM is not evaluated.

Initial parameters, numerical training sources and all 80000 actual geometry
draws agree exactly. Every training geometry is visited 123..201 times, with no
held-out draws or future fields in optimizer payloads. CPU replays of all 24
labeled fields agree with GPU saves within 4.83e-15 (Adam) and 3.56e-15 (SOAP).

Direct run directories under `/root/autodl-tmp/tf-logs/confined_ch2d` are
`20261030-confined-geometry-smallscale10-01` and
`20261030-confined-geometry-smallscale10-soap-01`. Each has saved configuration,
source, `model_final.eqx`, actual draws, predicted fields and smooth
`evaluation/fields_512.png`, `fields_516.png`, `fields_519.png` panels. Metrics
use raw fields; smoothing supplies no extra solver accuracy. The SOAP run's
`comparison/convergence_updates.png` and `convergence_wall_time.png` show both
full curves, and `training_report/` contains individual histories and tail trends.

#### Architecture and untested capacity controls

| Setting | Value |
|---|---:|
| Boundary/domain stream dimensions |(5,3) |
| Width / heads |64 / 4 |
| Conditional interaction blocks |4 |
| Temporal FiLM decoder layers |3 |
| Fixed spatial Fourier directions / scale |24 / 1 |
| Temporal Fourier frequencies |4 |
| Output increment scale |.1 |
| Trainable scalars, excluding 48 fixed Fourier entries |224705 |

Boundary features are (x,y,nx,ny,type); domain features are (x,y,phi0).
Queries share spatial preparation across times. The architecture has no fixed
point-distance bias or hole-specific identities.

`configs/train_operator_smallscale10_512_soap_wide.py` changes only width
64->96 (493729 trainable scalars). Width 128 would have 867201 scalars. The
width-96 preset is prepared but has not been trained or memory-benchmarked.

Another small generic control is `--config.arch.time_features=16` on the SOAP
preset: it adds only 1536 weights, for 226241 trainable scalars. Explicit temporal
features are sin(j*pi*t/T), cos(j*pi*t/T). Their maximum input angular frequency
with four modes falls from 16.755 at T=.75 to 4.189 at T=3. Sixteen modes restore
the old maximum physical input frequency while retaining slower modes.
Nonlinear MLPs are not band-limited, so this observation is an optimization
hypothesis rather than a proven cause or hard representation limit. Neither
this temporal control nor the wider model has a measured accuracy benefit yet.
Existing defaults remain unchanged.

[Experiment, validation and regeneration record](../../docs/tasks/archive/confined-ch-smallscale10-operator.md).

#### Width128 and spatial-refinement controls

The opt-in `configs/train_operator_smallscale10_512_width128.py` changes only
the selected cold30k recipe's backbone width96->128 (522145->905089 trainable
parameters). It retains the512input bank, B16/Nt16, Fourier settings, SOAP,
FP64ordinary policy and physical/loss scheme. Use the canonical entry:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_512_width128.py
```

This is an experimental control, not a promoted default. A separate two-domain
native h=.035->.030 check keeps holes/IC/epsilon/delta fixed and reports common-
query P0/P1 reconstruction sensitivity. It does not replace the training bank
or establish local/FEM convergence. Live state, GPU feasibility, full-budget
results and artifact paths belong in
[the completed capacity study](../../docs/tasks/archive/confined-ch-width128-refinement.md).

`configs/train_operator_smallscale10_512_width128_beta09.py` additionally offers
a cold SOAP beta2=.9 control of the same width128recipe. It changes only
`train.soap_b2`; the pinned library's inherited `shampoo_beta=-1` also makes
the Gram-statistics EMA follow .9. Other optimizer, data, architecture, sampler
and physical settings stay fixed. Use the same `train.py --config=...` entry.
This historical option does not select the current [operator SOTA](#geometry-operator-sota).

`configs/train_operator_smallscale10_512_width128_fp32.py` is a precision-only
control of that width128/beta09 recipe on the same GPU:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_512_width128_fp32.py
```

`train.compute_dtype` explicitly selects `float32` or `float64`. Geometry,
reference validation, graph construction and the frozen residual metric are
built in FP64; finalized floating training arrays and model parameters are
cast to the selected compute dtype. FP32 uses the same FP64 initialization and
time random draws followed by rounding, with highest matmul precision. JAX x64
stays enabled for preprocessing; the FP32 network/PDE/SOAP path is separately
audited for actual dtypes. No reference values are rounded for L2 evaluation.
The local precision adapter also supplies FP32 SOAP beta scalars, preventing
transient FP64 parameter-update arrays during bias correction while global x64
is enabled. Scalar configuration literals may be cast from FP64; network,
PDE, gradient, SOAP matrices and QR computations use the chosen precision.
Hard IC/mass identities hold to the selected precision's rounding error.
Native chemical reconstruction uses an explicit degree-dependent rounding
bound for FP32; FP64 checks retain their strict thresholds. The historical FP64
recipe is preserved; the current [operator SOTA](#geometry-operator-sota) explicitly
selects FP32 after the completed capacity and online-input studies.

`configs/train_operator_smallscale10_512_width192_fp32.py` provides a moderate
capacity control of the width128 FP32 recipe:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_512_width192_fp32.py
```

It uses width192/four heads (48features per head), 96spatial Fourier directions
(192sin/cos channels), four heterogeneous cross-attention blocks and three
temporal FiLM layers: 2021185trainable parameters, excluding fixed frequencies.
The512input bank, physics/constraints/residual metric, FP32, SOAP beta2=.9,
B16/Nt16, learning-rate schedule and30k cold budget remain the same. Explicit
SOAP axis cap512 keeps the384wide feedforward axes preconditioned; cap256 and
512 have identical coverage on the128baseline. The temporal modulation output
axis remains above both caps. This jointly expands backbone/Fourier capacity
and preserves preconditioning coverage; it is not an isolated head-dimension
or width-only ablation. Load checkpoints with their saved configuration.
Defaults stay unchanged pending independent validation; run-specific results
and direct artifact paths are in the [completed capacity study](../../docs/tasks/archive/confined-ch-width128-refinement.md).

`configs/train_operator_smallscale10_512_width256_fp32_50k.py` extends that
capacity family to width256/four heads (64features per head), 128spatial Fourier
directions/256sin-cos channels and a fixed50000step cold budget. It has
3579649trainable parameters, with the same four cross-attention blocks/three
FiLM layers and SOAP axis cap512, which still covers both512wide FFN axes.
Use the canonical entry:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_512_width256_fp32_50k.py
```

All other numerical/optimizer/data settings follow the192FP32 preset. A30k
paired checkpoint and native evaluation are retained; the research workflow
also assesses it at61times to distinguish common-budget capacity from the
extra20kupdates. Final fixed50k accuracy is not a best-heldout checkpoint
selection. Resource gates, direct run paths and results are recorded in
[the width256/50k task](../../docs/tasks/archive/confined-ch-width256-50k.md).
That run was intentionally stopped at the user's request; its partial logs and
complete40kmodel/SOAP pair are retained. It is not a completed50kresult.

`configs/train_operator_smallscale10_1024_width192_fp32_50k.py` tests geometry
coverage with the same192wide model and a fixed50000step cold budget:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_smallscale10_1024_width192_fp32_50k.py
```

It reuses the existing1024input-only bank, preserving the original512training
prefix,16training monitors and the exact same8reserved physical geometries.
Validation indices become1024..1031; accepted geometry draw IDs remain16..23.
The extra512inputs contain no future fields. No training reference trajectories
are solved. Architecture, initialization policy, physics, graph/point density,
residual metric, FP32ordinary policy, SOAPbeta2=.9/cap512, B16/Nt16 and LR schedule
match the512/192control. Actual geometry-ID samples change with the larger
sampling support; temporal draws and fresh model arrays remain identical.
Retain the30kcheckpoint and61time reference evaluation for a common-budget
comparison, and report the final50kbenefit separately. Existing defaults remain
unchanged. The historical launch was stopped before30kwhen the user switched
to the online25kcontinuation below; it has no30k/50kendpoint. Feasibility gates,
direct paths and retained partial results are in the
[closed geometry-bank study](../../docs/tasks/archive/confined-ch-width192-bank-growth.md).

#### Historical FP64 width96 small-scale10 recipe

The earlier FP64 width96 recipe reaches **9.66% pooled phase relative L2**,
**10.51% terminal phase L2**, and **9.19% over61native times** on the same8
retained development geometries. The16training monitors give5.92/6.30%.
The terminal result and individual geometries are not all below10%; worst
geometry pooledphase17.26%, mean chemicalpotential21.06%. This is one
initialization and a reused development cohort, with a native graph/BDF
reference; local FEM and freshfinal/seed-stability validation remain deferred.
The current [operator SOTA](#geometry-operator-sota) supersedes this recommendation;
this named preset retains its historical settings for reproducible controls.

| Selected setting | Value |
|---|---|
| Geometry inputs / reserved validation |512 / 8 |
| Fixed initial field / terminal time | small-scale10modes / T3 |
| Backbone / heads / conditionblocks / decoderlayers |96 / 4 / 4 / 3 |
| Spatial Fourier directions / channels / scale |64 / 128 / 2 |
| Time Fourier pairs / channels / maximumfrequency |32 / 64 / 4 |
| Parameters excluding fixed Fourier arrays |522145 |
| Condition batch / positive time points |16 / 16, plus exactt0PDEweight.2 |
| Residual metric | frozen initial-mean linear stable-part resolvent |
| Physical outputs | phi only; eliminate mu through the same graph |
| Constraints | hard initialfield and weightedmass; no duplicate IC/massloss |
| Optimizer / initialLR / arithmetic / compilepolicy | SOAP / 1e-3 / FP64 / ordinary |
| Budget |30000updates, one global time network |

The query encoder/spatial preparation is reused across time; temporal
partials use JVP. Boundary and initialfunction inputs use the generic
heterogeneous point streams, with no fixed query-hole distance bias. The
known-initial-RHS feature is off. Reference future fields never enter the
optimizer payload.

A fresh numerical-recipe entry, without parent-run dependencies, is:

```bash
source .venv/bin/activate
python examples/confined_ch2d/train.py --config=examples/confined_ch2d/configs/train_operator_smallscale10_best.py
```

Its configuration matches the measured winner after removing only runtime
seed/runname and continuationprovenance. The reported checkpoint was measured
as fullstate10k+10k+10k stages; a separate cold30000run is not claimed. Existing
single-IVP time-window defaults and all historical controlpresets are preserved.
The selected direct checkpoint is
`/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-30k-retry-01/model_final.eqx`,
with its exact template in the adjacent `config.json`. Selected-lineage walltime
is248.79min, warmupdate428.30ms; that lineage total excludes the earlier
incomplete30kattempt. All checkpoints/logs from the interruption remain available.
CPU24fieldreplay differs from GPU by<=4.34e-15; native61time massdrift<=1.44e-16
and sampledenergy has no increase. See the
[completed optimization record](../../docs/tasks/archive/confined-ch-fourier-optimization.md)
and `artifacts/index.yaml` for all controls, paths and validation receipts.

#### Fourier-width optimization toward10%

The `configs/train_operator_smallscale10_512_fourier.py` control keeps
the64wide backbone and expands spatial frequency vectors24->64 (128sin/cos
channels), temporal frequencies4->32 (64channels), and spatial Gaussian scale
1->2. These feature-count/bandwidth changes form one optimization bundle, not an
isolated ablation. Trainable parameters grow224705->243649. SOAP's max dimension
is256so the expanded133wide input axis remains preconditioned. Other physical,
loss, data and sampling choices remain. Target: roughly10% native validation
phase error. Its measured20k result is40.82%, so this wider/broader bundle alone
does not improve the prior39.56% SOAP result.

New operator runs additionally save `optimizer_final.eqx` alongside the final
network, without changing updates. Historical runs remain loadable without this
new output. At each configured intermediate checkpoint, new runs also save
the matching `optimizer_<step>.eqx` and publish `checkpoint_<step>.json` after
both model and optimizer payloads are complete. These backups preserve SOAP
state; a lone model or partial file is insufficient for exact recovery. They
remain intermediate snapshots, not completed-budget results.
An explicit continuation can import a complete new-run parent;
there is no missing-state fallback or in-place overwrite.

The explicit `train_operator_smallscale10_512_curriculum.py` research control
keeps that Fourier architecture and changes only the residual-time measure to
`log_uniform`, `time_scale=.05`, with an exponentially expanding training horizon
from `.15` to `3` over the first `8000` updates. The remainder uses the full
physical interval. It remains one global network with the same hard IC/mass;
there are no time windows, previous predicted endpoint inputs or future-field
training labels. Four positive times plus `t=0` keep fixed shapes. This changes
the optimization measure without inverse-density compensation. Its measured
20k native phase error is35.42% (terminal35.73%); it remains above the10%target. Configs without
`sample.temporal` preserve the historical operator random-time draws exactly.

Point operators also accept the optional `arch.time_max_frequency` (finite,
`>=1`, at least two time features). For example, 32 frequency pairs with maximum
4 give 64 Fourier channels over the original four-frequency bandwidth, rather
than extending to frequency32. Frequencies then span `[1,maximum]` uniformly in
`pi*t/T` units. Omission retains the historical integer frequencies exactly;
the metadata adds no learned weights or checkpoint array leaves. This allows
width and frequency range to be tested separately.

The `train_operator_smallscale10_512_stable.py` control retains
wide spatial features and bounds the 32 temporal frequency pairs at maximum4.
It returns to the historical uniform full-horizon time sampler and tests a
separate input-only `operator.metric=linear_stable_resolvent`, with six explicit
CPU construction workers. It changes the residual norm, not the physical model.

For the positive regional operator \(A=-\Delta_\delta\), the eliminated CH law is

$$
\phi_t+M A\{\beta(\phi^3-\phi)+\epsilon^2 A\phi\}=0.
$$

Freeze the homogeneous state at the known conserved mean
\(m_0=\sum_iV_i\phi_{0,i}/\sum_iV_i\). Linearizing gives
\(\eta_t+M A[\beta(3m_0^2-1)\eta+\epsilon^2 A\eta]=0\).
With \(S=V^{1/2}AV^{-1/2}\), its modal decay rates are
\(q(\lambda)=M\lambda[\epsilon^2\lambda+\beta(3m_0^2-1)]\).
The proposed physical residual map is

$$
P=V^{-1/2}\big[I+T\,q(S)_+\big]^{-1}V^{1/2},
\qquad q(S)_+=U\,\operatorname{diag}(\max(q(\lambda_j),0))\,U^T.
$$

A growing mode has \(q<0\) and keeps weight1; a stiff decaying mode receives
\(1/(1+Tq)\). All spectral weights are strictly positive, so the residual has
the same zero set. This differs from the previous biharmonic-only resolvent,
which can also downweight physically growing modes. The homogeneous
linearization is a frozen preconditioner, not a nonlinear reference solution
or an optimality claim. Its three-node spectra, finite-difference CH Jacobian,
volume symmetry and constant mode are independently checked. Matrices are
constructed once from the initial field and checked graph before training;
no future fields or duplicate dense bank are needed. `prepare` still uses the
original bank preset; this derived metric is an explicit training-only option.
Its completed20k result is22.73% pooled phase error and24.30% terminal error.
This is a combined bandwidth/metric control, not an isolated attribution to
either change.

Completed controls on the same physical case and original8native validation geometries
(512training inputs unless explicitly marked1024):

| Configuration | Updates | Phase L2, 5 saved times | Terminal phase L2 | Phase L2, 61 times |
|---|---:|---:|---:|---:|
| Original SOAP |20000|39.56%|39.17%|40.47%|
| Wider/broader Fourier |20000|40.82%|40.68%|41.63%|
| Wider Fourier + global log-time curriculum |20000|35.42%|35.73%|35.89%|
| Bounded temporal bandwidth + stable-part norm |20000|22.73%|24.30%|21.87%|
| Same norm + logarithmic decoder/lifting |20000|22.61%|24.07%|21.84%|
| Linear decoder/lifting + unit output gain |20000|22.82%|24.26%|22.01%|
| Unit-gain baseline +16geometries/16positive times |10000|18.95%|20.26%|18.27%|
| Same B16/Nt16, complete-state budget extension |20000|15.33%|16.47%|14.72%|
| Same B16/Nt16, additional5k with ongoing LR x.3 |25000|14.38%|15.45%|13.80%|
| Same cold10k B16/Nt16, backbone width96 |10000|15.33%|16.43%|14.77%|
| Same cold10k, known-initial-RHS condition |10000|19.14%|20.69%|18.23%|
| Width96, complete-state budget extension |20000|11.38%|12.28%|10.93%|
| Width64,1024input-only geometry bank |10000|20.21%|21.79%|19.32%|
| Width96, fixed30k final after exact-state recovery |30000|**9.66%**|**10.51%**|**9.19%**|

The61time independent native validation uses the same physical meshes and BDF
solver, passes time refinement (max phase6.58e-8/mu4.81e-7 relative difference),
and reproduces the original5anchors within3.49e-15. Its arrays are isolated from
the training bank and never enter optimizer payloads. The completed
checkpoints conserve mass and decrease sampled energy at these61times. The new
models replay all24saved fields on CPU within6.42e-15. These are one-seed
results on reused development geometries, not a fresh final generalization test.
The roughly10% pooledphase aim is reached by the selected30k recipe above. The two output/time-parameterization controls give
similar accuracy; the simpler linear-time coordinate remains preferred.
At the same10k budget, increasing only backbonewidth64->96 improves18.95/20.26%
to15.33/16.43%, with fixed Fourier arrays and actual geometry draws preserved.
Coldwall4981.73s and warm428.82ms versus2786.34s/219.25ms for64wide. All24
fields replay onCPU within5.04e-15. It remains a one-seed development result.
Adding the optional known-initial-RHS encoderchannel at64width gives19.14/20.69%
at10k; this does not beat the matched64width baseline, so it is not recommended.

The earlier64width25k model gives phase9.32/9.90% on16training monitors and14.38/15.45%
on8unseen domains (pooled/terminal). All512training PDEobjectives also show a
12.56x average development/training gap; this squared-residual ratio is not an
L2-error ratio. It motivates a separate1024input-bank coverage control. The
optional `train_operator_smallscale10_1024_batch.py` preserves the exact original
16monitors/8native references. Preparation takes129.56s for512newoperator-only
inputs; all520existing records are reused unchanged through explicit hardlinks,
with zero PDE/FEMsolves. This is a larger finite prior bank, not online geometry
construction. `precompute.reuse_mode=hardlink` is explicit, fails if unsupported,
and requires the original bank paths to remain immutable. The historical default
copy mode remains available. Its matched10k control gives20.21/21.79%phase,
worse than512inputs18.95/20.26%; trainingmonitor error also increases15.31->18.30%.
Both use160000geometry draws, so visits per input roughly halve. A smaller
train/development gap under these conditions is not sufficient evidence of
bettergeneralization. No larger-bank default or accuracy gain is claimed.
Actual1024GPUpeak32.44GB/live18.05GB remains below38.18GBallocatorlimit.

The earlier width-only20k comparison uses96width/512inputs, preserving the
same wide/bounded Fourier inputs, stable-part residual norm, B16/Nt16 andSOAP.
Native phase11.38/12.28%,61time10.93%; trainingmonitors8.10/8.64%. Its complete
checkpoint is
`/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-continue-01/model_final.eqx`.
Relative matched64width20k15.33/16.47%, increasingcapacity helps; it is not a
claim of lower walltime (cumulative166.20min vs93.49min). CPU24fieldreplay
4.22e-15, hardmass/61timeenergy and no-label gatespass; nativeonly, same8reused
development geometries/oneinitialization, no freshfinaltest orFEMclaim.
The explicit `train_operator_smallscale10_512_wide_30k.py`
continues the exact complete20k state/counters and unchangedLRschedule to30k;
its first attempt exited without a final result after26500logged evaluation
steps (exitcauseunknown, noOOM/Pythontraceback). The retry preserves that run
and starts from the complete20k model/SOAPstate with the same30k numerical
configuration. The retry completed30000 and fullvalidation, with its measured results listed above. Existingcanonical time-window defaults remain
unchanged; these are the separate geometry-operator research presets.

The25k state-preserving recipe costs7251.24s cumulatively, including both parent
stages. Its extra5k updates and LRchange are combined; no unchanged-LR25k control
isolates a learning-rate effect. Full512physics diagnostics show a training/
development residual gap, so decreasing training loss alone is insufficient
evidence of transfer to new geometries. All inputs retain the same physical
problem and original native validation cohort.
`train_operator_smallscale10_512_bounded.py` is a retained unrun bandwidth-only
curriculum control, not a measured recommendation.

For a complete parent run that has `optimizer_final.eqx`, the optional
`train.continuation.parent_run` names its absolute directory. Start a **new**
run with a strictly larger `train.iterations`, which denotes the total global
budget. By default all numerical settings, optimizer/sampler/seed and physical
inputs must match the parent; the option is rejected otherwise. The loader restores both
model and optimizer, retains global fold-in counters, and checks the imported
physical fields against every saved parent prediction before any new update.
`continuation.json` records direct parent paths; `model_initial.eqx` and
`optimizer_initial.eqx` preserve the imported state. Summary timing separates
this stage from cumulative parent-plus-child time. No time windows, optimizer
reset, schedule restart or new seed are involved. Fresh runs retain their old
initialization and draw sequence. Real SOAP/PDE serialization and a tiny
end-to-end trainer test verify the refresh boundary, global draws, initial
replay and cumulative timing. The10k B16/Nt16 run is the first full-benchmark
parent: the imported model and complete optimizer files match its final files
byte for byte, and all24physical predictions pass replay before new updates.

A narrowly scoped fine-tuning option is
`train.continuation.learning_rate_factor`, a finite value strictly between 0 and 1.
It allows only `train.lr_init` and `train.lr_min` to change, and both must equal
their parent values times the declared factor. Every other numerical setting
remains identical. Both states and the schedule counter are imported unchanged;
the ongoing schedule is scaled, rather than restarted. `continuation.json`
records the factor and both before/after values. The default branch still
rejects learning-rate changes. The prepared
`configs/train_operator_smallscale10_512_batch_finetune.py` preset uses.3
after the complete20k parent and requests25000total updates; it is not a measured
accuracy improvement until run. Equal-gradient SOAP updates are independently
checked to scale by that factor while the full adaptive state/counter stays
identical.

The optional `phase.lifting=logarithmic_additive`, with required positive
`phase.initial_time_scale=tau`, reuses the single-IVP time-coordinate convention
in the geometry operator:

$$
s(t)=\frac{\log(1+t/\tau)}{\log(1+T/\tau)},\qquad
\phi_\theta=\phi_0+s(t)\left[N_\theta(x,s(t),g)-\langle N_\theta\rangle_V\right].
$$

Use this same coordinate in decoder and lifting, including off-grid queries.
For omitted `phase`, the historical expression `s=t/T` is unchanged. The
physical-time JVP differentiates through the coordinate, so the physical PDE
and residual metric do not change. This is parameterization, not division by a
time-map Jacobian. Both coordinates vanish at0 and equal1 atT; they preserve
hard IC/mass and the free physical solution class. The bounded Fourier range is
in the chosen decoder coordinate: the earlier `pi*t/T` statement refers to
linear-time configurations.

The separate `train_operator_smallscale10_512_logtime.py` candidate adds only
this coordinate (`tau=.05`) to the stable-part preset. The initial RHS implies
required centered raw output RMS2.72..2.96 under linear lifting, versus.186..203
under log lifting, a14.595fold reduction. That follows from
`raw_c(0)=phi_t(0)/s'(0)` using only prescribed initial data and its graph RHS,
not evolved reference fields. It addresses an early-time scale mismatch without
prescribing a later interface. Closed-form initial derivatives, mass, query
consistency and physical-time finite differences pass. Its measured result is
listed above; the smaller initial scale requirement alone did not improve
overall accuracy materially.

The optional `loss.initial_residual_fraction` explicitly fixes the fraction of
the PDE objective evaluated at the initial node:

$$
L=\alpha\,\ell_{\mathrm{PDE}}(0)
 +(1-\alpha)\frac{1}{N_t}\sum_{j=1}^{N_t}\ell_{\mathrm{PDE}}(t_j).
$$

This is a PDE residual at0, distinct from the already exact initial condition.
Omission retains the historical mean over all nodes, including0. For the
original four positive times, `alpha=.2` reproduces that objective. When
increasing the number of uniform stratified positive samples, retaining `.2`
preserves the expected objective instead of implicitly reducing initial-time
weight. The explicit `train_operator_smallscale10_512_batch.py` research preset
uses16geometries and16positive times, a64wide model, unit output gain and
10000cold updates. It changes the sampling counts relative to the unit-gain
control, retaining the expected objective. Shared spatial preparation and a
common time-query set are retained. At10k its cold time is2786.34s and warm
update time219.25ms; the 512inputs are all visited268..369times. Geometry-response
error is40.25%, amplitude80.50% and correlation.923 on the same common probes.
Sample-count changes must be disclosed when comparing update counts or wall
time. `train_operator_smallscale10_512_batch_continue.py` is its explicit
full-state extension to20000total updates in a new run. Run-specific progress
and the exact resume point belong in the active task record.

An additional optional encoder control is
`configs/train_operator_smallscale10_512_initial_rhs.py`. For the first-order
evolution law \(\phi_t=F_g(\phi)\), the prescribed initial field also determines
the known instantaneous derivative. Here

$$
\mu_0=\beta(\phi_0^3-\phi_0)+\epsilon^2A_g\phi_0,
\qquad v_0=-T M A_g\mu_0.
$$

With `operator.initial_rhs_condition=True`, the domain-function encoder receives
`(x,y,phi0,v0)`, requiring `arch.input_dims=(5,4)` for the unchanged boundary and
domain streams. The multiplierT expresses the derivative over the characteristic
time horizon. These are known input functions, computed by two graph actions
without time integration or future fields. Coordinate queries, output lifting,
hard IC/mass and the PDE loss remain the same. This tests whether exposing the
discrete operator's initial response helps transfer to new geometries, rather
than requiring the encoder to infer that response from the point cloud alone.
It applies as an input representation for first-order evolution laws; accuracy
benefits remain to be measured. The option is off by default and currently
requires an unaugmented heterogeneous training input; transformed inputs would
need their known tangent recomputed. Checkpoints use their saved channel schema.

## Online geometry inputs with a rolling pool

The experimental `configs/train_operator_online_rolling.py` imports the explicit
paired25k checkpoint of the width192/1024input control and continues to50000total
updates. Start it through the canonical entry:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_online_rolling.py
```

The architecture, FP32 arithmetic, SOAP moments/schedule, hard IC/mass, eliminated
chemical potential and frozen linear-stable residual metric are retained.
`train.continuation.parent_run` and `train.continuation.checkpoint` are absolute
paths to the original run and its published `checkpoint_025000.json`; the marker
must name both complete model and optimizer files. The learning rate and global
time/slot sampling counters continue from25000. Parent files are never replaced.

`operator.online` explicitly declares6CPUworkers,32slot refreshes every100updates,
two refresh batches of prefetch,1536padded points,128padded neighbors and the
25000origin step. The pool initially holds1024old training inputs. The first
refresh is before update25001, followed by25101,25201,etc. Each refresh replaces
the next32slots cyclically. This introduces8000new geometries through50000while
keeping storage bounded; individual generated inputs are reused across updates.
The two-hole parameters use the same continuous valid prior. A stateless draw
identity and the runtime seed determine each geometry; FIFO consumption at fixed
updates makes training independent of worker completion order. A slow producer
causes an explicit wait. Oversized valid meshes fail instead of changing the prior
by resampling or cropping. The padding limits are checked envelopes, not a proof
of a global mesh-size bound for every parameter draw.

Workers generate only geometry, quadrature, known initial/encoder inputs, graph
coefficients and the current residual metric. Gmsh runs in its explicit isolated
meshing interpreter without importing DOLFIN or solving a FEniCS/MPI trajectory.
No future fields enter a training batch. Temporary mesh files are removed after
construction; full generated operators are not accumulated on disk. GPU pool
replacement is a functional donated-array update outside parameter AD; all times
retain the shared spatial grid and preparation within each sampled geometry.

The16original training-domain monitors and8reserved development domains are
stored separately from the replaceable pool. The reported training-domain L2
therefore measures those frozen old domains, not the current rolling pool. L2
uses the unchanged native nonlocal references. No accuracy improvement over a
fixed-bank continuation is established merely by enabling this sampler.

`online_inputs.jsonl`, `online_refresh.jsonl` and `training_draws.npz` record
physical geometry identities, refresh steps and actual sampled slots/geometries/
times. Intermediate paired checkpoints also publish `online_pool_<step>.json`;
restoring one reconstructs only the bounded active pool from its stateless draw
identities before continuing. Pending unused prefetch jobs need no serialization.
The initial model/SOAP bytes and physical-field scores are checked against the
imported checkpoint. The first25production updates check finite compute state
and record GPU memory/timing in the same process, avoiding duplicate compilation.
Run-specific results and status remain in the active task record.

For a fresh width256/Fourier256-channel run with online inputs from the first
update, use:

```bash
python examples/confined_ch2d/train.py \
  --config=examples/confined_ch2d/configs/train_operator_online_width256_cold50k.py
```

This preset has no `train.continuation`; `operator.online.origin_step=0` starts
50000 new updates with fresh model/SOAP/schedule state. The four heads each have
64 features; FP32 ordinary compilation, B16/Nt16, SOAP beta2=0.9 and the physical
scheme remain unchanged. The initial1024 known-input geometries are reused,
then32slots are refreshed before update1 and every100updates thereafter
(16000 fresh inputs installed through50000). Six CPUworkers and64pending inputs
bound generation memory, without saving a growing geometry library or solving
training trajectories. Fixed16/8evaluation payloads remain separate.

Complete paired checkpoints are saved every5000updates (10groups), together with
initial/final model and SOAP files and a pool-state marker for each checkpoint.
Every saved group is retained. Logs and actual input identities retain the full
training history. The completed run is now the accepted
[geometry-operator SOTA](#geometry-operator-sota) and is selected by
`configs/train_operator.py`. It changes network capacity and sampling together;
the comparison does not isolate either change's contribution.
