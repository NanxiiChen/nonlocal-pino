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
