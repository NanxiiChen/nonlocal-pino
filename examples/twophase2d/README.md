# Periodic phase-field transport pilot

This example exercises reference generation, mixed PINN training and local/nonlocal
spatial differentiation for a prescribed incompressible velocity. It is the first
stage towards two-phase flow. Velocity is given; Navier–Stokes, pressure, gravity
and variable density are not solved here. This is a small verified pilot, not a
tuned SOTA preset or a reproduction of Qiu2022's wall-vortex/bubble cases.

For the shape-changing vortex, the currently recommended tested preset is
`configs/train_vortex_conserved.py`, with direct phase output by default and
optional `--config.phase.map=inverse_cubic`; see the recommendation and
validation section below. The base remains the original constant-flow route check.

## Explicit problem

The periodic square is \([0,1)^2\), with time \(t\in[0,1]\). Let \(\phi=-1\)
denote the droplet phase, \(\phi=+1\) the surrounding phase, and \(q\) the scaled
chemical potential. The local model is

$$
\phi_t+\mathbf v\cdot\nabla\phi=M\Delta q,\qquad
q=\phi^3-\phi-\varepsilon^2\Delta\phi.
$$

The pilot uses \(\mathbf v=(0.2,0.1)\), \(M=0.002\), \(\varepsilon=0.05\).
These are explicit nondimensional pilot choices, not parameters inferred from
the attachment. The smooth, exactly periodic droplet-like initial condition is

$$
s(x,y)=\sin^2\!\bigl(\pi(x-0.35)\bigr)+\sin^2\!\bigl(\pi(y-0.5)\bigr),\qquad
\phi_0=\tanh\!\left(\frac{s-0.3}{0.12}\right).
$$

Its zero contour is a rounded droplet rather than an exact Euclidean circle.
No wall/contact-angle/flux boundary closure is needed on this periodic domain.
Constant velocity is divergence free. Continuum local-model phase mass is
conserved; its free energy decreases because constant translation supplies no
energy on a periodic cell. Finite-grid/time/PINN conservation is measured.

## Reversed-vortex deformation variant

`configs/train_vortex.py` adds a harder shape-changing case on the same periodic
square. Time is \(0\le t\le2\). The prescribed velocity follows the form of
Qiu2022 Eq23/24, with \(U=2\), \(T=2\), \(L=1\):

$$
v_x=-U\sin^2(\pi x/L)\sin(\pi y/L)\cos(\pi y/L)\cos(\pi t/T),
\qquad
v_y=U\sin(\pi x/L)\cos(\pi x/L)\sin^2(\pi y/L)\cos(\pi t/T).
$$

It is spatially periodic and divergence free. Velocity vanishes at \(t=1\),
then reverses; integrated advection reverses by \(t=2\). CH diffusion continues
throughout, so the final reference need not equal the initial condition.
Prescribed strain can supply energy: the local balance is
\(dF/dt=-M\int|\nabla q|^2-\int q\,\mathbf v\cdot\nabla\phi\).
Energy monotonicity is therefore not an acceptance condition for this variant.

The explicit pilot choices are \(M=10^{-4}\), \(\varepsilon=0.02\), initial
center \((0.5,0.75)\), level0.2 and width0.06 in the periodic IC above. Its
axis zero-contour radius is \(\arcsin\sqrt{0.2}/\pi\approx0.148\).
This is a periodic rounded droplet, not an exact Euclidean circle. The attached
paper's printed IC omits its stated radius and does not specify a complete
periodic reference setup; these choices are declared independently. This is
**a periodic variant**, not a repair or exact reproduction of the paper.

Both derivative controls use64x64 points,4 shared times,4x64 GELU layers, the
same embeddings/SOAP/FP32/ordinary compilation and5000 updates. The only hard
phase constraint is \(\phi=\phi_0+(t/T)N_\theta\); no future deformation is
encoded. Nonlocal horizon0.04 uses the same moment calibration. Independent
references use graph64, local128/256 and dt2e-4/1e-4, with relative-L2 tolerance
0.002 at **every** saved time, including maximum deformation. Variable
advection is explicit conservative Fourier flux divergence in IMEX-BDF2;
biharmonic diffusion remains implicit. Constant-flow splitting is unchanged.

```bash
source .venv/bin/activate
python -m examples.twophase2d.solver --config=examples.twophase2d.configs.train_vortex
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex.py --config.log.run_name=my-vortex-nonlocal
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex.py --config.nl.method=local_ad --config.log.run_name=my-vortex-local-ad
```

Reference path: `/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz`.
The evaluator also accepts these run paths and plots0/0.5/1/1.5/2. Use matched
references for optimization error, and the common local reference to compare
field accuracy; the finite-horizon model carries a separate discretization bias.
The explicit velocity configuration is either the historical constant two-vector
or a record with required `kind`, `amplitude`, `period` (the \(T\) above).
Existing saved constant-flow configurations/reference identities still load.
The [completed vortex record](../../docs/tasks/archive/twophase-vortex-pilot.md)
contains the paired5000-update results, full fields/curves, reference checks and
limitations; the original constant-flow default is retained.

## Spatial controls

`nl.method=local_ad` uses AD Hessians for the two negative Laplacians in
mixed form. The default `nl.method=nonlocal` replaces those Laplacians by a
periodic finite-horizon operator. Advection is a separate configuration choice:
legacy base/vortex presets use AD for the point-network phase gradient;
the recommended conserved/consistent presets set `advection=spectral_flux`
and compute the Fourier divergence of nodal velocity-phase flux instead.
Both diffusion controls retain the same selected advection scheme. Thus the
current recommended spatial method combines Fourier first derivatives with
nonlocal Laplacians, rather than nonlocal first derivatives. Fourier evaluation
uses all periodic nodes but approximates a local derivative, not a finite-horizon
interaction. Time derivatives use AD through a JVP of the physical field,
including the time-dependent mass projection. No NS/pressure/continuity
residual is trained: velocity is prescribed analytically, not a network output.

Known limitation, deferred: the current Fourier first derivative assumes a
uniform periodic grid and cannot be applied directly to general nonperiodic
walls or arbitrary geometry. A hard boundary constraint on the network does
not remove this operator assumption. The current nonlocal operators also wrap
periodically; switching to their research option does not solve general boundary
treatment. Retain FFT first derivatives and nonlocal Laplacians for this case;
further boundary work is deferred until explicitly resumed. See the
[deferred issue](../../docs/issues/twophase-fft-periodic-boundary.md).

$$
(L_\delta f)_i=\sum_jw_j(f_i-f_{i+j}),\qquad
w_j=a_h\frac{120h^2}{\pi\delta^4}\left(1-\frac{|z_j|}{\delta}\right)^2,
\quad0<|z_j|<\delta.
$$

Indices wrap periodically. For the square isotropic offset set,
\(a_h=2/\sum_j\widetilde w_jz_{j,x}^2\), so both diagonal second moments are2.
Taylor expansion then gives \(L_\delta\approx-\Delta\); paired positive weights
make it symmetric positive semidefinite and annihilate constants. This explicit
quadrature calibration differs from historical examples' stored coefficients;
none of those operators/defaults are changed. Here \(N_x=N_y=32\), \(h=1/32\),
\(\delta=0.08\). Finite-horizon and optimization errors remain distinct.

Both networks output \((\phi,q)\) and minimize the original mixed residuals:

$$
r_\phi=\phi_t+\mathbf v\cdot\nabla\phi+ML_\delta q,\qquad
r_q=q-\phi^3+\phi-\varepsilon^2L_\delta\phi.
$$

The local control substitutes \(-\Delta\) for \(L_\delta\). It also avoids direct
fourth-order AD, so the comparison separates mixed formulation from derivative
method. Unprojected legacy presets include an explicit phase-mass mean-square
loss with weights1/1/1 for phase/chemical/mass. Presets using exact grid-mass
projection have only the two PDE losses with weights1/1; they do not calculate
an additional mass penalty. No inverse-Jacobian residual transformation is used.

## Network and training

The network has3 hidden layers of48 GELU units, integer periodic sin/cos spatial
features through mode3, and8-dimensional random temporal embedding(scale1).
Only the prescribed phase IC is hard constrained:
\(\phi_\theta=\phi_0+tN_{\theta,\phi}\); \(q\) is an independent output.
There is no advected-coordinate input or prescribed evolved interface in the
trial function. Periodic features enforce all spatial periodicity exactly;
there is no duplicated BC/IC loss. The phase output is not clipped to[-1,1].

Both controls use the full32x32 spatial grid and4 shared random times per update,
time JVP, SOAP(lr1e-3,beta0.9/0.99,refresh5,clip10),5000 updates,FP32 and ordinary
GPU compilation. Seed is run provenance (`--seed`,common default42), omitted
from strategy configs. Reference snapshots are evaluation-only. Existing
explicit run names fail instead of overwriting/resuming a model.

```bash
source .venv/bin/activate
# CPU-only reference; no FEniCS dependency:
JAX_PLATFORMS=cpu python -m examples.twophase2d.solver
# Run the two GPU controls individually:
XLA_FLAGS='' python -m examples.twophase2d.train --config.log.run_name=my-nonlocal
XLA_FLAGS='' python -m examples.twophase2d.train --config.nl.method=local_ad --config.log.run_name=my-local-ad
# Direct-script train.py is also supported.
# Independent CPU reload and figures, using direct saved paths:
JAX_PLATFORMS=cpu python -m examples.twophase2d.evaluate --run=/root/autodl-tmp/tf-logs/twophase2d/my-nonlocal --run=/root/autodl-tmp/tf-logs/twophase2d/my-local-ad --output-dir=reports/my-transport-check
```

Default reference: `/root/autodl-tmp/data/twophase2d/results_periodic_pilot.npz`.
Default run root: `/root/autodl-tmp/tf-logs/twophase2d`.
Use `solver --output=<new-path>` and `train --config.reference.path=<same-path>`
for an alternate reference location; existing files fail explicitly. Models and
arrays load by direct paths, without hashes.

## Independent reference and acceptance

`solver.py` imports no JAX/PINN operator. NumPy/SciPy Fourier collocation and
IMEX-BDF2 integrate the phase field, using an implicit linear biharmonic/advection
part, explicitly extrapolated nonlinear reaction and an IMEX-Euler starter.
The independent nonlocal Fourier symbol comes from the specified kernel, not
training operator arrays. This unstabilized pilot integrator is checked by
actual refinement; no unconditional energy-stability claim is made.

Reference checks compare dt2e-4/1e-4 for both models and64/128 spatial grids for
the local model. The refined local solution is sampled onto the common32 grid;
the graph target uses the exact32-grid nonlocal symbol and time refinement.
Time/spatial refinement pooled and terminal phase L2 must be below0.002; zero-mode
mass drift below1e-10. Graph-versus-local discrepancy is recorded separately.
AD versus Fourier advection and unresolved interpolation contribute to the
finite-grid comparison; a graph target is not a continuum-exact reference.

Training logs report terminal/pooled phase L2, local-target error, interface-band
error, chemical-potential RMS and droplet-area drift on all21 reference times.
Initial/final/checkpoint models, config/source snapshots, loss/metric JSONL,
prediction arrays and summary are saved. Time0 velocity is prescribed here;
there is no learned velocity/pressure accuracy claim.

The formulation follows standard advective CH, as used in
[Qiu et al2022](https://doi.org/10.1063/5.0091063); the mixed-AD comparator is
also motivated by the same-group
[MIM follow-up](https://doi.org/10.6052/0459-1879-22-253).
Semi-implicit Fourier phase-field methods have established precedent in
[Chen and Shen1998](https://doi.org/10.1016/S0010-4655(97)00115-X).
The exact pilot data, moment correction and BDF2 split above are specified and
independently checked here; they are not asserted to reproduce those papers.
The [completed pilot record](../../docs/tasks/archive/twophase-transport-pilot.md)
contains full paired results, common-target limitations, timings and figures.
Run-specific detail stays in that task rather than an example STATUS file.

## Bounded residual-preconditioning experiment

`configs/train_vortex_residual.py` preserves the vortex PDE, reference, original
network/IC and5000-update budget; it adds explicit `residual.mode` and
`residual.eta`. Modes are `original`, `transport` (only phase transport), and
`both` (transport and chemical). Configurations without the optional record are
the historical original-residual schema; an existing record requires both fields.

Unlike melting's Allen–Cahn model, CH reference phases can exceed[-1,1] and
\(Lq\) does not factor \(1-\phi^2\). Strict tanh output would exclude part of
this verified solution. This experiment instead keeps the output unchanged and
uses an empirical bounded inverse-Jacobian continuation, with \(\eta=0.1\):

$$
w_\eta(\phi)=\frac{\sqrt{1+\eta^2}}
{\sqrt{\eta^2+\min\{(1-\phi^2)^2,1\}}},\qquad
1\le w_\eta\le\frac{\sqrt{1+\eta^2}}{\eta}.
$$

Selected residuals become \(\widetilde r=w_\eta r\). Positive weights preserve
pointwise PDE zeros but change the optimization norm; gradients pass through the
weight. The cap limits squared amplification to101, and the lower bound prevents
large excursions from reducing weights. This is not exact tanh latent training,
not a phase constraint and not an interface indicator: weights peak near the
pure phases. Mass loss stays unchanged. Whether this improves interface fields
is measured rather than assumed. The original vortex/base defaults stay intact.

```bash
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_residual.py --config.residual.mode=both --config.log.run_name=my-vortex-both
```

`configs/train_vortex_latent.py` provides a separate representation control that
allows all real phase values. Define \(f(u)=\tanh u+\alpha u\), \(\alpha=0.1\),
and compute \(u_0=f^{-1}(\phi_0)\) with8 fixed Newton steps. With
\(s=(t/T)N_{\theta,\phi}\), the exact numerical IC is preserved by

$$
\phi_\theta=\phi_0+\tanh(u_0+s)-\tanh(u_0)+\alpha s.
$$

There is no subsequent-motion prior. \(f'\ge\alpha\) and \(f\) is unbounded,
so reference excursions are representable. `residual.mode=original` tests only
this representation; `map_transport` / `map_both` multiply the selected original
physical residuals by \((1+\alpha)/f'(u_0+s)\). This is the exact inverse-map
Jacobian up to the explicit constant normalization, bounded by11. The chemical
potential remains independent. Spatial derivatives include the IC map terms;
the implementation differentiates the full physical output. `residual.eta` is
used only by the physical-field bounded modes above. The map preset validates
leak in[0.1,1] and at least8 inverse iterations; value/implicit-derivative and
residual-gradient checks cover the selected settings. Training still enters
`train.py`; select the new config path, then the residual mode explicitly.


The [completed six-control experiment](../../docs/tasks/archive/twophase-vortex-residual.md)
keeps both negative and positive outcomes. The unbounded map improves interface
representation in this single-seed test. Extra inverse-Jacobian residual scaling
trades interface sharpness against full-field/chemical/mass errors; physical-field
bounded weighting worsens this pilot. These remain explicit research controls,
with unchanged canonical defaults and physical reference. To reproduce the
sharper-interface representation-only control, use:

```bash
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_latent.py --config.residual.mode=original --config.log.run_name=my-vortex-latent-original
```

Interface diagnostics distinguish phi values in a reference band, zero-contour
distance and the width proxy area(|phi|<0.9)/zero-contour perimeter. The last is a
finite-grid diagnostic, not an exact normal interface thickness. Phase-area drift
uses the soft integral(1-phi)/2. Compare all metrics, not only full-domain L2.

## Convexified chemical coordinate and operator metrics

The optional `phase.map=inverse_cubic` uses the strictly monotone gradient
of the convexified potential, with coordinate \(u=\phi+\phi^3\), with analytic inverse

$$
g(u)=\frac{2}{\sqrt3}\sinh\!\left[\frac13\operatorname{asinh}\!\left(\frac{3\sqrt3}{2}u\right)\right].
$$

For \(F=(\phi^2-1)^2/4\), this is \(u=F'(\phi)+2\phi\),
so \(F'(\phi)=u-2g(u)\). Its derivative is
\(g'=1/(1+3\phi^2)>0\). It allows all real phase values, including CH
overshoots, with no iterative inverse or leakage coefficient. This is an
explicit trial representation, not a proven optimal coordinate. Set
\(u_0=\phi_0+\phi_0^3\), \(s=(t/T)N_{\theta,\phi}\), and use
\(\phi=\phi_0+g(u_0+s)-g(u_0)\). This preserves exactly the same IC and
encodes no evolved interface or reference values.

`configs/train_vortex_cubic.py` tests this map with the original residual.
`configs/train_vortex_schur.py` tests an operator metric on the additive map;
`train_vortex_cubic_schur.py` combines them. With \(L\approx-\Delta\),
freeze the pure-phase curvature at \(c=F''(\pm1)=2\):

$$
A_\tau=\begin{pmatrix}\tau^{-1}I & ML\\-(cI+\varepsilon^2L)&I\end{pmatrix},
\qquad s=\operatorname{diag}(\tau^{-1}I,I)A_\tau^{-1}r.
$$

`preconditioner.kind=ch_schur` requires explicit positive `time_scale`\(=\tau\)
and `curvature`\(=c\). A periodic FFT implements the block inverse; its
spectrum comes from the actual training graph weights (or \(|k|^2\) for local
AD). The Schur denominator \(1+\tau M\lambda(c+\varepsilon^2\lambda)\)
is positive, and the zero spatial mode remains present. Only the residual
metric changes; the nonlinear PDE, advection and reference remain unchanged.
It omits advection and variable curvature from the preconditioner and is not
an exact PDE inverse or a guaranteed accuracy improvement.

`preconditioner.kind=ch_causal` instead applies the inverse frozen backward-Euler
error operator across sorted training times. For \(B=cI+\varepsilon^2L\),
\(\Delta t_i=t_i-t_{i-1}\), \(z_{\phi,0}=0\):

$$
z_{\phi,i}=[I+\Delta t_i MLB]^{-1}
\{z_{\phi,i-1}+\Delta t_i(r_{\phi,i}-MLr_{q,i})\},
\qquad z_{q,i}=r_{q,i}+Bz_{\phi,i}.
$$

The loss uses \((z_\phi/\text{time_scale},z_q)\). This is a time-coupled
residual norm; continuous AD time derivatives still define the PDE residual.
It requires explicit `sample.time_rule=open_stratified`: one FP32 draw inside
each time bin, with an endpoint margin of `4*eps_float32*Nt` preventing equal
intervals after rounding, followed by promotion to the compute dtype. The
metric has no inverse Laplacian or dropped conserved mode. Its physical
approximation limitations are the same frozen curvature/omitted advection.

`*_refined.py` configs add an explicit optional `train.refinement` record for
FP64 SciPy L-BFGS after the SOAP budget. All optimizer limits/tolerances are
required; fixed midpoint time points and full spatial grids use the same training
objective: two PDE terms under hard mass projection, or PDE/soft-mass terms for
unprojected presets. They do not use reference fields. Save SOAP and final refined models/fields
separately; `refinement_summary.json` records convergence status, evaluations
and exact times. `summary.updates` sums SOAP updates and accepted L-BFGS
iterations, which have different costs. `metrics.jsonl` includes both stages.
`train_vortex_cubic_resolved.py` increases fixed modes to8, width to96 and
training times to8; `train_vortex_causal.py` uses that same architecture/budget
with the causal metric. All presets enter through `train.py`. These research
presets leave the canonical base choice unchanged.

The metric design is inspired by
[Liu et al., residual-preconditioned PINNs](https://arxiv.org/abs/2402.00531) and
[Wen et al., preconditioned neural-operator training](https://arxiv.org/abs/2609.36216);
the CH block formulas and cubic coordinate here are independently derived,
not copied Allen–Cahn formulas or claimed new general preconditioning methods.

## Periodic-frequency correction

As of2026-10-27, parameter gradients through the case-local integer spatial
feature modes are stopped, so new cold starts retain their prescribed integer
frequencies and periodic values/derivatives. Earlier code inadvertently trained
this floating array: several saved modes became noninteger, violating the
claimed exact periodic BC. Historical results above must be read with this
limitation. Their arrays and checkpoints are preserved; direct loading retains
the stored frequencies and predictions rather than rounding them or resuming
training. Temporal Fourier frequencies remain learnable by design. Numerical
checks exercise fixed mode gradients, parameter updates and opposing edges.


## Consistent conservative nodal controls

`configs/train_vortex_consistent.py` keeps the original additive phase output
and physical residual, fixes integer periodic frequencies, and computes
advection as the Fourier divergence of the nodal flux \(\mathbf v\phi\).
The even-grid Nyquist first-derivative multiplier is zero. Its zero mode is
zero to roundoff. This is the conservative derivative of the interpolated
nodal flux; it intentionally replaces the legacy AD point gradient. Since
\(\nabla\cdot\mathbf v=0\), the continuous equation is the same, but the
finite-grid discretization changes. The existing independent reference already
uses this conservative flux. This avoids using freely adjustable network point
slopes together with nonlocal operators that only see node values.

The optional `phase_mass=exact_grid_mean` projects onto the known conserved
mass on this fixed uniform periodic grid. For provisional output
\(\widehat\phi\), the physical phase is

$$
\phi_i(t)=\widehat\phi_i(t)-\frac1N\sum_j[\widehat\phi_j(t)-\phi_{0,j}].
$$

When this hard constraint is enabled, `losses.make_loss_fn` returns only
`[phase, chemical]` loss components and does not evaluate a mass penalty.
Canonical hard-mass configs omit `loss.mass_weight`. Older serialized hard-mass
configs may retain that obsolete field; it is ignored. Unprojected presets
still require the explicit mass weight and return three components. Historical
training logs with three components are retained unchanged; new hard-mass logs
contain two. Physical mass drift remains an evaluation diagnostic.

Use `fields.predict_fields(model, points, t, config)` for physical full-grid
inference. `model(x,t)` alone is the provisional point ansatz when mass
projection is enabled. Training, JVP time derivatives, saved fields and
checkpoint replay all use the same physical wrapper. The shift has zero
spatial query derivative and its time derivative is included. The constraint
conserves grid mass to roundoff and preserves IC/overshoots; it supplies no
future interface. For a general mesh the corresponding projection needs its
quadrature weights; for a nonconserved equation it is inappropriate.

`train_vortex_conserved.py` keeps the same additive output as the consistent
preset. Enable the cubic coordinate with `--config.phase.map=inverse_cubic`.
`train_vortex_causal_conserved.py` explicitly retains the cubic coordinate and
additionally changes the residual metric. These use64units/3fixed spatial modes,
8open-stratified times,FP64,SOAP5000 then at most1000L-BFGS iterations at32midpoint times.
They use the same reference and do not include extra initial-PDE weighting.

The general principles are compatible field/derivative discretizations,
constraints derived from actual conservation laws, and controlled optimizer
refinement. FFT implementation here relies on periodic geometry; it is not a
prescription for wall or irregular-domain cases. The archived task records all
negative controls as well as the measured recommended preset.


`configs/train_vortex_initial_time.py` adds `sample.include_initial_time=True`
to the conserved-cubic original-residual preset. Exactly t=0 becomes one
ordinary PDE time in addition to the positive random/midpoint times, with the
same per-time averaging; it introduces no separate loss weight or future
solution. Nt counts positive sampled times, so this option uses Nt+1 PDE times
and, in refinement,33 rather than32. It cannot be combined with the causal
metric's strictly positive intervals.
The initial phase is smooth in this case, so PDE endpoint collocation is valid.
Pointwise inverse-map residual modes also cannot be combined with grid-mass
projection: that projection couples all phase coordinates and its Jacobian
is not the pointwise map derivative.


## Recommended deformation preset and validation

Use `configs/train_vortex_conserved.py` through `train.py`. It keeps the vortex
physics/reference above, with64x64 grid, delta0.04,4x64 GELU layers,3fixed
integer spatial modes,8open-stratified times,FP64 and ordinary GPU compilation.
The phase uses direct additive output (`phase.map=additive`) plus exact
grid-mass projection; q remains independent. The cubic coordinate is off by
default to favor a common scheme across examples. Both physical mixed residuals
have their original norm and equal weights, with conservative Fourier-flux advection. No inverse-map,
Schur/causal residual preconditioning or extra initial-time point is enabled.

Train SOAP5000 at lr1e-3 (0.95 decay each1000, clip10, betas0.9/0.99,
preconditioner refresh5), then full-grid L-BFGS at32fixed midpoint times for
at most1000 accepted iterations/2000 function evaluations. History50, line
search limit40, gradient tolerance1e-9 and function tolerance1e-14 are explicit
in the config. These are two optimizer stages with different costs; neither
run reached the tolerance stopping criterion before its iteration limit.
No reference phase/q values enter the training loss.

```bash
source .venv/bin/activate
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_conserved.py --config.log.run_name=my-vortex-conserved
# Optional cubic coordinate; all other schemes stay the same:
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_conserved.py --config.phase.map=inverse_cubic --config.log.run_name=my-vortex-cubic
# Optional independent run provenance; seed is not in the strategy config:
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_conserved.py --seed=2026 --config.log.run_name=my-vortex-conserved-2026
```

The default direct-phase preset matches the fresh additive arm below apart
from declaring its previously implicit map explicitly: pooled phi3.20%,
terminal3.62% in one seed42 full run. This choice favors simplicity, not a
claim of uniformly better accuracy. The constant-flow train_base preset also
declares additive output. To switch back explicitly, use
`--config.phase.map=additive`. Historical configs without a phase record still
load as additive; saved cubic configs retain their own explicit map. Load old
checkpoints with their saved config.json, rather than today's default preset.

The matched default checkpoint is
`/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-map-ablation-additive/model_final.eqx`,
with its sibling config.json. The saved config omits phase; its additive
interpretation is equivalent to today's explicit setting.

### Historical opt-in cubic results

The following two complete independent cubic cold starts were evaluated at
all21times on0–2 against the SAME refined local reference
(relative L2, no best-checkpoint/seed search):

| Run provenance | Pooled phi L2 | Terminal phi L2 | Pooled interface-band phi L2 |
|---|---:|---:|---:|
| seed42 | 2.84% | 2.34% | 12.84% |
| seed2026 | 2.91% | 2.56% | 12.79% |

Matched finite-horizon-reference pooled/terminal errors are3.09%/2.54% and
3.21%/2.85%, respectively. Graph/local reference discrepancy is1.28% pooled
and1.17% terminal; this is separate from optimization error, not a strict
lower bound on finite-error local comparisons. Exact grid mass is imposed to
roundoff. The1% field target is not reached, and interface errors remain larger
than whole-field errors. This is evidence from two seeds of one prescribed-
velocity CH case, not a general accuracy/stability theorem or solved CHNS.

A historical same-budget additive-output control gives3.23% pooled/3.07%
terminal; its cubic control improves modestly. The fresh map-only pair below
does not reproduce a pooled-error advantage. The frozen causal metric fails
(25.17%/26.82%). An extra initial PDE point gives mixed results2.83%/2.53%
and is not selected. `train_vortex_accuracy.py` is the explicit3000-L-BFGS
budget experiment, not the recommended preset: training losses fall further
while final local pooled errors worsen to3.24%/6.34% for seeds42/2026.
Keep the shorter tested refinement; lower collocation loss is not sufficient
evidence for lower physical field error. Each recommended run takes about
213–215s on this machine, including training JIT/evaluation/refinement.

Historical cubic final model/config paths (use each saved config):

- `/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-conserved-conserved/model_final.eqx`
- `/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-conserved-conserved/config.json`
- Independent run: `/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-short-seed2026/model_final.eqx`

All16 controls, negative outcomes, unchanged reference validation, source
versions, exact initial arrays, CPU and frozen GPU checkpoint replays and
trained periodicity checks are in the
[completed study](../../docs/tasks/archive/twophase-operator-preconditioning.md).
[Fields](../../docs/tasks/archive/twophase-operator-recommended-phi.png),
[convergence](../../docs/tasks/archive/twophase-operator-convergence.png), and
[loss versus field error](../../docs/tasks/archive/twophase-operator-residual-vs-error.png)
use raw saved histories/fields. Old checkpoint contents and external paths
remain intact. The constant-flow base configuration and other example configurations are
retained; the case-local periodic-frequency bug fix applies to new networks.

### Fresh output-map ablation after mass-loss cleanup

Two new serial seed42 cold starts use the current two-term PDE loss, exact
phase IC/grid mass, conservative flux, nonlocal diffusion, FP64/ordinary and
SOAP5000 + L-BFGS1000. Both finish the full budget. Initial raw network bytes
and36frozen source files match; configs differ only in map and run name.
The train_vortex_consistent.py preset supplied the additive/direct-phase
control; at experiment time train_vortex_conserved.py enabled the inverse-cubic
coordinate. Today that latter preset defaults to additive; reproduce the cubic
arm with its saved config or the explicit inverse_cubic override. Physical
future fields at initialization need not match under different output maps.

Errors below use the SAME local reference and final-budget checkpoint:

| Quantity | Direct phase | Inverse cubic |
|---|---:|---:|
| Pooled phi relative L2 | 3.2006% | 3.2522% |
| Terminal t=2 phi relative L2 | 3.6215% | 2.9159% |
| Pooled interface-band phi relative L2 | 13.7798% | 14.2428% |
| Terminal interface-band phi relative L2 | 14.8632% | 11.6362% |
| Trainer wall, including JIT/evaluation/refinement | 212.38 s | 217.83 s |

The interface band is defined by \(|\phi_{\mathrm{ref,local}}|<0.9\).
Removing the map does not materially worsen pooled phase error in this pair;
terminal error rises by0.706percentage points (about24% relative). The cubic
map improves late times but worsens early times here. Both SOAP-only pooled
errors are about3.88% at5000 updates. Native nonlocal pooled/terminal errors
are3.5854%/4.0897% for direct output and3.5642%/3.2342% for cubic output.
Pooled local q errors remain60.0%/58.1%; these are phase-field results, not
an all-field3% claim. Exact grid mass holds to roundoff in both.

This single fresh pair, together with known same-seed trajectory variation,
does not establish a stable accuracy ordering or necessity of the cubic map.
Additive is the default for cross-example generality, with cubic available as
an opt-in. Reference, optimizer and physics
are unchanged. See the
[complete record](../../docs/tasks/archive/twophase-map-ablation.md),
[independent replay and metrics](../../docs/tasks/archive/twophase-map-ablation-analysis.json),
and [convergence curves](../../docs/tasks/archive/twophase-map-ablation-convergence.png).
Field figures in that record use the native nonlocal reference; the comparison
table and convergence curves use the common local reference.


## Matched local AD diffusion control

`configs/train_vortex_conserved_local.py` inherits the recommended conserved
vortex preset and changes only `nl.method` to `local_ad`. The negative
Laplacians of phi and q are spatial AD Hessian traces of the point network.
The exact grid-mass correction is a spatially constant shift; its temporal
change is included in the shared time JVP. Both arms retain the SAME conservative
nodal Fourier-flux advection, output map, constraints, samples and optimizer
budget. Thus this isolates diffusion differentiation; it is not a comparison
against all-AD advection or an independently retuned local architecture.

```bash
source .venv/bin/activate
XLA_FLAGS='' JAX_ENABLE_COMPILATION_CACHE=false python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_conserved_local.py --seed=42 --config.log.run_name=my-vortex-local-control
```

Use the SAME local reference for cross-method field errors; additionally report
native-reference errors because the finite-horizon nonlocal model differs from
the local PDE. Whole-process timing includes backend/setup, JIT, evaluation,
saves and both optimizer stages. The trainer timer starts after network/operator
setup. Compiled SOAP means omit the first100 updates and exclude time draws,
evaluation and saves; L-BFGS timing includes its separate JIT and callbacks.

A historical serial seed42 pair on the same RTX3090 used the cubic coordinate
in BOTH runs, with persistent compilation cache disabled, and completed
SOAP5000 + L-BFGS1000. Today's preset follows the additive default; reproduce
the cubic control by adding `--config.phase.map=inverse_cubic` to the command
above, or use the saved run config. The table is not an additive-control result:

| Quantity | Local AD diffusion | Nonlocal diffusion |
|---|---:|---:|
| Pooled phi relative L2, common local reference | 3.3643% | 3.7776% |
| Terminal t=2 phi relative L2, common local reference | 3.2233% | 3.2663% |
| Pooled phi relative L2, native reference | 3.3643% | 4.0638% |
| Trainer wall, including JIT/evaluation/refinement | 924.23 s | 211.15 s |
| Process launch to exit, additionally including setup | 931.02 s | 217.42 s |
| First SOAP update, including its JIT | 68.16 s | 30.51 s |
| Mean compiled SOAP update, after100 updates | 83.16 ms | 16.73 ms |
| L-BFGS stage, including JIT/callbacks | 425.20 s | 83.61 s |
| SciPy function/gradient evaluations | 1075 | 1079 |

Initial checkpoint bytes and36source snapshots are identical; configs differ
only in method/run name. Both L-BFGS runs stop at the1000-iteration budget.
This pair gives slightly lower pooled error for local AD and nearly equal
terminal error; nonlocal is4.38x faster by trainer wall and4.97x by compiled
SOAP update time. These are timings of this implementation, grid, architecture,
precision and hardware. They do not establish superiority for all local AD
implementations or independent local hyperparameter tuning.

The fresh nonlocal3.78% pooled error is higher than the earlier same-config
seed42 result2.84%. Direct comparison verifies identical initialization/config
and first loss, with later trajectory divergence; source changes are inactive
optional branches/validation/metadata for this configuration. Ordinary GPU
training does not promise bitwise trajectory reproducibility, and a unique
cause was not isolated. Retain both outcomes; the modest accuracy difference
in one matched pair is not sufficient evidence of a stable local advantage.
Defaults, physical parameters and references remain unchanged.

Direct new checkpoints are
`/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-local_ad/model_final.eqx`
and
`/root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-nonlocal/model_final.eqx`.
Both pass independent CPU field replay and GPU initialization/field checks with
the established FP64 field tolerance1e-10. CPU maximum difference is9.99e-15;
GPU maximum is4.44e-16. See the
[complete comparison](../../docs/tasks/archive/twophase-matched-local-comparison.md)
and [common-reference convergence curves](../../docs/tasks/archive/twophase-matched-local-convergence.png).

### Same-seed repeatability diagnosis

Optimizer stability follow-up is deferred at the user's request and is not a
current research blocker. See the standalone
[deferred SOAP issue](../../docs/issues/soap-repeatability.md); the records below
remain historical evidence, not an instruction to resume experiments.

A bounded diagnosis (six GPU processes,12short trajectories,814total updates)
locates a concrete early amplifier of the rerun variation. With EXACT saved
initialization and identical time samples, ordinary nonlocal transpose
accumulation gives identical scalar loss but gradients differing by up to
1.39e-17. Repeating SOAP on an IDENTICAL frozen gradient gives identical states;
feeding the near-identical measured gradients instead changes its initial Gram
matrices by about1e-21. For example the output-weight64x64Gram has numerical
rank2, so most eigendirections are nearly degenerate. After an identical second
gradient, actual SOAP updates differ25.3–26.5% in relative norm. Canonical
restarts first differ at update2;100-step parameter differences are32.1%.
These are gradient/update/parameter norms, NOT physical-field L2 errors.

A probe-only exact symmetric pullback, L-transpose=L for this reciprocal
periodic operator, removes the scatter accumulation without changing its forward
operator, PDE or residual norm. Two100-step restarts then match exactly WITHIN
each tested process. A deterministic-policy5-step control also matches. However,
identical-source independent symmetric-adjoint processes still show6.94e-18
initial gradient differences and diverging trajectories: this is a partial
isolation, NOT a completed cross-process fix. The exact compiler/kernel source
of those remaining differences is not isolated. The custom VJP is confined to
the diagnostic, with no production/default change or new full-run L2 claim.

The evidence justifies checking SOAP initialization/weak eigendirections before
expanding accuracy statistics. First validate a stable numerical treatment and
two identical-seed complete runs; then use predeclared paired seeds, report all
outcomes/mean/spread and accuracy at equal wall time. Do not infer a stable
accuracy ordering from the single3.36% versus3.78% pair. See the
[diagnosis and validation](../../docs/tasks/archive/twophase-repeatability.md).

### SOAP source audit

The current factory uses pinned `soap-jax==0.2.2` from
[SOAP_JAX](https://github.com/haydn-jones/SOAP_JAX), whose installed source
matches both its pinned upstream file and the prior diagnostic snapshot.
An audit against the [author's PyTorch implementation](https://github.com/nikhilvyas/SOAP)
and [pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer) finds no
core translation/call error explaining the observed same-seed variation in the
tested configuration. First-call initialization/skip, effective bias count,
update-before-refresh, first-moment basis transport and second-moment permutation
agree across16gradient-stream cases per implementation. Shared-basis FP64
parameter differences are at most2.22e-16; the actual project clipping/LR/bias
factory passes a1005-step independent recurrence check.

Rank-deficient basis sensitivity is also reproduced in the author implementation
at its native arithmetic precision, so swapping implementations alone is not a
validated remedy. Author eig/QR use FP32, current JAX follows parameter precision,
and the audited independent version uses parameter-precision eig but FP32 QR.
The upstream JAX README documents an OLD off-refresh-step stale-EMA issue;
current0.2.2 already includes its fix and the audited moment histories agree.

A real difference remains for NONZERO weight decay: author code decays the
updated parameter, while JAX decays the old parameter. The current project
factory uses zero decay, so this does not explain the current case. This audit
changes no dependency/default or training behavior and establishes no new field
L2 improvement. Exact sources, scopes, shared-basis controls, native precision,
sensitivity experiments and raw arrays are in the
[SOAP audit](../../docs/tasks/archive/soap-implementation-audit.md).

## All-nonlocal spatial derivative control

The explicit research preset `configs/train_vortex_nonlocal_flux.py` inherits
the conserved direct-phase strategy and changes advection to `nonlocal_flux`,
with a distinct matching native reference. All PDE spatial derivatives then
use finite-horizon neighbor sums; time differentiation remains JVP and parameter
training remains AD. The default conserved preset continues to use FFT flux.
This remains prescribed-flow CH, without NS or a pressure unknown.

On the full uniform periodic grid, the first derivative uses the same cubic
radial profile and horizon as diffusion, corrected by its discrete moment:

$$
K=\sum_jw_{ij}\xi_{ij}\xi_{ij}^{T},\qquad
(G_\delta f)_i=K^{-1}\sum_jw_{ij}(f_j-f_i)\xi_{ij},\qquad
D_\delta=-G_\delta^T.
$$

The transpose refers to the full discrete gradient and uniform-grid inner
product. Symmetric periodic weights make each component skew-adjoint. Apply
the divergence to the nodal flux, giving

$$
r_\phi=\phi_t+D_\delta(\mathbf v\phi)+ML_\delta q.
$$

The chemical residual and independent pair Laplacian are unchanged. Do not
replace that Laplacian by the gradient/divergence composition: the centered
nodal first derivative has checkerboard null modes. This transport control
does not establish pressure stability for a future NS solver.

```bash
source .venv/bin/activate
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_nonlocal_flux.py --config.log.run_name=my-vortex-nonlocal-flux
```

Native reference:
`/root/autodl-tmp/data/twophase2d/results_periodic_vortex_nonlocal_flux.npz`.
The solver evaluates independently derived nonlocal first-derivative sine
symbols for its native model; the common local target still uses local Fourier
derivatives. New-mode identity records the advection choice and rejects the
old native reference. The unsupported combination with local AD diffusion is
rejected because a matching mixed-method reference is not provided. The solver
CLI can generate this explicit reference from scratch; the experiment record
also preserves generation that explicitly reuses the unchanged, already-refined
common local fields and recomputes native fields with time-step refinement.

Compare both common-local and native-reference errors: first-derivative
discretization changes the reference dynamics at finite horizon. Full-budget
results, numerical checks and source/config/checkpoint replay provenance are in
the [experiment record](../../docs/tasks/archive/twophase-nonlocal-flux.md).

A fresh serial seed42 pair at64x64,delta0.04,FP64/ordinary completes the same
SOAP5000 + L-BFGS1000 budget with identical raw initial parameters and37source
snapshots. Both use additive output, hard IC/grid mass and the original two
PDE losses. Final-budget checkpoint results:

| Metric | FFT flux + nonlocal diffusion | Nonlocal flux + nonlocal diffusion |
|---|---:|---:|
| Pooled phi L2, common local target | 2.9236% | 8.5448% |
| Terminal phi L2, common local target | 3.1270% | 4.5147% |
| Pooled interface-band phi L2, common local target | 12.5945% | 38.7513% |
| Terminal interface-band phi L2, common local target | 12.5530% | 19.5499% |
| Pooled phi L2, each native target | 3.2435% | 2.7625% |
| Terminal phi L2, each native target | 3.4617% | 3.0188% |
| Trainer wall including JIT/evaluation/refinement | 213.41 s | 217.63 s |

The interface band uses the common local reference with absolute phi below0.9.
Native targets are different, so their L2 values are not a common-PDE accuracy
ranking. New native-reference time refinement max discrepancies are1.34e-5 for
phi and1.47e-4 for q; mass drift1.11e-16. Its common-local phi gap is9.219%
pooled/3.858% terminal, versus1.28%/1.17% for the FFT-flux native reference.
At t0.5 the two flux derivatives on the SAME common reference differ21.1%.
This supports a substantial finite-horizon transport bias, not just poor neural
fitting. It is not a strict error floor or an isolated unique causal mechanism.

The new native reference overshoots to[-1.546,1.586]; its PINN spans
[-1.415,1.378], versus the FFT-flux PINN[-1.054,1.035]. These values are
reported without clipping. Common-local pooled q errors are55.3%/147.6%
(native53.2%/43.5%); no all-field accuracy gain is claimed. Both CPU checkpoint
replays pass at the unchanged FP64 tolerance1e-10, with field discrepancies
at most1.22e-14. Both mass projections hold to roundoff. This single pair
shows no speed benefit or reason to replace the default FFT flux at this
grid/horizon/kernel; other nonlocal first-derivative designs remain untested.
See [convergence](../../docs/tasks/archive/twophase-nonlocal-flux-convergence.png),
[common-local field comparison](../../docs/tasks/archive/twophase-nonlocal-flux-fields.png)
and [full metrics/replays](../../docs/tasks/archive/twophase-nonlocal-flux-analysis.json).
