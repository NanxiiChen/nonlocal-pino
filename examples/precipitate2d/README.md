# Periodic coherent elastic precipitate

Small two-dimensional phase-field/elasticity pilot. The phase evolves through
conserved Cahn–Hilliard dynamics; the elastic field is equilibrated by an exact
differentiable spectral solve at every queried time. Elastic stress feeds back
into the chemical potential. The network predicts phase and mixed chemical
potential; **displacement is solved, not independently predicted by a network**.

The model follows coherent phase-field microelasticity and the energy structure
illustrated by [PFHub4](https://pages.nist.gov/pfhub/benchmarks/benchmark4.ipynb/)
and [Hu & Chen2001](https://doi.org/10.1016/S1359-6454(01)00118-5).
This is a simplified periodic homogeneous-isotropic-modulus model, with linear
eigenstrain interpolation and quartic wells. It is **not a reproduction** of
PFHub4's anisotropic, traction-free, tenth-order-potential benchmark.

## Explicit model

On the periodic unit square, phi=+1 denotes precipitate and phi=-1 matrix. Let
h=(1+phi)/2, E=diag(0.01,0), epsilon(u)=sym(grad(u)). Define

$$
F=\int_\Omega\left[\frac\beta4(\phi^2-1)^2+
\frac{\varepsilon^2}{2}|\nabla\phi|^2+
\frac12(\epsilon(\mathbf u)-hE):C:(\epsilon(\mathbf u)-hE)\right]dx.
$$

The isotropic 2D constitutive law is sigma=lambda tr(e) I+2 mu e, with
e=epsilon(u)-hE. Macroscopic strain is fixed to zero; u is periodic with zero
mean. No external body force, evolving displacement inertia or plasticity is
included. The nondimensional constants are beta=1, epsilon=0.025, M=0.01,
lambda=2450 and mu=1225; the eigenstrain is 1%. These choose elastic/interface
competition for a route test, not a calibrated material.

Stationarity in u and conserved gradient flow in phi give

$$
\nabla\cdot\sigma=0,\qquad
q=\beta(\phi^3-\phi)-\varepsilon^2\Delta\phi-\frac12\sigma:E,
\qquad \phi_t=M\Delta q.
$$

The elastic derivative follows by varying h: dh/dphi=1/2. After eliminating
equilibrated u, its implicit variation contributes zero by stationarity. The
continuum periodic balance is dF/dt=-M integral |grad(q)|^2, with conserved
integral(phi). Elastic relaxation time is assumed negligible; the phase time
is a model relaxation time without an experimental time calibration.

## Nonlocal and mechanical discretization

The default replaces -Delta in phase energy/CH diffusion with the symmetric
positive finite-horizon periodic operator L_delta from twophase2d/operators.py,
with delta=0.04 and discrete second moments equal to 2. Thus

$$
q=\beta(\phi^3-\phi)+\varepsilon^2L_\delta\phi-\frac12\sigma:E,
\qquad \phi_t=-ML_\delta q.
$$

The corresponding discrete interfacial energy is epsilon^2 phi^T L_delta phi/2
times the grid quadrature weight. Mechanical gradients remain Fourier spectral:
this is a **hybrid** scheme, not fully nonlocal elasticity. The explicit
`nl.method=local_spectral` control uses the Fourier -Delta in phase diffusion.
It is distinct from the `nl.method=local_ad` control, which evaluates both
mixed point-network Laplacians with AD Hessian traces. That control retains
the same full-grid phase projection and time JVP: the spatially uniform shift
has zero spatial-query Hessian, and its time derivative is included. Mechanical
equilibration remains spectral in both local controls. At finite collocation
resolution, AD Laplacians do not inherit an exact nodal summation-by-parts
identity; mass is enforced by the common projection and energy is measured.

For each nonzero wavevector k, set S=C:E, b=S k and
A=mu |k|^2 I+(lambda+mu) k k^T. Then

$$
\widehat{\mathbf u}(k)=-iA^{-1}b\,\widehat h(k),\qquad
B(k)=E:C:E-b^TA^{-1}b.
$$

The reduced elastic chemical multiplier on phi is B/4. The k=0 mode uses
u_hat=0 and the fixed-macrostrain elastic energy, including its constant
chemical-potential term. Odd grids avoid spurious real first-derivative
Nyquist null modes. Both independent matrix inversion and energy derivatives
are checked against the training implementation's analytic inverse.

## Initial condition and training

The initial rounded particle uses s=sum sin^2(pi(x-center)), center=(0.5,0.5),
phi0=-0.95+1.95[1-tanh((s-0.35)/0.09)]/2. The background offset supplies solute;
the IC embeds no future shape or velocity.

The default fixed grid is 63x63, with eight open stratified random times per
update over t in [0,1]. A four-layer width64 GELU network uses three fixed
integer spatial modes, exact periodicity and an IC-preserving inverse-cubic
coordinate map. Full-grid mean projection enforces discrete phase mass; all
training/evaluation/loading must use network.predict_fields. No strictly
bounded tanh is imposed, since conserved CH fields may overshoot pure-phase
values. Both loss terms are original mixed residuals, with explicit unit
weights; no inverse-Jacobian residual transformation is enabled.

Training uses ordinary GPU compilation, FP64, SOAP5000 at initial lr=1e-3,
then PDE-only L-BFGS up to500 accepted iterations at16 fixed midpoint times.
Reference field values are used only for evaluation/checkpoint selection,
never inside the training loss. `model_best.eqx` minimizes evaluated pooled
native phase relative L2; `model_final.eqx` preserves the final iterate. Seed
is run provenance via `--seed`, not part of the strategy preset. This is a pilot
preset; cross-seed robustness and optimality have not been established.

## Usage and reference

```bash
source .venv/bin/activate
python -m examples.precipitate2d.solver
python examples/precipitate2d/train.py --config=examples/precipitate2d/configs/train_base.py
python examples/precipitate2d/train.py --config=examples/precipitate2d/configs/train_local_spectral.py
python examples/precipitate2d/train.py --config=examples/precipitate2d/configs/train_local_ad.py
python -m examples.precipitate2d.evaluate --run /absolute/path/to/run
```

Canonical training starts in train.py. References/logs are under the existing
external data/tf-logs roots, in new precipitate2d subdirectories; existing case
paths remain unchanged. The reference path is explicit in configs/train_base.py.
The CPU NumPy/SciPy solver imports no JAX/PINN operator, assembles the elastic
acoustic tensor independently, and advances CH with IMEX-BDF2. It rejects a
reference unless time/grid refinements pass for phase, chemical potential,
displacement and stress at every saved time. A no-eigenstrain control measures
whether elastic feedback changes morphology. No FEniCS/MPI process is needed
for this periodic homogeneous-modulus reference.

Use native finite-horizon reference error to measure training, and common
refined local reference error separately to expose operator bias. Inspect
phase, displacement and stress errors, mass/energy and interface shapes.
Checkpoint loading uses direct paths and physical/grid identity checks, with
no file-hash gate.

## First validated cold start

The current pilot preset completed SOAP5000+L-BFGS500 in183.1 s including
training/evaluation/refinement compilation, excluding reference generation and
subsequent CPU replay. Seed42 is run provenance; one run establishes no
cross-seed robustness. L-BFGS reached its iteration budget, not its convergence
tolerance. Best and final checkpoints coincide at update5500.

| Physical field | Pooled relative L2 | Terminal relative L2 |
|---|---:|---:|
| Phase | 0.0387% | 0.0309% |
| Displacement, solved from predicted phase | 0.0185% | 0.0152% |
| Stress, xx/yy/xy component norm | 0.0354% | 0.0287% |
| Mixed network chemical potential | 3.807% | 0.3162% |

These use the native finite-horizon reference. Against the refined local model,
phase error is0.5420% pooled and0.5593% terminal; operator bias is reported
separately. Native interface-band phase error (|phi_ref|<0.9) is0.1145% pooled.
Chemical potential is less accurate at t=0 (16.42%); exact phase IC does not
enforce the algebraic chemical relation there. This remains an explicit
limitation. Mass drift is2.22e-16 and sampled predicted energy decreases.

The native reference terminal x/y semiaxis ratio is0.95860, PINN0.95876,
and the no-elastic-feedback control1.00000. The deformation in this first
short-time pilot is modest. Reference phase time/grid refinement differences
are at most5.49e-7/2.10e-9; all other saved fields also pass refinement checks.

Run: `/root/autodl-tmp/tf-logs/precipitate2d/20261027-periodic-elastic-pilot-01/`.
Use `model_best.eqx` with its direct `config.json` and canonical reference path.
[Full record and validation](../../docs/tasks/archive/elastic-precipitate-pilot.md)
include physical fields, convergence, independent replay and scope limits.

## Matched local controls

Two further seed42 cold starts retain the same initialized parameter bytes,
network, IC/mass constraints, physical model, time-sampling protocol,
FP64/ordinary policy and SOAP5000+L-BFGS500 budget. Only phase spatial
differentiation changes; mechanical equilibration is spectral in every run.
All three selected checkpoints coincide with their final update5500.

| Phase derivative | Pooled phi L2, common refined local target | Terminal phi L2, same target | Trainer time |
|---|---:|---:|---:|
| Nonlocal | 0.5420% | 0.5593% | 183.1 s |
| Local spectral | 0.0599% | 0.0529% | 183.1 s |
| Local AD | 0.0677% | 0.0590% | 680.1 s |

Against its own finite-horizon target, nonlocal pooled phi remains0.0387%.
Its larger common-local error primarily reflects the separately measured
0.5378% finite-horizon/local reference gap; the native/local targets must not
be mixed in a purported common-model comparison. Local spectral/AD target
the same local model. Their pooled displacement errors are0.0278%/0.0299%,
stress component errors0.0516%/0.0622%, and chemical-potential errors
7.273%/7.279%. q at t0 remains~29.98% inaccurate for both local controls;
terminal q errors are0.4467%/0.3642%. Phase accuracy does not imply all-field
subpercent accuracy.

Mean compiled SOAP step after100 updates is19.70ms nonlocal,19.95ms spectral,
91.53ms AD. Wall time includes JIT/evaluation/refinement, excludes reference
generation/CPU replay, and compares a historical nonlocal run with two new
serial GPU controls. This is one untuned seed, not a statistical ranking or a
dedicated kernel benchmark. No default is changed by this comparison.

Local runs live under the same run root as
`20261027-periodic-elastic-local-spectral-01` and
`20261027-periodic-elastic-local-ad-01`; explicit presets are shown above.
Both direct CPU checkpoint replays pass. The manufactured-field residual and
parameter-gradient comparison verifies the AD implementation independently.
[Comparison record](../../docs/tasks/archive/precipitate-local-comparison.md),
[convergence figure](../../docs/tasks/archive/precipitate-local-comparison.png)
and [local AD fields](../../docs/tasks/archive/precipitate-local-ad-fields.png)
retain numerical evidence and limitations.

## Multiple interacting particles

The explicit variant in configs/train_multiple.py keeps the same energy,
homogeneous elastic equilibrium, periodic zero macrostrain, hard phase IC/mass,
original mixed residuals and training budget. It introduces three unequal
particles at non-symmetric positions, stronger elastic/interface competition
and a longer observation interval. It does not prescribe their later motion.
The simple train_base.py pilot remains available unchanged.

For periodic particle profiles

$$
s_k(x)=\sum_d\sin^2\left(\frac{\pi(x_d-c_{k,d})}{L}\right),\qquad
a_k(x)=\frac12\left[1-\tanh\left(\frac{s_k(x)-\ell_k}{w}\right)\right],
$$

the prescribed smooth union is

$$
\phi_0(x)=\phi_b+(1-\phi_b)\left[1-\prod_k(1-a_k(x))\right].
$$

Centers are(0.35,0.32),(0.67,0.36),(0.40,0.68); levels are0.18,0.11,0.065,
with shared width0.035 and background-0.95. Moduli are lambda9800,mu4900
(four times the pilot), eigenstrain remains1% in x, and terminal time is4.
These are explicit dimensionless model choices, not a calibrated material or
PFHub4 reproduction. Initial particle shape data are not future training labels.
Displacement remains an exact solved equilibrium field, not a separate PINN.

Use the same canonical entry:

    XLA_FLAGS='' python examples/precipitate2d/train.py --config=examples/precipitate2d/configs/train_multiple.py --config.log.run_name=my-multiple-nonlocal
    XLA_FLAGS='' python examples/precipitate2d/train.py --config=examples/precipitate2d/configs/train_multiple.py --config.nl.method=local_ad --config.log.run_name=my-multiple-local-ad

The latter changes only phase diffusion to local AD Hessians.
Both retain63x63 collocation, three fixed spatial modes,4x64 network, eight
stratified times,FP64/ordinary,SOAP5000 then L-BFGS500. Mechanical gradients
remain Fourier spectral in both. The explicit reference path is
/root/autodl-tmp/data/precipitate2d/results_periodic_multiple.npz. Its local
time/space checks use191/255 grids and dt0.0005/0.00025 with the same all-field
0.2% threshold as the pilot. A sharper multiparticle IC needs the denser
coarse validation grid, especially for the chemical potential.

    python -m examples.precipitate2d.solver --config=examples.precipitate2d.configs.train_multiple

Legacy single-particle configs and checkpoints retain their original arithmetic
and direct loading. The new IC schema explicitly declares kind=multiple,
centers and levels, rather than keeping unused single-particle center/level
fields. See the [complete extension record](../../docs/tasks/archive/precipitate-multiparticle.md)
for reference controls, cold starts, field errors and morphology validation.

### First matched nonlocal/local AD validation

Both seed42 cold starts finish SOAP5000+L-BFGS500, with byte-identical initial
parameters and21 identical frozen source files. Only nl.method/run name differ.
The primary comparison uses final-budget model_final.eqx at update5500 and the
same refined **local** target for every field:

| Field | Nonlocal pooled L2 | Local AD pooled L2 | Nonlocal terminal L2 | Local AD terminal L2 |
|---|---:|---:|---:|---:|
| phi | 8.0304% | 12.8981% | 9.1820% | 19.2435% |
| Solved displacement | 9.4308% | 13.1430% | 12.0771% | 21.0310% |
| Stress components | 8.0802% | 13.8471% | 9.4585% | 21.5787% |
| Chemical potential q | 25.3784% | 25.8049% | 6.6135% | 6.6958% |

Trainer time is183.34s nonlocal versus674.27s local AD (3.68x); mean compiled
SOAP update is19.81ms versus91.53ms. Time includes compilation, evaluation and
L-BFGS, excludes reference generation/CPU replay; both run serially on the same
GPU. Both L-BFGS runs reach the500-iteration limit rather than convergence.
Best native-phi checkpoints occur at5000/4750, respectively: common-local
pooled phi8.0200%/12.8345%, slightly lower than the final-budget values.
Reference values select/report checkpoints but do not enter the training loss.

Nonlocal relative to its own finite-horizon target has pooled/terminal phi
7.3559%/8.9354%; the separately measured native/local reference gap is1.2353%
pooled. Thus the remaining error cannot be attributed solely to that operator
gap. This single untuned pair favors nonlocal here; it does not establish a
stable ordering across seeds or cases. Local AD refers to the phase/chemical
spatial Hessians; mechanical equilibrium remains spectral in both runs.

The refined reference shows the smallest zero-contour component disappearing
near t2.4(local)/2.6(nonlocal), with pronounced elongation of the largest one.
The zero-eigenstrain control instead merges the lower pair and ends with one
component. Elastic/control terminal phi differs39.14%. PINN disappearance is
late: nonlocal reaches two components only at t4, local AD still has three.
Initial q error remains61.8%/77.0% against each native target. These are useful
harder-case baselines, not a solved high-accuracy case; the simple default is
therefore retained without SOTA promotion or further tuning.

CPU final-checkpoint replays match saved fields to3.84e-13 maximum absolute
error. Predicted mass drift is below1e-15 relative to the directly prescribed
training IC mean, sampled energy decreases, and independently reconstructed
mechanical force residual is below1.02e-11. The resampled fine local reference
has a2.8808e-6 initial-mean offset and0.2091% initial phase difference from
that direct coarse sample; these are reported separately from conservation.
Seven focused tests and legacy checkpoint replay pass.

[Common-reference fields](../../docs/tasks/archive/precipitate-multiple-common-fields.png),
[convergence/time/energy](../../docs/tasks/archive/precipitate-multiple-comparison.png)
and the [full audit](../../docs/tasks/archive/precipitate-multiple-analysis.json)
retain the raw all-field results and scope. New runs are at the existing run
root with suffixes20261027-multiple-elastic-nonlocal-01 and
20261027-multiple-elastic-local-ad-01.
