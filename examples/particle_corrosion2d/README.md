# Ceramic-particle corrosion 2D — geometry-conditioned PINO

A physics-only neural operator predicts corrosion around **six random, smooth,
inert ceramic particles**. Particle positions, sizes, orientations and shapes
vary; the initial pit, reservoir, rectangular domain and material coefficients
are shared. Ceramic interiors are excluded from the reactive domain and remain
solid after the surrounding metal dissolves.

The representative result uses the explicit
[width256 preset](configs/train_physical_spaced_w256.py): a 50 × 30 μm region,
350 s of physical evolution, and a cold 20,000-step Adam run with **rolling
geometry and core/halo inputs**. Its mean phase relative L2 on four development
geometries is **2.93% overall / 5.76% terminal against local FEM**, and
**1.45% / 2.78% against direct nonlocal evolution**.

![Corrosion setup: gray metal, blue initial electrolyte pit and six warm-ivory ceramic particles](../../assets/pino-setups/particle_corrosion2d.png)

This schematic shows one prescribed geometry and its boundary conditions;
only the narrow pit mouth has reservoir data. Warm-ivory ceramic, gray metal
and blue electrolyte distinguish the three materials; no future interface is drawn.
[Editable setup PDF](../../assets/pino-setups/particle_corrosion2d.pdf) ·
[Source/style specification](../../artifacts/particle_corrosion_setup_materials.json) ·
[Regeneration](../../assets/pino-setups/README.md).

![Four six-particle layouts with predicted phase and local-FEM interfaces](../../assets/pino-main/particle_corrosion2d.png)

Rows show all four development geometries in saved order at actual times
\(t^*=0,0.2,0.5,0.8,1\), or0,70,175,280,350s. Coolwarm shows predicted phase;
warm-ivory interiors with dark outlines are inert ceramic; dashed curves are independent **local-FEM**
\(\phi=0.5\) interfaces. Each panel reports instantaneous unweighted nodal
phase relative L2; the headline historical volume-weighted metrics remain
separate. Original physical triangles and equal spatial scales are retained.
The accepted width256 final20k model is unchanged.
[Editable170mmPDF](../../assets/pino-main/particle_corrosion2d.pdf) ·
[Source/style specification](../../artifacts/particle_corrosion_main_figure.json).

## Problem setup and parameterized geometry

The dimensionless domain is \([0,1]\times[0,0.6]\) minus the ceramic interiors;
depth \(y\) increases downwards. The shared semicircular pit has center
\((0.5,0)\), radius 0.12, and a reservoir mouth
\(y=0,\ |x-0.5|\le0.055\). Physical lengths are obtained with
\(L_c=50\,\mu\mathrm m\).

Each particle is a smooth radial graph, normalized to its equivalent area,
stretched by an area-preserving ellipse and rotated:

$$
\mathbf x(\theta)=\mathbf c+
Q(\alpha)\operatorname{diag}(\sqrt a,1/\sqrt a)
\,r(\theta)(\cos\theta,\sin\theta)^T,
\qquad
r(\theta)=R_s\exp\!\left[
\sum_{k=3}^{5}A_k\cos(k\theta)+B_k\sin(k\theta)\right].
$$

| Geometry setting | Current value |
|---|---|
| Particle count | 6 |
| Equivalent radius / aspect ratio | 0.045–0.075 / 1–1.45 |
| Minimum pit-surface gap | 0.075, or 3.75 μm |
| Minimum particle gap / outer-wall gap | 0.055 / 0.025 |
| Near-pit requirement | At least 2 particles within pit-surface distance 0.16 |
| Minimum sampled curvature radius | 0.015 |

See [six-particle settings](configs/train_physical_spaced.py) and the inherited
[shape/validation bounds](configs/geometry_base.py). Invalid proposals are
rejected and regenerated with the same bounds. No future solution is used to
admit a geometry.

The larger initial gap separates particles from the initial diffuse interface
and delays strong particle interaction. It does not impose a future interface
or make the initial wall-normal derivatives exactly zero. The direct references
still corrode about **29.5%** of the reactive domain on average and interact
with 3–5 particles by \(t^*=1\).

## Physical model and nondimensionalization

The KKS fields are metal phase \(\phi\in[0,1]\) and normalized total
concentration \(c\). Define

$$
h(\phi)=3\phi^2-2\phi^3,\qquad
g(\phi)=\phi^2(1-\phi)^2,\qquad
\Delta c=c_{Se}-c_{Le},\qquad
q=c-\Delta c\,h(\phi)-c_{Le}.
$$

The dimensional equations are

$$
\partial_t\phi
=L\left[\alpha_\phi\Delta_x\phi-\omega_\phi g'(\phi)
+2A\Delta c\,q\,h'(\phi)\right],
\qquad
\partial_t c=D\Delta_xq.
$$

| SI input | Value |
|---|---:|
| Diffusivity \(D\) | \(8.5\times10^{-10}\,\mathrm{m^2/s}\) |
| Free-energy parameter \(A\) | \(5.35\times10^7\,\mathrm{J/m^3}\) |
| Double-well coefficient \(\omega_\phi\) | \(1.76\times10^7\,\mathrm{J/m^3}\) |
| Gradient coefficient \(\alpha_\phi\) | \(2.575\times10^{-5}\,\mathrm{J/m}\) |
| Actual matrix mobility \(L\) | \(2\times10^{-9}\,\mathrm{m^3/(J\,s)}\) |
| \(c_{Se},c_{Le}\) | \(1,\ 5100/143000\) |
| \(L_c,T_c\) | \(50\,\mu\mathrm m,\ 350\,\mathrm s\) |

Thermodynamic constants and diffusivity follow the
[electropolish setup](../electropolish2d/README.md); the mobility is chosen for
this more moderate reaction regime. With \(x^*=x/L_c,\ t^*=t/T_c\), derive all
four groups together:

$$
N_{\mathrm{CH}}=\frac{DT_c}{L_c^2}=119,\qquad
N_1=2ALT_c=74.9,\qquad
N_2=L\omega_\phi T_c=12.32,\qquad
N_3=\frac{L\alpha_\phi T_c}{L_c^2}=0.00721.
$$

For the rest of this README, derivatives are in dimensionless coordinates:

$$
\phi_t=N_3\Delta\phi-N_2g'(\phi)
       +N_1\Delta c\,q\,h'(\phi),
\qquad
c_t=N_{\mathrm{CH}}\Delta q.
$$

The additional mobility multiplier is \(\kappa=1\): `physical.L` already
contains the actual mobility. The
[dimensional factory](dimensional.py) also derives the initial interface
coefficient

$$
k=L_c\sqrt{\omega_\phi/(2\alpha_\phi)}=29.2295900,\qquad
\phi_0=\tfrac12\left[1+\tanh\!\left(
k\left(\sqrt{(x-0.5)^2+y^2}-0.12\right)\right)\right],
\qquad c_0=c_{Se}h(\phi_0).
$$

The 10–90% interface thickness is 3.7586 μm. The initial pit radius is 6 μm;
particle equivalent radii are 2.25–3.75 μm.

At the reservoir mouth, impose \(\phi=c=0\), including at \(t=0\).
On all other exterior and ceramic walls impose
\(\partial_n\phi=\partial_nq=0\): neutral wall energy and impermeable solute
transport. The reservoir exchanges concentration, so no constant-total-mass
constraint is imposed.

## Network, derivatives and losses

The generic GNOT-inspired
[HeterogeneousPointOperator](../../src/nonlocal_pinn/nn/point_conditional.py)
reads two point streams:

| Encoder stream | Features | Measure |
|---|---|---|
| Domain/initial field | \((x,y,\phi_0)\) on a masked 25 × 41 ambient map | Physical area weights |
| Boundary | \((x,y,n_x,n_y,\mathrm{kind})\) | Physical arc lengths |

Independent lifts feed two heterogeneous integral cross-attention blocks.
**Coordinate-only queries** read both streams without particle IDs or a
particle-specific distance bias. Spatial preparation is shared across times;
three temporal FiLM layers decode the time dependence. Different geometries
have their own physical query nodes and masks.

The selected network has width **256**, four heads of dimension **64**, 128
spatial Fourier directions (256 sine/cosine features), spatial scale 2, and
eight temporal frequencies spanning 1–4 in \(\pi t^*\). It contains 2,248,450
stored model parameters/array entries.

Two raw channels \(z,s\) give the physical fields:

$$
\phi=\tfrac12(1+\tanh z),\qquad
c=c_{Le}s+\Delta c\,h(\phi),\qquad q=c_{Le}(s-1).
$$

The liquid-coordinate channel \(s\) is unrestricted. The KKS relation is
satisfied algebraically, without an extra KKS loss. This recipe uses the
**original PDE residuals**; it does not divide by the tanh Jacobian.
Initial and reservoir conditions are soft losses.

Spatial diffusion uses visible symmetric nonlocal exchanges with physical
nodal volumes and horizon \(\delta=0.026\). Segments crossing a particle are
excluded. The regional operator closes exchange at ceramic/exterior walls;
this finite-horizon closure is not an exact pointwise local Neumann condition.
The implemented operator approximates the positive negative-Laplacian,
\(D_\delta\approx-\Delta\). In particular, evaluate
\(D_\delta q=c_{Le}D_\delta s\) to reduce FP32 cancellation.

Time derivatives use **JVP in the supplied dimensionless time**, through the
complete physical output map. The 32 times are randomly resampled at every
update; they are collocation times, not a finite-difference time grid.

There are five losses: AC residual, concentration-transport residual, mouth
phase, mouth concentration normalized by \(c_{Le}\), and initial physical fields
with concentration error normalized by \(c_{Le}\). PDE terms use mean absolute
residuals; IC/BC terms use squared errors. Both PDE prescales are 1.
Stopped inverse-L1 parameter-gradient balancing refreshes every 25 updates,
with `alpha_gradnorm=0.9` and cap 1000. AC/CH are trained jointly, without
staggering, causal weighting or time windows.

## Selected training recipe and rolling inputs

| Setting | Value |
|---|---|
| Entry / explicit preset | [train.py](train.py) / [train_physical_spaced_w256.py](configs/train_physical_spaced_w256.py) |
| Optimizer / budget | Adam, cold 20,000 updates |
| Learning rate | \(10^{-3}\), halved every 3,000 updates, floor \(10^{-5}\) |
| Precision / compilation | FP32 / ordinary GPU compilation; gradient clip 10 |
| Batch | 4 geometry draws × 32 freshly sampled times |
| Mesh / contour | Target spacing 0.015 / 128 edges per particle |
| Known-input pool | 64 rolling geometry slots |
| Spatial sampling | 8 core/halo sets per slot, 256 volume-sampled cores per set |
| IC sampling | 1,024 volume-sampled points per slot |
| Mouth sampling | Every actual Dirichlet node including endpoints, with boundary quadrature |
| Geometry refresh | Replace 4 slots every 250 updates |
| Spatial refresh | Refresh 4 slots every 50 updates, including core/halo sets and IC draws |
| Background preparation | 4 CPU workers, 4 prefetched events |
| Fixed padding | Halo 3,072; 64 neighbors; complete stencils, no truncation |

Background workers construct **known geometry, operators, quadrature and
initial/boundary data only**. They never solve evolved training trajectories.
The input family is continuously sampled, rather than restricted to a fixed
64-geometry library. Invalid candidates are regenerated; padded shapes stay
fixed inside JIT.

On the 32 GiB RTX5090, the completed run used
`XLA_PYTHON_CLIENT_PREALLOCATE=true` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`. Active peak was **20.69 GiB**; total
`nvidia-smi` usage was about 29.5 GiB including reservation and device overhead.
The initial non-preallocating attempt failed before its first update and is
retained in the study record.

## Results and validation

All results below are means over the same **four development geometries**,
excluded from the rolling training stream. Overall relative L2 uses physical
spatial quadrature and trapezoidal time weights over all 11 saved snapshots;
terminal error uses only \(t^*=1\).

| Reference | Phase overall / terminal | Concentration overall / terminal |
|---|---:|---:|
| Independent local P1 FEM | **2.933% / 5.757%** | **3.566% / 6.871%** |
| Direct production nonlocal solver | **1.450% / 2.781%** | **1.763% / 3.278%** |

Terminal corrosion-region IoU is **94.99% against local FEM** and **98.10%
against direct nonlocal**. The local corroded-field \(1-\phi\) error is larger,
6.98% overall / 8.88% terminal; whole-field L2 does not measure every interface
or transport error equally.

| Same-stream cold 20k control | Width 192 | Width 256 |
|---|---:|---:|
| Local phase overall / terminal | 3.091% / 5.985% | 2.933% / 5.757% |
| Native phase overall / terminal | 1.637% / 3.110% | 1.450% / 2.781% |
| Model size | 1.269 M | 2.248 M |
| Complete training time | 23.35 min | 28.47 min |
| Held-update median | 49.06 ms | 62.94 ms |
| Active peak | 15.66 GiB | 20.69 GiB |

Both width and Fourier capacity change. Physics, optimizer, actual condition/
pool/time draws, rolling identities and complete initial/final known inputs
match. Parameter shapes and initialization differ. On each of the four
geometries, local phase/concentration errors improve. This is one initialization
per architecture, not a cross-seed robustness result.

The lowest recorded width256 local phase error was **2.805% at 15k**
(5.346% terminal), with a saved checkpoint. The figure and main table deliberately
use the fixed-budget 20k endpoint.

The independent local solver uses P1 weak-form stiffness, lumped mass and sparse
BDF integration. On the first geometry, spacing 0.01→0.0075 changes phase by
0.1007% overall; tighter/halved time stepping has negligible sensitivity.
Production direct-nonlocal/local-FEM phase differences are 1.72–1.80% over the
four geometries. These errors cannot be subtracted to form scalar error
contributions.

All 20k actual draws, 380 sampled geometry identities, 1,900 spatial versions
and all 64 final input slots pass replay audits, with no evaluation overlap
or evolved training labels. All four local and four native saved prediction
files independently reproduce their metrics. Remaining limitations include
finite-horizon error, slight late-time growth underestimation, and the small,
already-inspected development cohort.

## Ceramic-wall sampling experiment

The [wall-core control](configs/train_physical_spaced_w256_wall_cores.py)
reserves 128 of 256 PDE draws for free nodes within one nonlocal horizon of
the actual ceramic polygon edges. The other 128 draws cover the remaining
free nodes. Each draw uses the node's lumped area within its stratum.
The stratum means are weighted by their total areas, so the estimator
retains the full-domain PDE measure. Complete visible-pair stencils and
the original initial and reservoir samples are used in both arms.

Prepare the wall-core bank with the same input seed as its area-sampled
partner and check the geometry and non-PDE arrays before comparing runs.
The [study record](../../docs/tasks/archive/pino-spatial-sampling-study.md) gives
the data paths and matched GPU procedure.

## Run the documented case

Use `.venv` from the repository root; CPU meshing needs the optional
`gmsh==4.15.2` dependency. Every output directory must be new.

~~~bash
# Independent evaluation geometries; no trajectories are generated here.
.venv/bin/python -m examples.particle_corrosion2d.preview \
  --config examples.particle_corrosion2d.configs.train_physical_spaced_w256 \
  --output /path/to/preview --count 4

# Initial geometry/operator/IC bank only.
.venv/bin/python -m examples.particle_corrosion2d.prepare \
  --config examples.particle_corrosion2d.configs.train_physical_spaced_w256 \
  --output /path/to/input-bank --workers 4

# Independent local FEM reference for evaluation only.
.venv/bin/python -m examples.particle_corrosion2d.qualify \
  --config examples.particle_corrosion2d.configs.reference_physical_spaced \
  --preview /path/to/preview/manifest.json --backend local \
  --spacing 0.01 --boundary-points 256 --output /path/to/local-reference

# Optional direct nonlocal reference on the production mesh.
.venv/bin/python -m examples.particle_corrosion2d.qualify \
  --config examples.particle_corrosion2d.configs.reference_physical_spaced \
  --preview /path/to/preview/manifest.json --backend nonlocal \
  --spacing 0.015 --boundary-points 128 --output /path/to/nonlocal-reference

# Cold physics-only training on the rolling input pool.
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
.venv/bin/python -m examples.particle_corrosion2d.train \
  --config examples/particle_corrosion2d/configs/train_physical_spaced_w256.py \
  --inputs /path/to/input-bank --reference /path/to/local-reference \
  --output /path/to/new-run
~~~

`prepare`, `preview` and `qualify` take dotted config modules; `train` takes a
config file. Reference fields are used by the diagnostic evaluator, not the
loss. The evaluation geometries are excluded from training automatically;
`--exclude-reference` adds a separate evaluation cohort.

Use the explicit preset above to reproduce this documented result.
[train_default.py](configs/train_default.py) still selects the separate,
higher-mobility 5 s physical research case. This README update changes the
representative documentation, not numerical defaults.

Seeds are runtime provenance, not strategy configuration fields. Omit `--seed`
for a fresh run; use it only for deliberate replay. Intermediate model
checkpoints do not provide exact rolling-pool/optimizer continuation.

### Opt-in matched rolling nonlocal/local-AD study

The research presets `train_physical_spaced_w256_method_nonlocal.py` and
`train_physical_spaced_w256_method_local_ad.py` inherit the selected six-particle,
350 s, width256, Adam20k recipe. They retain the same five shared losses and
inverse-gradient balancing; AD adds the two explicit Neumann terms. These
presets do not change the production or historical defaults. Intermediate
models are saved every5,000 updates; they are diagnostics, not resumable states.

Run the nonlocal preset with the usual `--inputs`, `--reference`, `--output`
and optional runtime `--seed`. After completing its independent
`python -m examples.particle_corrosion2d.online_validation --run RUN --workers 4`
audit, run the AD preset with `--rolling-method-control RUN`. This imports only
the cold initialization provenance and sampling seed, never trained weights.
Both sides require the same input-bank/reference paths and geometry exclusions.
Audit the completed AD inputs with `online_validation`, then use
`python -m examples.particle_corrosion2d.method_validation --run AD_RUN --control NL_RUN`.
The existing `compare --runs NL_RUN AD_RUN --output NEW_DIRECTORY` includes the
same strict audit and recomputes metrics from saved fields without inference.

The audit compares cold model/optimizer bytes, numerical sources, actual
geometry/revision/time draws, installed geometries, complete initial/final
inputs, and the actual PDE core coordinates/weights at every update.
`core_trace_incomplete.npy` temporarily occupies234.4MiB at20k/B4/K256 and is
compressed only on successful completion. A failed run retains its incomplete
trace. Trace I/O and rolling waits are reported separately from optimizer
updates. AD evaluates only cores, without halo network evaluations. For strict
input pairing both methods still build/store the same graph payload; this is
explicit unused graph-preparation overhead in the AD comparison, not an
intrinsic requirement of local differentiation. The completed two-initialization
study and same-device measurements are in the
[comparison record](../../docs/tasks/archive/pino-rolling-local-ad-controls.md).
Both particle pairs pass the full input/state audit. The comparison preserves
the production defaults; historical results below use a different recipe.

#### Completed 20,000-step comparison

These are new cold runs of the six-particle, 350 s recipe: width256, four
geometry draws, 32 random shared times, 256 cores and eight core/halo sets per
geometry, with rolling geometry/spatial pools. Both methods use ordinary
FP32, Adam1e-3 (decay0.5 every3000updates, floor1e-5), the same five common
losses and inverse-gradient balancing refreshed every25updates. Local AD adds
the two explicit positive-time Neumann groups. The displayed pair uses runtime
seed 42, the first strictly audited initialization rather than an error-selected seed.

The comparison below uses the **final 20,000-step seed-42 weights on 64 fresh
geometries**. All layouts follow the original six-particle prior and were frozen
after training but before reference solves or checkpoint inference. They exclude
all 700 unique geometries from the initial/rolling training banks and the old
four-case development set. Both methods use exactly the same 64 inputs.

The common reference is independent local P1 FEM with lumped mass: mesh spacing
0.01, 256 polygon nodes per hole, BDF rtol/atol 2e-8/2e-10 and maximum normalized
step 0.00625. Four predeclared mesh-refinement checks at spacing 0.0075 keep the
hole polygons fixed; phase differences are 0.0903–0.1093% under the original
physical-quadrature diagnostic. A time-step-halving check changes that phase
metric by 1.31e-7%. These checks qualify the sampled refinements, not a continuum
bound for every geometry. No reference fields enter physics training.

For each test input \(i\), first compute the relative L2 for each physical
solution channel over the complete space-time evaluation set \(Q_i\), then
average the two channels \(u=(\phi,c)\):

$$
e_{i,k}=\frac{\sqrt{\sum_{q\in Q_i}|\hat u_{i,k}(q)-u_{i,k}(q)|^2}}
{\sqrt{\sum_{q\in Q_i}|u_{i,k}(q)|^2}},\qquad
e_i=\frac{e_{i,\phi}+e_{i,c}}{2},\qquad
\bar e=\frac{1}{64}\sum_{i=1}^{64}e_i.
$$

The violin displays the 64 individual \(e_i\); the reported mean is \(\bar e\).
Each channel is normalized separately before averaging. We do not concatenate
channels before normalization or average instantaneous relative errors. The
KDE uses all64samples and Scott bandwidth in log-error coordinates; thin dashed
lines indicate the25th,50thand75thpercentiles. The compact figure displays raw
error ratios on a logarithmic axis, without percentage scaling, dots or mean
annotations. Tails extend three bandwidths beyond the observed log extrema;
they represent smoothed density, not extra samples. Matplotlib autoscales y.
These describe variation across inputs, not confidence intervals.
The figure omits the example-name title; both exports are named
`particle_corrosion2d_distribution.pdf` / `.png`. The source manifest below records
the editable PDF path and the matching README PNG.

Here \(Q_i\) contains every physical FEM node at all 11 uniformly saved times
from \(t^*=0\) to 1 (0–350 s), including endpoints. The new metric uses ordinary
**unweighted discrete L2 over evaluation points**. For this irregular mesh,
this is a point-sampling measure: it does not use nodal volume or trapezoidal
time weights. Previous phase-only/physical-quadrature scores remain historical
metrics in the evaluation records and must not be substituted for these values.

| Method | Mean error | Median | Interquartile range |
|---|---:|---:|---:|
| Nonlocal | 4.554% | 4.420% | 4.204%–4.957% |
| Local AD | 32.515% | 32.500% | 32.182%–32.945% |

![Final particle-corrosion weights: distribution of channel-averaged space-time relative L2 on 64 inputs](../../assets/pino-comparisons/particle_corrosion2d_distribution.png)

Only the completed seed-42 model pair is displayed. The sparse intermediate
checkpoints are retained in the archive; no convergence curve is shown here.

On the same49140MiB RTX4090, the two-seed mean warm
update costs93.50ms(nonlocal) versus80.40ms(AD); complete cold child processes
take34.38 versus33.52minutes. Active peaks are20.44 versus11.21GiB. Process
time includes setup/output but excludes subsequent independent evaluation and
audits; AD retains the shared graph-preparation cost for exact input pairing.
These runs show an accuracy advantage, **not** a nonlocal per-update speed or
memory advantage.

![Three particle geometries: local FEM, nonlocal and local AD phase evolution](../../assets/pino-comparisons/particle_corrosion2d_fields.png)

Each group of three rows shows local FEM, nonlocal PINO and local AD PINO for
one geometry. These are the first three **new 64-case cohort** entries (zero-based0,1,2), using
the strict seed42pair's final20kmodels, without error-based sample selection.
Columns are actual saved times \(t^*=0,0.2,0.5,0.8,1\), or0,70,175,280,350s.
Gray holes are the original ceramic geometry; all panels share the phase
scale0..1. Black contours show the displayed field's \(\phi=0.5\) interface;
white dashed contours show the local reference. Panel labels give the original
volume-weighted, phase-only spatial L2 at that time, separate from the two-channel
space-time violin metric. Original FEM triangles are retained; Gouraud interpolation is
display-only, with no field smoothing or temporal interpolation.

Nonlocal tracks the expanding pit with some late interface-position error.
AD remains near the original pit size and also shows an initial-condition
fitting shortfall, as in the old four-geometry inspection. These experiments
do not isolate the Neumann boundary treatment as the cause or show that local
AD is intrinsically inaccurate. The fresh64cohort confirms the same accuracy
ordering under the declared final-checkpoint metric. The previous phase-only,
other-initialization and small-cohort results remain in the original comparison
and 64-case evaluation records. No new model or default is selected here.

The [figure source manifest](../../artifacts/pino_derivative_comparisons.json)
stores all four full saved configurations, direct model/source/data paths,
sample identities and actual times. The
[validation and plotted values](../../artifacts/pino_derivative_comparison_validation.json)
record every plotted per-channel and channel-averaged error, numerator and
denominator norms, evaluation grids, KDE settings, rendering versions and image sizes. The [64-case evaluation record](../../docs/tasks/archive/pino-derivative-evaluation64.md)
contains exact reference/inference commands and cohort/replay checks. The
[distribution record](../../docs/tasks/archive/pino-final-violin-figures.md)
records the new reduction order and the superseded-figure archive.
To regenerate just these two images from the repository root:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py --case particle_corrosion2d
```

The command reads the existing external artifacts and checks their saved
configurations and field metrics; it performs no training, model inference or
reference solve. Use the saved configurations, not a later default, to replay
these checkpoints. Raw weights/fields remain outside Git.

## Artifacts and historical comparisons

The selected [20k checkpoint](/root/autodl-tmp/tf-logs/particle_corrosion2d/20261107-spaced-six-w256-cold20000-02/model_final.eqx),
[summary](/root/autodl-tmp/tf-logs/particle_corrosion2d/20261107-spaced-six-w256-cold20000-02/summary.json),
[local reference](/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/local_dev4/manifest.json),
[native reference](/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/native_dev4/manifest.json),
and [capacity comparison](/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-width256-02/capacity_comparison.json)
remain external. The current main figure's inputs, warm-ivory fill and dark outlines
are recorded in [its specification](../../artifacts/particle_corrosion_main_figure.json).
Regenerate its editable PDF and matching PNG on CPU:

```bash
.venv/bin/python -m scripts.render_particle_corrosion_main_figure \
  --spec artifacts/particle_corrosion_main_figure.json
```

The [publication collection](../../assets/pino-main/README.md) describes copying
approved exports into the shared README/website assets. The older gallery and
its renderer remain recorded in
[pino_readme_figures.json](../../artifacts/pino_readme_figures.json). Both routes
read saved fields without inference, training or new solves.

Earlier results use different physical regimes or geometry priors:

| Preset / record | Scope |
|---|---|
| [train_physical_spaced.py](configs/train_physical_spaced.py) | Same six-particle SI problem, width192 control |
| [train_physical_moderate.py](configs/train_physical_moderate.py) | Same SI coefficients but eight closer particles; cold20k local phase error 34.12% |
| [High-mobility physical pilot](../../docs/tasks/archive/particle-corrosion-physical-pilot.md) | \(L=2\times10^{-6},T_c=5\) s; eight particles; unresolved cold-training failure |
| [Dimensional audit](../../docs/tasks/archive/particle-corrosion-dimensional-audit.md) | Why older directly chosen dimensionless groups are not the same electropolish material in different units |
| [Historical rolling case](../../docs/tasks/archive/particle-corrosion-online.md) | Different dimensionless material regime; eight reused evaluation geometries; phase 1.983% overall / 3.121% terminal |
| [Whole-grid/core-halo comparison](../../docs/tasks/archive/particle-corrosion-full-grid.md) | Historical matched sampling comparison, not the current six-particle SI result |
| [Fixed-bank/local-AD comparison](../../docs/tasks/archive/particle-corrosion-pino.md) | Historical optimization controls; failed local-AD training is not intrinsic AD inaccuracy |

The user-accepted physical study, qualifications and negative controls are
retained in the [study record](../../docs/tasks/archive/particle-corrosion-physical-optimization.md).

## Experimental core coverage controls

The accepted default retains its existing rolling geometry and area-sampled
core/halo pools. Two explicit alternatives isolate coverage from wall allocation:
[train_physical_spaced_w256_cdf.py](configs/train_physical_spaced_w256_cdf.py)
uses inverse-CDF strata in physical-coordinate order;
[train_physical_spaced_w256_stratified.py](configs/train_physical_spaced_w256_stratified.py)
adds64near-particle-wall and192bulk cores with physical-area weights. Both retain
the original PDE, IC, mouth BC and full encoder context. Input banks must be
prepared with their matching sampler configuration; mismatches are rejected.
These are ongoing controls, not SOTA replacements. Source pairing, failed
controls and the64case final evaluation are tracked in the
[active study](../../docs/tasks/pino-unified-rolling-core-halo.md).
