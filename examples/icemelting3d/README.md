# Ice Melting 3D

A shrinking ice sphere driven by an Allen-Cahn equation with a curvature-independent
source term. The linear-radius tanh profile below is an approximate comparison:
the three-dimensional Laplacian adds a curvature term, so this profile is not an
exact spherical solution. See the
[diagnosis](../../docs/tasks/archive/icemelting-late-time-diagnosis.md).

## SOTA default

Seed is run provenance, not a strategy parameter. `get_config()` omits
`train.seed`; `train.py --seed=2026` sets it for one run. The common untuned
default is42. Resolved saved configs retain it for replay/old model loading.

The canonical entry is [train.py](train.py), with
[configs/train_base.py](configs/train_base.py):
`python -m examples.icemelting3d.train` (or `python examples/icemelting3d/train.py`).
The numerical strategy matches the selected5 s/5000-update SOAP result
(the recorded run used seed42):
IC-only latent residual, exact prescribed IC, initial lr1e-3,
128^3 sparse cubic grid,delta0.01,K_core2000,K_pool200,Nt64,
4-layer64-unit tanh MLP,64-dimensional spatial/temporal embeddings at scales2/1.
SOAP uses beta1/beta2=0.9/0.99, refresh5, gradient clip100 and continuous
0.95-per1000 learning-rate decay to a1e-5 floor; matmul precision is `highest`.
Causal/interface weighting is disabled. Detailed options and error definitions
follow below.

Selected checkpoint:
`/root/autodl-tmp/tf-logs/icemelting3d/20261026-defaultcheck-ordinary-float32-seed42/ckpt_005000.eqx`.
This new explicit ordinary-FP32 cold start completes5000 updates; its full-grid
terminal phi relative L2 against the same-graph numerical reference is
**3.9052947454328864e-5** (independent CPU3.90581606e-5). Solid-fraction relative
L2 is3.54455e-4; predicted/reference solid-volume node fractions both equal
0.00407791. Late4000–5000 sampled graph phi median is3.48832e-5, with
range2.89452e-5–6.54558e-5. See the
[completed default validation](../../docs/tasks/archive/ordinary-policy-defaults.md). The direct reference is
`/root/nonlocal-pinn/reports/icemelting-weak-prior-20261024/reference.npz`;
pass it with `--config.eval.graph_reference_path=...` to enable graph-L2
monitoring. This optional monitoring does not change training.
SOTA here means the selected verified5 s graph-error result, distinct from
error against the approximate analytic profile and the historical4 s run
whose directory retains a `sota` tag. The earlier seed42 run
`20261024-5s-soap-lr1e3-seed42` had full-grid L2=6.10970e-5; it originally declared
20000 updates and stopped at5000. The new validation and default both declare5000. See the
[selected-default record](../../docs/tasks/archive/icemelting-default-preset.md).

### Independent seed check —2026-10-26

With the same IC-only latent/SOAP strategy at5 s/5000 updates, the new
predeclared `--seed=2026` gives **full-grid terminal graph phi L2=5.98573384e-5**,
compared with6.10970108e-5 for the recorded seed42 run. Independent CPU replay
gives5.98538783e-5; the reference and predicted solid-volume node fractions
both equal0.00407791. Solid-fraction relative L2 is5.43177e-4, distinct from
whole-cube phase L2. Last1000-update sampled graph phi median is4.97345e-5,
range2.73270e-5–5.92417e-5; full-grid late monitoring has only two snapshots,
4000/5000. This supports initialization stability in these two runs, not a
statistical guarantee. Sparse time keys, graph-pool and evaluation randomness
remain fixed. See the [validation](../../docs/tasks/archive/phasefield-seed-stability.md)
and [curves](../../docs/tasks/archive/phasefield-seed-stability-results-convergence.png).

### GPU compilation policy and speed

`train.py` now explicitly applies `train.gpu_policy="ordinary"` before backend
initialization. It retains unrelated `XLA_FLAGS` and rejects an inherited
`--xla_gpu_deterministic_ops=true` conflict. Use
`--config.train.gpu_policy=deterministic` to run a deterministic control.
The default arithmetic stays FP32. Earlier versions inherited external flags;
historical run records do not establish every shell's policy.

The [paired GPU audit](../../docs/tasks/archive/phasefield-gpu-policy-audit.md)
uses the complete128^3 sparse grid,2000 core points,200 pool entries,Nt64,
ic_latent residual and production SOAP step. Complete synchronized updates cost
26.62ms ordinary versus468.26ms with `--xla_gpu_deterministic_ops=true`, a17.6x
slowdown. Each mode runs125 updates (50 warm-up/75 measured), with identical
initial parameters, complete pool and time samples. These are steady updates
including sampling/optimizer, excluding first compile/evaluation/logging/saves.

The isolated latent diffusion uses11929 padded edges. Current deterministic
HLO introduces one11929-iteration loop in forward and three in its input-gradient
computation; ordinary retains scatter. Independent FP64 NumPy edge/transpose
formulas agree with both policies at better than3e-7 relative in the measured
controls. Optimizing only forward would miss substantial backward overhead.

For an explicit ordinary-policy run in a shell that previously set the flag:

```bash
XLA_FLAGS='' python -m examples.icemelting3d.train
```

No formulation, default sampling or optimizer was changed. The full ordinary5000-update replay above confirms low graph error. The short
paired benchmark alone does not establish equal long-run accuracy/reproducibility
across policies; FP64 melting was not part of this FP32 comparison.

## Current strategy

The production entry point is [train.py](train.py), with defaults in
[configs/train_base.py](configs/train_base.py). It combines an exact initial
condition in a tanh latent representation with the exactly transformed PDE
residual. Subsequent motion is learned from the PDE.

| Component | Current strategy |
| --- | --- |
| Default formulation | `phase.formulation="ic_latent"` |
| Initial condition | Embed the prescribed tanh profile through \(u_\theta=u_0+t^*N_\theta\), \(\phi_\theta=\tanh u_\theta\) |
| Future evolution | Learned correction; the trial function contains no prescribed speed or evolved radius |
| Loss | MAE of the transformed residual over valid core points and sampled times |
| IC handling | No IC sampling, IC MSE or IC GradNorm in hard-IC modes; PDE/IC weights are 1/0 |
| Spatial discretisation | Sparse cubic graph, \(128^3\) nodes, \(\delta=0.01\), six axial neighbours |
| Sampling | Precomputed fixed-shape core/halo subgraph pool; fresh uniform time samples each update |
| Network | Fourier-embedded MLP, four hidden layers of 64 units with tanh activations |
| Optimizer / runtime | SOAP, FP32 `matmul_precision="highest"`; `train.gpu_policy="ordinary"` |
| Additional weighting | Interface weighting zero; causal weighting disabled by default |
| Training data | PDE and prescribed IC; reference fields are evaluation-only |
| Time convention | Physical 0–5 s mapped to \(t^*\in[0,1]\) |
| Monitoring | Separate analytic/graph errors and pooled/terminal phase-field relative L2 |
| Loading | Explicit config/checkpoint/reference paths, without file-hash requirements |

The canonical default is the selected SOAP configuration from the
[five-second optimizer comparison](../../docs/tasks/archive/icemelting-soap-adam-5s.md):
5000 updates, initial learning rate 0.001, 2000 spatial core points and
64 time samples per update. Run `python -m examples.icemelting3d.train` to start
this configuration. The [artifact index](../../artifacts/index.yaml) records its
selected checkpoint under `default_training`; the older `sota` directory tag
belongs to the historical 4 s bounded run.

## Random initial ellipsoid neural operator (experimental)

The opt-in [configs/train_operator.py](configs/train_operator.py) is the current
recommended NO preset: online B16, Adam1e-3/5000updates, ordinary FP32/highest,
hard IC and latent PDE loss. It matches the saved Adam winning recipe except
run-name/runtime-seed provenance. Selection uses the current local-FEM whole-field
accuracy/cost compromise; SOAP remains available explicitly. See the comparisons
below. Generated figures/arrays are local artifacts, with a
[restoration manifest](../../README.md#artifacts-and-recovery).

This preset uses the same
physical Allen-Cahn melting equation,5s horizon,128-cube cubic graph, exact IC
and transformed residual. It does not change the single-sphere PINN default.
The encoder receives an initial3Dphase volume; queries contain XYZ and time,
with `local_dim=0`. The known initial function enters the physical hard-IC
wrapper, not the raw decoder query features:

$$
u_\theta(x,t^*;\phi_0)=\frac{d_E(x)}{\sqrt{2}\epsilon^*}
+t^*N_\theta(\operatorname{Encoder}(\phi_0),x,t^*),
\qquad \phi_\theta=\tanh u_\theta.
$$

Here \(d_E\) is positive inside the prescribed ellipsoid. Subsequent geometry
is unconstrained; no future ellipsoid, contraction speed or analytic trajectory
is built into this expression. Online training draws fresh independent
semiaxes uniformly in[0.24,0.36], center coordinates in[0.45,0.55], and a Haar
rotation from a normalized Gaussian quaternion. These bounds guarantee strict
containment in the unit cube. Dimensional semiaxes are24..36 and center
coordinates are-5..5 in the original physical cube[-50,50]^3. This is an ellipsoid-family
pilot, not arbitrary-topology training.

[ellipsoids.py](ellipsoids.py) computes Euclidean signed distance through the
nearest-surface Lagrange multiplier root, including center/axis/equal-semiaxis
cases. It uses a surface-constraint identity near a singular smallest-axis
projection rather than dividing tiny coordinates by an unresolved tiny root.
It is not a scaled radial level-set value. Sphere identities, rotated/interior
cases against constrained SciPy minimization, and FP32-versus-FP64 distances
are independently checked. See [Eberly's nearest-point construction](https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf).

|Component|Operator pilot|
|---|---|
|Condition volume|One physical initial phase channel,32x32x32, CZYX order|
|3Dpatch encoder|4x4x4patches,512tokens,width48,2blocks,4heads|
|Spatial conditioning|Coordinate cross-attention to input tokens; XYZ queries|
|Temporal work|Shared spatial preparation, then3-layer temporal FiLM decoder/time JVP|
|Batch/sampling|16fresh ellipsoids/update,8shared uniform times,256cores,32fixed core/halo pools|
|Training|Adam1e-3,5000updates, ordinaryFP32/highest matmul|
|Physical output/loss|Exact prescribed IC, stable latent residual MAE only|
|Held-out bank|2monitor and8final-test ellipsoids; no future-field training labels|
|Reference|Independent full-cube RK4 with step halving; no sphere/octant symmetry|

The actual graph has2097152nodes and12484608directed edges, degree6; padded
training queries are1788core+halo nodes for this pool. Neither attention nor
the training decoder materializes the full spatial grid across all times.
Reference evaluation encodes once per condition and queries in1024-point
chunks. The shared `ConditionalFieldOperator` also keeps historical2Dpatch
ordering, initial parameters and checkpoint shapes unchanged;3Dwidth is
divisible by6 for three-axis sine/cosine token positions.
For scale, explicitly storing a B16/four-head attention matrix between all
128-cube queries and512tokens would alone require256GiB in FP32, before
backward or decoder activations. Core/halo sampling and chunked inference are
therefore part of this implementation, not an assumption that full-grid
batched attention fits on the48GiB card.

Eight-update GPU probes on the48GiB RTX3090 measure the full residual,
parameter gradient and Adam update:

|Batch|JAX peak active allocation|JAX memory pool|Warm mean update|
|---|---:|---:|---:|
|8|0.63GiB|2GiB|14.8ms|
|16|1.33GiB|4GiB|21.8ms|

JAX active allocation excludes CUDA context/driver memory; pool size is reported
separately. These are short memory probes with7warm updates, not long-run speed
or accuracy benchmarks. First-update compile+call takes111/113s; setup~10..11s.

```bash
# Prepare the independent held-out references first, on CPU in FP64.
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python examples/icemelting3d/train.py \
  --config=examples/icemelting3d/configs/train_operator.py \
  --config.operator.action=prepare
# Train with continuously sampled ellipsoids; reference fields are evaluation-only.
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/icemelting3d/train.py \
  --config=examples/icemelting3d/configs/train_operator.py
# Repeat a bounded training-memory probe explicitly.
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/icemelting3d/train.py \
  --config=examples/icemelting3d/configs/train_operator.py \
  --config.operator.action=memory --config.operator.condition_batch=16
```

The explicit dataset path is
`/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1`; new runs remain under
the existing melting tf-logs root. Preparation refuses overwriting that folder;
change the explicit path for a new reference identity. Saved configs identify
the3Darchitecture/IC distribution; checkpoint loading uses
`operator_data.build_network(saved_config,key)` and direct file paths.
Training records actual ellipsoid parameters/time draws, source paths,
checkpoints, loss, allocator statistics, timings and held-out predictions.

Reference gates concern time integration at the declared native graph, not
local-PDE spatial convergence. Report phiL2 together with solid-indicator L2,
interface absolute error and solid fraction: mostly-water full-domain phiL2
can conceal lost small particles. A relative solid metric is explicitly null
if the reference norm is exactly zero; it uses no artificial denominator floor.

First cold start:5000Adam updates/B16,80000distinct online ellipsoids, full
process300.57s (5.01min), trainer257.17s. The first update includes108.24s of
compilation/call; subsequent synchronized updates average22.43ms and online
geometry sampling3.41ms. Full-run JAXpeak active1.332GiB/pool4GiB. All10native
128-cube references pass time refinement (maximum1.422e-7); future fields never
enter training. The two monitor shapes are separate from the eight final-test
shapes. Final checkpoint only, no test-based selection.

|Eight held-out shapes, mean physical errors|Pooled times|At5s|
|---|---:|---:|
|phi relative L2 on24-cube diagnostic nodes|0.2266%|0.0922%|
|Solid-indicator relative L2 on same nodes|0.6425%|4.5017%|
|phi relative L2 on128x128central z slice|0.4201%|0.2861%|

Mean terminal solid-fraction relative error is1.2133% (maximum3.3453%); mean
absolute error is1.227e-5 of cube volume on diagnostic nodes. Interface-band
physical phiRMSE averages0.01088 for \(|\phi_{ref}|<0.9\). Small late particles
remain less accurate: maximum terminal solid-field L2 is10.594%. The stationary
IC baseline has45.62% pooled phiL2 on the same test bank. Reported3Dmodel errors
use24^3diagnostic nodes, not full128^3inference;128^3is the direct-reference
integration grid. High-resolution slices provide an additional field check.
One run/eight tests do not establish seed stability or arbitrary-shape accuracy.

Run/checkpoint:
`/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01/model_final.eqx`.
Source/input-stream/metric audits pass; checkpoint replay max gap9.56e-7, and
the original2Dmodel/checkpoint behavior is retained. [Full task and verification](../../docs/tasks/archive/icemelting-random-operator.md),
[convergence](../../docs/tasks/archive/melting-operator-convergence.png),
[phase fields](../../docs/tasks/archive/melting-operator-fields.png),
[held-out3Dinitial shapes](../../docs/tasks/archive/melting-operator-initial-shapes.png).

The operator also accepts `--config.train.optimizer=soap` through the same
`train.py` entry. A matched cold start keeps5000updates/B16/lr1e-3, all79initial
arrays,80000actual online ellipsoids and40000time draws identical to Adam;
only optimizer and output run name change. SOAP uses the shared upstream
SOAP_JAX0.2.2 factory, beta1/beta2=.9/.99, refresh5, clip100 and no1D
preconditioning. No reference labels or test checkpoint selection are used.

|Eight held-out inputs, final5000 checkpoint|Adam|SOAP|
|---|---:|---:|
|Pooled phi relative L2|0.2266%|0.0785%|
|Terminal phi relative L2|0.0922%|0.1291%|
|Pooled solid-indicator relative L2|0.6425%|0.2235%|
|Terminal solid-indicator relative L2|4.5017%|6.5036%|
|Terminal solid-fraction relative error|1.2133%|4.0156%|
|Full cold process|300.57s|492.06s|
|Warm synchronized update, excludes sampling|22.43ms|33.66ms|

SOAP improves early/intermediate fields and pooled errors in this pair, while
late small-solid errors worsen. Time integration/reference and diagnostic-grid
scope are unchanged. Peakactive allocation is1.332/1.590GiB for Adam/SOAP,
both with4GiB JAXpools; first-update compile/call is108.24/247.12s. Keep the
Adam operator default; one pair does not establish optimizer stability or
universal superiority. Single-sphere SOAP defaults remain separate.
SOAP checkpoint:
`/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-soap-01/model_final.eqx`.
[Matched control and validation](../../docs/tasks/archive/melting-operator-soap.md),
[monitor convergence](../../docs/tasks/archive/melting-operator-soap-convergence.png).

## Dimensional Governing Equation

$$
\frac{\partial\phi}{\partial t} = M\left(\Delta\phi - \frac{F'(\phi)}{\epsilon^2}\right) - \lambda\frac{\sqrt{2F(\phi)}}{\epsilon}
$$

with \(F(\phi)=\frac14(1-\phi^2)^2\), using \(|\phi|\le1\) for the source expression:

$$
F'(\phi) = \phi^3 - \phi,\qquad \frac{\sqrt{2F(\phi)}}{\epsilon} = \frac{1 - \phi^2}{\sqrt{2}\,\epsilon}
$$

- **Domain**: \(\Omega = [-50, 50]^3\)
- **IC**: sphere of radius \(R_0 = 35\) at origin:

$$
\phi(\bm x, 0) = \tanh\!\left(\frac{R_0 - |\bm x|}{\sqrt{2}\,\epsilon}\right)
$$

- **Approximate comparison profile**:

$$
\phi(\bm x, t) = \tanh\!\left(\frac{R(t) - |\bm x|}{\sqrt{2}\,\epsilon}\right),\qquad R(t) = R_0 - \lambda t
$$

---

## Non-Dimensionalisation

Map to unit cube \([0,1]^3\) and \(t^* \in [0,1]\):

$$
\bm x^* = \frac{\bm x + 50}{L_c},\quad t^* = \frac{t}{T_c}
$$

with \(L_c=100\) and production \(T_c=5\,\mathrm{s}\). The sphere centre is
\(\bm x_c^*=(0.5,0.5,0.5)\). Production, FEM and research-pilot defaults now all
use physical5 s; saved historical runs retain their own recorded time scales.

Non-dimensional parameters:

$$
N_1 = \frac{M T_c}{L_c^2},\quad N_2 = \frac{N_1}{\epsilon^{*2}},\quad N_3 = \frac{\lambda^*}{\sqrt{2}\,\epsilon^*}
$$

where \(\epsilon^* = \epsilon / L_c\), \(\lambda^* = \lambda T_c / L_c\), \(R_0^* = R_0 / L_c\).

### Non-Dimensional Governing Equation

$$
\frac{\partial\phi}{\partial t^*} = N_1 \Delta^*\phi - N_2\,(\phi^3 - \phi) - N_3\,(1 - \phi^2)
$$

### Non-Dimensional Comparison Profile

$$
\phi(\bm x^*, t^*) = \tanh\!\left(\frac{R_0^* - \lambda^* t^* - |\bm x^* - \bm x_c^*|}{\sqrt{2}\,\epsilon^*}\right)
$$

with \(R^*(t^*) = R_0^* - \lambda^* t^*\).

---

## Nonlocal Formulation

Replace \(\Delta^*\) with the nonlocal operator \(D_\delta\) (\(D_\delta \approx -\Delta^*\)):

$$
\frac{\partial\phi}{\partial t^*} + N_1\,D_\delta\phi + N_2\,(\phi^3 - \phi) + N_3\,(1 - \phi^2) = 0
$$

PINN residual:

$$
r_\phi = \frac{\partial\phi}{\partial t^*} + N_1\,D_\delta\phi + N_2\,(\phi^3 - \phi) + N_3\,(1 - \phi^2)
$$

---

## Parameters

| Symbol | Value | Description |
|---|---|---|
| \(M\) | 0.1 | Mobility |
| \(\lambda\) | 5.0 | Interface speed (dimensional) |
| \(R_0\) | 35 | Initial sphere radius (dimensional) |
| \(L_c\) | 100 | Characteristic length |
| \(T_c\) | 5.0 s | Production characteristic time |
| Grid | \(128^3\) nodes | Production sampling grid, including faces |
| \(h^*\) | \(1/127\) | Production node spacing |
| \(\epsilon^*\) | 0.0225 | Fixed independently of the training grid |
| \(R_0^*\) | \(0.35\) | Non-dimensional initial radius |
| \(\lambda^*\) | \(0.25\) | Non-dimensional source parameter (\(\lambda T_c / L_c\)) |
| \(T^*\) | \(1.0\) | Non-dimensional total time |

---

## IC-only formulation and transformed residual

With \(d=\sqrt{2}\epsilon^*\), the latent trial function is

$$
u_\theta(x,t^*)=\frac{R_0^*-|x-x_c|}{d}+t^*N_\theta(x,t^*),
\qquad \phi_\theta=\tanh(u_\theta).
$$

It embeds the given initial condition exactly. No front speed, later radius,
Eikonal equation or evolved profile is prescribed. [phase.py](phase.py) initializes
the correction network's final layer to zero, giving a stationary untrained
field. The factor \(t^*\) fixes its initial value while allowing
\((u_\theta)_{t^*}(x,0)=N_\theta(x,0)\) to be learned. The source parameter enters
through \(N_3\) in the PDE. The transformed nonlocal residual is

$$
r_u=(u_i)_{t^*}+N_1\sum_jw_{ij}
\frac{\sinh(u_i-u_j)\cosh(u_i)}{\cosh(u_j)}-N_2\tanh(u_i)+N_3.
$$

For finite latent values, \(r_\phi=(1-\phi_i^2)r_u\). The code evaluates the
cosh ratio using log-domain arithmetic, preserving small bulk-phase changes
without dividing by a rounded zero. Minimizing \(|r_u|\) changes the training
objective while preserving the pointwise equation: its loss magnitude cannot
be compared directly with \(|r_\phi|\). Pilots use the same full-grid phase
residual, geometry and reference errors for evaluation. Their reference checks
time integration error at fixed spatial discretisation; they do not establish
convergence to the continuous local PDE.

Output tanh with the original residual still optimizes \((1-\phi^2)r_u\),
which suppresses latent errors in saturated phases. The transformed objective
removes that factor from the optimized residual. The graph has
\((D_\delta\phi)_i=\sum_jw_{ij}(\phi_i-\phi_j)\), with \(w_{ij}\ge0\) and zero
row sums. [losses.py](losses.py) evaluates the edge ratio using
`sinh(u_i-u_j) * exp(logaddexp(u_i,-u_i)-logaddexp(u_j,-u_j))`.
It uses neither a denominator floor nor residual clipping.

## Default production training and L2 monitoring

`configs/train_base.py` now defaults to `phase.formulation="ic_latent"`: the
IC-only trial function above and the stable transformed residual. The final
correction layer starts at zero (stationary IC); subsequent motion is learned.
Hard-IC modes optimize only the PDE MAE, with no redundant IC loss or IC GradNorm
weight. The selectable controls are:

| `phase.formulation` | Representation | Training objective |
| --- | --- | --- |
| `ic_latent` (default) | Exact IC with tanh latent field | MAE of \(r_u\) |
| `ic_phase` | Same exact IC and tanh field | MAE of \(r_\phi\) |
| `raw_phase` | Raw phase output | PDE MAE + soft-IC MSE, with GradNorm |

The separate legacy `arch.use_hard_constraint` flag is output-only tanh and must
remain false for both IC-only modes. Sparse nonlocal, finite differences and local AD are supported;
the local transformed diffusion is \(-\Delta u+2\tanh(u)|\nabla u|^2\).

For the default mode, the loss is

$$
\mathcal L=\frac{1}{K_{\rm core}N_t}
\sum_{i\in\mathrm{core}}\sum_{k=1}^{N_t}|(r_u)_i(t_k^*)|.
$$

Halo nodes supply neighbouring values; halo-only rows and padding do not enter
the average. `sample.N_ic` is used only in `raw_phase`. Reference fields, radius
targets and interface-width targets are absent from the training loss. The graph
retains only in-domain neighbours, with no separate BC loss.

### Training settings and sampling

The scalar correction MLP has four hidden layers of 64 units, with spatial and
temporal Fourier embedding dimensions 64, scales 2.0/1.0, and tanh hidden
activations. `arch.tensor_split=True` splits the first layer's spatial/time
matrix products without changing its mathematical function.

The training grid spacing is \(h^*=1/127\). The cubic horizon
\(\delta/h^*\approx1.27\) has six axial neighbours per interior node. The graph
and 200 uniformly padded subgraphs are precomputed, and updates cycle through
the pool. Each update draws new uniform times in \([0.001,1)\). Network fields
are evaluated on core plus halo nodes; valid PDE residuals are reduced over the
core mask, keeping shapes fixed under JIT. Finite differences (`nl.method="fd"`)
are an alternative;
`"local"` uses AD derivatives and resampled spatial LHS points. The supplied
nonlocal graph reference applies to `"sparse"` only.

| Setting | Base training | Bounded cold-start preset |
| --- | ---: | ---: |
| Config | `configs/train_base.py` | `configs/train_cold_start.py` |
| Physical horizon | 5 s | 5 s |
| Optimizer updates | 5000 | 2000 |
| Spatial core points | 2000 | 256 |
| Time samples per update | 64 | 16 |
| Initial learning rate | 0.001 | 0.001 |
| Subgraph pool entries | 200 | 200 |

Both use SOAP with \(\beta_1=0.9\), \(\beta_2=0.99\), precondition refresh
every 5 updates, and global gradient-norm clipping at 100. The learning rate
decays continuously exponentially by 0.95 per 1000 updates, with floor
\(10^{-5}\). `train.matmul_precision="highest"` is explicit. Interface
weighting (`loss.iface_gamma_*`) is zero and causal weighting
(`loss.use_causal`) is false by default.

`train.optimizer="adam"` selects Adam for an optimizer comparison. The current
factory uses its installed Optax defaults (\(\beta_1=0.9\), \(\beta_2=0.999\),
\(\epsilon=10^{-8}\)); SOAP uses the explicit settings above. Compare them with
the same initial model, samples, time horizon, learning rate/schedule and update
budget, and record both phase-field error and elapsed time. A shared-rate
comparison evaluates these configured choices rather than separately tuned
optima for each algorithm.
The [five-second optimizer comparison](../../docs/tasks/archive/icemelting-soap-adam-5s.md)
records the matched5000-update protocol, checkpoint selection, L2 histories,
runtime and interpretation of the approximate analytic reference.

### Commands and outputs

```bash
source .venv/bin/activate
python -m examples.icemelting3d.train
# Same defaults with Adam:
python -m examples.icemelting3d.train --config.train.optimizer=adam
# Default training with the same graph-reference L2 monitoring as the comparison:
python -m examples.icemelting3d.train --config.eval.graph_reference_path=reports/icemelting-weak-prior-20261024/reference.npz
# Restore original soft-IC formulation explicitly:
python -m examples.icemelting3d.train --config.phase.formulation=raw_phase
# Bounded production-entry-point cold start (2000 updates, K_core=256, Nt=16,
# lr=0.001) using the previously generated local graph reference:
python -m examples.icemelting3d.train --config=examples/icemelting3d/configs/train_cold_start.py
```

The bounded cold-start preset changes the listed training budget and batch sizes,
and enables the evaluation reference path. It uses the same default learning rate.
Its `eval.graph_reference_path` points to a local artifact produced by the pilot
reference command; choose a matching reference path for another checkout. It is
evaluation data only. An empty path explicitly disables graph-reference metrics.
The loader checks physical parameters, length scale, centre, graph, grid, shapes
and physical snapshot times; different stored time scales are matched in seconds,
without changing the governing parameters or interpolating reference fields.

Each run uses a fresh directory under the configured log root and refuses an
existing run directory. It saves config/source snapshots and a source file-path
list (`source/files.json`), initial weights, checkpoints, final weights,
`evaluation.json`, `metrics.jsonl`, and `summary.json`. The runtime `--seed` controls the
cold-start network and local sampler. Sparse-mode time keys, graph-pool and
evaluation seeds remain fixed; changing it mainly probes initialization.
Checkpoint filenames now count completed optimizer updates; `model_initial.eqx`
is update 0. Historical weights need their original raw builder, while new IC-only
weights need `_build_phase_network(config,key)` before leaf deserialization.
No existing checkpoint files or artifact locations are rewritten.

Artifact loading uses explicit file paths: reference arrays use
`np.load(config.eval.graph_reference_path)`, and checkpoints use
`eqx.tree_deserialise_leaves(checkpoint_path, model_template)`. New summaries
record `model_path="model_final.eqx"`, relative to the run directory. No expected
file hashes or checksum manifests are required, including for older checkpoints.
Historical reports may retain old checksum fields; loaders do not read them.

### Relative L2 definitions

For the physical final time \(T\), terminal spatial relative L2 is

$$
E_{\rm terminal}
=\frac{\sqrt{\sum_i[\phi_\theta(x_i,T)-\phi_{\rm ref}(x_i,T)]^2}}
{\sqrt{\sum_i\phi_{\rm ref}(x_i,T)^2}}.
$$

Pooled L2 applies the same norm over all five evaluation times together; it is
neither terminal-only error nor the average of per-time relative errors. These
are errors of \(\phi\) over the whole cube, including water, and differ from
solid-fraction, interface-only, radius or volume errors. Low residual alone does
not establish low L2, and late L2 need not decrease monotonically.

The graph reference is direct float64 RK4 integration of the same fixed spatial
graph, with a time-step-halving check. Its octant storage uses centred-sphere
symmetry to reconstruct the full cube. It evaluates PINN error at that spatial
discretisation. Analytic error includes the approximate profile's discrepancy;
the FEM solver is a separate local-PDE comparison. Changing the grid or horizon
changes the graph model and requires separate spatial validation.

TensorBoard and JSONL distinguish these phase-field relative L2 metrics:

| JSONL key (TensorBoard prefix `eval/`) | Meaning |
| --- | --- |
| `analytic_pooled_rel_l2` | One pooled norm over 8000 fixed grid nodes and five times, vs approximate analytic profile |
| `analytic_terminal_rel_l2` | Same nodes at the final time only, vs analytic |
| `graph_pooled_rel_l2`, `graph_terminal_rel_l2` | Corresponding metrics vs matching graph integration, when supplied |
| `analytic_terminal_full_rel_l2`, `graph_terminal_full_rel_l2` | Final-time spatial field over every grid node |

The 8000 evaluation nodes are selected once with seed 123. The five evaluation
times are \(t^*=0,0.25,0.5,0.75,1\). Full-grid inference uses fixed chunks of
32768 nodes by default.

`eval/rel_L2` retains the pooled analytic meaning. Full-grid terminal evaluation
runs at update 0, every `eval.full_grid_every` updates (default 1000), and the final
update; sampled metrics run every `log.eval_every` updates (default 200).
`eval.chunk_size` fixes inference batch shape, including padding on a short final
chunk. Nonfinite loss/gradients cause a visible failure instead of sanitization.
Analytic and numerical-reference errors remain distinct because the nominal
linear-radius profile is approximate, even for a perfectly trained graph solver.


## Reference generation and research controls

An isolated research pilot compares the raw phase field with a hard-initial-value
tanh latent representation and an exactly transformed nonlocal residual. It
keeps the default six-neighbour graph, recomputes its weights in float64 for the
direct reference, and never reads reference labels during training. The default
pilot embeds only the prescribed initial condition. Its untrained field is
stationary; velocity and later geometry must be learned from the PDE. This is
the validated basis for the production default described above.

```bash
# Independent CPU reference, then isolated GPU pilots; no historical resume.
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m examples.icemelting3d.accuracy reference
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m examples.icemelting3d.accuracy check
python -m examples.icemelting3d.accuracy train
```

The pilot configuration is `configs/accuracy_weak_prior.py`. Generated fields,
checkpoints and metrics go to its explicit, separate `reports/` directory.
Existing outputs are protected from overwrite; choose another output directory
in a copied configuration for a new run. `--config PATH` selects that file and
`--method ic_balanced` runs only the selected method.

The earlier `configs/accuracy_pilot.py` remains an explicit strong-prior control:
its `latent_*` methods also subtract the known forcing term \(N_3t^*\) from
the trial function. That builds in near-reference nominal front motion, so
their low errors cannot establish autonomous evolution learning. Use that
configuration explicitly only when studying the effect of this prior; the
default `ic_*` controls remove it.
See the [IC-only validation record](../../docs/tasks/archive/icemelting-weak-prior.md)
for matched-budget controls, reproducibility and limits of the accuracy claims.

## Independent FEM comparison

The rewritten independent FEM solver reads physical coefficients and IC
metadata directly from the saved PINO config and held-out manifest declared
in [configs/solver_base.py](configs/solver_base.py). It fixes
\(\epsilon^*=0.0225\) independently of mesh size. It uses full-cube CG2
tetrahedral finite elements, consistent mass, degree8reaction quadrature,
IMEX BDF2 time integration and natural homogeneous Neumann conditions
on every face. The exact rotated/shifted ellipsoid SDF is shared as prescribed
initial data; no neural residual, nonlocal operator or future labels enter FEM.

Run one explicit mesh/time resolution in the separate environment:

```bash
conda activate fenicsenv
mpirun --allow-run-as-root -np 12 python examples/icemelting3d/solver.py \
  --config examples.icemelting3d.configs.solver_base \
  --index 5 --subdivisions 48 --dt 0.005 \
  --output /root/autodl-tmp/data/icemelting3d/operator_local_fem_v1/case005-n48-dt005.npz
```

Production and FEM defaults use \(T_c=5\,\mathrm{s}\) and \(t^*\in[0,1]\), so
their endpoint is physical time5 s. Match interface thickness and the saved
physical scales before comparing outputs. Changing \(T_c\) recomputes all
time-scaled coefficients; the normalized endpoint remains \(t^*=1\).

Every single-resolution output starts with `validated_reference=false`;
the aggregate refinement/acceptance record must accompany any reference claim. Available
adjacent mesh levels32/48/64/96 and dt.005/.0025/.00125/.000625 retain identical physics;
per-time phase/solid relative-L2 gates are5e-4/1e-2. These are explicit
validation choices, not a claim that an untested coarse output is converged.
Evaluate at the same physical points/times as both saved neural checkpoints,
and distinguish total local-FEM errors from fitting errors against the native
graph target. Their scalar error norms do not add in general. Run FEM MPI and
JAX GPU work in their separate environments without concurrency.

IMEX BDF2 treats local diffusion implicitly, extrapolates the weak reaction
load as \(2G(\phi^n)-G(\phi^{n-1})\), and uses BE/explicit-reaction startup.
It reuses constant consistent mass/stiffness matrices, without mass lumping
or nodal reaction collocation. Homogeneous-ODE step refinement verifies
second-order convergence. The explicit `time.scheme="implicit_midpoint"`
control retains a nonlinear fully implicit alternative. See
[Ascher, Ruuth and Wetton (1995)](https://doi.org/10.1137/0732037).
The N24 pilot was rejected for insufficient resolution (initial interpolation
already undershoots -1.005); its result is not a validated reference.

### Local versus nonlocal reference check (one held-out ellipsoid)

For randomly selected held-out input5, the matching N96 CG2 FEM (7,189,057
DOFs, dt*=0.00125) and existing full128-cube native nonlocal reference give:

| Dimensionless time | 3D diagnostic phi relative L2 | Central-slice phi relative L2 |
|---|---:|---:|
| 0.00 | 0.0070% | 0.0103% |
| 0.25 | 0.0930% | 0.1495% |
| 0.50 | 0.1754% | 0.3210% |
| 0.75 | 0.1993% | 0.4548% |
| 1.00 | 0.1670% | 0.5351% |
| Five-time pooled | **0.1474%** | **0.3575%** |

Errors use the local FEM norm at identical24^3diagnostic nodes and128^2slice
points, not continuous/full-grid L2. Spatial64->96 max phase difference is
0.0378%; time.0025->.00125 is0.0295% on3Dnodes and0.0737% on the slice.
The user accepts this accuracy and stops the conditional finer solve; the
original strict0.05% slice gate remains recorded as failed, with explicit
`status=accepted_by_user`, not relabeled as passed. Native time refinement
is much smaller (max1.176e-7 relative).

Continue using the native reference for routine monitoring of the current
whole-field phase objective; no operator optimization is required for that
purpose from this one-case check. Retain local checks when making finer
accuracy comparisons. Reference choice can change model ordering at this
small error scale: reloaded final Adam pooled phi is0.2383% against native,
0.1479% against FEM; SOAP is0.0709% against native,0.1861% against FEM.
These errors need not add: fitting and reference-gap fields can partly cancel.
The late small-solid metric remains in the detailed report but is not the
user's current optimization target. No training/default/checkpoint changes.

[Reference fields](../../docs/tasks/archive/melting-local-fem-reference-fields.png),
[all metrics](../../docs/tasks/archive/melting-local-fem-evaluation.json),
[refinement and error curves](../../docs/tasks/archive/melting-local-fem-refinement-errors.png),
[formulation, validation and accepted scope](../../docs/tasks/archive/melting-operator-local-fem.md).
Local artifact: `/root/autodl-tmp/data/icemelting3d/operator_local_fem_v1/case005-imex-n96-dt0p00125.npz`;
its acceptance record is linked in the task. Native artifact:
`/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1/condition_005.npz`.
This is a single-input engineering validation, not uniform accuracy across
all ellipsoids or evidence for a local-AD speed comparison.

## Three-dimensional interface figures

The existing Adam operator checkpoint also has true3Dvisualizations for the
previously randomly selected held-out ellipsoids5/9/2. Query61times on a96^3XYZ
grid, then extract the actual phi=0 interfaces using marching cubes. No future
shape law or geometric smoothing is added. These are prediction visualizations,
not new reference solves or full-grid accuracy measurements.

For paper figures, use the matte orthographic
[five-time strip](../../docs/tasks/archive/melting-paper-3d-single.png)
([PDF](../../docs/tasks/archive/melting-paper-3d-single.pdf)) or the
[three-input gallery](../../docs/tasks/archive/melting-paper-3d-snapshots.png)
([PDF](../../docs/tasks/archive/melting-paper-3d-snapshots.pdf)). Sample2 is the
most elongated of this existing random trio based only on initial semiaxes.
All panels retain identical bounds, aspect and camera; time is0,1.25,2.5,3.75,5s.
The [time-colored3Dcontour overlay](../../docs/tasks/archive/melting-paper-3d-overlay.png)
([fully vector PDF](../../docs/tasks/archive/melting-paper-3d-overlay.pdf))
combines silhouettes and three orthogonal interface-intersection curves.
An optional [nested cutaway](../../docs/tasks/archive/melting-paper-3d-cutaway.png)
removes a front quadrant only for display; it must not be presented as a physical
hole. Different colors represent different times, not simultaneous phases.

PDF/SVG keep labels/axes/curves vector; dense filled surface layers are
rasterized at high resolution. PNG figures use450dpi. Reusable field/mesh data:
`/root/autodl-tmp/data/icemelting3d/operator_render3d_v1`.
[Task, scripts and verification](../../docs/tasks/archive/melting-operator-3d-render.md).
Optional mesh/render packages were installed in an isolated temporary environment;
training dependencies and the checkpoint remain unchanged. The earlier decorative
animation is retained as a preview; paper-style figures are the recommended output.

## Project split (2026-10-24)

This example belongs to `nonlocal-pinn`. Install the project and run from its
repository root; existing external data/output paths are retained. Shared
implementation paths changed during the split; see
[the source map](../../docs/migration/source-map.json) and
[migration notes](../../docs/migration/README.md).
