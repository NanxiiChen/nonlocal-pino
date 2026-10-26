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
`/root/autodl-tmp/tf-logs/icemelting3d/20261024-5s-soap-lr1e3-seed42/ckpt_005000.eqx`.
Its full-grid terminal phi relative L2 against the same-graph numerical
reference is **6.10970108092214e-5**. The direct reference is
`/root/nonlocal-pinn/reports/icemelting-weak-prior-20261024/reference.npz`;
pass it with `--config.eval.graph_reference_path=...` to enable graph-L2
monitoring. This optional monitoring does not change training.
SOTA here means the selected verified5 s graph-error result, distinct from
error against the approximate analytic profile and the historical4 s run
whose directory retains a `sota` tag. The saved run originally declared20000
updates and was stopped at the user-selected5000 checkpoint; the default
correctly declares5000. See the
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
| Optimizer | SOAP, explicit FP32 `matmul_precision="highest"` |
| Additional weighting | Interface weighting zero; causal weighting disabled by default |
| Training data | PDE and prescribed IC; reference fields are evaluation-only |
| Time convention | Physical 0–5 s mapped to \(t^*\in[0,1]\) |
| Monitoring | Separate analytic/graph errors and pooled/terminal phase-field relative L2 |
| Loading | Explicit config/checkpoint/reference paths, without file-hash requirements |

The canonical default is the selected SOAP configuration from the
[five-second optimizer comparison](../../docs/tasks/archive/icemelting-soap-adam-5s.md):
5000 updates, initial learning rate 0.001, seed 42, 2000 spatial core points and
64 time samples per update. Run `python -m examples.icemelting3d.train` to start
this configuration. The [artifact index](../../artifacts/index.yaml) records its
selected checkpoint under `default_training`; the older `sota` directory tag
belongs to the historical 4 s bounded run.

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

The independent FEM solver runs in its separate environment:

```bash
conda activate fenicsenv
mpirun --allow-run-as-root -np 12 python examples/icemelting3d/solver.py
```

Production and FEM defaults use \(T_c=5\,\mathrm{s}\) and \(t^*\in[0,1]\), so
their endpoint is physical time5 s. Match interface thickness and the saved
physical scales before comparing outputs. Changing \(T_c\) recomputes all
time-scaled coefficients; the normalized endpoint remains \(t^*=1\).

The FEM default uses 64 elements per axis, normalized time step 0.001,
Crank–Nicolson (`theta=0.5`) and natural homogeneous Neumann boundary conditions.
Its interface thickness is computed from the mesh in [solver.py](solver.py);
production training fixes \(\epsilon^*=0.0225\). See
[configs/solver_base.py](configs/solver_base.py). Run FEM MPI and JAX GPU work
in their separate environments without concurrency.

## Project split (2026-10-24)

This example belongs to `nonlocal-pinn`. Install the project and run from its
repository root; existing external data/output paths are retained. Shared
implementation paths changed during the split; see
[the source map](../../docs/migration/source-map.json) and
[migration notes](../../docs/migration/README.md).
