# 2D KKS Phase-Field Electropolishing — PINN + FEM Reference

## SOTA default

Seed is not a strategy parameter. The canonical `get_config()` has no
`train.seed`; choose a run with `train.py --seed=2026`. With no override, the
common untuned run seed is42. Saved resolved config metadata still includes
`train.seed` for reproducibility and old checkpoint compatibility. The best
observed run below is provenance, not a seed to optimize or an accuracy promise.

The matched default requires the sparse graph/minibatches and does not accept
legacy causal/interface weighting, bias schedules or optional diagnostic
switches; unsupported overrides fail explicitly. Use a legacy config for those
controls. Changing physical/operator/grid settings requires a matching reference.

Use [train.py](train.py) with [configs/train_base.py](configs/train_base.py):

```bash
source .venv/bin/activate
python -m examples.electropolish2d.train
# Equivalent direct script:
python examples/electropolish2d/train.py
# Give a fresh cold start an explicit name:
python -m examples.electropolish2d.train --config.log.run_name=my-electropolish-run
```

The default matches the **best observed terminal phi+c relative L2** at5000
updates among11 completed matched-reference runs. It also has the smallest
terminal phi error among45 saved checkpoints in the current nine-run cohort.
This is the original-residual, random-data **seed7** model shown in the
[three-row field panel](../../docs/tasks/archive/electropolish-best-phi-panel.md).
The data mode defines the strategy; seed7 identifies this historical observation.
Default invocation uses the common run seed42, which did not reproduce this
best observation. Seed is deliberately not selected to promise low error.

| Setting | Default SOTA |
|---|---|
| Physical horizon / updates | 5 s /5000 |
| Network | MLP,6 layers x64,gelu; tensor_split=True |
| Spatial / temporal embeddings | 64 /64 dimensions, scales2 /2 |
| Operator / grid | Sparse cubic graph,delta0.02,101x51 |
| Core / pool / times | 256 /100 /16; uniform t* in[0.001,1) |
| IC samples | 1000 global +1000 near-interface random points |
| BC samples | 128 random top +128 random bottom points |
| Objective / concentration map | Original phase residual; physical phi,c data; KKS |
| CH evaluation / norm | `physical_difference` /L1 |
| AC / CH scale | 1 /1 |
| Optimizer / initial rate | SOAP /1e-3 |
| SOAP beta1 / beta2 / refresh | 0.9 /0.99 /5 updates |
| Rate schedule / gradient clip | Continuous0.95 per1000 updates,floor1e-5 /50 |
| GradNorm alpha / refresh / cap | 1 /25 updates /1000 |
| Stagger / causal / interface weight / time window | Off /off /0 /off |
| Late rate / CH-RMS continuation | Off /off |
| Matmul / GPU runtime | `highest`; `--xla_gpu_deterministic_ops=true` |

Network/time RNG uses the runtime `--seed`; the prescribed rough surface and fixed random
IC/BC samplers retain their original seed42. The entry enables the audited GPU
flag before backend initialization, keeps unrelated XLA flags and rejects a
conflicting setting. This is not a cross-hardware exact-replay guarantee.
Both IC and BC are soft losses. Positive-time interior reference fields enter
only evaluation; no hard IC or interface-motion prior is introduced.

| Verified final5 s metric | Selected checkpoint |
|---|---:|
| phi relative L2,all5151 nodes | 0.0011378188435460965 |
| c relative L2,all5151 nodes | 0.0013578160145092115 |
| q relative L2,free nodes | 0.0115080 |
| Dq relative L2,free nodes | 0.0377613 |

Checkpoint:
`/root/autodl-tmp/tf-logs/electropolish2d/20261025-data-physical-icrandom_interior-bcrandom-phase-stagger0-seed7/ckpt_005000.eqx`.
Reference:
`/root/autodl-tmp/data/electropolish2d/results_matched_graph_20261024.npz`.
The reference is independent SciPy integration of the same fixed graph, with
time-refinement discrepancy1.52e-8; it is not the older local-FEM array or a
continuum-convergence result. Pooled phi/c L2 is0.00213641/0.00260170, distinct
from the terminal values above.

**Seed limitation:** the same random-data setting with seed42 ends at
phi/c0.29093/0.29901. The all-node IC/BC + `s_bc` candidate instead gives
0.002106/0.002518(seed42) and0.002389/0.002869(seed7), but does not beat the
selected seed7 endpoint. To reproduce that candidate through the same entry,
set `experiment.initial_sampling=matched_graph`,
`experiment.boundary_sampling=matched_graph`, `experiment.data_loss_mode=s_bc`,
and the runtime argument `--seed=42`. The best observed seed is not a general strategy guarantee.
See the [optimization record](../../docs/tasks/archive/electropolish-field-optimization.md),
[convergence curves](../../docs/tasks/archive/electropolish-convergence-curves.md)
and [current promotion](../../docs/tasks/archive/phasefield-sota-defaults.md).

Empty `log.run_name` generates a fresh timestamped directory. Existing explicit
names fail; no saved checkpoint is overwritten or automatically resumed.
Outputs include config/source snapshots, initial/1000-update/final weights,
TensorBoard, loss/metric JSONL, `prediction.npz` and `summary.json`.
Models/references load by direct paths with compatibility checks, no file hashes.
The validated trainer is shared internally with corrosion; `train.py` is the
canonical user entry. The old301x151/6000-update preset and FD/local controls
remain under [configs/train_legacy.py](configs/train_legacy.py).


### Independent seed check —2026-10-26

The same strategy with new predeclared `--seed=2026`,5000 updates and5 s gives
terminal phi/c **0.00283171/0.00342674**, with pooled phi/c0.00225795/0.00275241.
Independent CPU q/Dq free errors are0.0185384/0.0549065. Every column at all21
times retains one interface. Late4000..5000 phi median is0.00463667,
range0.00275226–0.00712757: physical fields are accurate, with visible late
fluctuations. This additional run reproduces low error, but the existing
seed42 result0.29093/0.29901 remains a counterexample to strategy-wide stability.
Do not promote a successful seed to a strategy parameter. No fixed-control
setting was retuned; see the [validation](../../docs/tasks/archive/phasefield-seed-stability.md)
and [curves](../../docs/tasks/archive/phasefield-seed-stability-results-convergence.png).

## Motivation

This example compares nonlocal, finite-difference (FD) and local AD treatments
of a rough dissolving interface. Left/right walls have no external transport.
The graph implements this through in-domain pairs; the FD control truncates
its stencil. Neither is pointwise identical to a classical Neumann condition
at finite resolution. A local AD model needs explicit boundary treatment.

The initial interface is a **rough, randomly-generated surface** described by
a signed-distance function (SDF) to a piecewise-linear profile.

## Physical Setup

A metal workpiece is held between **insulating (inert) fixtures** on the left
and right sides; only its top surface is exposed to the electrolyte.  The
fixture walls are electrochemically inactive — no reaction, no current.

```
                   electrolyte  φ=0, c=0 (top, Dirichlet)
  ┌─────────────────────────────────────────────┐
  │                                             │
  │  electrolyte   φ≈0,  c≈0                   │ ← inert wall
  ├──────── rough interface ~~~~~~~~~~~~~~~~~~~─┤   ∂φ/∂n = 0
  │                                             │   ∂c/∂n = 0
  │  solid metal   φ≈1,  c≈1                   │ ← inert wall
  │                                             │
  └─────────────────────────────────────────────┘
                   solid metal  φ=1, c=1 (bottom, Dirichlet)
```

- **Top** ($y=0.5$): liquid electrolyte, $\phi=0$, $c=0$
- **Bottom** ($y=0$): solid metal, $\phi=1$, $c=1$
- **Left / Right** ($x=0$, $x=1$): inert insulating fixtures, $\partial\phi/\partial n=0$, $\partial c/\partial n=0$
- **Interface**: initial rough surface at $y\approx 0.33$; metal dissolves at the interface

The Neumann (zero-flux) condition on the side walls is physically justified:
the insulating fixture does not participate in the electrochemical reaction,
so there is no flux of either the phase-field or the concentration field across
the fixture surface.  This is the standard inert-wall assumption used in
phase-field corrosion and electropolishing models.

### Boundary interpretation

In-domain pair interactions have no exchange with points outside the graph.
This is a discrete no-external-flux model; it does not impose an exact zero
normal derivative on the neural field at each boundary point. The truncated
FD graph also supplies discrete boundary treatment without a separate loss.

The rough initial profile may not satisfy the classical local Neumann trace
at time zero. A local PDE can develop an initial boundary layer. Graph and
local FEM solutions can therefore differ near walls and interfaces, even when
physical parameters agree. Compare the PINN to direct integration of its own
graph to isolate training error, and assess convergence toward a local PDE
separately. This example alone does not establish operator superiority.

## Governing Equations (Local / Classical)

2D KKS phase-field model for electropolishing. Domain: $\Omega = [0,1] \times [0,0.5]$ (non-dimensional).

With $\Delta c_0 = c_{Se} - c_{Le}$ and
$q = c - h(\phi)\Delta c_0 - c_{Le}$, the free-energy density is

$$f = A q^2 + \omega_\phi g(\phi) + \frac{\alpha_\phi}{2}|\nabla\phi|^2.$$

**Allen-Cahn:**

$$
\frac{\partial \phi}{\partial t}
= L\left[\alpha_\phi \nabla^2\phi
- \omega_\phi g'(\phi)
+ 2Aq\Delta c_0 h'(\phi)\right]
$$

**Cahn-Hilliard:**

$$\frac{\partial c}{\partial t} = D\,\nabla^2\!\left[c - h(\phi)\Delta c_0 - c_{Le}\right]$$

The constant $c_{Le}$ is inert under the CH Laplacian, but remains part of
$q$ in the Allen-Cahn driving force.

**Interpolation functions:**

$$h(\phi) = -2\phi^3 + 3\phi^2,\quad g(\phi) = \phi^2(1-\phi)^2$$

### Geometry & Boundary Conditions

```
y=0.5  ┌─────────────────────────┐  Top: φ=0, c=0 (liquid electrolyte)
       │                         │
       │   domain                │  Left/Right: Neumann (∂φ/∂n=0, ∂c/∂n=0)
       │                         │  → no BC loss terms — key ablation point!
       │  ~~~~ rough ~~~~        │  Initial interface (SDF-based tanh)
       │                         │
y=0    └─────────────────────────┘  Bottom: φ=1, c=1 (solid metal)
```

| Boundary | Condition |
|---|---|
| Top ($y=0.5$) | $\phi = 0,\; c = 0$ (liquid electrolyte) |
| Bottom ($y=0$) | $\phi = 1,\; c = 1$ (solid metal) |
| Left ($x=0$) | $\partial\phi/\partial n = 0,\; \partial c/\partial n = 0$ (Neumann) |
| Right ($x=1$) | $\partial\phi/\partial n = 0,\; \partial c/\partial n = 0$ (Neumann) |

**Initial condition** — SDF-based tanh profile on rough surface:

$$y_{\text{surf}}(x) = y_0 + \sum_{i=1}^{n_\text{modes}} a_i \bigl[\sin(2\pi f_i x + \theta_i) + \cos(2\pi f_i x + \theta_i + 0.3)\bigr]$$

$$\phi_0(x,y) = \frac12\bigl[1 + \tanh(k \cdot \text{sdf}(x,y))\bigr],\quad c_0 = h(\phi_0) \cdot c_{Se}$$

where $k = \sqrt{\omega_\phi / (2\alpha_\phi)}\cdot L_c$, and the SDF is the signed perpendicular distance to the piecewise-linear profile $y_{\text{surf}}(x)$ (positive below / solid side).

---

## Non-Dimensionalisation

Characteristic length $L_c = 100~\mu\text{m}$ and time $T_c$ define:

$$\bm{x}^* = \frac{\bm{x}}{L_c},\qquad t^* = \frac{t}{T_c}$$

Four non-dimensional parameters:

$$N_{\text{CH}} = \frac{D T_c}{L_c^2},\quad N_{\text{AC,1}} = 2A L T_c,\quad N_{\text{AC,2}} = L\omega_\phi T_c,\quad N_{\text{AC,3}} = \frac{L\alpha_\phi T_c}{L_c^2}$$

### Non-dimensional Allen-Cahn

$$\frac{\partial\phi}{\partial t^*} - N_{\text{AC,3}}\,\nabla^{*2}\phi + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,1}}\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr](c_{Se}-c_{Le})h'(\phi) = 0$$

### Non-dimensional Cahn-Hilliard

$$\frac{\partial c}{\partial t^*} - N_{\text{CH}}\,\nabla^{*2}\!\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr] = 0$$

with $\Delta c_0 = c_{Se} - c_{Le}$.

---

## Nonlocal (Peridynamics-Inspired) Formulation

Replacing $\nabla^{*2}$ with the nonlocal operator $D_\delta$:

$$\frac{\partial\phi}{\partial t^*} + N_{\text{AC,3}}\,D_\delta\phi + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,1}}\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr]\Delta c_0\,h'(\phi) = 0$$

$$\frac{\partial c}{\partial t^*} + N_{\text{CH}}\,D_\delta\!\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr] = 0$$

In the local limit $D_\delta \approx -\nabla^{*2}$. The nonlocal operator is defined by a compactly-supported kernel $J(r)$:

$$(D_\delta u)(x_i) = \sum_{j \in N_\delta(i)} w_{ij}\,(u_i - u_j),\quad w_{ij} = -D_{ij}$$

Edge weights pre-computed from the `SparseD` operator. Supported kernels: `cubic`, `quartic`, `wendland`.

### Finite Difference (Ablation)

With `configs/train_legacy.py`, setting `--config.nl.method=fd` switches to a 5-point FD Laplacian on the structured grid. The FD stencil is truncated at left/right Neumann boundaries, while the nonlocal operator requires no boundary handling.

---

## Physical Parameters

| Symbol | Value | Unit | Description |
|---|---|---|---|
| $D$ | $8.5 \times 10^{-10}$ | m²/s | Diffusion coefficient |
| $A$ | $5.35 \times 10^7$ | J/(m³·s) | Reaction constant |
| $L$ | $1.0 \times 10^{-6}$ | m/(J·s) | Interface mobility |
| $\alpha_\phi$ | $1.03 \times 10^{-4} / 4$ | J/m | Gradient energy coefficient |
| $\omega_\phi$ | $1.76 \times 10^7$ | J/(m³·s) | Barrier height |
| $c_{Se}$ | 1.0 | — | Equilibrium concentration (solid) |
| $c_{Le}$ | $5100 / 1.43 \times 10^5$ | — | Equilibrium concentration (liquid) |

### Non-Dimensionalisation

| Scale | Value | Description |
|---|---|---|
| $L_c$ | $100~\mu\text{m}$ | Characteristic length |
| $T_c$ | 5.0 s | Characteristic time |

---

## Rough Surface Configuration

| Parameter | Default | Description |
|---|---|---|
| `y0` | 0.33 | Mean interface height |
| `n_modes` | 6 | Number of Fourier modes |
| `amplitude` | 0.03 | Max amplitude of roughness |
| `freq_lo` | 1.0 | Min spatial frequency |
| `freq_hi` | 5.0 | Max spatial frequency |
| `seed` | 42 | RNG seed for reproducibility |

The signed distance is computed via the perpendicular distance to each segment of the piecewise-linear profile, using a sliding window (`window=16`) over query points for efficiency.

---

## Project Structure

```
electropolish2d/
├── solver.py               # FEM reference solver (FEniCS, MPI)
├── train.py                 # PINN training (MLP / FiLM)
├── losses.py                # Loss functions (AC, CH, BC, IC)
├── configs/
│   ├── solver_base.py       # FEM solver config
│   ├── train_base.py        # Default observed-field SOTA
│   └── train_legacy.py      # Historical large-grid/FD/local controls
├── README.md
```

---

## Usage

### Run FEM Solver

```bash
conda activate fenicsenv
cd examples/electropolish2d

# Serial
python solver.py --config=configs/solver_base.py

# MPI (18 processes)
mpirun --allow-run-as-root -np 18 python solver.py --config=configs/solver_base.py
```

Output: `.npz` file at `/root/autodl-tmp/data/electropolish2d/results_rough.npz` with `X`, `Y` meshgrids, `time` array, and `phi`, `c` fields over time. Also saves snapshot plots (`snapshot_phi.png`, `snapshot_c.png`).

### Train PINN

```bash
# From project root
python examples/electropolish2d/train.py --config=examples/electropolish2d/configs/train_base.py

# FD ablation (legacy loop, separate reference validation required)
python examples/electropolish2d/train.py --config=examples/electropolish2d/configs/train_legacy.py \
    --config.nl.method=fd --config.log.run_name=fd_run

# Spatial minibatch
python examples/electropolish2d/train.py --config.sample.K_core=400 --config.sample.K_pool=400
```

### Legacy-loop Config Options

```bash
# Architecture
--config.arch.type=film          # or "mlp"
--config.arch.hidden_dim=64
--config.arch.spatial_scale=2.0   # single scale or tuple, e.g. (1.0, 3.0, 6.0)

# Nonlocal operator
--config.nl.delta=0.02
--config.nl.kernel=cubic
--config.nl.method=sparse         # "sparse" | "fd"

# Spatial minibatch (core+halo sub-sampling)
--config.sample.K_core=400        # 0 = full grid
--config.sample.K_pool=400

# IC sampling
--config.sample.N_ic_global=2000
--config.sample.N_ic_local=2000   # oversampling near interface

# Staggered training
--config.loss.stagger_steps=1000
--config.loss.stagger_switch_every=25

# Hard constraint (enforce c bounds)
--config.arch.use_hard_constraint=True

# Causal training
--config.loss.use_causal=True
--config.causal.n_chunks=0
```

### Key Design Decisions

- **Side boundaries have no separate loss term.** The graph omits exterior
  edges and FD truncates its stencil; these are discrete boundary treatments,
  without an exact pointwise local Neumann guarantee.
- **Rough initial interface** is defined by SDF to a piecewise-linear profile, not simple y-distance, for smooth $\phi$ at t=0
- **IC oversampling** near the interface (`N_ic_local`) ensures the sharp tanh transition is adequately resolved
- **Output constraint** (`use_hard_constraint=True`) uses the KKS mixture
  \(c=c_L+h(\phi)(c_{Se}-c_{Le})\), with bounded \(c_L\); it is independent
  of the soft initial-condition loss.


## Matched residual experiment

The shared [experiment runner](../corrosion2d/residual_experiment.py) compares
original and transformed phase residuals with the existing KKS output map and
soft IC/BC losses. Its separate 101x51 sparse-graph preset retains this case's
physical parameters, rough surface and 5 s horizon. The canonical default
now uses this validated engine with the SOTA settings above; the research CLI
retains its historical defaults for archived commands. See the [corrosion experiment instructions](../corrosion2d/README.md#matched-residual-experiment)
for settings, reference checks and commands; use `--case electropolish2d`.
The [paired pilot record](../../docs/tasks/archive/coupled-latent-residual-experiments.md)
documents field errors and extra-interface diagnostics; algebraic equivalence
does not establish a precision improvement.

The FEM solver now evaluates the same SDF initial function directly, applies
Dirichlet traces to its initial vector, and uses actual time increments when
hitting save/final times. Existing reference files are historical: a source
fix does not update their contents. New graph references have distinct names
and explicit model/BC/physical-time metadata.

### Research data and temporal-window controls

The shared runner also supports all graph-node analytic ICs (with the same
Dirichlet overrides as the graph reference), all nodal Dirichlet BCs, and
`--data-loss-mode s_bc`. This last option fits the existing KKS concentration
variable \(s=c_L/c_{Le}\) at the boundaries; it changes data-loss coordinates
and concentration scale while preserving the prescribed physical BCs at zero
loss. Away from zero loss this is not merely a constant multiplier on total-c
error, since \(c=c_L+\Delta c\,h(\phi)\) couples the two outputs.
`s_bc_ic` additionally normalizes IC
concentration; neither option imposes hard initial conditions.

`--time-ramp-steps R --time-ramp-early a` is a **research-runner sampling**
curriculum. At zero-based update \(k\), sample the same fixed-size time batch
uniformly from \([t_{min},T_k]\), with

$$
T_k=a+(1-a)\min(k/R,1),\qquad R>0.
$$

The final \(5000-R\) updates cover the full5 s horizon. Disabled by default
(`loss.t_ramp_steps=0`), it reproduces the original random time samples exactly.
The early sampling objective intentionally emphasizes the current time window;
there is no importance correction making it a full-horizon loss before \(R\).
It introduces no interface-motion/reference-trajectory targets, residual
weight, network constraint or optimizer reset. Enabled settings require
\(t_{min}<a<1\), integer \(0<R<\text{iterations}\), and full-horizon endpoint1.
In this research runner, `loss.t_ramp_early`/`t_ramp_late=1` specify sampling
endpoints. Canonical matched configs use this sampling mechanism; the helper
in the explicitly selected legacy loop is a separate time-weight mechanism.

For example, from the repository root with no FEniCS/MPI process running:

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m examples.corrosion2d.residual_experiment train \
  --case electropolish2d --formulation phase --seed 42 \
  --data-loss-mode s_bc --initial-sampling matched_graph \
  --boundary-sampling matched_graph
```

This candidate uses SOAP, lr1e-3,5000 updates, the original phase residual and
full-horizon time sampling. Both initial and boundary values remain soft loss
terms. Full-grid IC sampling uses5151 analytic initial-value points, rather than
2000 random IC points; graph-node BCs use101 top and101 bottom points, rather
than128 random points on each. Positive-time interior reference fields are
used for evaluation, not supervised training. The update budget is equal;
computational work is not, because IC sample counts differ.

To test the separate curriculum, add `--time-ramp-steps 3000
--time-ramp-early 0.05`: the upper time bound starts at0.25 s, reaches5 s at
index3000, and the last2000 updates cover the full interval. The tested window
does not beat its original-residual counterparts at the5000 endpoint; it
reduces extra bands in the full-latent arm, whose physical errors remain higher.
Compare `--formulation latent` with the same data/time settings when assessing
the transformation, retaining phi/c L2, q/Dq and interface-count diagnostics.

The [optimization record](../../docs/tasks/archive/electropolish-field-optimization.md)
contains two-seed validation and direct checkpoint/config paths. The candidate
avoids the large-field-error seed42 result and has a single interface in both
tested seeds; seed7's already accurate random-data baseline has smaller terminal
phi/c and Dq errors, while the candidate improves q. These two seeds do not
establish universal superiority. The current default explicitly selects the
best observed random-data seed7 terminal field instead of that candidate.

Completed run directories are preserved; the CLI refuses existing names. For
a fresh replay, use `train.py` with explicit `--config.experiment.*` overrides
and a new `--config.log.run_name`. Load saved configs/checkpoints by their
explicit paths.

## Project split (2026-10-24)

This example belongs to `nonlocal-pinn`. Install the project and run from its
repository root; existing external data/output paths are retained. Shared
implementation paths changed during the split; see
[the source map](../../docs/migration/source-map.json) and
[migration notes](../../docs/migration/README.md).
