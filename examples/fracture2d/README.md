# 2D Rounded-Notch Fracture: Local FEM Reference and GPU Direct Nonlocal Solver

This example is the 2D stage of the fracture program. The implementation order
is local FEM reference, direct nonlocal solver, then nonlocal PINN. The
**adopted direct scheme is one self-contained GPU-only file**,
[`nonlocal_baseline.py`](nonlocal_baseline.py); the next stage (PINN) is
tracked in
[`docs/tasks/fracture2d-nonlocal-pinn.md`](../../docs/tasks/fracture2d-nonlocal-pinn.md).

The historical research stack (per-module implementation, monolithic /
arc-length / reduced solvers, durable workflow, stage-specific test suites) is
parked under the ignored `_legacy/` directory and in git history. Only the
adopted scheme is maintained here.

**Both available reference solutions — the local FEM archive and the nonlocal
direct solution — are warning-mode, non-strict diagnostics.** Neither is a
strictly converged reference; the exact nonlocal issues are listed in
[Validation status and accepted limitations](#validation-status-and-accepted-limitations).

## Reference solutions and data

The case ships with two independent reference solutions and the frozen model
inputs. Load all archives with `np.load(..., allow_pickle=False)`.

| What | Path | SHA256 |
|---|---|---|
| Adopted solver entry (GPU-only) | `examples/fracture2d/nonlocal_baseline.py` | — |
| Nonlocal material cloud (3395 nodes, `d048` graph) | `/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz` | `fa2ebf165f2e89ec88c63116346401b5039263e8e4c1a1917f6f13a3f4541267` |
| Nonlocal reference solution (compact, fresh) | `/root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz` | `5f3bc12bc4bc31718cce4c4aed18d47314fd80e3a207bec0180ac6cdd93f7bab` |
| Nonlocal reference archive (research) | `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz` | `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc` |
| Local FEM reference (point anchor, warning mode) | `/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz` | `e9634f1f8fdb21116b50ad453d37115746c4a5a09f828e76245a9c59f6107d17` |
| FEM figures + config | `/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/` (`config.json`, `reaction_displacement.png`, `representative_fields_*.png`, `stress_strain.png`) | — |
| PINN task | [`docs/tasks/fracture2d-nonlocal-pinn.md`](../../docs/tasks/fracture2d-nonlocal-pinn.md) | — |
| Archived direct-stage record | [`docs/tasks/archive/fracture2d-lean-direct-solver.md`](../../docs/tasks/archive/fracture2d-lean-direct-solver.md) | — |

Archive schemas (see [Adopted solution and figures](#adopted-solution-and-figures)
for the provenance and limitations):

- **material cloud** — `coordinates (3395,2) f8`, `triangles (6284,3) i8`,
  `volumes (3395,) f8`, `bottom`/`top (3395,) bool`, `anchor_particle int`,
  `d048 (E,2) i8`.
- **nonlocal compact solution** — `loads (76,) f8` (`0 -> 0.0075`),
  `union_q (76, 7402) f8` (material prefix is the first `2*3395 = 6790`
  columns), `damage (76, 3395) f8`, `history (76, 3395) f8`,
  `top_reaction (76,) f8`, `acceptance_status (76,)`, `warning (76,) bool`,
  `coordinates`, `triangles`, `metadata_json` (contains the exact `scheme`).
- **local FEM archive** — `u_bar (127,) f8` (`0 -> 0.0075`),
  `reaction_y (127,) f8`, `snapshot_u_bar (76,) f8`, `u (76, 3395, 2) f8`,
  `d (76, 3395) f8`, cellwise `H (76, 6284)` with `H_coords (6284,2)`,
  `sigma_xx/yy/xy (76, 6284)`, `gate_warning (127,) bool`, plus the material
  and geometry scalars (`E`, `nu`, `Gc`, `l0`, `k_res`, `L0`,
  `notch_*`, `history_method`, `plane_assumption`, `gate_mode`). It is
  `plane_strain`, `gate_mode="warning"` with `13` warning loads of `127`; peak
  reaction `642.4857295` at `0.0057`, terminal `3.97718` at `0.0075`.

## Problem

The material domain is the unit square with a finite-width slot opening on the
left boundary. The slot centerline is `y=0.5`, its total width is `2*l0`, and
its right end is a semicircle of radius `l0`. The foremost point of the tip is
`(0.5, 0.5)`, so the semicircle center is `(0.5-l0, 0.5)` with `l0=0.024`.

The calculation uses plane strain. The local FEM reference applies a vertically
constrained bottom edge, one horizontal point anchor at `(0,0)`, and a vertical
displacement on the top edge:

$$
u_y=0\quad\text{on }y=0,
\qquad u_x(0,0)=0,
\qquad
u_y=\bar u\quad\text{on }y=1.
$$

In the adopted direct nonlocal scheme the unit square is not constrained
directly. Structured exterior collars ("volume grips") are added below `y=0`
and above `y=1`, and the prescribed displacement is transferred through every
added nonmaterial collar particle; material interface particles remain free.
The point anchor and loading contract are given in
[Constraints, load coordinate, and reaction](#constraints-load-coordinate-and-reaction).

## Nonlocal discretization

### Material particle cloud and visibility graph

The production material cloud is **frozen** at
`/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz`
and is loaded directly by the solver with `np.load(..., allow_pickle=False)`.
It contains the tagged Gmsh mesh data: `coordinates`, `triangles`, barycentric
volumes, `bottom`/`top` masks, the lower-left `anchor_particle`, and the frozen
`d048` visibility graph.

The historical loader (`nonlocal_geometry.py`, with `load_gmsh_particle_cloud`,
`ROUNDED_SLOT_TAGS`, the polygonized-slot visibility contract, and the
`64*eps*L_geom` machine floor) is parked under `_legacy/` and remains in git
history. The cloud's barycentric nodal volumes are

$$
V_i=\frac{1}{3}\sum_{e\ni i}A_e .
$$

The graph is reciprocal, self-edge-free, and strictly inside the horizon
(`0 < r_ij < delta`); every retained or removed pair appears in both directions.
The fracture2d direct path is frozen to float64 geometry, volumes, graph
arrays, and operator inputs with JAX x64 enabled before construction.

### Weighted nonlocal damage operator

The nonlocal damage operator is the shared
`pinn.operators.ExplicitVolumeSparseD` (the same operator the PINN stage
reuses), constructed from the material cloud with the frozen cubic kernel:

```python
from pinn.operators import ExplicitVolumeSparseD

operator = ExplicitVolumeSparseD(
    cloud["coordinates"],
    cloud["volumes"],
    cloud["d048"],
    delta,
    "cubic",
)
```

For the 2D cubic kernel,

$$
(D_\delta d)_i=\sum_{j:(i,j)\in\mathcal E}
J_\delta(r_{ij})V_j(d_i-d_j),
\qquad
J_\delta(r)=\frac{120}{\pi\delta^4}(1-r/\delta)^2
\quad(0<r<\delta).
$$

With `M=diag(volumes)` the reciprocal graph satisfies `M @ D = (M @ D).T`, so
the fixed-history damage problem is the Euclidean SPD system

$$
S=M\left[\operatorname{diag}(G_c/\ell_0+2H)+G_c\ell_0D_\delta\right],
\qquad
S d=2MH .
$$

### Bond-associated correspondence mechanics

The solver builds one target-inclusive bond-associated support per directed
bond of the union graph. For each directed bond `(i,j)`:

$$
\mathcal C_{i|j}=\{j\}\cup\{k\ne i,j:(i,k)\in\mathcal E\ \text{and}\ (j,k)\in\mathcal E\}.
$$

The support shape tensor `K`, the endpoint coefficients `beta`, the target
vectors `xi`, and the normalized support weights `eta` are frozen per directed
bond from the cubic kernel and the particle volumes. Each bond deformation
gradient is `F=A K^{-1}`; the small-strain tensor is `epsilon=sym(F-I)`. With
the plane-strain Lamé constants `lambda`, `mu`,

$$
\psi=\tfrac12\lambda(\operatorname{tr}\epsilon)^2+\mu\,\epsilon:\epsilon,
\qquad
P=\lambda(\operatorname{tr}\epsilon)I+2\mu\epsilon .
$$

Rank-deficient shape tensors fail before inversion; there is no pseudo-inverse,
determinant clipping, kernel floor, stabilization, or fallback.

### Engineering volume grips (union cloud)

The solver copies the material cloud as an **exact prefix** and adds structured
collar particles without remeshing the material triangles. Collar offsets are

```text
k * h_g  for k*h_g < delta,  followed by the exact delta
```

so `h_g=0.02`, `delta=0.048` give offsets `[0.02, 0.04, 0.048]`: three layers
below `y=0` and three above `y=1`. The union volume measure is the sum of two
phase measures,

$$
V_i^{\star}=V_i^{m}+V_i^{g},
$$

where `V_i^m` is the material barycentric volume (zero on collar particles) and
`V_i^g` is the grip-phase volume (zero on material particles). Each grip pair
has total area exactly `2*delta`, so the union area is the material area plus
`2*delta`. The union graph is rebuilt by the same visibility rule on the union
triangles, and the material-induced subgraph must equal the frozen `d048` graph.

## Governing equations (adopted engineering-grip scheme)

### Phase-measured union energy

Mechanics uses the project-owned phase-measured A2 energy on the union cloud.
For union displacement `q` and material damage `d_m`,

$$
\Pi_{A2}(q;d_m)
=\sum_{i}\left[V_i^{m}g(d_i)+V_i^{g}g(0)\right]
\sum_{j}\eta_{ij}^{\star}\,\psi_{i|j}^{\star}(q),
\qquad
g(d)=(1-d)^2+k_{\mathrm{res}} .
$$

Material particles degrade with their own damage; grip particles stay intact and
never receive damage or history fields. This is the direct-baseline hybrid
convention: the complete undamaged elastic energy is degraded in mechanics,
while the history driver uses the Amor volumetric-deviatoric positive split.

### Amor history

For every union bond strain embedded in 3D with `epsilon_zz=0`,

$$
\psi^{+}=\frac12K\langle\operatorname{tr}\boldsymbol\epsilon^{(3)}\rangle_+^2
+\mu\,\boldsymbol\epsilon_{\mathrm{dev}}^{(3)}:
\boldsymbol\epsilon_{\mathrm{dev}}^{(3)},
\qquad
K=\lambda+\frac{2\mu}{3}.
$$

The source-centered driver is eta-averaged on union edges but retained only at
material source centers:

$$
\Psi^{+}_{m,a}(q)=\sum_{j:(a,j)\in\mathcal E^{\star}}
\eta^{\star}_{aj}\,\psi^{+}_{a|j}(q),\qquad a<N_m ,
$$

and the committed history is the pure maximum update

$$
H_n=\max\left(H_{n-1},\Psi^{+}_{m,a}(q_n)\right).
$$

Material-to-grip target edges contribute to material history; grip-source
centers construct no history. No history is accumulated across sweeps.

### Nonlocal AT2 damage

At fixed history, the material damage equation is the nonlocal AT2 system on the
material cloud and its frozen `d048` graph:

$$
\frac{G_c}{\ell_0}d
+G_c\ell_0\,D_\delta d
+2H d
=2H ,
\qquad\text{i.e.}\qquad
S d=2MH .
$$

Damage is never clipped or projected; raw range and healing violations are
reported and are hard failures in strict mode. Mechanics uses the hybrid stress
`sigma=g(d) sigma_0` through the center weights above; no second
irreversibility constraint or damage bound is added.

### Constraints, load coordinate, and reaction

- every added nonmaterial collar particle has its vertical DOF prescribed:
  bottom collar `u_y=0`, top collar `u_y=\bar u`; collar horizontal DOFs are
  free;
- material particles on `y=0` and `y=1` remain completely free vertically;
- only the original material anchor particle `(0,0)` has `u_x=0`;
- all other boundaries are natural traction boundaries;
- damage uses natural zero-flux conditions.

The continuation coordinate is the prescribed collar displacement `u_bar`. The
top reaction is the residual sum over the complete top volume grip.

## Numerical algorithm

A committed state is `(u_bar, q, d, H)`. For each target load and each stagger
sweep:

1. build a fresh fixed-damage union view at `d_iter` and solve mechanics-before
   at the target load on the GPU;
2. set `H_used=max(H_committed, Psi_plus_m(q_before))`;
3. solve the exact fixed-history damage equation for `d_solved`;
4. under-relax the candidate, `d_trial = d_iter + omega * (d_solved - d_iter)`
   with `omega=0.5`;
5. solve mechanics-after at `d_trial` on the GPU;
6. set `H_next=max(H_committed, Psi_plus_m(q_after))` and evaluate every gate.

A sweep is accepted in strict mode only when every coupled gate passes. The
declared gates are displacement, damage, history, reaction, committed damage
residual, raw damage range, healing, history monotonicity, force, and moment,
plus the finite/mechanics/damage-equation/globalization hard checks. When the
stagger budget is exhausted, the separate **stabilized-warning** acceptance may
commit the last complete finite candidate and records the exact bypassed gate
list; non-finite values, failed mechanics/CG solves, structural errors, and
failed work checks remain hard failures. Cutbacks halve the increment
(`cutback_factor=0.5`, up to `8`), down to an endpoint-snapped
`du_min=6.25e-6`. A one-time centered zero-damage work check validates the
mechanics work at `load=1e-4`, `epsilon=1e-5`, `relative_limit=1e-7`.

The solver is **GPU-only**: `JaxA2CPUReplayAdapter` is required, every device
solve is replayed through a fresh CPU A2 view for energy, residual, reactions,
force/moment, and the explicit free-residual gate, and only that
CPU-authoritative result backs the acceptance gates. The CPU CG reference path
and all non-adopted relaxation policies (Anderson and backtracking variants),
matched-load schedules, and archive-budget machinery were removed.

### Parameters

| Symbol / field | Value | Role |
|---|---|---|
| `delta` | `0.048` | nonlocal horizon (material and union graph) |
| `h_g` | `0.02` | collar layer spacing; offsets `[0.02, 0.04, 0.048]` |
| material particles | `3395` | material cloud from the frozen Gmsh cloud |
| `l0` | `0.024` | phase-field length; notch half-width and tip radius |
| `Gc` | `2.7` | fracture energy |
| `youngs_modulus` | `210.0e3` | `E` |
| `poisson_ratio` | `0.3` | `nu` |
| `k_res` | `1.0e-8` | residual stiffness in `g(d)` |
| `rank_rtol` | `1.0e-12` | BA shape-tensor rank tolerance |
| `visibility_tolerance` | `1.0e-12` | graph visibility input tolerance |
| kernel | `cubic` | `J_delta` above |
| `du_initial` | `1.0e-4` | nominal load increment |
| `du_min` | `6.25e-6` | minimum increment |
| `cutback_factor` / `max_cutbacks` | `0.5` / `8` | cutback schedule |
| `max_stagger_iterations` | `144` | sweeps per attempt |
| `omega` | `0.5` | fixed damped-Picard relaxation |
| `cg_rtol` / `cg_atol` / `cg_maxiter` | `1e-10` / `1e-13` / `3000` | CG |
| coupled gate tolerance | `atol=1e-9, rtol=1e-8, floor=1.0` | displacement/damage/history/reaction/force/moment |
| damage residual tolerance | `atol=1e-10, rtol=1e-8, floor=1.0` | committed damage residual |
| `healing_atol` / `damage_range_atol` | `1e-10` / `1e-10` | raw range and healing |
| work check | `1e-4` / `1e-5` / `1e-7` | centered zero-damage work |
| `acceptance_mode` | `stabilized_warning` | honest warning labeling |
| `load_final` | `0.0075` | adopted endpoint |

All values above are the defaults of `nonlocal_baseline.py`; they reproduce the
adopted archived scheme field-by-field.

## Adopted solution and figures

The adopted complete solution covers `u_bar = 0 -> 0.0075`:

- fresh re-solve:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz`,
  SHA256 `5f3bc12bc4bc31718cce4c4aed18d47314fd80e3a207bec0180ac6cdd93f7bab`;
- original research archive:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz`,
  SHA256 `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc`.

Statistics: `76` states (initial + `65` strict + `10` warning-accepted); peak
reaction `641.34017` at `0.0061` (local FEM warning reference: `642.48573` at
`0.0057`, archive path in [Reference solutions and data](#reference-solutions-and-data));
terminal reaction `69.15426` at `0.0075` (FEM: `3.97718`). Drift
points (`|u_i|/u_bar > 1.08`) are `0` through `0.0061`, `2` at `0.0062`,
`3/5/10` at `0.0064/0.0065/0.0066`, and `28`-`31` from `0.0067` to `0.0075`.
At the endpoint the drift-excluded ranges are `d in [6.57e-6, 0.98954]` and
`|u| in [3.26e-7, 7.726e-3]`, while the drift-inclusive maximum `|u|` is
`0.142`. Figures (coolwarm, per-panel color bars, drift points drawn as black
open circles but excluded from the color ranges):

- `.../direct_nonlocal_final/fields_drift_excluded_ranges.png`
- `.../direct_nonlocal_final/response_curve.png`
- `.../direct_nonlocal_final/fresh_v1_figures/` (fresh re-solve)

The archived solution is explicitly `field_equations_validated=false` and
`validated=false`; it is a diagnostic warning reference, not a strict or
field-validated production artifact.

## Nonlocal PINN (in progress)

### Operator and model reuse

The FP32 PINN retains the same material particles, union-collar construction,
hard constraints, A2 energy/residual assembly, Amor and AT2 laws, and physical
parameters as the direct solver. Because float32 rounding changes strict
`r < delta` membership, [`fp32.py`](fp32.py) independently rebuilds the
union/material graph, phase quadrature, BA supports and cubic damage operator.
Its graph is **not identical** to the frozen FP64 direct graph. The adopted
direct implementation remains [`nonlocal_baseline.py`](nonlocal_baseline.py):

| Need | Symbol |
|---|---|
| Union cloud, collars, constraints, `d048` prefix identity | `build_union_geometry_from_arrays`, `UnionGripGeometry`, `AffineConstraints`, `build_volume_grip_constraints`, `physical_interface_line_weights` |
| Frozen model (supports, shape tensors, static identity) | `build_engineering_grip_model`, `EngineeringGripModel`, `A2StaticMechanics` |
| Fixed-damage A2 energy / residual / tangent evaluation | `A2FixedDamageView` (`.energy`, `.residual`, `.tangent_action`, `.evaluate`) |
| Amor material-source history driver | `compute_material_source_amor` |
| Nonlocal damage operator (2D cubic kernel) | `pinn.operators.ExplicitVolumeSparseD` |
| Adopted reference generator (GPU) | `run_stabilized_engineering_grip`, `JaxA2CPUReplayAdapter` |
| Rebuilt FP32 PINN graph and residual-only mechanics backend | `fp32.build_fp32_model`, `fp32.FP32A2Backend` |

The reference model is still built in FP64 with JAX x64 enabled; the network,
new graph, BA A2/Amor residuals, and material AT2 operator are all FP32 during
training. The original `d048` graph is not cast into FP32. On this material
cloud the rebuilt union candidate pairs differ by `+209/-76` and the material
directed edges by `+76/-152`; maximum FP32 BA condition is `20.24`. The first
archived state produces FP32 reaction `12.548` versus `12.675` when evaluated
with the original FP64 operator. This is a discrete-model difference, not a
training error.

### Strong-form residuals

With union displacement `q`, material damage `d`, committed history `H_c`, the
A2 energy residual `r_{A2}`, and the Amor material-source driver
`\Psi^+_{m,a}`, the training residuals are

$$
r_q=\bigl[\,r_{A2}(q;d)\,\bigr]_{\mathrm{free}},\qquad
\rho=\frac{G_c}{\ell_0}d+G_c\ell_0D_\delta d+2H(d-1),
\qquad
r_H=H-\max\bigl(H_c,\Psi^+_{m,a}(q)\bigr).
$$

The network displacement is affinely lifted to impose bottom nonmaterial collar
`u_y=0`, top nonmaterial collar `u_y=u_bar`, and the original material anchor
`u_x=0` **exactly**, with no BC penalty. Collar `u_x` and all material top/bottom
`u_y` remain free; all other boundaries are natural. Damage has the same
material-only natural zero-flux graph treatment, without hard boundary values.
See [Constraints, load coordinate, and reaction](#constraints-load-coordinate-and-reaction).
Every symbol above is defined in
[Governing equations](#governing-equations-adopted-engineering-grip-scheme);
no new kernel, split, or degradation is introduced by the PINN.

### Reference comparison protocol

- The nonlocal compact solution and the FEM archive share the same `3395`
  material nodes and `6284` triangles, so per-state field comparison is
  node-wise after reshaping: `union_q[:, :6790] -> (76, 3395, 2)`.
- **Strict prefix** `u_bar <= 0.0059`: `59` strictly accepted reference states.
  Use these for initial quantitative field/curve comparisons.
- **Drift-free warnings** `0.0060` and `0.0061`: warning-accepted even though
  no material particle exceeds the drift envelope; compare diagnostically.
- **Drifting tail** `u_bar >= 0.0062`: `2 -> 31` drift points
  (`|u_i|/u_bar > 1.08`). Acceptance labels are mixed: warning at `0.0062`
  through `0.0068` and `0.0071`, strict at the other tail loads. Compare
  qualitatively only; never present this tail as converged physical truth or
  as an exact-solution training target. The archive contains `65` strict and
  `10` warning accepted states *in total*, not as consecutive regions.
- Curve targets: pre-peak reaction follows the FEM reference (`642.48573` at
  `0.0057`); the nonlocal reference peak is `641.34017` at `0.0061`. The
  terminal mismatch (`69.15` vs FEM `3.98` at `0.0075`) is an accepted
  direct-solver limitation, not a PINN target.

### Training convergence criteria

- The direct FP64 solver's reference gates (not yet demonstrated achievable by
  the separately rebuilt FP32 PINN) are:
  displacement/damage/history/reaction/force/moment `atol=1e-9, rtol=1e-8,
  floor=1.0`; committed damage residual `atol=1e-10, rtol=1e-8, floor=1.0`;
  raw damage range and healing `1e-10` each; linear solves `cg_rtol=1e-10,
  cg_atol=1e-13, cg_maxiter=3000`.
- Field comparison milestones before claiming a trained strict-prefix state:
  relative `L2` error versus the reference on all material nodes below `1e-3` for
  `u`, `d`, and `H` at `u_bar = 1e-4, 1e-3, 0.0058, 0.0059`. At `0.0061` the
  warning-accepted comparison is diagnostic only, regardless of field error.
- Report residual norms, field errors, and reaction together. A low training
  loss alone is not convergence evidence, and gates must not be widened to make
  a failing state pass; record failures explicitly.
- Keep training/evaluation shapes fixed in JIT paths and precompute the graph,
  supports, constraints, and kernel data outside the training step (see
  `docs/JAX_GUIDELINES.md`).

## Usage

The solver requires a visible JAX GPU; there is no CPU fallback.

```bash
source .venv/bin/activate
python -m examples.fracture2d.nonlocal_baseline run \
  --output /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution.npz \
  --progress --progress-every 20
python -m examples.fracture2d.nonlocal_baseline verify \
  --archive /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution.npz
python -m examples.fracture2d.nonlocal_baseline plot \
  --archive /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution.npz \
  --output-dir /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/figures
```

The first PINN implementation is a **single-load full-graph experiment**, not
an accepted load-continuation curve. It uses the separately rebuilt FP32 model
and the previous archived history as context, never the current fields as loss targets.
Run on a strictly accepted prefix state (indices `1..59`) with the JAX virtual
environment and `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.train \
  --config=examples/fracture2d/configs/train_base.py \
  --config.data.state_index=1 --config.train.steps=100 \
  --config.train.report_every=10
```

Both entries load `configs/train_base.py` through the shared
`ml_collections` config flag (`--config=<file>`), as in the other examples;
any field can be overridden with `--config.<path>=<value>`.
`configs/train_base.py` declares the frozen cloud/reference hashes, physical
parameters, FP32 network, optimizer, and fixed **optimization-only** loss scales.
`--config.loss.scaling=fixed` retains the original explicit scales; the default
`initial_driver_rms` freezes the initial A2 residual RMS, Amor-driver RMS and
twice the driver RMS for mechanics, history and damage respectively.
Networks are built by `_build_network` from `pinn.arch.make_pinn` with the
explicit `arch.*` fields (`type`, `hidden_dim`, `num_layers`, `activation`,
`use_spatial_emb`, `spatial_emb_dim`, `spatial_scale`, `multi_head`,
`tensor_split`); the single-network `train.py` uses four outputs
(`u_x, u_y, d, H`), while the stagger diagnostic uses separate two-output and
one-output networks. The final output layer is initialized to zero so each
field starts at the affine lift.
The script prints raw A2/AT2/Amor residual maxima, material `u/d/H` errors,
reaction, range/healing diagnostics, and the **entire** warning/drift tail with
each load's archived acceptance label. A short smoke run is not evidence of
convergence. Strict-prefix accepted-state advancement is available in
`continuation.py`.

The two governing equations are **A2 displacement equilibrium and AT2 damage**.
Amor history is an algebraic committed-history maximum, not a third PDE. An
additional single-load stagger diagnostic alternates two independent
networks (mechanics with damage fixed, then damage with history fixed), uses the
same direct-scheme damage under-relaxation, re-solves mechanics, and recomputes
history from the *unchanged* committed previous state:

$$
H_{n,a}=\max\!\left(H_{n-1,a},\Psi^{+}_{m,a}(q_n)\right).
$$

Here \(H_{n-1}\) is the **accumulated history at the last accepted load**, not
just that load's instantaneous energy. It and each fixed stagger subproblem's
other field are explicitly `stop_gradient` inputs; the current Amor driver
remains differentiable with respect to the current displacement network.

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.stagger \
  --config.train.dtype=float64 --config.data.state_index=1 \
  --config.stagger.mechanics_steps=12000 \
  --config.stagger.damage_steps=12000 --config.stagger.sweeps=3 \
  --config.output.dir=/root/autodl-tmp/data/fracture2d/pinn_stagger_new_run \
  --config.output.log_every=500 --config.output.checkpoint_every=1000
```

Resume an interrupted run with the same command plus
`--config.output.resume=True`.

For strict-prefix continuation, use `continuation.py` with an explicit final
state (indices `1..59` only). The first load starts from the archived zero
state; subsequent loads reuse only the previous **accepted PINN state**,
including both networks, Adam states, displacement, damage, and cumulative
Amor history. The archive supplies the load values and diagnostic comparators,
not the current-load training targets. Each load receives a separate
`state_0001/`, `state_0002/`, ... directory under `output.dir`, an atomic
`accepted.ckpt.zip` only after the final sweep passes every gate, and
independent TensorBoard events under `<tensorboard_root>/<run-name>/state_0001/`,
etc. The first rejected load stops without
changing its accepted parent. Resume the same root with
`--config.output.resume=True` to continue an interrupted load, or increase
**only** `--config.data.end_state_index` and add the resume flag to run more
strict-prefix loads. Previously accepted loads and their numerical configs
are verified and reused without retraining; their checkpoint identities are
never rewritten. A completed rejected load requires a new run with a
different numerical policy or budget.
With `stagger.accept_on_first_passing_report=True`, only the **last** sweep's
mechanics-after reporting points may trigger online acceptance. All configured
gates use that point's same-pass post-update fields; the network, both Adam
states, damage and cumulative history are saved at its **actual** step.
Earlier sweeps always complete; old event curves are never searched for a
favourable state after training. Set the flag to `False` to require the
final optimizer step instead.

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.continuation \
  --config=examples/fracture2d/configs/train_base.py \
  --config.train.dtype=float64 --config.data.end_state_index=2 \
  --config.stagger.sweeps=3 \
  --config.stagger.mechanics_steps=12000 \
  --config.stagger.damage_steps=12000 \
  --config.output.dir=/root/autodl-tmp/data/fracture2d/pinn_continuation_fp64
```

After loads 1–2 are accepted, run the same command with
`--config.data.end_state_index=3 --config.output.resume=True` to train only
state 3 from the accepted state-2 networks and optimizer states. All other
numerical settings (including budgets, gates and output root) must match.
If state 3 completes with a failed gate, this root remains committed only
through state 2 and further resume stops there.

To change the mechanics scaling **without retraining the accepted prefix**,
start a separate continuation root using
`--config.stagger.mechanics_scale_policy=first_load_times_load_ratio`,
`--config.output.fork_from=<previous-root>`,
`--config.output.fork_after_state_index=2`,
`--config.data.end_state_index=3` and a new `--config.output.dir`. Keep the
remaining optimizer, budget and gate settings identical to the source. The
old root and its checkpoint hashes are not rewritten; the new load starts
from the verified accepted state-2 networks, optimizer states and q/d/H.
Both root and per-state source configs and the accepted predecessor digest
are checked. A rejected trial in the old root is **not** imported.
For a later policy change, the `fork_from` root can itself be a fork. The
ancestry is followed read-only for each prefix state and its exact per-state
configuration and predecessor digest are verified. For example, to change
acceptance starting at state 7 without retraining accepted states 1–6, use
`--config.output.fork_after_state_index=6` and point `fork_from` to the
accepted state-6 continuation root, with a new output directory.

The base completed-sweep gate limits are normalized mechanics loss `1e-2`,
material displacement relative L2 `5%`, and reaction relative error `5%`,
plus finite values, zero drift, damage range `1e-10`, damage healing
`stagger.healing_atol=1e-3`, and monotone history. The healing tolerance is
separate from the damage-range tolerance: it permits finite damage decreases
of up to `1e-3` relative to the previously accepted damage field and is **not**
an exact damage-irreversibility guarantee. Raw healing is always logged.
Reference-field and reaction comparisons are acceptance diagnostics only;
the optimizer loss remains the governing-equation residual.
For damage-field accuracy, `d_rmse = sqrt(mean((d_PINN - d_ref)^2))`
over all material nodes is the primary reported comparison. It is defined at
zero reference damage and is in dimensionless damage units (`0..1`).
`d_max_abs_error = max(abs(d_PINN - d_ref))` also reveals localized mismatch
near the notch that full-field RMSE can dilute. The previous `d_relative_l2`
is retained as an auxiliary trend: at early nearly-undamaged loads it can
become large solely because the reference norm is small. These fields are
diagnostics; no reference damage is used to train a network and no numerical
damage-field acceptance threshold is set by this reporting change.
For the user-approved reaction/damage-focused continuation, set
`stagger.gate_displacement_l2=False` (still report the displacement error) and
`stagger.damage_rmse_limit=1e-2`. The RMSE gate applies on the *same final
reported state* as the reaction and other physics gates. The maximum absolute
damage error remains diagnostic-only. The base config uses `True` and `0.0`
respectively (zero disables the RMSE gate) to preserve its earlier acceptance
policy. These settings change numerical identity and require a new fork root;
the previously rejected state is never retroactively accepted.
With `stagger.mechanics_scale_policy=first_load_initial`, every load
recomputes the same first-load zero-output affine free-row residual RMS
(about `0.05103` on the adopted graph). This is an **absolute path-wide**
force-residual normalization, not a load-relative accuracy criterion.
`first_load_times_load_ratio` instead divides by the same first-load RMS
times the ratio of the current prescribed displacement to the first strict
load. It changes the objective and normalized gate in the same way; raw A2
force residuals, the governing equation and the `0.01` normalized limit do
not change. The load ratio is a chosen optimization/acceptance convention,
not a proof of exact load invariance after damage evolves. An archived rejected
state is not retroactively accepted under this new rule.
`per_load_warm_start` remains an explicit alternative. A rejected completed
load cannot be resumed or silently advanced. Changing **numerical settings**
requires a new output root; increasing only the final state does not.

TensorBoard tags follow the repository convention (`loss/*` optimization
losses, `eval/*` reference comparisons, `diag/*` residual and state
diagnostics), matching `corrosion2d`/`fracture1d`:

- `loss/mechanics_before`, `loss/damage`, `loss/mechanics_after` — the three
  stage objectives, evaluated at the reporting step from the same post-update
  fields used by every other logged value. `loss/mechanics_before` is the
  free-row A2 residual MSE with damage held at `d_iter`; `loss/damage` is the
  AT2 residual MSE with fixed `H_used`; `loss/mechanics_after` solves the
  **same A2 equation** at under-relaxed `d_trial`.
- `eval/rel_L2_u`, `eval/rel_L2_d`, `eval/rel_L2_H`, `eval/d_rmse`,
  `eval/d_max_abs_error`, `eval/reaction` — material-node comparison against
  the FP64 archived reference, plus `eval/reaction_reference` at step zero.
- `diag/mechanics_residual_inf`, `diag/damage_residual_inf`,
  `diag/history_residual_inf` — full-graph raw residuals of the current
  (q, d, trial-history) state; `diag/history_residual_inf` is zero because the
  trial history is the exact same-pass Amor maximum.
- `diag/coupled_loss` — diagnostic sum of the normalized A2 and AT2 MSEs at
  that state; **not** the stage optimizer's loss or an acceptance gate. Its
  fixed scales are logged once as `diag/mechanics_scale` and
  `diag/coupled_damage_scale`. During the damage phase, `loss/damage` uses
  fixed `H_used`, so `diag/damage_residual_inf` (current q-driven history) can
  differ.
- `diag/drift_count`, `diag/damage_min`, `diag/damage_max`,
  `diag/damage_healing_max` — state diagnostics. `gate/*` records the
  **configured** per-sweep checks and `diag/accepted` their conjunction.
  The base policy gates material displacement L2 at 5%; the reaction-relative
  limit is 5% in both the base and reaction/damage-focused policies.

Every sweep also prints the same values as JSON, including the final coupled
diagnostics. Logs are written to `/root/autodl-tmp/tf-logs/fracture2d/<run-name>/`, where `run-name`
is the output directory's final component. View all fracture2d runs with
`tensorboard --logdir /root/autodl-tmp/tf-logs/fracture2d/`. The bound
network/optimizer/trial state is atomically updated at
`<output-dir>/latest.ckpt.zip`. Restart the same command with
`--config.output.resume=True` (same settings and output directory) to continue
an interrupted run. Use a new output directory for each run; do not overwrite
an existing run. On the first
load the earlier short stagger experiments were unconverged; this does not
establish what a many-thousand-step run will achieve. A proposed fixed-damage
energy-only diagnostic was interrupted and is not an active training option.

`run` writes a compact solution NPZ (loads, material fields, top reaction,
per-state acceptance status, geometry, metadata); `verify` reports state and
warning counts, peak/terminal reaction, drift counts and drift-excluded ranges;
`plot` draws `u_x`, `u_y`, `|u|`, and `d` with drift-excluded color ranges while
keeping every drift point in the solution and drawing it.

Retained focused tests:

```text
tests/test_fracture2d_nonlocal_baseline_entry.py
tests/test_fracture2d_bond_associated_algebra.py
tests/test_fracture2d_geometry.py
tests/test_fracture2d_pinn_residual.py
```

## Validation status and accepted limitations

**Both reference solutions are warning-mode, non-strict diagnostics.** The
local FEM archive is `gate_mode="warning"` (`13` of `127` loads carry
`gate_warning=True`), and the nonlocal direct solution is
`stabilized_warning` (`10` of `75` accepted states are warning commits, with
`2 -> 31` drift points). Neither is a strictly converged reference, and the
warning tail of neither solution may be presented as physical truth.

The nonlocal direct solver's remaining issues, explicitly accepted for now and
deferred:

1. **Terminal reaction is not physical.** Nonlocal `69.15` versus local FEM
   warning `3.98` at `0.0075`; the nonlocal response collapses too slowly after
   localization. Peak and approximate onset are close (`641.34` vs `642.49`),
   which is why the solution is still used as the direct reference.
2. **Ten of seventy-five accepted states are warning-accepted.** All raw gate
   defects are retained (`field_equations_validated=false`); no failed gate is
   relabeled as converged.
3. **Particle drift after severe localization.** Two material points first
   exceed the diagnostic envelope `|u_i|/u_bar > 1.08` at `0.0062`, growing to
   `31` by `0.0075` with a drift-inclusive tip displacement of `0.142`
   (amplification about `19`). Drift points stay in the equations and the
   solution; only the plotted color ranges exclude them.
4. **Healing in the unbounded warning path.** Up to `0.02915` after `0.0067`;
   accepted only as raw diagnostic provenance, never as a physical state.
5. **The original strict path never completes.** Original strict gates first
   fail near `0.0060` (damage-map defect), and the minimum-step target
   `0.006118125` stalls after `144` sweeps. No strict `0 -> 0.0075` path exists
   under the current increments and gates.
6. **Mesh and horizon convergence is open.** On an actually refined 3711-node
   mesh with `delta=0.032`, strict continuation reaches `0.0059` but `0.0060`
   still fails. Frozen-damage diagnostics at `delta=0.040/0.032` reduce the tip
   amplification `21.84 -> 10.76 -> 6.74` but do not remove the outliers.
7. **The local FEM reference is itself warning-mode.** Its strict convergence
   and mesh refinement follow-ups are also open
   ([archived local-FEM task](../../docs/tasks/archive/fracture2d-local-fem.md)).
8. **Onset/drift trigger.** The published pre-damage/damage-jump event rule
   misses the onset in both strict and warning paths; a drift-sensitive trigger
   still requires an explicit bounded numerical choice.
9. **Model-combination provenance.** Silling's intact correspondence basis,
   bond-associated stabilization, and the nonlocal AT2/Amor split are
   source-backed separately, but the damage-aware combination is a
   project-owned discrete choice; `k_res` sensitivity (`1e-8 -> 1e-5`) was
   falsified as a drift cure rather than resolved.

## Implementation layout

`nonlocal_baseline.py` is the single maintained implementation. The merged
sections are marked with `# ===== <name> =====` comments and use module-prefixed
private names (`_g_`, `_a_`, `_j_`, `_e_`, `_s_`) from the mechanical merge:

- union geometry, constraints, BA supports, and `MechanicsEvaluation`;
- A2 fixed-damage view, CG-free CPU evaluation, Amor, and fixed-history damage;
- the JAX GPU PCG backend and `JaxA2CPUReplayAdapter`;
- the engineering-grip model, gates, state, diagnostics, and work check;
- the fixed damped-Picard continuation with `stabilized_warning` acceptance;
- the `run` / `verify` / `plot` CLI.

`src/pinn/operators.py` remains the shared operator library
(`ExplicitVolumeSparseD` and the kernel registry); this file does not redefine
it. Parked material lives in `_legacy/` (ignored) and git history: the
per-module implementation, monolithic/arc/reduced solvers, the durable
workflow, the historical Gmsh loader, and the stage-specific test suites.
Historical audit detail and rejected alternatives live in
`docs/tasks/archive/` and `docs/DECISIONS.md`.
