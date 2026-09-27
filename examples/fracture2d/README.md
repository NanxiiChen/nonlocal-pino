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
-G_c\ell_0\,D_\delta d
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

## Nonlocal PINN (next stage)

### Operator and model reuse

The nonlocal PINN is trained on the same graph, operator, energy, history
driver, and residual assembly as the direct solver, taken from
[`nonlocal_baseline.py`](nonlocal_baseline.py) and `pinn.operators`:

| Need | Symbol |
|---|---|
| Union cloud, collars, constraints, `d048` prefix identity | `build_union_geometry_from_arrays`, `UnionGripGeometry`, `AffineConstraints`, `build_volume_grip_constraints`, `physical_interface_line_weights` |
| Frozen model (supports, shape tensors, static identity) | `build_engineering_grip_model`, `EngineeringGripModel`, `A2StaticMechanics` |
| Fixed-damage A2 energy / residual / tangent evaluation | `A2FixedDamageView` (`.energy`, `.residual`, `.tangent_action`, `.evaluate`) |
| Amor material-source history driver | `compute_material_source_amor` |
| Nonlocal damage operator (2D cubic kernel) | `pinn.operators.ExplicitVolumeSparseD` |
| Adopted reference generator (GPU) | `run_stabilized_engineering_grip`, `JaxA2CPUReplayAdapter` |

Enable JAX x64 before building the operator/model. The `d048` graph is
float64-specific; any float32 PINN branch must rebuild and revalidate its own
graph (do not cast this one, do not add kernel floors or cutoff tolerances).

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

The hard constraints (bottom/top collar `u_y`, anchor `u_x`) and the natural
boundary conditions are the same as in
[Constraints, load coordinate, and reaction](#constraints-load-coordinate-and-reaction).
Impose them exactly if the architecture permits; otherwise add them as explicit
penalties and report their violation separately from the PDE residuals. Every
symbol above is defined in
[Governing equations](#governing-equations-adopted-engineering-grip-scheme);
no new kernel, split, or degradation is introduced by the PINN.

### Reference comparison protocol

- The nonlocal compact solution and the FEM archive share the same `3395`
  material nodes and `6284` triangles, so per-state field comparison is
  node-wise after reshaping: `union_q[:, :6790] -> (76, 3395, 2)`.
- **Strict region** `u_bar <= 0.0061`: `65` strict reference states. Use these
  for quantitative field/curve convergence.
- **Warning region** `u_bar >= 0.0062`: `10` warning states, `2 -> 31` drift
  points (`|u_i|/u_bar > 1.08`). Compare qualitatively only; never present this
  tail as converged truth or as a training target for an exact solution.
- Curve targets: pre-peak reaction follows the FEM reference (`642.48573` at
  `0.0057`); the nonlocal reference peak is `641.34017` at `0.0061`. The
  terminal mismatch (`69.15` vs FEM `3.98` at `0.0075`) is an accepted
  direct-solver limitation, not a PINN target.

### Training convergence criteria

- Residual gates: reuse the direct-solver numbers as the convergence contract —
  displacement/damage/history/reaction/force/moment `atol=1e-9, rtol=1e-8,
  floor=1.0`; committed damage residual `atol=1e-10, rtol=1e-8, floor=1.0`;
  raw damage range and healing `1e-10` each; linear solves `cg_rtol=1e-10,
  cg_atol=1e-13, cg_maxiter=3000`.
- Field milestones before claiming a trained state: relative `L2` error versus
  the reference on the retained material nodes below `1e-3` for `u`, `d`, and
  `H` at `u_bar = 1e-4, 1e-3, 0.0058, 0.0061`.
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
