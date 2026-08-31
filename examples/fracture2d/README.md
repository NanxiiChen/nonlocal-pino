# 2D Rounded-Notch Fracture: Local FEM and Direct Nonlocal Baseline

This example is the first 2D stage of the fracture program. It is an
independent local FEM reference for a later direct nonlocal solver and
nonlocal PINN. It does not import or call the repository nonlocal operators.

## Problem

The material domain is the unit square with a finite-width slot opening on the
left boundary. The slot centerline is `y=0.5`, its total width is `2*l0`, and
its right end is a semicircle of radius `l0`. The foremost point of the tip is
`(0.5, 0.5)`, so the semicircle center is `(0.5-l0, 0.5)`.

The calculation uses plane strain, a vertically constrained bottom edge, one
horizontal point anchor at `(0,0)`, and a vertical displacement on the top edge:

$$
u_y=0\quad\text{on }y=0,
\qquad u_x(0,0)=0,
\qquad
u_y=\bar u\quad\text{on }y=1.
$$

The top horizontal displacement is free. All other boundaries are natural
traction boundaries in this local FEM reference.

## Formulation

The solver follows the hybrid implementation used by Kristensen and
Martinez-Paneda (2020), without that paper's quasi-Newton or adaptive-step
algorithm. The mechanics problem uses the full undamaged elastic energy with
the degradation function, while the history field uses the Amor
volumetric-deviatoric positive energy:

$$
\psi_0^+
=\frac12K\langle\operatorname{tr}\boldsymbol\varepsilon\rangle_+^2
+\mu\boldsymbol\varepsilon_{\mathrm{dev}}:
\boldsymbol\varepsilon_{\mathrm{dev}},
\qquad
K=\lambda+\frac{2\mu}{3}.
$$

The mechanics energy density is

$$
\psi_{\mathrm{mech}}(\boldsymbol\varepsilon,d)
=g(d)\psi_0(\boldsymbol\varepsilon),
\qquad
g(d)=(1-d)^2+k_{\mathrm{res}}.
$$

Thus the mechanics stress is the paper's hybrid form

$$
\boldsymbol\sigma=g(d)\boldsymbol\sigma_0.
$$

The tensile history is updated only after the mechanics state at a staggered
iteration has been solved:

$$
\mathcal H_n
=\max\left(\mathcal H_{n-1},\psi_0^+(\boldsymbol\varepsilon_n)\right).
$$

At fixed history, the damage equation is linear:

$$
\frac{G_c}{\ell_0}d
-G_c\ell_0\Delta d
+2\mathcal H d
=2\mathcal H.
$$

The FEM weak form uses natural zero-flux damage conditions. No clipping,
pointwise damage bound, or second irreversibility constraint is added.

Mechanics is solved from the derivative of the declared hybrid elastic energy.
The staggered algorithm is retained for this first reference; the paper's
quasi-Newton monolithic method and adaptive step rule are deliberately not
implemented.

## Geometry And Mesh

`mesh.geo` is a Gmsh geometry with explicit physical curves for the outer
boundary and notch faces. The solver passes all geometry and mesh-size values
with `gmsh -setnumber`, converts triangle cells to XDMF with `meshio`, and
loads the mesh with DOLFIN.

The default mesh sizes are:

```text
l0       = 0.024
notch    = 0.048 wide, radius 0.024
lc_near  = 0.004
lc_far   = 0.020
```

The mesh is intentionally fixed during the load path. Dynamic refinement and
nonlocal graph construction belong to later stages.

## Usage

Use the mutually exclusive FEniCS environment:

```bash
conda activate fenicsenv
python examples/fracture2d/solver.py \
  --config=examples/fracture2d/configs/solver_base.py
```

For a short smoke run, use a separate config override or copy the base config
and set `loading.u_final` to a small value. The default configuration uses
explicit `solver.gate_mode="warning"` for a curve-diagnostic run, matching the
accepted 1D baseline. It commits only finite candidates after the stagger
budget and records `gate_warning` plus all convergence defects in the archive.
Use `solver.gate_mode="strict"` to retain rollback/cutback and stop at the
first unresolved load state. No clipping or damage-bound projection is used.
The base configuration uses `loading.u_final=0.0075` for the default curve.

The default archive is written to:

```text
/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz
```

It contains load histories, accepted increments, stagger diagnostics, warning
flags, absolute/normalized damage-history-reaction defects, damage and healing
diagnostics, elastic/fracture energies, displacement and damage snapshots,
history fields, stress components, and the explicit geometry/model metadata.
PNG response and final-field plots are written beside the archive.
The generated field plots include `representative_fields_initial.png`,
`representative_fields_before_crack.png`,
`representative_fields_crack_development.png`,
`representative_fields_near_complete.png`, and
`representative_fields_fully_open.png`. These use continuous CG1 triangular
interpolation for `u_x`, `u_y`, `|u|`, and `d`; every panel uses that field's
actual minimum and maximum rather than a shared manual color range. This keeps
the local notch-tip concentration visible. `stress_strain.png` includes the
explicit zero-load point; the history plot remains cellwise because its FEM
space is DG0. When the archive covers the requested loads, `selected_fields_grid.png`
combines `u_x`, `u_y`, and `d` in three rows. The base configuration selects
`u_bar=0.0055`, `0.0058`, `0.0059`, `0.0060`, `0.0061`, and `0.0062`; pass a
different ordered list through `c.output.field_grid_loads` to change the
columns without changing the solver.

## Validation Order

The point-anchor warning-mode archive is the accepted working local reference
for developing the next direct nonlocal baseline. It is a curve diagnostic,
not a strict converged fracture reference. Before treating it as a strict
quantitative reference, verify:

1. Gmsh geometry and total material area;
2. mesh refinement near the rounded tip;
3. load-step refinement;
4. staggered convergence and rollback behavior;
5. damage range and healing diagnostics;
6. elastic/fracture energy and reaction histories;
7. comparison with an independent local FEM mesh or formulation.

The next stage will compare this local reference with a direct nonlocal
discretization. It must not reuse this solver's weak form as a hidden part of
the nonlocal implementation. Current numerical validation and open limitations
are tracked in the archived task record
[`docs/tasks/archive/fracture2d-local-fem.md`](../../docs/tasks/archive/fracture2d-local-fem.md).

## Direct-nonlocal Stage 1 geometry API

The independent Stage 1 module is
`examples/fracture2d/nonlocal_geometry.py`. It reads the original Gmsh
`.msh` file with `meshio`; do not use the FEM solver's XDMF conversion for this
path because that conversion intentionally keeps only triangle cells and
therefore drops the physical-group contract. The mesh now includes the
explicit physical point `anchor` in addition to the named `bottom`, `top`,
outer-side, slot-face, slot-tip, and `domain` groups.
Every tagged line cell must be a triangle edge with incidence one, and the
tagged line cells must cover the complete material boundary exactly once. The
slot groups `notch_upper`, `notch_tip`, and `notch_lower` must form one ordered
open chain with one endpoint at each opening lip. The loader closes that chain
with the unmeshed opening segment to construct the polygonized slot; malformed,
internal, duplicated, inconsistently tagged, or untagged boundary edges fail
explicitly. For an already-read `meshio.Mesh`, call `build_particle_cloud` with
the same explicit arguments instead of the file loader.

```python
from examples.fracture2d.nonlocal_geometry import (
    ROUNDED_SLOT_TAGS,
    load_gmsh_particle_cloud,
)

cloud = load_gmsh_particle_cloud(
    msh_path,
    horizon=delta,
    boundary_tags=ROUNDED_SLOT_TAGS,
    visibility_tolerance=1.0e-12,
)
```

The `make_rounded_slot` helper is not part of this Stage 1 call: visibility is
defined by the retained mesh topology, not by the analytic circle.

`cloud.coordinates` and `cloud.triangles` use deterministic lexicographic
particle ordering. `cloud.volumes` are the barycentric nodal areas

$$
V_i=\frac{1}{3}\sum_{e\ni i}A_e,
$$

and `cloud.first_moments` stores the area-weighted first moments. The
`cloud.boundary` masks are tag-derived; `cloud.boundary.anchor_particle` is the
unique tagged lower-left particle. `candidate_edges`, `visible_edges`, and `removed_edges`
are sorted directed arrays, so every retained or removed pair appears in both
directions and no self edge is inserted.

Visibility removes a pair only when an open subsegment enters the strict
interior of the polygonized slot, or when it has positive overlap with the
unmeshed opening segment. Tagged slot-face and notch-tip chord edges, material
boundary edges, and tangent segments are retained. The explicitly supplied
`visibility_tolerance` is the caller input recorded in
`cloud.visibility_tolerance`. Each predicate uses the effective absolute
tolerance `tau_eff=max(tau_input,64*eps*L_geom)`, where `L_geom` is the largest
length among the polygon extent/edges, opening closure, and candidate segment.
The `64*eps*L_geom` term is a numerical machine floor, not a physical clearance;
therefore input `0` remains explicit while avoiding scale-dependent roundoff.
A non-collinear classification that remains ambiguous at `tau_eff` fails rather
than falling back to the analytic circle or a coordinate-derived boundary.
The exported `segment_is_visible` diagnostic expects the ordered simple
polygon already validated by `build_particle_cloud`; it does not repeat that
topology check for every graph edge. This module stops before the weighted
scalar operator, correspondence mechanics, and coupled fracture solve. A
production horizon is intentionally not selected by Stage 1; callers must pass
it explicitly.

## Direct-nonlocal Stage 2 weighted scalar API

The Stage 2 scalar damage operator is a separate general-purpose class in
`pinn.operators`. It does not read Gmsh data or infer geometry, and it does not
change the existing `SparseD` semantics. Construct it from the Stage 1 cloud
with an explicitly selected horizon and the retained directed graph:

```python
from pinn.operators import ExplicitVolumeSparseD

operator = ExplicitVolumeSparseD(
    cloud.coordinates,
    cloud.volumes,
    cloud.visible_edges,
    delta,
    "cubic",
)
```

The constructor requires floating-point coordinates and volumes with the same
dtype, a nonempty integer `(E, 2)` directed graph, a finite positive `delta`,
and a registered kernel. It checks finite unique points, positive volumes,
reciprocal self-edge-free in-range edges, strict `0 < r_ij < delta` support,
and a positive outgoing family at every node. Edges are normalized to
deterministic lexicographic `(source, target)` order. Matching float32 and
matching float64 inputs are supported; mixed coordinate/volume dtypes fail
explicitly. Float64 construction requires JAX 64-bit mode and fails rather
than silently downcasting when that mode is disabled.

The fracture2d direct baseline is frozen to float64 geometry, barycentric
volumes, graph arrays, and operator inputs, with JAX x64 enabled before
operator construction. Graph membership is dtype-specific because strict
`r_ij < delta` can change after coordinate casting. A future float32 PINN must
rebuild and revalidate its graph in float32; it must not cast this float64 graph
or add a cutoff tolerance or kernel floor.

For the frozen 2D cubic kernel,

$$
(D_\delta d)_i=\sum_{j:(i,j)\in\mathcal E}
J_\delta(r_{ij})V_j(d_i-d_j),
\qquad
J_\delta(r)=\frac{120}{\pi\delta^4}(1-r/\delta)^2
\quad(0<r<\delta).
$$

`operator @ values` accepts only `(N,)` and `(N, T)` arrays with the operator's
coordinate/volume dtype. The fixed
`diagonal`, `volumes`, `shape`, `directed_edges`, and `to_scipy()` CSR export
are available for diagnostics and linear solves. With
`M=diag(volumes)`, the reciprocal graph satisfies
`M @ D = (M @ D).T` and its directed pair energy is non-negative. Therefore
the fixed-history damage solve uses the Euclidean SPD system

$$
S=M\left[\operatorname{diag}(G_c/\ell_0+2H)+G_c\ell_0D_\delta\right],
\qquad
S d=2MH.
$$

Persistent unittest coverage covers nonuniform-volume graph identities,
SciPy/JAX matvecs, fixed-history residuals, dtype behavior, invalid inputs, and
full-square triangular barycentric-volume manufactured fields. Rounded-slot
full/deep values are independent review evidence, not persistent unittest
coverage. They are generated through the two mutually exclusive environments:
`fenicsenv` generates the tagged Gmsh mesh and extracts explicit Stage 1 arrays
for each horizon; `.venv` enables JAX x64 and instantiates/checks the operator
from those serialized arrays. No binary fixture or cross-environment package
installation is part of the repository. Boundary distances use polygonized
triangle edges with incidence one, and the absent left opening is not added as
a material boundary edge. No production common horizon is selected by these
checks. The exact temporary-array regeneration workflow, measured values, and
verification commands are recorded in
[`docs/tasks/fracture2d-direct-nonlocal.md`](../../docs/tasks/fracture2d-direct-nonlocal.md).

## Direct-nonlocal Stage 3 intact mechanics API

Stage 3 is an independent NumPy/SciPy float64 mechanics baseline. It does not
import `ElasticD`, the local FEM solver, or JAX, and it contains no damage,
history, continuation, or PINN coupling. Construct
`BondAssociatedMechanics` with every required value explicitly:

```python
from examples.fracture2d.nonlocal_mechanics import BondAssociatedMechanics

mechanics = BondAssociatedMechanics(
    cloud.coordinates,
    cloud.volumes,
    cloud.visible_edges,
    delta,
    youngs_modulus=210.0e3,
    poisson_ratio=0.3,
    k_res=1.0e-8,
    rank_rtol=1.0e-12,
)
```

Coordinates must be a finite unique `(N,2)` float64 array, and volumes must be
a finite positive `(N,)` float64 array. The graph must be integer,
self-edge-free, reciprocal, and satisfy the strict `0 < r_ij < delta`
horizon. Directed edges are copied and normalized to deterministic
lexicographic `(source,target)` order. Every particle must have positive
center mass. `E>0`, `-1<nu<0.5`, `k_res>0`, and `0<rank_rtol<1` are checked
before construction; a rank-deficient bond-associated shape tensor fails
before inversion. There is no pseudo-inverse, determinant clipping, kernel
floor, stabilization, or fallback.

For each directed bond `(i,j)`, the valid padded support is target-first:

$$
\mathcal C_{i|j}=\{j\}\cup\{k\ne i,j:(i,k)\in\mathcal E\ \text{and}\ (j,k)\in\mathcal E\}.
$$

The exposed fixed arrays are `support_indices`, `support_mask`, `beta`, `xi`,
`shape_inverse`, `eta`, `center_mass`, `support_sizes`, `shape_rank`,
`shape_condition`, and `target_sensitivity`. The target term is exactly
`omega_ij * V_j`; a common-neighbour term is `omega_ik * V_k`. Padded entries
are masked, and the fixed arrays are read-only. The cubic kernel is

$$
J_\delta(r)=\frac{120}{\pi\delta^4}(1-r/\delta)^2
\qquad (0<r<\delta).
$$

The intact model means `d=0` and `g=1+k_res`. With
`F=A K^{-1}`, `epsilon=sym(F-I)`, and the plane-strain Lamé constants, the
center energy is

$$
\Pi_{\mathrm m,h}(u)=\sum_i V_i(1+k_{\mathrm{res}})
\sum_{j:(i,j)\in\mathcal E}\eta_{ij}\,\psi_{i|j}(u),
$$

where

$$
\psi=\tfrac12\lambda(\operatorname{tr}\epsilon)^2+\mu\,\epsilon:\epsilon,
\qquad
P=\lambda(\operatorname{tr}\epsilon)I+2\mu\epsilon.
$$

`energy(u)` evaluates this expression. `residual(u)` independently assembles
the energy gradient from the endpoint terms
`beta * P @ K^{-T} @ xi`; `internal_force(u)` returns its negative. The
`tangent_action(v)` path computes `delta F` and applies the plane-strain
constitutive tangent directly; it is not a residual finite difference. All
displacements and directions are flat interleaved float64 arrays `(2*N,)` in
`ux,uy` order.

`evaluate(u)` returns the energy, residual/internal force, edge `F`, edge strain,
edge stress, and source-centered `eta`-averaged particle strain/stress. The
reported `edge_stress` and `particle_stress` are the undegraded constitutive
stress `P`; `edge_effective_stress` and `particle_effective_stress` expose the
intact stress `(1+k_res)P` without changing that convention.

Build the frozen material-only hard constraints explicitly from boolean masks:

```python
constraints = mechanics.build_constraints(
    bottom_mask=cloud.boundary.bottom,
    top_mask=cloud.boundary.top,
    anchor_particle=cloud.boundary.anchor_particle,
)
result = mechanics.solve(
    constraints,
    u_bar=0.01,
    cg_rtol=1.0e-11,
    cg_atol=1.0e-13,
    cg_maxiter=3000,
)
```

The constraints impose `uy=0` on bottom particles, `uy=u_bar` on top
particles, and `ux=0` at the bottom anchor; all other horizontal boundary DOFs
remain free. `solve` injects free values through a SciPy
`LinearOperator` and calls CG without forming a dense global tangent. It fails
on nonconvergence or nonfinite values and records `cg_iterations`, `cg_info`,
and the free-system `free_rhs_norm`. Its exact particle-local block-Jacobi
preconditioner is assembled from the same tangent blocks and changes only CG
convergence, not the declared mechanics equations. The result includes the full
displacement and response fields,
free-residual norms, energy, separate `top_reaction`, `bottom_reaction`, and
`anchor_reaction`, plus total force and moment diagnostics. Reactions use the
residual-gradient sign: `top_reaction=sum(r_iy)` on prescribed top vertical
DOFs, so the actuator force has the opposite sign. The reduced-energy reaction
check re-solves the free problem independently at `u_bar +/- epsilon`.

Persistent Stage 3 coverage is in
`tests/test_fracture2d_nonlocal_mechanics.py`. The 41-by-41 quadratic
full-square check uses an explicit `delta=0.08` diagnostic and a strict mask
more than `2*delta` from the boundary; it is evidence for the residual sign
and finite-cloud interior behavior, not a production horizon selection.

## Direct-nonlocal Stage 4 fixed subproblems

Stage 4 is implemented separately in
`examples/fracture2d/nonlocal_baseline.py`. It contains fixed-subproblem APIs
only; it does not run a staggered continuation or select a common production
horizon.

### Fixed history damage

`solve_fixed_history_damage(operator, Gc, l0, history)` requires an
`ExplicitVolumeSparseD`, positive finite `Gc` and `l0`, and an exact-float64
non-negative history array of shape `(N,)`. It exports the operator as
`D=operator.to_scipy()`, forms `M=diag(V)`, and solves

$$
S d=b,

\qquad
S=M\left[\operatorname{diag}(G_c/\ell_0+2H)+G_c\ell_0D\right],
\qquad
b=2MH.
$$

The returned immutable `FixedHistoryDamageResult` contains the damage/history
copies, crack-local, history, graph, and total energies, strong and
volume-multiplied Euclidean residual diagnostics, system symmetry diagnostics,
and damage extrema/counts. The graph energy is evaluated with the directed
coefficient

$$
\frac{G_c\ell_0}{4}\sum_{(i,j)\in\mathcal E}
V_iV_jJ_\delta(r_{ij})(d_j-d_i)^2
=\frac12G_c\ell_0d^TM Dd.
$$

The direct solve fails on non-finite or singular systems. It never clips or
projects damage; values outside `[0,1]` are reported by the result instead.
This fracture2d API also independently reconstructs the frozen 2D cubic
coefficients and rejects a wrong dimension/kernel, graph/CSR mismatch,
non-constant-preserving export, or violation of `M D = D.T M` before solving.

### Fixed damage mechanics

`FixedDamageBondAssociatedMechanics` is a separate immutable-factor view of the
Stage 3 `BondAssociatedMechanics` implementation:

```python
from examples.fracture2d.nonlocal_mechanics import (
    FixedDamageBondAssociatedMechanics,
)

mechanics = FixedDamageBondAssociatedMechanics(
    intact_mechanics,
    damage=damage,
)
constraints = mechanics.build_constraints(
    cloud.boundary.bottom, cloud.boundary.top, cloud.boundary.anchor_particle
)
result = mechanics.solve(
    constraints,
    u_bar=0.01,
    cg_rtol=1.0e-11,
    cg_atol=1.0e-13,
    cg_maxiter=3000,
)
```

The `damage` input must be finite exact float64 with shape `(N,)`; it is not
clipped, and `damage_min`, `damage_max`, `damage_below_zero_count`, and
`damage_above_one_count` report its range. For source particle `i`,

$$
g_i=(1-d_i)^2+k_{\mathrm{res}},
\qquad
\text{center factor}_{i|j}=V_i g_i\eta_{ij}.
$$

The fixed view shares the intact instance's read-only geometry, graph, BA
supports, and constitutive data, and reuses the approved deformation gradient,
energy-gradient residual, tangent action, source-centered field averaging,
hard-constraint CG solve, and reaction code. Edge effective stress is
`g_i*P_edge`; particle effective stress is `g_i` times its source-centered eta
average. Its exact particle-local block-Jacobi cache is separate from the
intact instance and is solver-only. Fixed state cannot be rebound or deleted;
construct a new view for a new damage field. A fixed `damage=0` view reproduces
the intact path exactly.

### Optional JAX fixed-damage mechanics backend

`JaxBondAssociatedMechanics` is a separate float64 accelerator backend. It
consumes an already validated intact `BondAssociatedMechanics` instance and its
explicit hard constraints; it does not replace the NumPy/SciPy reference or use
`ElasticD`. Enable x64 before construction and select the device explicitly:

```python
import jax

jax.config.update("jax_enable_x64", True)

from examples.fracture2d.nonlocal_mechanics_jax import (
    JaxBondAssociatedMechanics,
    JaxCPUReplayMechanicsAdapter,
)

backend = JaxBondAssociatedMechanics(
    intact_mechanics,
    constraints,
    jax.devices("gpu")[0],
)
jax_result = backend.solve_strict(
    damage,
    u_bar=0.01,
    cg_rtol=1.0e-11,
    cg_atol=1.0e-13,
    cg_maxiter=3000,
)
replay = backend.replay_cpu(
    jax_result,
    damage,
    u_bar=0.01,
    cg_rtol=1.0e-11,
    cg_atol=1.0e-13,
    cg_maxiter=3000,
)
cpu_result = replay.cpu_result
cpu_replay_adapter = JaxCPUReplayMechanicsAdapter(backend)
```

The backend keeps the flattened target-inclusive supports, analytic tangent,
exact masked block-Jacobi preconditioner, and PCG loop on the selected device.
Changing same-shape damage, load, CG values, or initial displacement does not
change the compiled shape. A supplied initial displacement must match all hard
boundary values exactly and follows the same strict warm-seed/early-exit policy
as the CPU reference.

Only `replay.cpu_result`, reconstructed through a fresh CPU fixed-damage view,
is authoritative for acceptance gates and physical archive diagnostics. The
retained JAX fields and iteration count are execution diagnostics. Continuation
may use only the restricted `JaxCPUReplayMechanicsAdapter`; generic solver hooks
are rejected, and backend provenance is stored in continuation metadata and
archives.
This integration does not approve a production horizon or peak/crack-growth
continuation. Its focused regression suites are
`tests/test_fracture2d_nonlocal_mechanics_jax.py` and
`tests/test_fracture2d_nonlocal_continuation.py`.

### Optional JAX engineering-grip mechanics backend

`examples/fracture2d/nonlocal_engineering_grip_jax.py` is the separate float64
backend for the A2 phase-measured engineering-grip path. It consumes an accepted
`A2StaticMechanics` union model and its exact `AffineConstraints`; it does not
accept material-only `BondAssociatedMechanics` geometries, and the material-only
backend above does not accept A2 views. The static graph, target-inclusive
supports, and constraints live once on one explicit device; only the material
damage is dynamic. Center weights are computed exactly as
`Vm*((1-d)^2+k_res)+Vg*(1+k_res)`, and the solver is the exact particle
block-Jacobi PCG with one fixed-shape compiled solve reused for every changing
damage/load/seed/tolerance input:

```python
import jax

jax.config.update("jax_enable_x64", True)

from examples.fracture2d.nonlocal_engineering_grip_jax import (
    JaxA2CPUReplayAdapter,
)

adapter = JaxA2CPUReplayAdapter.from_static(
    model.static,
    model.constraints,
    jax.devices("gpu")[0],
)
result = run_strict_engineering_grip(
    model,
    config,
    mechanics_adapter=adapter,  # Omit for the NumPy/SciPy CPU path.
)
```

`run_zero_damage_work_check` and `run_strict_engineering_grip` accept the
adapter as the explicit `mechanics_adapter` argument. The default `None` keeps
the CPU reference path used by the manufactured tests; generic callables and
adapters whose static/constraint identity does not match the model are
rejected. The adapter serves the work-check center/plus/minus solves and all
stagger solves from one compiled backend. Every returned production value is
CPU-authoritative: the GPU displacement is transferred, a fresh
`A2FixedDamageView` independently evaluates energy/residual/reactions/force/
moment and the explicit free-residual gate without rerunning CPU CG, and the
result is the existing `A2MechanicsSolveResult` with the full seed provenance
and `cg_info=0`. GPU iteration counts are retained separately as non-authority
diagnostics. The focused regression suite is
`tests/test_fracture2d_nonlocal_engineering_grip_jax.py`.

### Amor history driver

`compute_amor_history(mechanics, displacement, previous_history)` evaluates the
edge strain from `BondAssociatedMechanics`, embeds it in 3D with
`epsilon_zz=0`, and returns an immutable `AmorHistoryResult` containing
`edge_positive_energy`, the source-only eta-averaged `particle_driver`, the
committed `previous_history`, and the pure trial `trial_history` maximum:

$$
\psi^+=\frac12K\langle\operatorname{tr}\boldsymbol\varepsilon^{(3)}\rangle_+^2
+\mu\boldsymbol\varepsilon_{\mathrm{dev}}^{(3)}:
\boldsymbol\varepsilon_{\mathrm{dev}}^{(3)},
\qquad
K=\lambda+\frac{2\mu}{3}.
$$

`update_history_max(previous_history, particle_driver)` is the standalone pure
maximum update; it does not mutate its committed input. No mechanics/damage
coupling loop is part of Stage 4.

The focused regression suite is
`tests/test_fracture2d_nonlocal_baseline.py`. It independently checks the
fixed-history system and directed energy, spatially damaged mechanics,
source-centered effective stress, Amor tension/compression/shear/mixed states,
hard-constraint reactions, immutable fixed state, and loud validation paths.

Stage 4 passed independent numerical review. Fresh rounded-slot diagnostics at
the explicit non-production horizons `0.032`, `0.040`, and `0.048` compared
fixed-`d=0` loads `1e-4`, `5e-4`, and `1e-3` with the warning-mode FEM archive.
Across those loads, nonlocal/FEM displacement-norm ratios were
`0.9970`--`0.9996`, top-reaction ratios were `0.9684`--`0.9930`, and mean
`sigma_yy` ratios were `0.9674`--`0.9919`; all trends were monotone and nearly
linear. Amor-driven fixed-history damage remained in `[0,1]`, with maximum
`0.01234`--`0.01361` and strong residual infinity norms below `7.2e-15`.
Fixed-damage force/moment diagnostics passed, and the `delta=0.048`
reaction-work relative error was `9.24e-13`. These values approve the fixed
subproblems only; they do not select a common production horizon or validate a
coupled continuation.

## Direct-nonlocal Stage 5 strict continuation API

Stage 5 is implemented in `examples/fracture2d/nonlocal_baseline.py` as an
explicit controller around the approved Stage 3/4 APIs. It does not modify the
bond-associated equations, `ElasticD`, graph, clipping policy, or fixed
subproblem solvers, and it does not select a production horizon.

Construct every setting explicitly with `GateTolerance(atol, rtol,
scale_floor)` and `StrictContinuationConfig`. A gate limit is
`max(atol, rtol * max(scale, scale_floor))`. The configuration requires the
load span, initial/minimum increments, cutback factor and budget, stagger and
CG limits, seven gate tolerances (`displacement`, `damage`, `history`,
`reaction`, `damage_residual`, `force`, and `moment`), `healing_atol`, and
`damage_range_atol`. The load start must be exactly zero; no production config
is added by this stage.

```python
from examples.fracture2d.nonlocal_baseline import (
    GateTolerance,
    StrictContinuationConfig,
    SupplementalArchiveMetadata,
    run_strict_continuation,
    write_continuation_archive,
)

gate = GateTolerance(1.0e-9, 1.0e-8, 1.0)
config = StrictContinuationConfig(
    load_start=0.0,
    load_final=0.01,
    du_initial=0.005,
    du_min=0.00125,
    cutback_factor=0.5,
    max_cutbacks=4,
    max_stagger_iterations=20,
    cg_rtol=1.0e-11,
    cg_atol=1.0e-13,
    cg_maxiter=3000,
    displacement=gate,
    damage=gate,
    history=gate,
    reaction=gate,
    damage_residual=gate,
    force=gate,
    moment=gate,
    healing_atol=1.0e-10,
    damage_range_atol=1.0e-10,
)
result = run_strict_continuation(
    operator,
    mechanics,
    constraints,
    Gc,
    l0,
    config,
    mechanics_solver=cpu_replay_adapter,  # Omit for the NumPy/SciPy CPU path.
)
```

`CommittedState` contains only `(u_bar, displacement, damage, history)`.
Its float64 arrays are independent bytes-backed read-only copies: attempting to
re-enable their NumPy write flags fails. `ContinuationResult.states` contains
the archived zero state followed by accepted states, while `attempts` retains
every accepted and rejected trial with cutback/load information, failure stage
and message, and before/after SHA256 state digests. `AcceptedStep` pairs the
accepted state with `H_used`, the Amor driver, fixed-history result, final
mechanics result, the exact final-sweep `damage_before` iterate and
`mechanics_before_result`, all diagnostics, and its cutback count. Every
result also contains immutable `ContinuationMetadata` built by the
controller. It records the exact float64 coordinates, volumes, sorted
reciprocal directed edges, boundary masks, anchor and constrained/free DOFs,
material parameters, cubic kernel, plane-strain/no-stabilization declaration,
mechanics backend, acceptance mode, dtype, and the exact
`StrictContinuationConfig`. Accepted-step and result validation cross-check
loads and state/audit pairing, rebuild authoritative mechanics from the frozen
metadata for both the before and after final-sweep solves, replay all Amor
fields (including `H_used` from the mechanics-before displacement and the
preceding committed history), and then recompute the complete final-sweep
gate diagnostics — every error, scale, limit, pass flag, residual vector
(including the volume-weighted used/committed residuals), force/moment value,
rejection reason, and `complete_sweep` flag — comparing them field by field
against the stored summaries. Complete-sweep diagnostics must be entirely
finite; incomplete failure diagnostics may retain `inf` sentinels but never
NaN, and they can never be paired with an accepted step. Fixed-history
residuals, energies, and symmetry diagnostics are reconstructed from the
frozen damage system.

Archive writing requires a separate `SupplementalArchiveMetadata` instance;
these fields are intentionally explicit because the controller cannot infer
mesh triangles or provenance from the operator. Supply exact int64 triangle
indices, volume-weighted first moments, visibility tolerance/rule, quadrature
rule, study label, and model provenance. Triangle indices/areas and first
moments are validated against the result metadata; barycentric quadrature also
requires triangle area and first-moment agreement. The required call is:

```python
archive_metadata = SupplementalArchiveMetadata(
    triangles=triangles,
    first_moments=first_moments,
    visibility_tolerance=visibility_tolerance,
    visibility_rule=visibility_rule,
    quadrature_rule=quadrature_rule,
    study_label=study_label,
    model_provenance=model_provenance,
)
write_continuation_archive(path, result, archive_metadata)
```

The archive stores static metadata, every zero/accepted state, complete
mechanics/damage/Amor fields and energies, reactions/force/moment diagnostics,
gate values, work/history quantities, and the complete numerical attempt audit
in JSON-compatible arrays. It is designed to load with `allow_pickle=False`.
Accepted steps additionally archive the exact final-sweep before-execution
data — `accepted_damage_before`, `accepted_mechanics_before_displacement`, and
`accepted_mechanics_before_free_residual` — which together with the frozen
graph, states, and `H_used` is sufficient to reconstruct every final-sweep
gate without rerunning the sweep.

For each target and each stagger sweep, the controller performs exactly:

1. a fresh fixed-damage view at `d_iter` and `mechanics_before`;
2. `H_used=max(H_committed, Psi_plus(u_before))`;
3. one fixed-history damage solve for `d_next`;
4. raw range and healing checks, without clipping;
5. a fresh fixed-damage view at `d_next` and `mechanics_after`;
6. `H_next=max(H_committed, Psi_plus(u_after))`, followed by every gate.

`H_committed` is immutable throughout an attempt and is never replaced by a
trial history. If the history or committed direct damage-residual gate fails,
the controller completes the configured stagger budget rather than performing
a standalone final damage re-solve. Mechanics `cg_info=0` and its own free
residual check remain authoritative. Force and moment are conservation gates;
there is no reaction or total-energy monotonicity gate. Failed fixed-subproblem
`RuntimeError`/`FloatingPointError` exceptions are audited and cut back, while
unknown exceptions propagate. At `du_min` or the cutback budget,
`ContinuationFailure` carries the exact last committed state and complete
attempt audit. A cutback increment is retained after acceptance; no automatic
increment-growth rule is hidden in this API.

The focused suite
`tests/test_fracture2d_nonlocal_continuation.py` additionally compares one
clean step with independent dense BA mechanics, damage, and Amor reference
helpers and forces explicit force/moment gate rejections. Independent Sol review
approved the Stage 5 controller implementation after the audit findings were
closed and all `134` explicit Stage 0--5 CPU tests passed. A rounded-slot pilot
at the temporary nonproduction `delta=0.048` subsequently reached `u_bar=1e-3`
in ten accepted increments without cutback. All strict gates passed; the largest
error/limit ratios were `0.7791` for committed damage residual and `0.7442` for
history. Independent review approved this pre-localization evidence only. It
does not approve a production horizon, peak/crack-growth continuation, mesh or
horizon refinement, or PINN work.

### Warning-mode all-gate diagnostic continuation

`run_warning_continuation(operator, mechanics, constraints, Gc, l0, config,
mechanics_solver=None)` is a separate exploratory entry point for staged curve
diagnostics toward peak load. It accepts the existing `StrictContinuationConfig`
and optional reviewed `JaxCPUReplayMechanicsAdapter` exactly as the strict runner,
and both modes share one private controller, so the sweeps, seeds, rollback,
cutback, and audit semantics are identical. Strict behavior and strict archive
semantics are unchanged: strict results still accept only fully passing sweeps.

Warning mode differs only at acceptance. A trial runs the full configured
stagger budget unless every gate passes earlier. Finite raw damage-range and
healing failures are never short-circuited, so the mechanics-after solve and
`H_next` update always complete. At the stagger limit, any failed finite
post-sweep acceptance gate may be bypassed — the canonical bypassable set
`WARNING_GATE_NAMES` is exactly `displacement`, `damage`, `history`,
`reaction`, `damage_residual`, `damage_range`, `healing`,
`history_monotonic`, `force`, and `moment`. A warning commit requires a
complete finite mechanics-before, `H_used`, raw damage, mechanics-after,
`H_next` sweep with complete diagnostics and commits the exact unmodified
`(u_after, d_next, H_next)`. No clipping, projection, tolerance change,
standalone final damage solve, history accumulation, or increment growth is
introduced. Missing/non-finite candidates, structural `ValueError`s, and
mechanics/CG failures that return no valid result remain the existing audited
hard/cutback failures; a warning-mode run at `du_min` or the cutback budget
still raises the typed `ContinuationFailure`.

Warning commits are labeled, never relabeled as strict evidence:

- `ContinuationMetadata.acceptance_mode` is `strict` or `warning`.
- `StaggerDiagnostics.complete_sweep` distinguishes full-sweep diagnostics
  from incomplete failure diagnostics.
- `AcceptedStep.acceptance_status` is `strict_passed` or `warning_accepted`;
  strict steps keep `diagnostics.passed=True` with no bypassed gates, while
  warning steps keep `diagnostics.passed=False` with the exact canonical
  `bypassed_gates` list (validated to match the failed gates exactly).
- `AttemptRecord` carries the matching `acceptance_status`, `sweep_complete`,
  and `bypassed_gates`; `ContinuationResult.validate` cross-pairs them with
  the accepted steps and rejects a warning-accepted step inside strict
  metadata.

Archives written by `write_continuation_archive` now record
`archive_schema_version` (currently `2`), `acceptance_mode`, the fixed
canonical `gate_names`, per-state `state_acceptance_status` (the initial zero
state first), `accepted_acceptance_status`, `accepted_warning`, the boolean
`accepted_bypassed_gate_mask` aligned with `gate_names`, and the attempt-audit
`sweep/status/bypassed` fields. The existing per-gate pass arrays keep their
actual values, so warning states retain `False` for every bypassed gate. All
arrays remain non-object and load with `allow_pickle=False`. No production
configuration is added by this mode; warning runs are diagnostics only and the
first authorized endpoint is `u_bar=2e-3`.

The warning regression class in
`tests/test_fracture2d_nonlocal_continuation.py` covers the budget-exhaustion
commit with an exact bypass list, endpoint/load pairing, unmodified
out-of-range damage commits with mechanics-after still executing, cutback and
typed failures for incomplete candidates, structural error propagation, strict
compatibility, CPU/JAX warning equivalence (including an actual GPU adapter
run), forged warning metadata rejection, archive reload/provenance, and
coherently forged final-sweep gate summaries (force pass plus shrunken bypass
list, volume-weighted used residual, `H_used`, and before-force) plus the
finite-complete-sweep diagnostics contract. It also rejects post-construction
diagnostic dtype changes and mechanics-backend relabeling that lacks the
CPU/JAX solve result's process-local provenance token.

The first authorized fresh `delta=0.048` warning-mode checkpoint reached
`u_bar=2e-3` in 20 nominal `1e-4` increments. Every step actually passed all
strict gates, so no warning bypass or rounded-cloud rollback occurred. Final
reaction was `253.5305191`, maximum damage was `0.0513580`, and reaction remained
monotone. Independent archive replay approved this only as temporary pre-peak
evidence; it does not select a production horizon or validate peak load, crack
growth, refinement, or PINN work.

A second fresh checkpoint reached exact `u_bar=3e-3` in 30 nominal `1e-4`
increments. An explicit eight-ulp endpoint snap prevents floating-point load
accumulation from creating a duplicate terminal solve; it does not change any
physical or convergence gate. Two complete finite steps were warning-accepted
for damage/history convergence gates. Final reaction was `371.4762173`, maximum
damage was `0.1204572` at the notch tip, and every reaction increment remained
positive. Independent replay approved the archive only as staged pre-peak
evidence. Near-threshold warning trigger loads vary with GPU/stagger execution
and are not claimed reproducible.

## Durable engineering-grip workflow

The reusable orchestration entry point is
`examples/fracture2d/engineering_grip_workflow.py`. It has four explicit
subcommands:

- `run` starts from exact zero, runs the strict GPU/CPU-replay controller, and
  writes a completed archive or a typed failure audit. Pass
  `--acceptance-mode warning --warning-safety-limit <positive-value>` for the
  separately archived warning diagnostic mode;
- `replay` independently replays a completed strict or warning archive, or a
  stabilized completed/failure archive, without calling the production solver;
- `analyze` derives load, reaction, gate-ratio, monotonicity, and crack-region
  summaries from either archive kind;
- `plot` compares accepted material `d`, `u_x`, and `u_y` fields and the
  nominal response with the warning-mode point-anchor FEM archive.

The production source cloud is generated from the tagged Gmsh mesh, not from a
temporary archive. The durable inputs used by the authorized engineering-grip
path are:

```text
/root/autodl-tmp/data/fracture2d/engineering_grip/source/rounded_slot.msh
/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz
/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz
```

To regenerate the cloud in the mutually exclusive mesh environment, use the
current `mesh.geo` with its physical `anchor` point and preserve the exact
float64/int64/bool arrays:

```bash
conda activate fenicsenv
gmsh -2 \
  -setnumber Lx 1.0 -setnumber Ly 1.0 \
  -setnumber tip_x 0.5 -setnumber center_y 0.5 \
  -setnumber r_tip 0.024 -setnumber lc_far 0.02 -setnumber lc_near 0.004 \
  examples/fracture2d/mesh.geo \
  -o /root/autodl-tmp/data/fracture2d/engineering_grip/source/rounded_slot.msh
python - <<'PY'
from pathlib import Path
import numpy as np
from examples.fracture2d.nonlocal_geometry import (
    ROUNDED_SLOT_TAGS,
    load_gmsh_particle_cloud,
)

mesh = Path(
    "/root/autodl-tmp/data/fracture2d/engineering_grip/source/rounded_slot.msh"
)
out = mesh.with_name("material_cloud_d048.npz")
cloud = load_gmsh_particle_cloud(
    mesh,
    horizon=0.048,
    boundary_tags=ROUNDED_SLOT_TAGS,
    visibility_tolerance=1.0e-12,
)
np.savez_compressed(
    file=out,
    coordinates=np.asarray(cloud.coordinates, dtype=np.float64),
    triangles=np.asarray(cloud.triangles, dtype=np.int64),
    volumes=np.asarray(cloud.volumes, dtype=np.float64),
    bottom=np.asarray(cloud.boundary.bottom, dtype=bool),
    top=np.asarray(cloud.boundary.top, dtype=bool),
    anchor_particle=np.asarray(cloud.boundary.anchor_particle, dtype=np.int64),
    d048=np.asarray(cloud.visible_edges, dtype=np.int64),
)
PY
```

Use the JAX environment for the strict workflow. Every required physical,
geometry, source, FEM, and continuation setting is explicit; the strict path
has no CPU fallback and no warning acceptance:

```bash
source .venv/bin/activate
python -m examples.fracture2d.engineering_grip_workflow run \
  --source-cloud /root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz \
  --fem-archive /root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz \
  --output-directory /root/autodl-tmp/data/fracture2d/engineering_grip \
  --run-id engineering_grip_v2_u1e4 \
  --load-final 1e-4 --max-stagger-iterations 20 --gpu-device-index 0 \
  --delta 0.048 --h-g 0.02 --du-initial 1e-4 --du-min 5e-5 \
  --visibility-tolerance 1e-12 \
  --youngs-modulus 210000.0 --poisson-ratio 0.3 --k-res 1e-8 \
  --rank-rtol 1e-12 --Gc 2.7 --l0 0.024
```

Longer studies use the same explicit command with a new run stem and authorized
endpoint/budget changes. They must start from exact zero and stop loudly on a
typed strict continuation failure; do not warning-accept or resume from a
serialized intermediate state. The controller classifies a trial increment as
`du_min` within eight float64 ULPs at the participating load/increment scale, so
subtraction roundoff cannot create a duplicate retry of the same minimum target.

For an explicitly requested warning diagnostic, use the same command and add a
new run stem plus a positive raw damage safety limit:

```bash
source .venv/bin/activate
python -m examples.fracture2d.engineering_grip_workflow run \
  --source-cloud /root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz \
  --fem-archive /root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz \
  --output-directory /root/autodl-tmp/data/fracture2d/engineering_grip \
  --run-id engineering_grip_warning_u1e4 \
  --acceptance-mode warning --warning-safety-limit 1e-3 \
  --load-final 1e-4 --max-stagger-iterations 20 --gpu-device-index 0 \
  --delta 0.048 --h-g 0.02 --du-initial 1e-4 --du-min 5e-5 \
  --visibility-tolerance 1e-12 \
  --youngs-modulus 210000.0 --poisson-ratio 0.3 --k-res 1e-8 \
  --rank-rtol 1e-12 --Gc 2.7 --l0 0.024
```

Warning acceptance is considered only on a final complete sweep. It may bypass
finite failed post-sweep gates, but incomplete, non-finite, mechanics, structural,
and work failures remain hard failures. Intermediate finite range/healing
violations may continue and recover; a warning commit retains the exact raw
candidate and requires both final range/healing excesses to be at most the
explicit safety limit. Warning archives and all derived reports are always
`validated=false` and are diagnostic evidence only.

The separately authorized unbounded warning diagnostic removes only the numerical
range/healing magnitude cutoff. It still requires a complete finite final sweep,
records the raw excesses, never clips or projects the candidate, and keeps all
non-finite, mechanics/CG, incomplete, structural, and work-check failures hard.
Use a new run stem and the exact fresh-zero command below; this is not a
production or validation command, and every result remains `validated=false`:

```bash
source .venv/bin/activate
python -m examples.fracture2d.engineering_grip_workflow run \
  --source-cloud /root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz \
  --fem-archive /root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz \
  --output-directory /root/autodl-tmp/data/fracture2d/engineering_grip \
  --run-id engineering_grip_warning_unbounded_u7p5e3_s72 \
  --acceptance-mode warning --warning-safety-policy unbounded \
  --load-final 0.0075 --max-stagger-iterations 72 --gpu-device-index 0 \
  --delta 0.048 --h-g 0.02 --du-initial 1e-4 --du-min 5e-5 \
  --visibility-tolerance 1e-12 \
  --youngs-modulus 210000.0 --poisson-ratio 0.3 --k-res 1e-8 \
  --rank-rtol 1e-12 --Gc 2.7 --l0 0.024
```

Unbounded mode can commit a finite damage field below `0` or above `1`, and
can commit finite healing of any magnitude after the final stagger budget.
Such fields are evidence of the unconstrained diagnostic path, not admissible
phase-field states or evidence that the full load path is physically valid.

### Stabilized convergence-first experiment

The separate `--continuation-strategy stabilized` path is an experimental
fresh-zero controller. It changes only the numerical fixed-point iteration:
the damage candidate is under-relaxed under an explicit
`--relaxation-policy`. `sufficient_decrease` is the existing safeguarded merit
backtracking policy. `fixed` is a damped-Picard policy: every sweep evaluates
exactly one candidate with the explicit `omega_initial`, does not require merit
decrease, and records the candidate merit only as a diagnostic. For `fixed`,
`omega_min` must equal `omega_initial`, `max_backtracks` must be `0`, and the
`omega_reduction`/`sufficient_decrease` flags are forbidden. The original
mechanics, damage equation, constitutive law, `k_res`, graph, boundary
conditions, and raw fields are retained. Strict stabilized mode keeps every
existing gate unchanged. The separate stabilized-warning mode requires fixed
relaxation and explicit unbounded warning policy. Only a complete final sweep
may bypass coupled displacement, damage, history, reaction, post-history damage
residual, fixed-point, range, and healing gates. Finite/mechanics/CG,
force/moment, history monotonicity, fixed-history damage-equation,
globalization/completeness, structural, and work-check failures remain hard.
No range projection, healing clipping, or displacement clipping is performed.

The sufficient-decrease controls are required explicitly and currently use
`omega_initial=0.5`, `omega_min=0.03125`, `omega_reduction=0.5`,
`sufficient_decrease=0.01`, and `max_backtracks=5`. The only authorized short
comparison is the fixed-policy fresh-zero run through `u_bar=0.0063` with
`omega=0.5`, `du_initial=1e-4`, and `du_min=1.25e-5`; it is not a production
run or a physical crack-growth validation. Compare it with the unbounded
warning diagnostic before considering any longer endpoint.

Before any GPU adapter or continuation work is created, the stabilized
workflow preflights durable archive storage from the exact model dimensions
and the accepted-state layout. `estimate_stabilized_archive_bytes` returns a
conservative uncompressed bound for the configured run: accepted physical
states through the configured load/cutback bounds, one compact solver-summary
row per retained state, work-check scalars, the failure-summary reserve, and
JSON/NPY container reserves. Static geometry and operators are rebuilt from the
source cloud and are not included in the v5 archive estimate. Every stabilized
run must also provide `--archive-budget-bytes`. Startup requires the nominal
archive estimate to fit that budget and the budget plus the named
`STABILIZED_FILESYSTEM_RESERVE_BYTES` (2 GiB) to fit the free bytes of the
output filesystem. The theoretical worst-case estimate is reported for
visibility; runtime checks stop before a prospective accepted-state row would
exceed the budget and write a replayable failure archive without committing it.
The flag is rejected for the legacy continuation strategy.

```bash
source .venv/bin/activate
python -m examples.fracture2d.engineering_grip_workflow run \
  --source-cloud /root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz \
  --fem-archive /root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz \
  --output-directory /root/autodl-tmp/data/fracture2d/engineering_grip \
  --run-id engineering_grip_damped_cached_o5_u6p3e3_s144 \
  --continuation-strategy stabilized \
  --relaxation-policy fixed \
  --archive-budget-bytes 500000000 \
  --progress-every-sweeps 10 \
  --load-final 0.0063 --max-stagger-iterations 144 --gpu-device-index 0 \
  --delta 0.048 --h-g 0.02 --du-initial 1e-4 --du-min 1.25e-5 \
  --visibility-tolerance 1e-12 \
  --youngs-modulus 210000.0 --poisson-ratio 0.3 --k-res 1e-8 \
  --rank-rtol 1e-12 --Gc 2.7 --l0 0.024 \
  --omega-initial 0.5 --omega-min 0.5 \
  --max-backtracks 0
```

The authorized damped stabilized-warning diagnostic uses a separate run stem
and schema:

```bash
source .venv/bin/activate
python -m examples.fracture2d.engineering_grip_workflow run \
  --source-cloud /root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz \
  --fem-archive /root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz \
  --output-directory /root/autodl-tmp/data/fracture2d/engineering_grip \
  --run-id engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144 \
  --continuation-strategy stabilized --relaxation-policy fixed \
  --acceptance-mode warning --warning-safety-policy unbounded \
  --archive-budget-bytes 500000000 --progress-every-sweeps 10 \
  --load-final 0.0075 --max-stagger-iterations 144 --gpu-device-index 0 \
  --delta 0.048 --h-g 0.02 --du-initial 1e-4 --du-min 6.25e-6 \
  --visibility-tolerance 1e-12 \
  --youngs-modulus 210000.0 --poisson-ratio 0.3 --k-res 1e-8 \
  --rank-rtol 1e-12 --Gc 2.7 --l0 0.024 \
  --omega-initial 0.5 --omega-min 0.5 --max-backtracks 0
```

The physical, discretization, load-step, fixed-relaxation, and input-path values
shown above are now the `run` parser defaults. They remain overridable. The
controller selection is deliberately not inferred from those numbers: the
default acceptance remains `strict`, while stabilized execution still requires
an explicit continuation strategy, archive budget, and progress cadence;
warning acceptance additionally requires explicit `warning` and `unbounded`
flags.

The completed warning path also exhibits suspected effectively zero-stiffness
particle drift after severe localization: material displacement grows far beyond
the prescribed grip displacement while the response collapses. This is an open,
deferred numerical issue. The warning archive is retained for diagnosis and
must not be treated as a converged default solution.

The stabilized command requires a positive explicit `--progress-every-sweeps`
cadence. It emits transient, flushed, compact JSON lines without adding logs to
the v5 archive; the first, periodic, final, and passing sweeps are reported:

```text
STABILIZED_PROGRESS {"committed_load":0.0,"cutback_count":0,"elapsed":0.02,"event":"attempt_start","increment":0.0001,"max_sweeps":144,"omega":0.5,"policy":"fixed","step_index":1,"target_load":0.0001}
```

The progress option is rejected for the legacy continuation strategy. A stdout
write or flush failure is an explicit operational error; logging is never
silently disabled while a long continuation keeps running invisibly.

Stabilized archives and workflow reports remain explicitly experimental and do
not claim the generic `validated=true` status used by the legacy strict path.
Independent replay reports instead expose `field_equations_validated`: this is
true only after every retained accepted state passes the independently rebuilt
field-equation, mechanics, history, damage, conservation, and work checks.
For the separate stabilized-warning mode, replay still performs those checks.
It reports `field_equations_validated=false` when a warning-accepted state fails
the independently rebuilt static field equations, but may report true when the
stored equations pass and only transition-convergence, range, or healing gates
were bypassed. Warning gate/bypass provenance remains available separately.
Failure archives remain unvalidated and the workflow does not promote this
short experiment to a production horizon.

Strict stabilized archives use the version-5 schema and replay contract. The
stabilized-warning archives use a separate warning-v1 identity while retaining
the same lean accepted-state design:

- **Physical rows** retain the exact-zero state and each accepted state:
  `state_u_bar`, union displacement `state_union_displacement`, material
  damage `state_material_damage`, and material history
  `state_material_history`, together with exact state digests and acceptance
  statuses.
- **Summary rows** retain one compact canonical `solver_summary_json` row for
  the initial state and each accepted state. Accepted reaction scalars,
  cumulative `state_external_work`, and the centered work-check scalars are
  stored as typed arrays. A failure archive additionally stores one
  `failure_summary_json` record and the last committed-state summary. No
  intermediate sweep/trial vectors, attempt/sweep/trial logs, or static
  geometry/operators are duplicated in the archive.

Each retained solver summary records the selected `relaxation_policy`, applied
`omega`, candidate `merit`, and the policy-specific globalization diagnostic.
For the fixed policy, `sufficient_decrease_limit` is explicitly `null` and
`globalization_pass` means that the single complete hard-gated candidate was
selected; it does not assert merit decrease.

Replay reloads the external source cloud by its absolute path, field
descriptors, size, and SHA256, then rebuilds the geometry and operators from
that source. Reports are bound to the exact archive SHA256, schema, completion
status, source-bundle hash, and the `states`, `accepted_steps`, and
`solver_summary_rows` counts. Because v5 does not retain rejected-trial `q`,
`d`, or `H` vectors, it cannot replay the exact rejected numerical path:
successful replay reports
`path_replayed=false` and `rejected_trial_path_available=false`. This is a
field-equation validation of retained accepted states, not a claim that the
full rejected continuation path was replayed.

All run artifacts are written beside the selected run stem. A completed run
has `.npz`, `.summary.json`, `.replay.json`, `.analysis.json`, `.fields.png`,
`.stress_strain.png`, `.curve.npz`, `.plot.json`, and `.manifest.json` files.
A failure has corresponding `.failure.*` audit/summary/replay/analysis/plot files and
the same manifest. Writes are atomic, existing artifacts require
`--overwrite`, and the manifest records source/code hashes without copying a
temporary source binary into the output directory.

For existing artifacts, use the independent operations explicitly:

```bash
python -m examples.fracture2d.engineering_grip_workflow replay \
  --archive /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.npz \
  --report /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.replay.json

python -m examples.fracture2d.engineering_grip_workflow analyze \
  --archive /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.npz \
  --report /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.analysis.json

python -m examples.fracture2d.engineering_grip_workflow plot \
  --archive /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.npz \
  --fem-archive /root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz \
  --target-load 1e-4 \
  --fields /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.fields.png \
  --curve /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.stress_strain.png \
  --curve-data /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.curve.npz \
  --report /root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u1e4.plot.json
```

The FEM archive is a warning-mode diagnostic reference only. Its reaction and
near-grip fields are not engineering-grip acceptance gates; strict finite,
mechanics, damage, history, range, healing, force, moment, and crack-region
requirements remain authoritative for the direct nonlocal result.
