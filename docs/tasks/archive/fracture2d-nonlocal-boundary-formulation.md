# Fracture2D Nonlocal Boundary Formulation

## Goal

Derive and independently audit a nonlocal mechanics boundary formulation for
the rounded-slot fracture2d problem before implementing a new operator. The
formulation must support component-wise mixed boundary data, preserve a clear
energy/work and reaction contract, and keep mechanics free-surface treatment
separate from AT2 damage boundary treatment.

## Acceptance Criteria

- Cite primary sources with page/equation references for the selected nonlocal
  kinematics, equilibrium, and boundary treatment.
- Define the top `u_y=u_bar`, bottom `u_y=0`, lower-left `u_x=0`, and otherwise
  free `u_x` conditions without assigning hidden collar displacement data.
- Define real/fictitious particle coordinates, volumes, corner ownership,
  rounded-slot visibility, and kinematic/stress neighborhoods.
- Define plane-strain constitutive response and frozen phase-field degradation
  on every real/fictitious center or bond used by mechanics.
- Derive either a conservative discrete energy and its residual/tangent, or
  explicitly reject the formulation for this project.
- Define work-conjugate top, bottom, and anchor reactions and force/moment
  conservation checks.
- Define affine patch, manufactured-boundary, free-surface, and frozen
  `0.0055 -> 0.0056` comparison tests with quantitative pass criteria.
- Obtain an independent numerical-method review and explicit implementation
  go/no-go decision.

## Relevant Context

- The current material-only Chen--Spencer-inspired BA energy develops an
  unstable top damage/history fixed point at `u=0.0056`; the warning branch is
  not accepted as converged physical evidence.
- The current mechanics constraints are component-wise: top and bottom prescribe
  only `u_y`, one lower-left particle prescribes `u_x`, and all other horizontal
  boundary DOFs are free.
- Behzadinasab Part I/II sources define separate kinematic/stress neighborhoods
  and zero-stress fictitious nodes, but not all data required by this BVP. In
  particular, they do not source the mixed-component collar, phase-field
  degradation on fictitious nodes, global discrete energy/reaction, or exact
  rounded-slot collar quadrature.
- The existing BA operator, continuation, damage graph, and raw diagnostic
  archives must remain unchanged during derivation.

## Status

Completed and independently reviewed. Implementation is accepted only as
an isolated NumPy/SciPy float64 frozen-mechanics prototype at `delta=0.048`.
The manufactured gates and both actual union-cloud hard-gate spacings pass. The
frozen transition is numerically admissible but does not satisfy the
project-owned diagnostic-usefulness criterion, so Candidate A2 remains a
nonproduction counterfactual. No coupled fracture, continuation, JAX/PINN, or
production operator is authorized.

## Blocker

The isolated prototype has no unresolved implementation blocker. The transition
comparison is a substantive diagnostic blocker: at `h_g=0.02`,
`R_eff=667.0949973747717` versus mapped FEM `637.5547029659703` (within the
10% guard), but the material-top `H_used` error reduction is only about 3.8%,
not the required 50%. The implementation must not be promoted to a physical or
coupled boundary model. This failed irreversible-history criterion is dominated
by the transplanted raw history floor: A2 reduces instantaneous top `Psi_plus`
from the material-only `78.8966` to `5.6326` (`h_g=0.02`) or `6.2479`
(`h_g=0.01`), while inherited top history is already `76.1040`.

## Next Action

No action remains in this task. Any clean-start use of A2 requires a new
coupled-formulation decision defining material-only damage/history with
phase-measured grip mechanics; do not infer that authorization from the strong
instantaneous-driver result.

## Implementation And Validation Record — 2026-08-24

- Added `examples/fracture2d/nonlocal_grip_diagnostic.py` and
  `tests/test_fracture2d_nonlocal_grip_diagnostic.py`; existing material-only
  mechanics, damage, continuation, JAX/PINN, and production configuration files
  were not modified by this implementation.
- The focused Candidate A2 suite passes all 11 tests. The related geometry,
  mechanics, baseline, and Candidate A2 suites pass all 42 tests. The changed
  module and test compile successfully, and `git diff --check` passes.
- The manufactured contract passes: three rigid modes, affine error
  `6.66e-16`, energy-gradient error `8.66e-12`, residual-JVP error `3.88e-12`,
  and work error `1.72e-12`.
- The final `h_g=0.02` replay passes all hard gates with 3701 union particles,
  79060 directed edges, minimum constrained Ritz value `9.977418624358839`,
  `R_ext=608.5826594600637`, `R_eff=667.0949973747717`, reaction-work relative
  error `1.9522e-10`, and opening-chain error `4.2119e-08`. The prior staged
  `h_g=0.01` run also passed all hard gates with 3905 particles and 89004
  directed edges before transition authorization.
- Canonical nonproduction artifacts are under
  `/tmp/opencode/fracture2d_candidate_a2_canonical_20260824_r2/`, with manifest
  `fracture2d_candidate_a2_canonical_manifest.json`, comparison report
  `fracture2d_candidate_a2_canonical_comparison.json`, and per-spacing JSON/NPZ/
  PNG files. The artifact schema is `candidate-a2-run-v2`; the diagnostic module
  SHA256 is
  `21210e40b2c689279e3ce952b73bcff88a5cdf780fd2249c660fa957dd99208a`.
- Primary review found no high-severity implementation defect after phase-weight,
  visibility, reaction-balance, history-floor, and canonical-artifact fixes.
  The hard-gate pass is credible. The `3.7654%` `H_used` rejection is valid under
  the predeclared gate but is not the instantaneous mechanics improvement.

## Candidate A: Conservative Energy-Bearing Grip Collar

This candidate is a project-owned variational extension, not the
Behzadinasab strong-form natural-boundary method. It is restricted to an intact
or explicitly frozen-damage mechanics diagnostic. It is not yet authorized for
coupled fracture.

### Domain And Quadrature

Let the physical rounded-slot domain be `Omega_m`. For the diagnostic horizon
`delta`, define two full-width exterior grip regions

$$
\Omega_B=[0,1]\times[-\delta,0],\qquad
\Omega_T=[0,1]\times[1,1+\delta],
$$

and `Omega_star = Omega_B union Omega_m union Omega_T`. Mesh this union as one
conforming Gmsh domain so interface vertices at `y=0` and `y=1` are not
duplicated. Use the existing characteristic length and triangular barycentric
nodal volumes. The required quadrature identities are the total area and first
moments of `Omega_star`; no volume may be counted twice at a grip/material
interface. The rounded slot remains material-only and unchanged. The outer grip
sides and outer top/bottom are graph-truncated free surfaces except for the
volume displacement constraints below.

Build one reciprocal strict-horizon graph over all particles. Material--grip
edges across `y=0` and `y=1` are admissible. Segment visibility must remain
inside the closure of `Omega_star`; no edge may cross the rounded slot or leave
and re-enter the union. Existing target-inclusive BA supports, cubic influence,
shape tensors, and rank tolerance are unchanged.

### Frozen Degradation

For a raw committed material damage field `d_m` at `u=0.0055`, define

$$
d_i^\star=
\begin{cases}
(d_m)_i,&i\in\mathcal P_m,\\
0,&i\in\mathcal C_B\cup\mathcal C_T.
\end{cases}
$$

Thus both grips are explicitly intact, energy-bearing elastic material. This is
not an interpolation, natural damage condition, or phase-field collar; it is a
counterfactual non-fracturing grip assumption. No damage or history solve is
performed on `Omega_star`. Amor strain energy is evaluated and reported only at
the original material particles.

### Energy And Constraints

On the enlarged particle set and graph, retain the existing BA definitions
without changing constitutive or support formulas:

$$
\Pi^\star(q;d^\star)
=\sum_{i\in\mathcal P^\star}V_i g(d_i^\star)
\sum_{j\in\mathcal H_i}\eta_{ij}\psi_{i|j}(q).
$$

Use component-wise affine elimination

$$
q=P_fq_f+q_0+a_T\bar u.
$$

All bottom-grip particle `u_y` DOFs are fixed to zero and all top-grip particle
`u_y` DOFs are fixed to `u_bar`. Every grip `u_x` remains free except one
horizontal anchor at the original material particle `(0,0)`. No physical
material top/bottom particle is directly constrained merely because it lies at
the grip interface.

The free equilibrium, tangent, and generalized top reaction are

$$
P_f^T r=0,\qquad r=\nabla_q\Pi^\star,
$$

$$
K_{ff}=P_f^T\nabla_q^2\Pi^\star P_f,
$$

$$
R_T=\frac{d\Pi_{\mathrm{red}}^\star}{d\bar u}=a_T^Tr.
$$

The final equality is valid only at free equilibrium. Bottom and anchor
reactions are the corresponding constrained residual sums. Translation and
rotation invariance require total force and moment over all constrained DOFs to
vanish within roundoff/linear-solve error.

### Required Verification Before Transition Replay

1. Validate exact dtype, finite values, coordinate uniqueness, positive volumes,
   area/first moments, reciprocal strict-horizon edges, visibility, rank, and
   condition-number reporting.
2. On a manufactured regular enlarged cloud, reproduce rigid translations,
   infinitesimal rotation, and affine strain to float64/operator tolerance.
3. Check the residual against directional finite differences of `Pi_star` and
   check independent tangent actions against residual finite differences.
4. Check full and free tangent symmetry, rigid nullity before constraints, and a
   positive constrained first non-rigid eigenvalue.
5. Check `R_T` against centered finite differences of the reduced equilibrated
   energy, and check total force/moment conservation.
6. Repeat at at least two collar mesh spacings before interpreting the physical
   transition diagnostic; fixed `delta` alone is not a refinement claim.

### Frozen Transition Decision Metrics

At material `d_m,H_m` from raw `u=0.0055`, solve only fixed-damage mechanics at
`u_bar=0.0056`. Report material-particle displacement, strain, instantaneous
Amor driver, reaction, and all numerical gates. Compare with both the current
material-only BA solve and mapped FEM at the same nominal load.

The counterfactual is numerically admissible only if every mechanics, energy,
work, symmetry, rank, and conservation gate passes. It is diagnostically useful
only if the top-to-tip Amor-driver ratio is below one and its error relative to
mapped FEM is smaller than the material-only BA error, without moving the global
driver maximum away from the notch region to another artificial grip/interface
location. Reaction error and physical top/interface displacement must be
reported, not silently normalized away. These are comparison criteria, not
production validation or proof that the grip model is correct.

## Open Approval Items

- Adding energy-bearing grips changes the BVP and contradicts the previously
  frozen material-only boundary choice unless explicitly approved as a separate
  counterfactual.
- `d=0` in the grips is an explicit non-fracturing-grip assumption; it cannot be
  reused for coupled damage without a new formulation decision.
- The outer grip side/free-edge behavior is variational graph truncation, not
  Behzadinasab zero-stress fictitious-node enforcement.
- Exact Gmsh construction, mesh size, interface tagging, and visibility contract
  require implementation review after this derivation is approved.
- No source audit found a published formulation that combines this enlarged BA
  energy, the exact mixed-component constraints, and phase-field-degraded
  fictitious particles. Candidate A must remain labeled project-owned.

Independent review rejects Candidate A as written. A conforming remesh would
not preserve the raw material particle IDs/fields, shared interface particles
have ambiguous material-versus-grip damage, the specialized constraint class
requires an anchor in its bottom mask, and a volume-fixed grip does not provide
a like-for-like physical opening or reaction. Candidate A is retained above as
the rejected first derivation, not as an implementation contract.

## Candidate A2: Phase-Measured Elastic Grip Prototype

Candidate A2 resolves the first review's algebraic and mapping blockers. It is
still a project-owned frozen-mechanics counterfactual and remains subject to a
second independent review.

### Exact Material Preservation And Collar Mesh

Preserve all existing material coordinates, material triangles, material
barycentric nodal volumes, directed material graph data, and raw material field
indices bitwise. Do not remesh `Omega_m`.

Construct each grip as a separate structured triangular strip whose interface
row reuses the existing ordered physical top/bottom coordinates. For the current
uniform boundary spacing `h=0.02` and `delta=0.048`, use exterior offsets
`0.02`, `0.04`, and `0.048`. More generally use all positive `k*h<delta` plus
the exact final offset `delta`. Split every interface-aligned quadrilateral into
two triangles with a deterministic alternating diagonal. Reject the geometry if
the physical boundary coordinates are not ordered, unique, and uniformly spaced
within the declared float64 geometry tolerance.

The combined geometric particle set contains one displacement DOF pair per
unique coordinate. Interface coordinates are not duplicated. Partition the
triangles, however, into material and grip phases and compute two barycentric
center measures:

$$
V_i^m=\sum_{T\in\mathcal T_m,\,i\in T}\frac{|T|}{3},\qquad
V_i^g=\sum_{T\in\mathcal T_g,\,i\in T}\frac{|T|}{3}.
$$

The kinematic neighbor volume is the union measure

$$
V_i^\star=V_i^m+V_i^g.
$$

At interface particles both phase measures may be positive, but the raw
material field remains attached to the unchanged geometric particle ID.

At `delta=0.048`, the required union quadrature targets are

$$
A_\star=1.0722509374813683,
$$

$$
(M_x^\star,M_y^\star)
=(0.5421241109732579,0.5361254687406843).
$$

Tests must also recover the original material area/moments exactly to the
existing geometry tolerance, each grip area `delta`, no duplicate triangles,
and no phase-volume double counting.

### Phase-Measured Frozen Energy

Use union volumes `V_star` in family masses, `eta`, shape tensors, and all BA
kinematics. Replace the one-value center factor only in the outer energy sum by

$$
W_i(d_m)=V_i^m g((d_m)_i)+V_i^g g(0),
$$

where `(d_m)_i` is required only when `V_i^m>0`. Since the current degradation
is `g(d)=(1-d)^2+k_res`, the grip factor is explicitly `g(0)=1+k_res`, not one.
The frozen mechanics energy is

$$
\Pi^{A2}(q;d_m)
=\sum_i W_i(d_m)
\sum_{j\in\mathcal H_i^\star}\eta_{ij}^\star\psi_{i|j}^\star(q).
$$

This is a conservative two-measure center-energy extension. A material--grip
interface center contributes its material and grip center measures with their
respective degradation, while sharing one displacement and one union BA
kinematic reconstruction. Residual and tangent are the exact first and second
derivatives of `Pi_A2`; no phase-field or history DOF exists in the grip.

### Outer-Face Constraints And Matched Opening

Use a generic immutable affine-constraint object rather than overloading the
current material-boundary-specific `HardBoundaryConstraints` contract. Prescribe
only

- `u_y=0` on the outer bottom face `y=-delta`;
- `u_y=U_ext` on the outer top face `y=1+delta`;
- `u_x=0` at the unchanged original material anchor `(0,0)`.

Every other component, including all grip `u_x` and all material/grip interface
DOFs, is free. The generalized outer reaction is

$$
R_{ext}=a_{ext}^Tr=\frac{d\Pi_{red}^{A2}}{dU_{ext}}.
$$

Define line-quadrature-weighted physical interface opening

$$
\Delta_m(U_{ext})
=\frac{1}{L_T}\sum_{i\in\Gamma_T}w_i u_{i,y}
-\frac{1}{L_B}\sum_{i\in\Gamma_B}w_i u_{i,y},
$$

where `w_i` are trapezoidal weights from the unchanged physical boundary
segments and `L_T=L_B=1`. Because frozen mechanics is linear, solve once at a
unit nonzero `U_ext`, require `Delta_m/U_ext>0`, and scale `U_ext` so
`Delta_m=0.0056` exactly to the load-comparison tolerance. The opening-conjugate
effective reaction is

$$
R_{eff}=R_{ext}\left(\frac{d\Delta_m}{dU_{ext}}\right)^{-1}.
$$

Report both `R_ext` and `R_eff`. Only `R_eff` may be compared with the FEM and
material-only reactions at prescribed physical opening, and even that
comparison retains a different boundary-displacement distribution.

### Corrected Conservation And Field Contract

The small-strain energy must pass translation and infinitesimal-rotation null
tests; no finite-rotation objectivity is claimed. Conservation is evaluated in
reference coordinates over the complete residual:

$$
\sum_i r_i=0,\qquad \sum_i X_i\times r_i=0.
$$

At free equilibrium these become balances among outer-top, outer-bottom, and
anchor constrained reactions.

Map raw `u=0.0055` damage and history by exact original material particle ID;
reject any coordinate or material-volume mismatch. Label the input a
`validated=False` transplanted frozen diagnostic, not an accepted Candidate A2
state. Mechanics uses only `d_m`. On original material particles report both
instantaneous Amor driver `Psi_plus` and

$$
H_{used}=\max(H_m,\Psi^+).
$$

No grip history is constructed.

### Quantitative Prototype Gates

- Geometry: exact material coordinates/triangles; float64 area and first-moment
  defects within the existing scale-aware `512*eps64` reconstruction bound.
- BA support: zero rank failures at `rank_rtol=1e-12`; report support and
  condition distributions and reject any non-finite condition.
- Patch: affine and infinitesimal-rigid-mode maximum reconstruction error no
  greater than `5e-12` on the actual enlarged cloud.
- Variational consistency: centered energy-gradient and residual-JVP relative
  errors no greater than `1e-7`; full/free tangent relative symmetry defect no
  greater than `1e-10`.
- Stability: exactly three unconstrained infinitesimal rigid modes before the
  anchor/load constraints and a positive constrained minimum eigenvalue above
  its eigensolver residual/error bound.
- Solve: retain `cg_rtol=1e-7`, `cg_atol=1e-10`, `cg_maxiter=3000`; require
  `info=0` and the authoritative free-residual gate.
- Work/conservation: centered reduced-energy checks for `R_ext` and `R_eff`
  within `1e-7` relative error; force/moment within the existing scale-aware
  reaction gate.
- Mesh sensitivity: run at least `h_g=0.02` and `h_g=0.01` while preserving
  `delta=0.048` and original material particles; report rank, conditioning,
  `Delta_m/U_ext`, `R_eff`, and material top/tip drivers.

After all numerical gates pass, Candidate A2 is diagnostically promising only
if material top `H_used` remains below notch-tip `H_used`, its absolute top error
against mapped FEM is reduced by at least 50% from the material-only BA error,
and no new global `H_used` maximum appears at a grip interface or side corner.
Require `R_eff` to remain within 10% of the mapped FEM reaction as a diagnostic
guard, not a production calibration. Failure of these project-owned comparison
criteria rejects Candidate A2 as a useful boundary counterfactual but does not
invalidate its numerical implementation.

### Executable Contract Corrections

The collar refinement parameter `h_g` controls only the outward-normal rows;
the tangential interface spacing remains the unchanged material spacing. For
each `h_g`, offsets are all positive `k*h_g<delta` plus exact `delta`. Thus
`h_g=0.01` uses `0.01,0.02,0.03,0.04,0.048` without adding material/interface
particles. Quadrilaterals are indexed by increasing physical-boundary segment
index and outward row. Split even-indexed cells from lower-left to upper-right
and odd-indexed cells from upper-left to lower-right. Interface coordinates are
the original unsnapped values; a boundary is accepted as uniform only if every
spacing differs from its median by at most the existing scale-aware float64
geometry tolerance.

Store explicit `material_indices`, `material_volumes`, `grip_volumes`,
`union_volumes`, phase triangle arrays, interface masks, and outer-face masks.
Union arrays begin with the original material-particle ordering. Require exact
equality of that coordinate prefix and material triangles, bitwise equality of
copied material volumes, and exact equality of the induced material--material
`d048` directed graph with the original graph. Build all new union edges from an
independent union-polygon segment-containment predicate using the existing
effective geometry floor; require reciprocal lexicographic order and strict
`0<distance<delta`.

The frozen source artifacts are

- `/tmp/opencode/fracture2d_d048_warning_ufinal_0075.raw.npz`, SHA256
  `48d92621955289c51710296569362c50db30e28f63d622ee447c11a2317f66e5`;
- `/tmp/opencode/stage3_cloud_float64.npz` for coordinates, triangles,
  barycentric volumes, tags, and `d048` edges.

Require `validated=False`, exact raw/cloud coordinate and edge equality, exact
raw/cloud triangle equality, and exact reconstructed/cloud material-volume
equality. Select the unique raw load satisfying `isclose(u_bar,0.0055,rtol=0,
atol=1e-14)`; record its stored value and copy `damage[index]` and
`history[index]` without interpolation. Grip-only field storage must not use
NaN sentinels.

The generic immutable affine-constraint prototype contains `n_particles`,
sorted unique `constrained_dofs`, complementary `free_dofs`, exact float64
`q_zero`, exact float64 `load_direction`, disjoint outer-top/outer-bottom masks,
the original anchor index, and explicit top/bottom/anchor residual selectors.
It validates a complete DOF partition, zero `q_zero` on top-load DOFs, unit
vertical `load_direction` only on outer-top DOFs, bottom `u_y=0`, anchor
`u_x=0`, and no other constrained component. The prototype solver accepts this
object separately; it does not weaken the existing `HardBoundaryConstraints`.

An independent reference loop must reconstruct every union support and verify

$$
\text{center factor}_{ij}=W_i\eta_{ij}^\star,
$$

while family mass, `eta`, `beta`, and shape tensors use only `V_star`. Shared
production helpers are not sufficient for this algebra gate. Use centered
energy/residual differences with the existing mechanics test convention
`epsilon=2e-5`, deterministic normalized directions, and the existing strict
relative-error evidence rather than relaxing algebra checks to `1e-7`; record
the measured defects and require each to be no worse than `1e-10` on the
manufactured prototype. Reaction/work checks use `epsilon=1e-5` and a `1e-7`
relative gate because they include independent re-solves.

On the actual enlarged cloud, verify analytic x/y translations and the
infinitesimal rotation `(-y,x)` by tangent action with the existing scale-aware
mechanics comparator. Orthonormalize those three vectors, project the free
spectral operator onto their complement before constraints, and require a
converged smallest positive Ritz value whose normalized residual is below
`1e-7` and whose lower bound `lambda-Ritz_residual` is positive. After affine
constraints, require the same positive lower-bound condition without removing
additional modes. Do not claim a dense count of exactly three zero eigenvalues
on the production cloud.

Order each unchanged physical top/bottom segment by x and form trapezoidal
weights as half the sum of adjacent segment lengths, with one half-segment at
each endpoint. Require each weight positive and each line-weight sum equal to
the polygonized boundary length within the geometry tolerance. After scaling,
require `Delta_m=0.0056` within `512*eps64*max(1,abs(Delta_m))`. Check
`R_ext` by centered reduced-energy differences and `R_eff` by

$$
\frac{\Pi_{red}(U+\epsilon)-\Pi_{red}(U-\epsilon)}
{\Delta_m(U+\epsilon)-\Delta_m(U-\epsilon)}.
$$

For field comparisons, `material top` is the exact original top mask, the notch
tip is the unique original particle exactly at `(0.5,0.5)`, and all reported
regional values are maxima unless named otherwise. The mapped FEM source is
`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz`
with SHA256
`e9634f1f8fdb21116b50ad453d37115746c4a5a09f828e76245a9c59f6107d17`;
retain the independently validated DG0-to-particle area-mass-lumped mapping.
Compare A2 at `Delta_m=0.0056` with material-only BA/FEM at physical opening
`u_bar=0.0056`, using the same reaction sign. The primary field is `H_used`; also
archive instantaneous `Psi_plus`. The no-new-maximum check applies only to
original material particles: its global `H_used` maximum must lie within one
`delta` of the notch tip and not on the top/bottom interface or physical side
corners. Grip particles may archive instantaneous `Psi_plus` separately but do
not carry or participate in `H_used`.
