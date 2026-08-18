# 2D Rounded-Notch Local FEM Reference

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
