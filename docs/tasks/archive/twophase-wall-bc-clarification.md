# Constricted-channel wall-BC clarification

## Goal and context

Answer how the proposed indented walls influence a full NS--CH flow through
boundary conditions. User wants the physical mechanism, not another general
feasibility report. Continue the independent-wall feasibility analysis. No
implementation, numerical BC choice or training is requested.

Read progress and last five decisions; current numerical/JAX/development rules
and the preceding source/task remain applicable. Current twophase2d is still
periodic prescribed CH and has no implemented wall NS boundary path.

## Status and explanation

Complete. Stationary upper/lower curves satisfy ux=uy=0 at their actual physical
coordinates. This is velocity Dirichlet data and includes both no penetration
and no slip. Indentation changes the domain and where that same data applies;
wall values stay zero while interior profiles, transverse flow and pressure
adjust under incompressibility and momentum balance. Fixed inlet flow gives
mean axial Q/h, not a prescribed whole velocity field. The wall itself does
not inject extra flow/work or require an arbitrary prescribed wall pressure.

Recommended conceptual boundary partition: a common inlet velocity profile
with the desired fixed flow law, no-slip on both shaped walls, and a compatible
pressure-type open/traction outlet. Bare p=0 is not a complete universal outlet
specification. Concrete traction convention, phase inlet/outlet and possible
outflow stabilization remain future implementation choices. No duplicate
outlet velocity/no-slip or pressure-gauge prescription is selected here.

In a NO, geometry encoding and BC enforcement are separate: actual per-domain
wall points enter both condition data and appropriate boundary constraints or
residuals. A velocity boundary loss could average |ux|^2+|uy|^2 using wall
quadrature; an exact boundary construction is also possible but not designed
here. Domain PDE sampling stays inside the actual geometry. Geometry tokens or
regional scalar nonlocal sums alone do not enforce no-slip. Phase-wall wetting
and chemical-potential no-flux data are separate from the velocity constraint.

## Primary references checked

- NGSolve official Navier--Stokes tutorial:
  https://ngsolve.org/ngsolve/docs/i-tutorials/unit-3.2-navierstokes/navierstokes.html .
  Opened lines66--72 confirm inlet profile, open outlet and homogeneous velocity
  Dirichlet walls; no cylinder geometry or time scheme is adopted. Its displayed
  integral normalization has a typo relative to its mean-one profile; no numeric
  inlet coefficient is copied into the proposed model or response.
- FEniCSx tutorial, NS stress/variational boundary discussion:
  https://jsdokken.com/dolfinx-tutorial/chapter2/navierstokes.html .
  Opened lines71--99 confirm momentum/incompressibility, stress and traction
  boundary roles, and that outflow treatment depends on the chosen formulation.
  No split-method outlet condition is promoted as a complete coupled NS--CH BC.
- Earlier primary NS--CH boundary references and the geometric flux derivation
  remain recorded in `twophase-independent-wall-nsch.md`.

## Verification, diff and task-end

Concept/source/document review and `git diff --check` only. No implementation,
numeric tests, solver/training, default/dependency or Git-history change. Update
`docs/PROGRESS.md`, append `docs/DECISIONS.md`, archive this task; prior diagrams
and uncommitted documentation remain untouched. No blocker, required resume or
queued numerical job.
