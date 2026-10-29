# Phase-field case with complex geometry and Neumann boundaries

Goal: recommend a phase-field example suited to nonlocal natural no-flux
boundary handling on modestly complex geometry. This turn is assessment only;
no implementation/training/default/Git change. Preserve precipitate results.

Acceptance: primary-source basis, explicit local/graph boundary distinction,
physical scenario and simple-to-hard geometry proposal, fair local AD comparison
and independent-reference validation route. No promised accuracy advantage.

Status: complete; recommendation and primary-source assessment delivered.
Blocker: none. Next action: none until user selects implementation. No training
process, numerical/default/reference change or Git operation.

Context: prior examples periodic or simple boundary geometry; user wants a
phase-field Neumann case where geometry handling can show actual nonlocal
advantages. Latest comparator preference is local AD, not local spectral.
Current FFT boundary issue remains deferred; this is a new case assessment,
not authorization to redesign the old two-phase case. No elastic/flow coupling
needed for a clean boundary experiment.

## Recommended scenario and equations

First geometry: smooth elliptical closed container with two unequal circular
impermeable holes at nonsymmetric positions. Schematic is illustrative only;
no physical dimensions, horizon or final initial field is frozen. Use smooth,
deterministic initial phase bands with evolving interfaces reaching both hole
and outer walls; check initial Neumann compatibility before training. Avoid
high-frequency random spinodal patterns in the first L2 comparison, since
pattern sensitivity can confound discretization and optimization errors.

Local constant-mobility mixed CH:

$$
\phi_t=M\Delta\mu,\qquad
\mu=\beta(\phi^3-\phi)-\epsilon^2\Delta\phi,
\qquad \partial_n\phi=\partial_n\mu=0.
$$

Zero normal phase derivative follows from variation of isotropic gradient
energy without a wall energy; zero chemical-potential normal derivative is the
impermeable mass-flux condition for constant positive M. Together they supply
the two local fourth-order boundary conditions. They must hold on all outer
and inner boundaries. Neutral wetting has a90-degree contact angle in the
smooth resolved-interface setting, unlike general nonzero wall energy.

Proposed discrete structural formulation, not a chosen kernel/consistency proof:

$$
(A v)_i=\frac1{m_i}\sum_j a_{ij}(v_i-v_j),\quad
a_{ij}=a_{ji}\geq0,\quad m_i>0,
$$

$$
\dot\phi=-M A\mu,\qquad
\mu=\beta(\phi^3-\phi)+\epsilon^2 A\phi.
$$

Let V=diag(m_i) and L=VA be the symmetric pair Laplacian. Then A1=0,
1^T VA=0 and v^T VA v=(1/2)sum_ij a_ij(v_i-v_j)^2>=0, by interchanging
pair indices. Consequently the exact semidiscrete dynamics conserves
sum_i m_i phi_i. Its energy sum_i m_i F(phi_i)+(epsilon^2/2)phi^T L phi has
derivative -M mu^T L mu<=0, because its phi gradient is V mu.
These independently written algebraic identities establish graph closure and
energy structure, not continuum convergence or exact neural-PINN conservation.
Finite residual error may still change predicted mass; any hard mass projection
must be shared by both compared arms, without duplicate soft mass loss.

Use a justified regional finite-horizon/variational construction without
external bonds, ghost values or separate natural-boundary losses for the native
nonlocal problem. At finite delta this does not imply the network's AD normal
derivative equals zero pointwise on the classical boundary. Need h/delta and
horizon refinement with correct kernel moments and boundary quadrature.
Do not import single-nonlocal CH convergence theorems as a proof for replacing
both Laplacians by any arbitrary graph. Pure row normalization may break the
physical-volume pair symmetry; a reversible stationary graph measure need not
be physical volume.

Geometry caution: endpoint-inside tests can connect pairs across an insulating
hole in a nonconvex domain. Choose horizon well below obstruction/feature
lengths or derive a symmetric visibility/support rule and revalidate consistency;
do not silently add a new constitutive interaction rule. Existing DenseD
unstructured fallback uses bounding-box volume/N, inappropriate as a general
perforated-domain area quadrature. Reassess explicit masses before reusing it.
Do not copy the periodic twophase/precipitate FFT stencil to this geometry.

## Experimental route if selected

1. Circle sanity check, then ellipse+twoholes. Reference preflight must show
   phase-wall interaction rather than all motion remote from boundaries.
2. Independent mixed FEM reference, mesh/time refinement and consistent initial
   field. A native graph reference separately measures finite-horizon/boundary
   bias; no FEM labels enter training. Reference choice/kernels remain unchosen.
3. Local AD mixed phi/mu with both Neumann losses, versus nonlocal native natural
   closure without these losses. Same architecture/init/interior data/IC/mass,
   physical parameters and budget. Boundary treatment is an explicit experimental
   difference, not a purported identical-continuum formulation at finite delta.
   A weak local PINN is a possible later control, since FEM/weak local methods
   also naturally handle homogeneous Neumann; advantage is not unique to nonlocal.
4. Primary shared-reference pooled/terminal phi and mu relative L2, boundary-band
   error, interface/contact location, physical weighted mass and energy. Record
   boundary sample/weight cost and wall time. AD normal derivatives of native
   solutions diagnose the local limit rather than requiring them identically0.
5. Nonlocal horizon/grid/reference refinement precedes PINN superiority claims.
   No accuracy improvement or universal boundary-free Neumann handling promised;
   the proposed advantage targets homogeneous no-flux, not prescribed nonzero flux.

Alternatives: smooth dumbbell container for neck-controlled coarsening (more
nonconvex support risks); Neumann Allen-Cahn domain relaxation if a cheaper
single-field baseline is desired (no CH mass-conservation diagnostic). First
recommendation remains confined CH, not full hydrodynamics or elastic coupling.

## Primary sources checked

- Hofmann, Mueller, Andrae2017, A fast immersed interface method for the
  Cahn-Hilliard equation with arbitrary boundary conditions in complex domains:
  https://doi.org/10.1016/j.commatsci.2017.08.025 . Publisher abstract describes
  mixed second-order form, Neumann/wetting and3D porous-domain demonstration.
  Full article fetch returned403; no unprovided full-method detail adopted.
- Abels, Hurm2024, Strong nonlocal-to-local convergence of the Cahn-Hilliard
  equation and its operator: https://epub.uni-regensburg.de/58342/ and
  https://arxiv.org/abs/2307.02264 . Smooth bounded Neumann domains and W1,1
  kernel; does not validate arbitrary finite graph or doubly replaced CH.
- Davoli, Scarpa, Trussardi2020, Neumann boundary conditions and viscosity terms:
  https://doi.org/10.1007/s00205-020-01573-9 . Nonlocal phase-energy operator but
  local chemical-potential diffusion with its own Neumann condition; distinguish
  emergence of phase Neumann in local limit from imposing all native conditions.
- On nonlocal problems with Neumann boundary conditions: scaling and convergence
  for nonlocal operators and solutions2025:
  https://doi.org/10.1186/s13662-025-03923-x . Discusses regional restricted-domain
  diffusion without a separate boundary equation and alternative Neumann models.
- Gal2018, Doubly nonlocal Cahn-Hilliard equations:
  https://archive.numdam.org/articles/10.1016/j.anihpc.2017.05.001/ . Confirms
  doubly nonlocal family; its strong-to-weak assumptions are not asserted for
  the unchosen finite-horizon numerical proposal.

Assessment-only artifacts: [proposed geometry](phasefield-neumann-geometry.png),
[plot source](phasefield-neumann-geometry.py). No solved physical field is shown.
No production code/config/README/API change or new training/test suite needed.
Memory changes: PROGRESS.md, DECISIONS.md, this archived task. Closure uses
local-link checks and git diff --check only; old artifacts/defaults stay intact.
