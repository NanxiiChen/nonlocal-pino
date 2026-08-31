# Fracture2D Direct Nonlocal Baseline

## Goal

Implement and validate a direct pure-nonlocal numerical baseline for the
rounded-notch fracture2d problem before introducing a PINN. The local FEM
remains the independent reference; the direct solver and later PINN will share
the validated nonlocal discretization so that representation/optimization error
can be measured separately from model/discretization error.

## Acceptance criteria

- Freeze a traceable discrete nonlocal model for quadrature, scalar damage,
  damage-aware correspondence mechanics, boundary treatment, and reaction.
- Validate a weighted 2D scalar operator on full-square and rounded-notch
  particle sets before coupling fracture.
- Validate intact and damaged correspondence mechanics through patch, energy,
  conservation, conditioning, and null-mode checks.
- Solve fixed-history damage and fixed-damage mechanics independently with
  explicit residual and energy/work diagnostics.
- Run a strict staggered continuation with rollback/cutback before considering
  any warning-mode curve diagnostic.
- Archive the same load, field, energy, reaction, and convergence quantities
  needed for comparison with the local FEM working reference.
- Complete fixed-horizon grid refinement and a separate local-limit sequence
  before starting the nonlocal PINN.

## Frozen context

- Geometry and material follow `examples/fracture2d/configs/solver_base.py`:
  unit square, rounded slot of width `2*l0`, tip radius `l0`, foremost tip
  `(0.5,0.5)`, plane strain, `E=210e3`, `nu=0.3`, `Gc=2.7`, `l0=0.024`, and
  `k_res=1e-8`.
- Boundary intent follows the local reference and is now frozen as
  boundary-particle hard constraints: bottom boundary representative particles
  have `uy=0`, top representatives have `uy=u_bar`, one lower-left particle has
  `ux=0`, and all other top/bottom `ux` DOFs are free.
- Use one common first-study horizon for mechanics and damage, but do not select
  its numerical value without a documented ratio/refinement argument.
- The local point-anchor FEM archive is a warning-mode working reference, not a
  strictly converged target to fit.
- No PINN, core+halo sampling, adaptive point refinement, or parameterized
  surrogate belongs in this task.
- The selected architecture is stable energy-consistent correspondence
  mechanics, weighted `SparseD` nonlocal AT2 damage, Amor positive-energy
  maximum history, history-field irreversibility, staggered coupling, and
  sparse or matrix-free linear fixed-subproblem solves. The later PINN will
  reuse the same strong-form nonlocal residuals.

## Blocking model decisions

Implementation must stop at the relevant stage until each item is supported by
a reference or independently checked derivation:

1. **Particle quadrature:** point layout and per-particle volumes on the rounded
   slot domain; verify total area and moments. Existing `SparseD` has only one
   scalar volume and is insufficient for a nonuniform/slot-removed cloud.
2. **Notch interactions:** whether the family is only the material-domain
   intersection or also removes bonds whose segments cross the slot/void.
3. **Damage-aware mechanics:** define a single discrete correspondence energy,
   its degradation/history placement, and the force obtained from that energy.
4. **Correspondence stability:** define and validate any stabilization; do not
   add an ad-hoc hourglass penalty after observing a bad fracture curve.
5. **Boundary-particle hard constraints:** define and validate exact physical
   boundary masks, the lower-left anchor, DOF elimination, affine lifting, and
   the top/bottom/anchor reaction diagnostics. This is a material-only,
   project-owned discrete BVP, not a volume collar.
6. **Reaction:** define the work-conjugate discrete reaction from constrained
   forces and verify its sign, weighting, and energy derivative.
7. **History discretization:** define particle/cell history and any projection;
   do not copy the 1D DG0 averaging rule without a 2D derivation.

## Execution plan

### Stage 0: Discrete model specification

Write the equations and data layout before production code:

- particle coordinates, volumes, active material mask, and deterministic
  boundary-particle masks;
- kernel, normalization, horizon, neighbor and visibility rules;
- weighted scalar operator and discrete damage energy;
- nonlocal deformation gradient, plane-strain constitutive response, degraded
  energy, force state, history, and stabilization if required;
- fixed-history and fixed-damage algebraic systems;
- constrained reaction and energy/work balance;
- fixed-`delta` refinement and local-limit sequences.

Deliverable: a reviewed formulation section in the example README or task file,
with references/derivation and no unresolved symbols.

## Stage 0 Draft Formulation (Pending Sol Review)

This section is the complete project-owned derivation proposed for independent
review. It is deliberately an energy-first specification: the mechanics force,
tangent, and reaction below are definitions obtained by differentiating the
same discrete energy. The bond-associated kinematics use the target-inclusive
finite-volume interpretation of Chen and Spencer (2019), Section 3,
Eqs. (16)-(23), while the normalized center-energy partition and its diffuse
AT2 coupling are project-owned choices. They must not be presented as equations
printed in that paper. The boundary realization remains a separate model
decision and is not selected by this draft.

### 0.1 Discrete domain, volumes, and graph

Let the material particle set be

$$
\mathcal P=\{1,\ldots,N\},
\qquad
\boldsymbol X_i\in\Omega_m\subset\mathbb R^2,
\qquad
V_i>0,
$$

with unit out-of-plane thickness. For a Gmsh triangular mesh, the first
candidate quadrature is barycentric nodal lumping:

$$
V_i=\frac13\sum_{e\ni i}A_e.
$$

The geometry checks must establish

$$
\sum_iV_i=|\Omega_m|=0.9762509375,
$$

up to the declared floating-point tolerance, and must check the first moments
against the measured rounded-slot geometry. The volume is a quadrature weight;
it is not a uniform replacement computed from the bounding box.

For the first study, mechanics and damage use one common horizon

$$
\delta_m=\delta_d=\delta.
$$

For distinct reference points define

$$
\boldsymbol\xi_{ij}=\boldsymbol X_j-\boldsymbol X_i,
\qquad
r_{ij}=|\boldsymbol\xi_{ij}|.
$$

The reciprocal material graph is

$$
\mathcal E=\{(i,j):i\ne j,\ r_{ij}<\delta,\ \chi_{ij}=1\},
\qquad
\chi_{ij}=\chi_{ji}.
$$

The first visibility rule is segment visibility through the material domain:

$$
\chi_{ij}=1
\quad\Longleftrightarrow\quad
[\boldsymbol X_i,\boldsymbol X_j]
\cap\operatorname{int}(\Omega_{\mathrm{void}})=\varnothing.
$$

Thus, a bond whose segment crosses the finite-width slot is removed, while a
segment contained in the material closure is retained. The graph is fixed from
the reference geometry; damage does not delete graph edges. This symmetric rule
is a model choice for the rounded slot and must be compared with the material-
intersection-only alternative in a later visibility sensitivity study.

Use the existing two-dimensional cubic kernel for both first-study operators:

$$
J_\delta(r)=
\begin{cases}
\displaystyle\frac{120}{\pi\delta^4}\left(1-\frac r\delta\right)^2,
&0<r<\delta,\\[6pt]
0,&r\ge\delta.
\end{cases}
$$

Set \(\omega_{ij}=\chi_{ij}J_\delta(r_{ij})\). In the full, untruncated
two-dimensional continuum neighborhood this kernel satisfies

$$
\int J_\delta(|\boldsymbol\xi|)
\boldsymbol\xi\otimes\boldsymbol\xi\,d\boldsymbol\xi
=2\boldsymbol I_2,
$$

so the difference operator defined below has interior limit
\(D_\delta d\to-\Delta d\). This moment statement does not hold pointwise
near the external boundary or slot; those regions require the explicit finite-
horizon validation rather than a local-PDE assumption.

Define the center normalization

$$
m_i=\sum_{j:(i,j)\in\mathcal E}\omega_{ij}V_j,
\qquad
\eta_{ij}=\frac{\omega_{ij}V_j}{m_i}.
$$

Every active center must have \(m_i>0\), and

$$
\sum_{j:(i,j)\in\mathcal E}\eta_{ij}=1.
$$

All graph and quadrature construction must fail loudly if a center has no
family or if a required bond-associated shape tensor below is rank deficient.

### 0.2 Boundary-particle hard constraints (frozen project-owned BVP)

The selected first-study BVP is material-only. No exterior fictitious particles
are added, and no horizon-thick interior material strip is constrained. The
physical material graph \(\mathcal E\), including its rounded-slot visibility
rule, is unchanged. Boundary representative particles remain graph nodes, BA
support points, and energy centers. The rounded slot and free side surfaces
therefore retain the natural graph-truncated interaction treatment.

Let \(\mathcal B_B\) and \(\mathcal B_T\) denote the bottom and top physical
boundary representative particle masks. In the production Gmsh path these masks
must be carried from the explicit Gmsh physical boundary tags before any mesh
data are discarded. If a nodal cloud does not retain tags, Stage 1 must
construct and retain deterministic masks from the mesh boundary entities and
geometry as part of the cloud data layout. A hidden coordinate tolerance or an
unstated nearest-boundary rule is not allowed. The regular-cloud check below
uses exact predicates \(X_{i,y}=y_{\min}\) and \(X_{i,y}=y_{\max}\) only because
that manufactured coordinate representation is exact.

The lower-left anchor is the unique tagged/constructed physical boundary
particle \(i_A\) at \((0,0)\). With DOF ordering
\(q_{2i}=u_{i,x}\), \(q_{2i+1}=u_{i,y}\), impose

$$
u_{i,y}=0\quad(i\in\mathcal B_B),
\qquad
u_{i,y}=\bar u\quad(i\in\mathcal B_T),
\qquad
u_{i_A,x}=0.
$$

All other top and bottom horizontal DOFs remain free. Define

$$
\mathcal D_c
=\{2i+1:i\in\mathcal B_B\cup\mathcal B_T\}
\cup\{2i_A\},
\qquad
\mathcal D_f=\{0,\ldots,2N-1\}\setminus\mathcal D_c.
$$

Let \(\boldsymbol P_f\) be the column-selector matrix whose columns are the
standard basis vectors indexed by \(\mathcal D_f\). Let \(\boldsymbol q_0\)
contain the fixed zero values on bottom vertical DOFs and the anchor, and let
\(\boldsymbol a_T\) be one on the top vertical DOFs and zero elsewhere. The
affine lifting is

$$
\boldsymbol q
=\boldsymbol P_f\boldsymbol q_f
+\boldsymbol q_0
+\boldsymbol a_T\bar u.
$$

This is direct DOF elimination/affine lifting, not a penalty or an additional
boundary energy. At fixed damage, free equilibrium is

$$
\boldsymbol P_f^T\boldsymbol r(\boldsymbol q)=0,
\qquad
\boldsymbol K_{ff}=\boldsymbol P_f^T\boldsymbol K\boldsymbol P_f,
$$

and the linear fixed-damage solve used by the algebra check is

$$
\boldsymbol K_{ff}\boldsymbol q_f
=-\boldsymbol P_f^T\boldsymbol K
\left(\boldsymbol q_0+\boldsymbol a_T\bar u\right).
$$

Damage receives no hard boundary condition in this BVP. Its graph-truncated
AT2 operator remains the declared natural graph treatment.

The top reaction is the constrained residual sum and the reduced-energy
derivative:

$$
R_T
=\boldsymbol a_T^T\boldsymbol r
=\sum_{i\in\mathcal B_T}r_{i,y}
=\frac{d\Pi_{\mathrm m,h}^{\mathrm{red}}}{d\bar u}.
$$

The bottom and anchor reactions are retained as separate diagnostics:

$$
R_B=\sum_{i\in\mathcal B_B}r_{i,y},
\qquad
R_A=r_{i_A,x}.
$$

The reported sign is the force required in the positive direction of the
corresponding prescribed displacement parameter; the actuator reaction has the
opposite sign. The fixed-damage reduced-energy finite difference and the free
residual checks are mandatory before production implementation.

**Historical alternatives and rejection.** The exterior fictitious interaction
collar remains documented for provenance, but is not adopted. Behzadinasab,
Foster, and Bazilevs Part II, DOI \(10.1007/s42102-020-00039-6\), Section 4 and
Eqs. (8)-(9), assumes an exterior horizon-thick layer with separate kinematic
and stress-divergence neighborhoods and essential/natural data at boundary
points; that is a different BVP and is not relabeled as this hard constraint.
The interior material grip is also rejected because it constrains a material
volume and changes the deformable BVP. The selected boundary-particle rule is
project-owned and must not be described as a continuum nonlocal volume
constraint or as Behzadinasab's exterior layer.

### 0.3 Bond-associated kinematics

For every directed graph edge \((i,j)\), let
\(\mathcal H_i=\{k:(i,k)\in\mathcal E\}\). Chen and Spencer (2019),
Eqs. (16)-(17), define the bond-associated integration domain from the
intersection of the family at the source point with a bond-associated
neighborhood at the target point. With the first-study choice that the
bond-associated radius is \(\delta\), the target-inclusive finite-volume
discretization is

$$
\mathcal C_{i|j}
=\{j\}\cup
\{k\ne i,j:(i,k)\in\mathcal E\ \text{and}\ (j,k)\in\mathcal E\}.
$$

The explicit singleton \(\{j\}\) is the target endpoint quadrature
contribution. It is not a self-edge and must not be added to \(\mathcal E\).
The old intersection omitted it because \((j,j)\notin\mathcal E\), so a
displacement supported only at \(j\) could not change \(\boldsymbol F_{i|j}\).
For a uniform notation define

$$
\beta_{i|j,k}=
\begin{cases}
\omega_{ij}V_j,&k=j,\\
\omega_{ik}V_k,&k\in\mathcal C_{i|j}\setminus\{j\}.
\end{cases}
$$

Thus the target term is explicitly \(\omega_{ij}V_j\), while all other terms
use the source-centered weight \(\omega_{ik}V_k\).

With the displayed two-dimensional kernel, \([J_\delta]=L^{-4}\),
\([\beta]=L^{-2}\), and the products \(\beta\,\boldsymbol\xi\otimes
\boldsymbol\xi\) and \(\beta\,\boldsymbol Y\otimes\boldsymbol\xi\) are
dimensionless, as required for \(\boldsymbol K\), \(\boldsymbol A\), and
\(\boldsymbol F\).

For \(k\in\mathcal C_{i|j}\), let

$$
\boldsymbol\xi_{ik}=\boldsymbol X_k-\boldsymbol X_i,
\qquad
\boldsymbol Y_{ik}=\boldsymbol\xi_{ik}+\boldsymbol u_k-\boldsymbol u_i.
$$

The fixed reference bond-associated shape tensor and current moment are

$$
\begin{aligned}
\boldsymbol K_{i|j}
&=\sum_{k\in\mathcal C_{i|j}}
\beta_{i|j,k}\,
\boldsymbol\xi_{ik}\otimes\boldsymbol\xi_{ik}
\\
&=\omega_{ij}V_j\,\boldsymbol\xi_{ij}\otimes\boldsymbol\xi_{ij}
 +\sum_{k\in\mathcal C_{i|j}\setminus\{j\}}
 \omega_{ik}V_k\,\boldsymbol\xi_{ik}\otimes\boldsymbol\xi_{ik},
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol A_{i|j}(\boldsymbol u)
&=\sum_{k\in\mathcal C_{i|j}}
\beta_{i|j,k}\,
\boldsymbol Y_{ik}\otimes\boldsymbol\xi_{ik}
\\
&=\omega_{ij}V_j\,\boldsymbol Y_{ij}\otimes\boldsymbol\xi_{ij}
 +\sum_{k\in\mathcal C_{i|j}\setminus\{j\}}
 \omega_{ik}V_k\,\boldsymbol Y_{ik}\otimes\boldsymbol\xi_{ik}.
\end{aligned}
$$

The bond-associated deformation gradient is

$$
\boldsymbol F_{i|j}(\boldsymbol u)
=\boldsymbol A_{i|j}(\boldsymbol u)
\boldsymbol K_{i|j}^{-1}.
$$

The inverse is computed only after an explicit rank and conditioning check;
there is no pseudo-inverse, determinant clipping, or identity fallback. At the
reference state \(\boldsymbol Y_{ik}=\boldsymbol\xi_{ik}\), so
\(\boldsymbol F_{i|j}=\boldsymbol I_2\). For an affine deformation
\(\boldsymbol y=\boldsymbol F\boldsymbol X+\boldsymbol c\),

$$
\boldsymbol A_{i|j}=\boldsymbol F\boldsymbol K_{i|j},
\qquad
\boldsymbol F_{i|j}=\boldsymbol F.
$$

This is the affine patch property. The target-inclusive common support is the
Chen-Spencer-inspired stability mechanism: each associated gradient uses a
bond-local family and directly sees the target bond rather than using one
single center gradient with the endpoint omitted. The absence of additional
null modes on the actual finite graph remains a required spectral test, not an
assumption.

### 0.4 Plane-strain constitutive energy

Let

$$
\boldsymbol G_{i|j}=\boldsymbol F_{i|j}-\boldsymbol I_2,
\qquad
\boldsymbol\varepsilon_{i|j}=\operatorname{sym}(\boldsymbol G_{i|j}).
$$

Embed the two-dimensional strain into plane strain:

$$
\boldsymbol\varepsilon^{(3)}_{i|j}
=
\begin{bmatrix}
\varepsilon_{xx}&\varepsilon_{xy}&0\\
\varepsilon_{xy}&\varepsilon_{yy}&0\\
0&0&0
\end{bmatrix}_{i|j}.
$$

Use

$$
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},
\qquad
\mu=\frac{E}{2(1+\nu)}.
$$

The full plane-strain elastic energy and in-plane work-conjugate stress are

$$
\psi_{i|j}
=\frac12\lambda
\left(\operatorname{tr}\boldsymbol\varepsilon^{(3)}_{i|j}\right)^2
+\mu\boldsymbol\varepsilon^{(3)}_{i|j}:
\boldsymbol\varepsilon^{(3)}_{i|j},
$$

$$
\boldsymbol P_{i|j}
=\lambda\operatorname{tr}(\boldsymbol\varepsilon^{(3)}_{i|j})\boldsymbol I_2
+2\mu\boldsymbol\varepsilon_{i|j}.
$$

For this small-strain model, \(\boldsymbol P\) is the derivative of \(\psi\)
with respect to the in-plane deformation-gradient entries. The associated
fourth-order tangent is the plane-strain map

$$
\mathbb C_{\mathrm{pe}}[\boldsymbol q]
=\lambda\operatorname{tr}(\operatorname{sym}\boldsymbol q^{(3)})\boldsymbol I_2
+2\mu\operatorname{sym}\boldsymbol q,
$$

where \(\boldsymbol q^{(3)}\) is the three-dimensional embedding of the
in-plane map \(\boldsymbol q\) with zero out-of-plane components. The tangent
therefore annihilates an infinitesimal skew perturbation, as required by the
small-strain energy.

The project-owned bond-associated center energy is

$$
\Psi_i(\boldsymbol u)
=\sum_{j:(i,j)\in\mathcal E}
\eta_{ij}\psi_{i|j}(\boldsymbol u).
$$

The mechanics degradation is

$$
g(d_i)=(1-d_i)^2+k_{\mathrm{res}},
\qquad k_{\mathrm{res}}>0,
$$

and the complete discrete mechanics energy is

$$
\Pi_{\mathrm m,h}(\boldsymbol u,\boldsymbol d)
=\sum_{i=1}^N V_i g(d_i)\Psi_i(\boldsymbol u).
$$

The degradation is applied to the complete center energy, including every
bond-associated contribution in \(\Psi_i\). No separate stabilization force is
added, and no damage average \(g(d_i,d_j)\) is inserted into a force formula.
This placement is what makes the following force the derivative of one declared
energy. The normalized \(\eta_{ij}\) preserves the local energy scale for every
affine deformation, including truncated families.

This is not the Bie PD-PFM fracture model. Bie's cohesive crack density,
rational degradation, tensile threshold, and plane-stress coefficients are not
used here. Only the general energy-first/stable-correspondence motivation is
shared.

### 0.5 Mechanics force and tangent

For a nodal perturbation \(\boldsymbol v_n\in\mathbb R^2\), define the linear
map

where \(\delta_{kn}\) below denotes the Kronecker delta, not the horizon.

$$
\mathcal B_{i|j,n}[\boldsymbol v_n]
=\left[
\sum_{k\in\mathcal C_{i|j}}
\beta_{i|j,k}(\delta_{kn}-\delta_{in})
\boldsymbol v_n\otimes\boldsymbol\xi_{ik}
\right]\boldsymbol K_{i|j}^{-1}.
$$

Then

$$
\delta\boldsymbol F_{i|j}
=\sum_n\mathcal B_{i|j,n}[\delta\boldsymbol u_n].
$$

Using \(\boldsymbol P_{i|j}=\partial\psi_{i|j}/\partial\boldsymbol F_{i|j}\),
the unconstrained mechanics residual is the energy gradient

$$
\boldsymbol r_n
=\frac{\partial\Pi_{\mathrm m,h}}{\partial\boldsymbol u_n}
=\sum_{(i,j)\in\mathcal E}V_i g(d_i)\eta_{ij}
\sum_{k\in\mathcal C_{i|j}}
\beta_{i|j,k}(\delta_{kn}-\delta_{in})
\boldsymbol P_{i|j}\boldsymbol K_{i|j}^{-T}
\boldsymbol\xi_{ik}.
$$

The direct internal force, if reported with the usual peridynamic sign, is
\(\boldsymbol f_{\mathrm{int},n}=-\boldsymbol r_n\). Equilibrium is
\(\boldsymbol r_n=0\) on free displacement degrees of freedom. The production
implementation must assemble \(\boldsymbol r\) from this derivative, not from
the existing conventional `ElasticD.elastic_matvec` formula.

At fixed damage, the tangent is the Hessian

$$
\boldsymbol K_{mn}
=\frac{\partial\boldsymbol r_m}{\partial\boldsymbol u_n}
=\sum_{(i,j)\in\mathcal E}V_i g(d_i)\eta_{ij}
\mathcal B_{i|j,m}^{*}
\mathbb C_{\mathrm{pe}}
\mathcal B_{i|j,n}.
$$

Here \(\mathcal B^*\) is the Euclidean adjoint of the displayed linear map.
Consequently,

$$
\boldsymbol K_{mn}=\boldsymbol K_{nm}^{T},
$$

and for any displacement perturbation \(\delta\boldsymbol u\),

$$
\delta\boldsymbol u^T\boldsymbol K\delta\boldsymbol u
=\sum_{(i,j)\in\mathcal E}V_i g(d_i)\eta_{ij}
\left(\delta\boldsymbol F_{i|j}\right)
:
\mathbb C_{\mathrm{pe}}
\left(\delta\boldsymbol F_{i|j}\right)\ge0.
$$

The non-negativity follows from \(g(d_i)>0\) and the positive semidefinite
linear elastic tangent. It does not by itself prove that the only null modes of
the finite particle matrix are translations and infinitesimal rotations; that
requires the stated rank, patch, and spectrum tests on the actual cloud,
boundary-particle masks, notch visibility, and damaged weights.

The mechanics/damage cross derivative of the declared mechanics energy is most
compactly written using the mechanics residual contribution of center \(\ell\):

$$
\boldsymbol r_{\ell,n}^{\mathrm m}
=V_\ell\sum_{j:(\ell,j)\in\mathcal E}\eta_{\ell j}
\frac{\partial\psi_{\ell|j}}
{\partial\boldsymbol u_n}.
$$

Then

$$
\frac{\partial^2\Pi_{\mathrm m,h}}
{\partial\boldsymbol u_n\partial d_\ell}
=g'(d_\ell)\boldsymbol r_{\ell,n}^{\mathrm m}.
$$

It is not needed by the selected staggered fixed-damage mechanics solve. This is
only the mechanics-energy `K_ud` block: the selected history-field system is not
one joint potential, so its future monolithic Jacobian is generally
non-symmetric and this block does not imply a transpose `K_du` block.

### 0.6 Amor history and nonlocal AT2 damage

For the same bond-associated strain, define the plane-strain Amor positive
energy

$$
\psi^+_{i|j}
=\frac12K
\left\langle\operatorname{tr}\boldsymbol\varepsilon^{(3)}_{i|j}\right\rangle_+^2
+\mu\boldsymbol\varepsilon_{\mathrm{dev},i|j}^{(3)}:
\boldsymbol\varepsilon_{\mathrm{dev},i|j}^{(3)},
$$

where

$$
K=\lambda+\frac{2\mu}{3},
\qquad
\boldsymbol\varepsilon_{\mathrm{dev}}^{(3)}
=\boldsymbol\varepsilon^{(3)}
-\frac13\operatorname{tr}(\boldsymbol\varepsilon^{(3)})\boldsymbol I_3.
$$

The particle history driver is averaged with the same normalized bond weights:

$$
\Psi_i^+(\boldsymbol u)
=\sum_{j:(i,j)\in\mathcal E}\eta_{ij}\psi^+_{i|j}(\boldsymbol u),
$$

and the committed history update is

$$
H_i^n=\max\left(H_i^{n-1},\Psi_i^+(\boldsymbol u^n)\right).
$$

During a trial load step, the damage subproblem uses the trial history

$$
H_i^{\mathrm{trial},k}
=\max\left(H_i^{n-1},\Psi_i^+(\boldsymbol u^{\mathrm{trial},k})\right),
$$

where \(k\) is the staggered iteration. Only a converged trial replaces
\(H^{n-1}\) by \(H^n\); a cutback restores the immutable committed field.

This is a particle history, not a copied DG0 projection. The history is updated
from the current trial mechanics state but committed only after an accepted
load step, and is restored on a failed load-step trial.

For a fixed committed history \(\boldsymbol H\), define the AT2 history
surrogate

$$
\Pi_{\mathrm d,h}(\boldsymbol d;\boldsymbol H)
=\sum_iV_i\left[
\frac{G_c}{2\ell_0}d_i^2
+H_i(1-d_i)^2
\right]
+\frac{G_c\ell_0}{4}
\sum_{(i,j)\in\mathcal E}
V_iV_jJ_\delta(r_{ij})(d_j-d_i)^2.
$$

Define the weighted graph operator

$$
(D_\delta d)_i
=\sum_{j:(i,j)\in\mathcal E}
J_\delta(r_{ij})V_j(d_i-d_j).
$$

Differentiating the damage surrogate and dividing by \(V_i\) gives

$$
\frac1{V_i}\frac{\partial\Pi_{\mathrm d,h}}
{\partial d_i}
=\frac{G_c}{\ell_0}d_i
+G_c\ell_0(D_\delta d)_i
+2H_id_i-2H_i.
$$

The fixed-history damage equation is therefore

$$
\frac{G_c}{\ell_0}d_i
+G_c\ell_0(D_\delta d)_i
+2H_id_i=2H_i.
$$

The directed matrix of \(D_\delta\) is self-adjoint in the volume inner
product. With \(\boldsymbol M=\operatorname{diag}(V_i)\), its off-diagonal
entries satisfy

$$
V_i(D_\delta)_{ij}=V_j(D_\delta)_{ji}
=-V_iV_jJ_\delta(r_{ij}),
$$

so \(\boldsymbol M D_\delta=D_\delta^T\boldsymbol M\). Equivalently, the
assembled Hessian \(\boldsymbol M D_\delta\) is symmetric and

$$
\boldsymbol d^T\operatorname{diag}(V_i)D_\delta\boldsymbol d
=\frac12\sum_{(i,j)\in\mathcal E}
V_iV_jJ_\delta(r_{ij})(d_i-d_j)^2\ge0.
$$

The strong-form fixed-history matrix is self-adjoint and positive definite in
the volume inner product because \(G_c/\ell_0+2H_i>0\); it is not generally
Euclidean-symmetric for nonuniform \(V_i\). The volume-multiplied matrix is
Euclidean SPD and may be solved directly, or the equivalent volume-weighted
operator may be used matrix-free. No clipping, pseudo-inverse, or hidden damage
bound is part of this formulation. Damage range and healing are hard
diagnostics at acceptance; a violation is reported rather than silently
modified.

The history-field formulation is staggered rather than the first variation of
the sum \(\Pi_{\mathrm m,h}+\Pi_{\mathrm d,h}\): \(H\) is an internal committed
history, and the mechanics equation uses the full degraded energy while the
damage equation uses the fixed-history surrogate. This distinction is required
for a correct later strong-form PINN residual.

### 0.7 Work-conjugate reaction

Let \(\boldsymbol q\) collect all displacement degrees of freedom and write the
prescribed displacement parameterization as

$$
\boldsymbol q=\boldsymbol P_f\boldsymbol q_f
+\boldsymbol q_0
+\boldsymbol a_T\bar u,
$$

where \(\boldsymbol a_T\) is one on the vertical DOFs of
\(\mathcal B_T\) and zero elsewhere, and \(\boldsymbol P_f\) is the selector
for all DOFs not in \(\mathcal D_c\) from Section 0.2. At fixed damage, free
equilibrium is

$$
\boldsymbol P_f^T\nabla_{\boldsymbol q}\Pi_{\mathrm m,h}=0.
$$

If \(\boldsymbol q_f^*(\bar u,\boldsymbol d)\) denotes an accepted solution of
this constrained mechanics problem, define

$$
\Pi_{\mathrm m,h}^{\mathrm{red}}(\bar u,\boldsymbol d)
=\Pi_{\mathrm m,h}\left(
\boldsymbol P_f\boldsymbol q_f^*(\bar u,\boldsymbol d)
+\boldsymbol q_0+\boldsymbol a_T\bar u,
\boldsymbol d\right).
$$

The reported top loading reaction is the work-conjugate derivative

$$
R_T
=\frac{d\Pi_{\mathrm m,h}^{\mathrm{red}}}{d\bar u}
=\boldsymbol a_T^T\nabla_{\boldsymbol q}\Pi_{\mathrm m,h}.
$$

This convention reports the positive force required to impose increasing
\(\bar u\). The reaction on the actuator has the opposite sign. The definition
must be checked by a centered finite difference of the reduced mechanics energy:

$$
R_T\approx
\frac{\Pi_{\mathrm m,h}^{\mathrm{red}}(\bar u+\epsilon)
-\Pi_{\mathrm m,h}^{\mathrm{red}}(\bar u-\epsilon)}{2\epsilon}.
$$

The damage surrogate is not included in this mechanics reaction derivative.
Energy/work checks for the complete staggered path must separately report
mechanics energy, fracture energy, external work, history updates, and the fact
that the history surrogate is not a jointly conservative potential.

### 0.8 Independent checks before production code

The following checks are required before Stage 1-4 production coupling:

1. **Quadrature:** positive volumes, total material area, first moments, and
   deterministic ordering.
2. **Graph:** reciprocal edges, symmetric visibility, positive family mass, and
   no bond crossing the slot under the selected visibility rule.
3. **BA support:** target-inclusive \(\mathcal C_{i|j}\), rank-two
   \(\boldsymbol K_{i|j}\), explicit condition numbers, nonzero target-bond
   sensitivity, and loud failure for insufficient supports.
4. **Affine patch:** every valid \(\boldsymbol F_{i|j}\) reproduces a prescribed
   affine map exactly within tolerance.
5. **Rigid modes:** translations and infinitesimal rotations have zero mechanics
   energy and residual; unconstrained spectra report exactly the three rigid
   modes and the first non-rigid eigenvalue, rather than only a minimum.
6. **Energy derivative:** for random perturbations \(\boldsymbol v\), assemble
   the mechanics residual independently from each BA term using current
   \(\boldsymbol F\), strain, \(\boldsymbol P\), \(\beta\), and explicit
   source/target endpoint signs, then compare
   \(\boldsymbol r_{\mathrm{exp}}\cdot\boldsymbol v\) with

   $$
    \frac{\Pi_{\mathrm m,h}(\boldsymbol u+\epsilon\boldsymbol v,\boldsymbol d)
   -\Pi_{\mathrm m,h}(\boldsymbol u-\epsilon\boldsymbol v,\boldsymbol d)}
   {2\epsilon}.
   $$

7. **Tangent:** verify
   \([\boldsymbol r_{\mathrm{exp}}(\boldsymbol u+\epsilon\boldsymbol v)-
   \boldsymbol r_{\mathrm{exp}}(\boldsymbol u-\epsilon\boldsymbol v)]/(2\epsilon)\)
   against \(\boldsymbol K\boldsymbol v\), independently verify
   \(\boldsymbol r_{\mathrm{exp}}(\boldsymbol u)=\boldsymbol K\boldsymbol u\),
   and check block symmetry and non-negative quadratic forms before damage
   localization.
8. **Damage:** verify constants, non-negative graph energy, volume-weighted
   symmetry, fixed-history residual, and positive definiteness without clipping.
9. **Reaction:** compare \(R_T\) with the reduced-energy finite difference and
   verify constrained residual signs.
10. **Damage coupling:** repeat force, tangent, and spectrum checks for constant
    and spatially varying damage, including the `k_res` conditioning regime.
11. **History:** verify that only the positive volumetric Amor term is truncated,
    while the deviatoric term remains active under shear and general in-plane
    compression; verify committed max update and exact rollback.
12. **Boundary/notch:** verify exact boundary-particle masks, hard-constraint
     DOF elimination, free-block SPD, top/bottom/anchor reactions, and reduced-
     energy work; retain rounded-slot/free-side natural graph truncation without
     silently changing the production graph.

The checks establish energy consistency and conditional stability of the
project-owned discretization. They do not convert the novel combination of
bond-associated mechanics and history-field AT2 into a published model; that
scope distinction remains explicit for Sol's review.

### 0.9 Corrected float64 algebra check of the derivation

The persistent check `tests/test_fracture2d_bond_associated_algebra.py` implements only the
displayed Stage 0 algebra; it is not a production solver. It uses a regular
`5 x 5` cloud in index coordinates, unit particle volumes, `delta=2.01`, the
cubic kernel, target-inclusive support, plane-strain `E=1`, `nu=0.3`,
`k_res=1e-8`, and deterministic random damage/displacement with seed `1234`.
Its mechanics energy evaluator and Hessian use the existing linear-map path,
while `_mechanics_residual_explicit` independently recomputes each BA term's
current moment, \(\boldsymbol F\), strain, \(\boldsymbol P\), and endpoint
assembly from \(\beta\boldsymbol P\boldsymbol K^{-T}\boldsymbol\xi\). The
energy directional FD uses this explicit residual; the Hessian JVP FD uses
explicit residuals at \(\boldsymbol u\pm\epsilon\boldsymbol v\), not a
finite difference of \(\boldsymbol K\boldsymbol u\). A separate residual versus
Hessian-gradient consistency assertion is also required.
The reproducible commands are:

```bash
source .venv/bin/activate
python tests/test_fracture2d_bond_associated_algebra.py
python -m unittest -v tests.test_fracture2d_bond_associated_algebra
```

The direct float64 report was:

- 204 directed edges; target-inclusive support size `3`--`7`; minimum shape
  rank `2`; shape-tensor condition-number range
  `[1.0000000000000000, 6.1024140507688145]`;
- target-only deformation sensitivity `0.8293015467410023` for the corrected
  support, versus exactly `0.0` for the deficient support;
- maximum affine-reproduction error `2.2204460492503131e-16`;
- translation energies `4.7281540841043204e-33` and
  `2.4636347523972630e-33`, with residual norms
  `2.1916133082149157e-16` and `1.5117049738328688e-16`; infinitesimal-rotation
  energy `6.6331573465818889e-39`, with residual norm
  `1.6481817651541565e-19`;
- centered energy-gradient finite-difference relative error using the explicit
  residual `5.1524182361008573e-13`; explicit-residual versus Hessian-gradient
  relative error `4.9797821209460204e-15`; Hessian JVP relative error from
  explicit residual FD `2.2562462860351144e-13`; Hessian symmetry relative error
  `6.6001468553380383e-17`;
- nonuniform-volume self-adjoint relation relative error
  `1.6052310978229946e-17`; directed-edge AT2 energy-gradient finite-difference
  relative error `6.3846707667200082e-12`;
- mechanics-Hessian PSD tolerance `1.7240874314182322e-12`; rigid-spectrum
  tolerance `1.7240874314182322e-10`; unconstrained Hessian rigid-zero-mode
  count `3`, minimum eigenvalue `-3.1849625516757346e-17`, first non-rigid
  eigenvalue `3.5559679223267984e-02`, and maximum eigenvalue
  `1.7240874314182322`.
- exact boundary-particle masks contain `5` bottom and `5` top particles;
  `11` DOFs are constrained and `39` remain free, including `9` non-anchor
  top/bottom horizontal DOFs. The eliminated free block
  has minimum eigenvalue `4.3220471171536131e-03` and condition number
  `3.1948527041858995e+02` for fixed damage at `u_bar=0.08`;
- fixed-damage hard-constraint free residual norm is
  `1.5669856032796044e-16`; top reaction is
  `3.0021111652864470e-02`, top residual-sum error is `0.0`, bottom reaction is
  `-3.0021111652864401e-02`, anchor reaction is
  `-9.2246346459494968e-18`, and top/bottom vertical balance error is
  `6.9388939039072284e-17`;
- after re-solving the free DOFs independently at
  \(\bar u\pm\epsilon\) with `epsilon=1e-5`, the reduced-energy FD versus top
  reaction relative error is `6.3111086703542940e-13`.

The earlier transient values used the deficient common-neighbor support and are
superseded for BA stability review. These corrected results establish the
algebraic structure and selected hard-constraint BVP only on a simple reciprocal
cloud. They do not establish stability after rounded-slot visibility,
tag-derived nonuniform Gmsh masks/volumes, or damage localization; those remain
mandatory implementation tests.

### 0.10 Revised internal-review status

The target-endpoint blocker is resolved in Section 0.3 and in the persistent
float64 check. The corrected support is

$$
\mathcal C_{i|j}=\{j\}\cup
\{k\ne i,j:(i,k)\in\mathcal E\ \text{and}\ (j,k)\in\mathcal E\},
$$

with the target term included in \(\boldsymbol K\), \(\boldsymbol A\),
\(\mathcal B\), force, and tangent sums. The old check is not evidence for
the corrected stability mechanism.

The boundary BVP is now frozen as the project-owned material-only
boundary-particle hard-constraint model in Section 0.2. Exterior fictitious
collars and interior material grips remain historical alternatives and are
explicitly rejected for this first study; neither is silently treated as the
selected BVP.

The review also reconfirmed the plane-strain constitutive and Amor formulas,
energy-derived mechanics force and tangent, directed-edge AT2 coefficient,
volume self-adjoint relation, and fixed-damage reduced-energy reaction. The
corrected spectrum report now includes the three unconstrained rigid modes and
the first non-rigid eigenvalue.

The coefficient/index audit is unchanged and explicit: \(\lambda\) and \(\mu\)
are the three-dimensional plane-strain Lamé constants; \(K=\lambda+2\mu/3\)
and only the positive volumetric trace is truncated in the Amor driver; the
directed AT2 graph energy has coefficient \(G_c\ell_0/4\), which yields
\(G_c\ell_0D_\delta\) after differentiating both reciprocal directed entries;
\(D_{ij}=-J_\delta(r_{ij})V_j\) gives
\(\operatorname{diag}(V)D=D^T\operatorname{diag}(V)\); and the reduced-energy
reaction has no extra particle-volume factor beyond the volumes already in
\(\nabla_q\Pi_{\mathrm m,h}\). The new directed-edge and nonuniform-volume
checks in Section 0.9 exercise these coefficients.

The follow-up independence review is addressed without changing the declared
boundary model: the explicit residual path is now the energy-gradient and
residual-JVP reference, while the linear-map Hessian path remains an independent
comparison. The boundary-particle hard-constraint gate is closed subject to the
Stage 1 tag/mask preservation check.

### 0.11 Source-backed versus project-owned choices

- **Chen-Spencer source-backed:** the bond-associated deformation gradient is
  formed for an individual target bond from the intersection of the source
  family with a target-centered bond-associated neighborhood, with shape tensor
  and force-state derivations in Eqs. (16)-(23) of Chen and Spencer (2019), DOI
  `10.1002/nme.5973`; the reviewed primary full text is the OSTI copy at
  <https://www.osti.gov/servlets/purl/1572458>. The direct target-bond
  kinematic relation is therefore retained in the discrete support.
- **Project-owned discretization:** the explicit finite-volume endpoint term
  \(\{j\}\), the use of the single first-study radius \(\delta\), reciprocal
  visibility graph, source-centered \(\beta_{i|j,k}=\omega_{ik}V_k\), normalized
  \(\eta_{ij}\), center-energy partition, diffuse \(g(d_i)\) degradation,
  plane-strain Amor-history AT2 coupling, boundary-particle hard constraints,
  DOF elimination, and constrained reaction are not
  printed as this combined discrete model in Chen-Spencer. They are defined and
  checked here as one project-owned energy-first model.
- **Behzadinasab Part II source-backed:** DOI
  `10.1007/s42102-020-00039-6`, Section 4 and Eqs. (8)-(9), describes an
  exterior horizon-thick fictitious layer, essential/natural boundary data, and
  separate kinematic/stress-divergence neighborhoods. It does not select the
  present mixed componentwise displacement BVP, material-only BA energy, or
  work-conjugate reaction.

### Stage 1: Geometry, particles, and quadrature

- Build a deterministic rounded-slot particle set and retain enough geometry
  data to compute explicit per-particle volumes.
- Evaluate the simplest viable cloud first. A Gmsh-derived nodal cloud with
  barycentric lumped areas is the leading candidate because it preserves the
  existing rounded geometry and exposes nonuniform volumes explicitly; it is
  not accepted until the Stage 2 manufactured tests converge.
- Validate positive volumes, unique coordinates, deterministic ordering,
  material membership, exact/tag-derived boundary-particle masks, and neighbor
  counts.
- Check volume sum against the measured material area `0.9762509375` and check
  first moments against geometry symmetry.

Implemented focused tests: `tests/test_fracture2d_nonlocal_geometry.py`.

### Stage 2: Weighted scalar nonlocal operator

- Extend or wrap `SparseD` with explicit `V_j` edge weights and the frozen
  geometry interaction rule while preserving SciPy export.
- Keep the weighted energy pairing explicit; verify constants, symmetry in the
  volume-weighted inner product, and non-negative quadratic energy.
- Test full-square constant/affine/quadratic manufactured fields, then the
  rounded-notch domain with full and boundary-excluded errors.
- Demonstrate convergence under fixed horizon refinement before using the
  operator in AT2 damage.

Proposed tests: `tests/test_scalar_operator_2d.py` and
`tests/test_nonlocal_geometry_graph.py`.

### Stage 3: Intact correspondence mechanics

- Implement a separate energy-first bond-associated path; do not extend the
  conventional `ElasticD` force residual into the fracture solver. Expose
  per-particle volumes, float64 strain/stress, bond-associated conditioning,
  energy, internal force, and a direct linear-solve interface.
- Validate rigid translation and rotation, affine strain reproduction, linear
  and angular momentum, and local Navier-Cauchy behavior away from boundaries.
- Check force as the finite-difference derivative of the same discrete energy,
  stiffness symmetry where expected, and reaction as the derivative with
  respect to imposed displacement.
- Inspect the constrained stiffness spectrum/null space before adding damage.
- Validate boundary-particle hard-constraint patch tests and free rounded-notch
  surfaces.

Proposed tests: `tests/test_fracture2d_nonlocal_mechanics.py`.

### Stage 4: Fixed fracture subproblems

- **Fixed `d=0`:** solve intact mechanics and compare pre-crack displacement,
  reaction, and stress trends with FEM.
- **Fixed history:** solve the dimensional AT2 algebraic system directly with
  SciPy; verify linear residual and damage-energy consistency without clipping.
- **Fixed damage:** solve mechanics, verify force residual, reaction/work
  balance, and finite diagnostics.
- Verify the Amor positive history computed from the frozen nonlocal
  plane-strain strain tensor under tension, compression, and mixed states.

Proposed tests: `tests/test_fracture2d_nonlocal_baseline.py`.

### Stage 5: Coupled strict continuation

- Reuse the fracture1d control pattern, not its 1D mechanics formula:
  immutable committed history, trial state, staggered mechanics/history/damage,
  acceptance gates, rollback, cutback, and healing diagnostics.
- Start with strict gates and a short pre-localization endpoint.
- Continue through peak/crack growth only after every fixed subproblem passes.
- Archive accepted increments, residuals, cutbacks, damage range/healing,
  energies, reaction, history, stress/strain, particle volumes, horizon,
  quadrature, boundary masks, visibility, stabilization, and model provenance.
- Compare at common load states, especially `u_bar=0.0055` through `0.0062`,
  while retaining the FEM warning flags in comparison plots.

#### Stage 5 implementation status — review approved; pilot deferred

The strict controller is now implemented in
`examples/fracture2d/nonlocal_baseline.py` through the public
`GateTolerance`, `StrictContinuationConfig`, `ContinuationMetadata`,
`SupplementalArchiveMetadata` (also exported as `ArchiveMetadata`),
`CommittedState`, `StaggerDiagnostics`, `AttemptRecord`, `AcceptedStep`,
`ContinuationResult`, `ContinuationFailure`, `run_strict_continuation`, and
`write_continuation_archive` APIs. Setup validation requires exact float64 2D
cubic operator/mechanics data, identical coordinates/volumes/sorted directed
edges and exact `delta`, compatible hard constraints, positive material and
configuration values, and strict zero-start loading relations.

The controller builds immutable `ContinuationMetadata` into every result. It
contains exact bytes-backed coordinates, volumes, reciprocal directed edges,
boundary masks, anchor, constrained/free DOFs, material parameters, kernel,
plane-strain/no-stabilization declaration, float64 dtype, and the exact
configuration. Result validation cross-checks accepted load/state pairing and
recomputes the committed damage residual from those frozen metadata fields.
The zero state is archived, committed arrays remain independent bytes-backed
read-only storage, and rollback is verified with SHA256 dtype/shape/byte
digests. A fresh fixed-damage view is created for each mechanics solve. Every
trial uses the immutable committed history for both Amor maxima; it performs a
complete before/damage/after sweep and gates the direct committed residual with
`H_next`. Raw damage range and healing violations reject without clipping.
Fixed-subproblem `RuntimeError` and `FloatingPointError` values are audited and
cut back; unknown exceptions propagate. There is no standalone final damage
re-solve, reaction/energy monotonicity gate, stabilization, or hidden increment
growth rule.

`write_continuation_archive(path, result, archive_metadata)` requires explicit
`SupplementalArchiveMetadata` for int64 mesh triangles, volume-weighted first
moments, visibility tolerance/rule, quadrature rule, study label, and model
provenance. It validates these fields against `ContinuationMetadata` and writes
static data, all state/accepted-step numerical fields, diagnostics, work and
history quantities, and the complete attempt audit without object arrays so the
archive can be loaded with `allow_pickle=False`.

`tests/test_fracture2d_nonlocal_continuation.py` covers the manufactured-cloud
controller semantics, an independent dense BA/damage/Amor clean-step reference,
exact final state pairing and residuals, committed immutability, trial-history
isolation, rollback/cutback equivalence, raw range rejection, residual
rejection, force/moment gate rejection, forged-step invariants, typed
minimum-increment failure, exception propagation, lazy-view isolation,
setup/config validation, coherent forged-result rejection, structural exception
propagation, and archive audit. Independent Sol review approved this controller
implementation after all findings were closed. A rounded-slot pilot with
temporary diagnostic values remains deferred and is not part of this review.
No production horizon or physical continuation approval is claimed.

#### 2026-08-22 independent review checkpoint — incomplete

The read-only independent review was continued on the manufactured 4-by-4 cloud
with `JAX_PLATFORMS=cpu`. The focused Stage 5 suite ran 24 tests in 4.641 s and
the complete Stage 0-5 verification list ran 130 tests in 107.080 s; both passed.
Compilation of the directly involved modules and `git diff --check` also passed.
The default JAX test invocation remains unavailable because the environment's
CUDA device initialization fails with `CUDA_ERROR_OUT_OF_MEMORY`.

The review does not approve Stage 5. It found these result/archive audit gaps:

- `ContinuationResult.validate` near `nonlocal_baseline.py:1926-2022` does not
  compare `step.amor_driver.previous_history` with the preceding committed
  state's history. Replacing the accepted first-step previous history with a
  zero array was accepted.
- `AcceptedStep.__post_init__` near `nonlocal_baseline.py:1724-1772` checks
  Euclidean fixed-history residual summaries against their forged vector, but
  neither it nor `ContinuationResult.validate` reconstructs that residual from
  the frozen damage system. A forged Euclidean residual was accepted and
  `write_continuation_archive` serialized it.
- A forged `MechanicsSolveResult.free_residual_relative_norm`, with the matching
  `StaggerDiagnostics` and attempt audit record, was accepted. The live solver
  validator near `nonlocal_baseline.py:2528-2537` is authoritative during a
  solve, but result validation near `:1955-1963` does not independently validate
  the relative scalar.
- `_failure_diagnostics` near `nonlocal_baseline.py:2649-2663` still swallows
  `ValueError` from `_damage_residuals`. A manufactured mechanics-after
  `RuntimeError` plus diagnostic `ValueError` was converted into a normal typed
  continuation failure instead of propagating the structural error. Direct
  `ValueError` from the fixed-history trial path does propagate.

The actual force/moment path is supported: the mechanics solve computes them
from the residual near `nonlocal_mechanics.py:1260-1275`, the live validator
recomputes them near `nonlocal_baseline.py:2541-2582`, and the tiny-tolerance
tests use the live gates rather than forged summaries. Bytes-backed nested
arrays, graph metadata, rollback digests, archive `allow_pickle=False` loading,
and the existing attempt-sequence checks passed their current tests.

Exact resume point: independently decide and implement (if authorized) the
four validation fixes above, add matching forged-state tests, then rerun the
same CPU-only suites. Do not run a rounded-slot pilot or select a production
horizon during this task.

#### 2026-08-22 independent review resolution — approved

All four checkpoint blockers and the earlier archive/immutability findings are
closed. `ContinuationResult.validate` now rebuilds authoritative intact and
fixed-damage mechanics from frozen metadata for every accepted state; recomputes
mechanics energy, residual, free RHS, reactions, force, and moment; replays all
Amor fields from accepted strain and the preceding committed history; and
reconstructs fixed-history residuals, energies, and symmetry diagnostics.
`MechanicsSolveResult` records and validates non-negative `free_rhs_norm`, all
accepted numerical arrays are bytes-backed, and structural `ValueError` values
propagate instead of entering the numerical cutback path.

Independent final review found no remaining issue and approved the Stage 5
controller implementation. With `JAX_PLATFORMS=cpu`, the explicit Stage 0--5
verification list passed all `134` tests in `106.700 s`; changed-module
`compileall`, `python src/pinn/operators.py`, and `git diff --check` passed. The
default-GPU equivalence benchmark remains independently blocked by CUDA memory
pressure and its pre-existing ConvD/SparseD update-step threshold, not by Stage
5. No rounded-slot pilot or production horizon was run or selected.

Exact resume point: wait for explicit authorization before choosing temporary
diagnostic values outside the repository and running the short pre-localization
rounded-slot pilot. Do not promote `delta=0.032`, `0.040`, or `0.048`, begin a
PINN, or claim physical continuation validation from the implementation review.

#### 2026-08-22 rounded-slot pilot — completed and approved

The user authorized the next Stage 5 step. The temporary pilot uses the existing
`stage3_cloud_float64.npz` `d048` graph, `delta=0.048`, `load_final=1e-3`,
`du_initial=1e-4`, `du_min=1e-5`, cutback factor `0.5`, at most eight cutbacks,
and 20 stagger sweeps. CG uses the Stage 4 rounded settings `rtol=1e-7`,
`atol=1e-10`, and `maxiter=3000`. Displacement, damage, history, reaction,
force, and moment gates use `(atol,rtol,scale_floor)=(1e-9,1e-8,1)`, while the
damage-residual gate uses `(1e-10,1e-8,1)`; healing/range tolerances are
`1e-10`. These are traceable operational values, not calibrated production
tolerances.

Acceptance requires reaching `u_bar=1e-3`, every accepted gate passing, exact
rollback digests for any rejected attempt, finite unmodified fields, complete
archive reload with `allow_pickle=False`, and independent review of gate ratios,
damage/history monotonicity, conservation, CG, energy, and Stage 4 trend
comparisons. Exact resume point: run the temporary pilot, preserve failures
without gate relaxation, and record the archive/review outcome here.

The one-step `u_bar=1e-4` preflight first passed in three stagger sweeps with no
cutback. The full run then completed ten `1e-4` increments in `6614.34 s`; all
attempts were accepted, stagger counts were `[3,3,3,5,4,8,10,4,7,8]`, and every
accepted mechanics solve used 243 CG iterations with `cg_info=0`. The maximum
gate error/limit ratios were displacement `0.00479`, damage `0.28653`, history
`0.74418`, reaction `0.02991`, committed damage residual `0.77914`, force
`1.46e-8`, and moment `1.39e-8`.

Final damage was `[2.4324e-7, 0.0125557]`, final history maximum was `1.49402`,
reaction was `128.5400361`, mechanics energy was `0.0642700`, fixed-history
energy was `0.0643696`, and cumulative trapezoidal reaction work was `0.0644170`.
Minimum accepted healing and history gaps were positive at `2.439e-9` and
`3.103e-10`. At `u_bar=1e-4`, `5e-4`, and `1e-3`, pilot/FEM reaction ratios were
`0.968387`, `0.968371`, and `0.968320`; displacement-norm ratios were `0.997036`,
`0.997031`, and `0.997016`; mean-`sigma_yy` ratios were `0.967317`, `0.967302`,
and `0.967254`.

Independent review reconstructed quadrature, digests, gates, fixed-history
residuals/energies, Amor fields, and mechanics fields and approved the pilot.
The archive contains 148 non-object fields and loads with `allow_pickle=False`:

- archive: `/tmp/opencode/stage5_rounded_pilot_delta048.npz`, SHA256
  `8a6ac2909d0b3a0148627013264e03cd1dc1f444018fb59e85635bbcea1ba6e0`;
- summary: `/tmp/opencode/stage5_rounded_pilot_delta048.json`, SHA256
  `164d864ca2d659e567d3426c2e80aeb8fdff0d5260aed2289016f4ba28a6a00b`.

Residual limitations are explicit. No rejected rounded attempt exercised the
runtime rollback branch, although manufactured failure injection covers it.
The archive stores complete gate diagnostics but not all final-sweep before
fields needed to reconstruct every gate without rerunning. The existing archive
encodes JSON booleans as integer `1`; numerical pass arrays are bool, and the
serializer ordering is fixed with a focused regression test for future archives.

Exact resume point: select whether to improve CPU runtime/restart support or pay
for a staged diagnostic extension toward peak/crack growth. The current
`delta=0.048` remains nonproduction; fixed-horizon refinement and local-limit
studies are still required before horizon selection or PINN work.

#### 2026-08-23 direct-mechanics optimization — in progress

The user selected optimization before peak continuation. The frozen contract is
unchanged: float64 graph/mechanics, exact fixed-damage linear equations, fresh
views, strict explicit residual validation, and all Stage 5 gates remain
authoritative. The existing NumPy/SciPy implementation remains the independent
reference.

The measured pilot performed 55 stagger sweeps and therefore 110 mechanics
solves. Every solve took 243 zero-start CG iterations, for 26,730 matrix-free
tangent actions over 72,934 directed edges and padded support width 47. Damage
solves took only about `0.13`--`0.19 s`; mechanics calls took about `48`--`64 s`.

Execution order is: add an explicitly validated full-displacement initial guess
to mechanics CG; check its free residual before constructing block-Jacobi; seed
new loads from the scaled committed solution and repeated sweeps from the prior
mechanics-after solution; prove cold/warm equivalence; benchmark the rounded
preflight; then vectorize/flatten support and block-Jacobi. Only after those
checks may a separate JAX float64 backend keep tangent, preconditioner, and CG on
GPU under fixed shapes. No tolerance, horizon, or physical model changes belong
to this optimization.

The CPU optimization is now validated. An initial implementation that accepted
an ordinary tolerance-level repeated-system displacement without replay was
rejected because its reaction error was about 30 times the Stage 5 reaction
gate. The accepted policy permits early exit only for an explicitly authorized
same-load, same-damage repeated system. A changed-system seed below the strict
CG limit falls back to cold CG; a seed at or above the limit enters SciPy as
`x0`. Cross-load seeds are scaled from the immutable committed displacement,
and cutback seeds never enter committed state.

The exact particle block-Jacobi assembly was also replaced by closed-form
`2 x 2` occurrence contributions and vectorized `bincount` reductions, without
changing the tangent. On the rounded `u_bar=1e-4` one-step benchmark, the
original cold path took about `326.0 s`, the accepted safe warm path took
`194.1 s`, and safe warm plus vectorized block-Jacobi took `164.3 s`. The final
reaction differed from cold by about `6.76e-10`; the focused CPU mechanics and
continuation suite passed all `42` tests.

`examples/fracture2d/nonlocal_mechanics_jax.py` now implements a separate
float64 backend constructed from the accepted intact CPU mechanics and explicit
hard constraints. It flattens only valid target-inclusive support occurrences,
keeps int32 indices and float64 graph data on one explicit device, and JITs the
analytic residual, tangent, exact masked block-Jacobi, and `lax.while_loop` PCG.
Damage, load, tolerances, maximum iterations, seed presence, initial
displacement, and early-exit permission are dynamic fixed-shape inputs. The
backend rejects disabled x64, invalid shapes/dtypes, non-finite degradation,
non-finite public outputs, PCG breakdown/nonconvergence, and an explicit final
residual at or above the unchanged strict CG limit.

The replay contract is deliberately asymmetric. `replay_cpu` exactly pairs the
constraints, damage, load, CG settings, initial displacement, and seed-policy
flags, then evaluates the GPU displacement through a fresh CPU fixed-damage
view. Only the returned CPU `MechanicsSolveResult` is authoritative for Stage 5
acceptance and physical archive fields. JAX fields and iteration counts remain
non-authoritative execution diagnostics; no heuristic CPU/GPU reduction
tolerance is used.

The real-cloud GPU benchmark consumed the final pilot damage on the diagnostic
`delta=0.048` graph: `3395` particles, `72934` directed edges, `1131622` valid
support occurrences, and padded support width `47`. On the RTX 5090 with JAX
preallocation disabled, construction retained about `652 MB` and a completed
solve about `986 MB`. Compile plus first solve took `7.92 s`; subsequent
243-iteration solves had median `0.0751 s`. The corresponding current CPU cold
solve took `45.16 s` in a separate same-cloud measurement. Warm primitive
medians were about `0.82 ms` for one tangent action, `0.48 ms` for block-Jacobi,
and `1.48 ms` for evaluation. The solve trace count stayed at one while damage,
load, tolerances, maximum iterations, and host/device seeds changed.

The final-damage GPU solve had explicit residual norm
`5.6001261029e-5`; fresh CPU replay gave `5.6001261295e-5`, below the common
limit, and CPU-authoritative top reaction `128.5400360735`. The focused suite
passed all `8` tests on GPU and `8` tests with one GPU-only skip on CPU. Full
CPU discovery passed `153` tests with one GPU-only skip; changed-module
`compileall` and `git diff --check` passed. Independent numerical re-review
found no remaining standalone-backend blocker and approved progression only to
a separately reviewed Stage 5 adapter.

The Stage 5 integration is now restricted to
`JaxCPUReplayMechanicsAdapter`; generic solver hooks are rejected. Every optional
solve receives bytes-backed immutable damage and seed arrays, returns a fresh
CPU replay result, and is then checked against another fresh fixed-damage view
by the unchanged common validator. Numerical `RuntimeError` and
`FloatingPointError` values retain cutback semantics, while structural replay
`ValueError` values propagate. `ContinuationMetadata`, JSON metadata, and the
numerical archive record `mechanics_backend` explicitly. Manufactured tests
cover CPU/JAX continuation equivalence, immutable inputs, before/after failure
rollback, structural replay errors, archive reload, and an actual GPU adapter
continuation.

The authorized full CPU-replayed GPU rerun used the exact prior temporary
`delta=0.048`, `u_bar=0` to `1e-3` configuration. It completed in
`430.1667 s` with one JAX trace, 236 mechanics calls, 102 safe repeated-system
early exits, 30 nonzero initial guesses used by PCG, and `16.4087 s` total
device solve time. The run had one strict rejection at target `6e-4` after 20
sweeps because history and committed damage-residual ratios were `2.8905` and
`5.4113`. Its committed-state rollback digest was unchanged. The controller
cut the increment to `5e-5`, accepted `5.5e-4`, and then completed through
`1e-3` in 15 accepted steps without changing any gate.

All accepted gates passed. Maximum error/limit ratios were displacement
`0.00773`, damage `0.46167`, history `0.83078`, reaction `0.03019`, committed
damage residual `0.98142`, force `1.34e-8`, and moment `1.11e-8`. Final damage
was `[2.4324e-7, 0.0125557]`, final history maximum was `1.4940184`, reaction
was `128.5400361`, mechanics energy was `0.0642700180`, fixed-history energy
was `0.0643696221`, and cumulative reaction work was `0.0644178638`.

Fresh independent CPU reconstruction reproduced every archived mechanics field
exactly; the direct damage-residual reconstruction differed by at most
`1.33e-15`. Relative to the prior approved CPU path, final maximum differences
were `7.76e-11` displacement, `2.51e-10` damage, `4.38e-8` history, and
`2.11e-8` reaction. These differences and the cutback path are retained rather
than hidden as exact path equivalence.

The 154-field archive loads with `allow_pickle=False`, contains no object arrays,
and records backend `jax_gpu_cpu_replay`:

- archive: `/tmp/opencode/stage5_rounded_jax_pilot_delta048.npz`, SHA256
  `4acf5ebf6e3c1426ec1395892cb8455b1a4cf6a979d1f20423eae864c9e27806`;
- summary: `/tmp/opencode/stage5_rounded_jax_pilot_delta048.json`, SHA256
  `21a8984e863ac56a5b2504bf1b51c53a84bb576471b9d7406300c18defc15989`;
- temporary driver: `/tmp/opencode/stage5_rounded_jax_pilot.py`, SHA256
  `b28e8c8d13dcbfff0c31d766a807714233eee985735becb75f512918fe3a9db6`.

Independent numerical review approved the adapter and this run only as
pre-localization evidence. Final verification ran 158 CPU tests with two
GPU-specific skips, all nine requested GPU backend/adapter tests, the shared
operator self-check, changed-module `compileall`, and `git diff --check`; all
passed. Exact resume point: explicitly choose either a staged peak/crack-growth
diagnostic or Stage 6 fixed-horizon refinement. Do not remove CPU replay, relax
the two close gates, select a production horizon, or begin PINN work from this
result.

#### 2026-08-23 staged warning continuation — completed through first checkpoint

The user selected staged progression toward peak with the first endpoint
`u_bar=2e-3` and explicitly authorized every finite post-sweep acceptance gate
to become a warning. This does not authorize committing a missing or non-finite
candidate: structural errors, non-finite values, and mechanics/CG failures that
return no valid displacement remain hard failures on the existing audited
cutback path.

Implement a separate `run_warning_continuation` entry point. Strict behavior and
strict archive semantics must remain unchanged. A warning commit is allowed only
after a complete mechanics-before, committed-history Amor, raw damage,
mechanics-after, and `H_next` sweep reaches the configured stagger budget. It
must commit the unmodified `(u_after, d_next, H_next)` candidate, retain
`diagnostics.passed=False`, and archive an exact canonical list/mask of every
bypassed gate. No clipping, projection, tolerance change, standalone final
damage solve, history accumulation, or increment-growth rule is permitted.

Exact resume point: implement and independently review the warning result and
archive invariants, prove strict-mode compatibility and CPU/JAX warning
equivalence on the manufactured cloud, then run a fresh zero-start temporary
`delta=0.048` diagnostic through `u_bar=2e-3`. Do not proceed farther until that
checkpoint is audited.

#### 2026-08-23 warning-continuation audit fixes — approved

The independent audit found two blockers: accepted steps trusted the stored
`StaggerDiagnostics` pass summaries and bypass list (a coherently forged force
pass/bypass or volume-weighted used residual could validate and archive), and
complete-sweep diagnostics allowed `inf` scalars.

`AcceptedStep` now carries the immutable exact final-sweep before data
`damage_before` (the final sweep's `d_iter`) and `mechanics_before_result`
beside the existing final mechanics result, populated for strict and warning
accepted steps. Step validation requires exact float64 finite shapes, u_bar
pairing, and exact before-side diagnostics relationships (reaction/force/moment,
CG info, free-residual norms, and `e_d` against `damage_before`).
`ContinuationResult.validate` now rebuilds a fresh fixed-damage view for
`damage_before`, fully validates the stored mechanics-before result against it,
enforces `damage_before == preceding committed damage` for single-sweep steps,
replays `H_used` from the mechanics-before displacement and the preceding
committed history, and recomputes the complete `_full_stagger_diagnostics`
from the authoritative before/after mechanics, `damage_before`/`d_next`,
`H_used`/`H_next`, frozen damage matrix/volumes/config/geometry. Every scalar,
pass flag, vector (including `rho_used_volume`/`rho_next_volume` and forces),
rejection reason, `complete_sweep`, and the warning bypass list must match the
replay (exact for booleans/integers/strings, scale-aware for floats/arrays).
`StaggerDiagnostics` now rejects any non-finite scalar when
`complete_sweep=True`; incomplete failure diagnostics retain `inf` sentinels
but never NaN and can never pair with an accepted step.

The archive adds `accepted_damage_before`,
`accepted_mechanics_before_displacement`, and
`accepted_mechanics_before_free_residual`, sufficient with the frozen graph,
states, and `H_used` to reconstruct every final-sweep gate without rerunning
the sweep. All arrays remain non-object and load with `allow_pickle=False`.

New regressions reproduce the forged force pass plus shrunken bypass list,
forged `rho_used_volume`, coherently forged `H_used`, forged before-force,
forged strict before summaries, the finite-complete-sweep contract, and the
archive before-data fields with strict compatibility.

Verification with `JAX_PLATFORMS=cpu`:
`tests.test_fracture2d_nonlocal_continuation.py` passed all `53` tests, and
the explicit Stage 0--5 verification list passed all `165` tests with GPU-only
skips; the full discovery run passed all `172` tests. With the GPU visible,
the warning and strict GPU adapter tests passed. Changed-module `compileall`
and `git diff --check` passed. The rounded `2e-3` warning diagnostic remains
deferred to the primary agent.

Exact resume point: independent re-review of the final-sweep gate replay and
archive before-data, then the authorized fresh zero-start temporary
`delta=0.048` warning diagnostic through `u_bar=2e-3`. Do not relax gates,
clip fields, or extend past `2e-3` before that checkpoint is audited.

#### 2026-08-23 warning continuation implementation — approved

`examples/fracture2d/nonlocal_baseline.py` now exposes
`run_warning_continuation` beside the unchanged `run_strict_continuation`. Both
delegate to one private `_run_continuation(..., acceptance_mode)` controller;
the strict entry point, sweeps, seeds, rollback digests, cutback rules, and
typed failures are byte-for-byte the previous strict behavior. The only
mode-dependent trial logic in `_run_trial_attempt` is: the strict-only raw
range/healing short-circuit (`damage_acceptance` stage) is skipped in warning
mode, and at the stagger budget a complete finite sweep whose failed gates are
all bypassable commits the exact unmodified `(u_after, d_next, H_next)` as
`warning_accepted`. Accepted-when-passed steps remain `strict_passed` in both
modes.

The immutable contracts were extended: `ContinuationMetadata.acceptance_mode`
(`strict`/`warning`, required), `StaggerDiagnostics.complete_sweep` (full-sweep
diagnostics only), `AcceptedStep.acceptance_status`/`bypassed_gates`
(the bypass tuple must exactly equal the canonical failed subset of
`WARNING_GATE_NAMES` = displacement, damage, history, reaction,
damage_residual, damage_range, healing, history_monotonic, force, moment),
`AttemptRecord.acceptance_status`/`sweep_complete`/`bypassed_gates`, and
`ContinuationResult.validate` cross-pairing that rejects warning-accepted steps
under strict metadata and any attempt/step status mismatch.

`write_continuation_archive` now records `archive_schema_version=2`,
`acceptance_mode`, fixed canonical `gate_names`, `state_acceptance_status`
(`initial` first), `accepted_acceptance_status`, `accepted_warning`,
`accepted_bypassed_gate_mask`, and attempt-audit
`acceptance_status`/`sweep_complete`/`bypassed_gates` entries. Existing per-gate
pass arrays keep their real values (`False` for bypassed warning gates); all
arrays remain non-object and load with `allow_pickle=False`. Strict archives
carry the same provenance with all-warning fields false/empty.

Verification with `JAX_PLATFORMS=cpu`: the focused suite
`tests.test_fracture2d_nonlocal_continuation.py` passed all `49` tests (35
strict unchanged + 14 warning), and the explicit Stage 0--5 verification list
passed all `165` tests with three GPU-only skips. With the GPU visible and
preallocation disabled, both the new warning GPU adapter test and the existing
strict GPU adapter test passed. Changed-module `compileall` and
`git diff --check` passed. The new tests cover budget-exhaustion commits with
exact canonical bypass lists, endpoint/load pairing, unmodified out-of-range
damage commits with mechanics-after still executing, cutback/typed failures for
incomplete candidates, structural `ValueError` propagation, strict
compatibility/statuses, clean-path warning/strict state equality, CPU/JAX
warning equivalence, forged warning metadata rejection, archive reload and
provenance, and the gate-name/status contract.

#### 2026-08-23 staged `2e-3` checkpoint — completed and approved

The final independent warning-mode review required two additional evidence
hardening changes before the real run. Authoritative replay comparisons now
reject dtype changes as well as non-finite values. Each CPU/JAX
`MechanicsSolveResult` also carries a process-local provenance token set by its
actual constructor; result validation pairs both final-sweep mechanics results'
tokens and labels with continuation metadata. Regressions reject
post-construction float32 diagnostic rebinding and coherent metadata/result
backend relabeling. Independent re-review then gave limited approval to run the
fresh zero-start `delta=0.048` diagnostic through `u_bar=2e-3` only.

The temporary CPU-replayed JAX GPU run used the frozen pilot values unchanged
except for `load_final=2e-3` and the separate warning entry point. It completed
20 nominal `1e-4` increments in `886.3862 s`, with 412 mechanics calls, 186
authorized repeated-system early exits, 50 PCG initial guesses, one JAX trace,
and `22.8238 s` total device solve time. Every attempt was accepted with zero
cutbacks. All 20 accepted steps actually passed every strict gate, so the
archive contains 20 `strict_passed` states, zero `warning_accepted` states, and
an entirely false `(20,10)` bypass mask.

The worst accepted error/limit ratios were displacement `0.017286`, damage
`0.822088`, history `0.890768`, reaction `0.045210`, committed damage residual
`0.861808`, force `2.09e-8`, and moment `2.06e-8`. Final damage was
`[9.64339e-7, 0.0513580]`, history maximum was `6.3174924`, reaction was
`253.5305191`, mechanics energy was `0.2535305191`, fixed-history energy was
`0.2556993274`, and cumulative trapezoidal reaction work was `0.2558902124`.
Reaction increased at every archived load; this remains pre-peak evidence and
does not demonstrate crack growth.

Independent audit reconstructed every archived CPU mechanics field exactly,
recomputed the Amor and fixed-history fields, matched all state/attempt SHA256
digests, and verified all range, healing, history, CG, conservation, energy, and
work data. The archive has 164 fields, contains no object arrays, records schema
2, mode `warning`, and backend `jax_gpu_cpu_replay`, and loads with
`allow_pickle=False`:

- archive: `/tmp/opencode/stage5_rounded_jax_warning_delta048_2e3.npz`, SHA256
  `bd67a11fcfcdffa97611d08aad0926489565a9c7d76ef7465af9c3a1061b2225`;
- summary: `/tmp/opencode/stage5_rounded_jax_warning_delta048_2e3.json`, SHA256
  `9d3700952e665a7672482f023db25563c90cd902f638f60e889fc26e5b10b36e`;
- temporary driver:
  `/tmp/opencode/stage5_rounded_jax_warning_delta048_2e3.py`, SHA256
  `cb06e138adb88c1e6beac55cf820053615782a2c3db284e33fd1d3bfa4beb27b`.

Final verification passed all 178 CPU-discovery tests with three GPU-only skips,
all 63 focused GPU mechanics/continuation tests, changed-module compilation,
and `git diff --check`. Remaining limitations are explicit: the real rounded
run exercised neither a warning bypass nor rollback; those paths remain covered
only by manufactured tests. No production horizon, peak/crack-growth behavior,
refinement, or PINN work is approved.

Exact resume point: ask the user to choose and authorize the next staged load
endpoint or switch to Stage 6 fixed-horizon refinement. Do not extend beyond
`u_bar=2e-3` implicitly.

#### 2026-08-23 semantic naming cleanup — completed

The persistent JAX adapter name no longer encodes the roadmap stage:
`JaxStage5MechanicsAdapter` was replaced by
`JaxCPUReplayMechanicsAdapter`, which states the actual contract that JAX
execution is followed by a CPU-authoritative replay. The associated local
adapter variables and regression method names were made CPU-replay-specific,
and manufactured study labels no longer contain `stage5`.

The persistent bond-associated algebra regression was also renamed from
`test_fracture2d_stage0_ba.py`/`Fracture2DStage0BondAssociatedTest` to
`test_fracture2d_bond_associated_algebra.py`/
`Fracture2DBondAssociatedAlgebraTest`. The fracture1d exploratory modules were
renamed from `frozen_pilot.py`, `transition_pilot.py`, and
`pseudo_time_pilot.py` to `fixed_history_diagnostics.py`,
`transition_diagnostics.py`, and `pseudo_time_diagnostics.py`, with matching
`run_*_diagnostic` entry points and test imports.

Roadmap labels remain in historical task/decision prose where they describe
validation order. `baseline`, `warning`, and `failure_stage` remain because
they describe reference-solver role, acceptance policy, and execution phase,
respectively. Existing temporary archive filenames and historical hashes were
not rewritten.

#### 2026-08-23 staged `3e-3` checkpoint — authorized

The user authorized the next fresh zero-start `delta=0.048` warning diagnostic
through `u_bar=3e-3`. Retain the prior nominal `du=1e-4`, `du_min=1e-5`, eight
cutbacks, 20 stagger sweeps, CG settings, all gate tolerances, unmodified fields,
and `JaxCPUReplayMechanicsAdapter` with CPU-authoritative replay. Use semantic
temporary artifact names rather than roadmap-stage labels.

Acceptance for this checkpoint means reaching exactly `3e-3` or preserving and
auditing any hard failure, then independently checking reaction slopes, the
first warning/bypass if one occurs, damage localization, history/healing,
rollback, CG/conservation, work/energy, and archive reconstruction. Do not
continue beyond `3e-3`, alter the horizon or tolerances, or infer a production
horizon/refinement/PINN approval.

Exact resume point: run the fresh-zero temporary diagnostic, preserve its raw
archive and summary, and obtain independent numerical audit before choosing a
farther endpoint.

The first execution exposed a pure endpoint-bookkeeping defect: repeated
`1e-4` addition reached `0.0029999999999999988`, after which the controller
solved an extra `1.301e-18` step to exact `0.003`. The archive was internally
valid, but that tiny denominator created a meaningless terminal reaction slope.
The original archive and summary are preserved with `_unsnapped` suffixes.

The controller now snaps a nominal target to exact `load_final` only when the
remaining difference is at most eight ulps of the endpoint. A regression proves
that `0 -> 0.003` with `du=1e-4` produces exactly 30 targets and a full-sized
last increment. No PDE, gate, cutback, warning, or archive tolerance changed.
Focused CPU continuation verification passed all 56 tests with two GPU-only
skips. The fresh-zero checkpoint was then rerun for independent audit.

The snapped rerun completed exact `u_bar=3e-3` in 30 accepted nominal steps and
`1344.8024 s`, using 590 mechanics calls, 265 authorized repeated-system early
exits, 94 PCG initial guesses, one JAX trace, and `28.3720 s` total device solve
time. There were no rejected attempts or cutbacks. Statuses were 28
`strict_passed` and two `warning_accepted`:

- near `1.6e-3`: bypassed history and committed damage residual;
- near `1.8e-3`: bypassed damage, history, and committed damage residual.

The maximum ratios were damage `1.19581`, history `4.94751`, committed damage
residual `4.78882`, displacement `0.02810`, reaction `0.07073`, force
`1.75e-8`, and moment `1.56e-8`. All warning candidates were complete and
finite; damage range, healing, history monotonicity, mechanics, force, and
moment passed. Final damage was `[2.13505e-6, 0.1204572]`, maximum history was
`15.7555227`, reaction was `371.4762173`, mechanics energy was `0.5572143262`,
fixed-history energy was `0.5686836102`, and reaction work was `0.5691179855`.
Damage remained maximal at the declared notch-tip particle `(0.5,0.5)`.

Independent audit found exactly 30 full-sized increments, positive reaction
increments/slopes and work increments, exact state/attempt digests, exact fresh
CPU mechanics reconstruction, finite archive fields, CG convergence, and
float64-level conservation and energy reconstruction. The warning loads differed
from the first unsnapped execution because GPU reductions and near-threshold
stagger trajectories are path-sensitive; archive-local statuses are valid, but
warning trigger reproducibility is not claimed.

The approved schema-2 archive contains 164 non-object fields and loads with
`allow_pickle=False`:

- archive: `/tmp/opencode/fracture2d_d048_warning_u3e3.npz`, SHA256
  `d8109cb854dc6952a2791165ac6bd9c39d09891639fd18d6ed739b293db4e511`;
- summary: `/tmp/opencode/fracture2d_d048_warning_u3e3.json`, SHA256
  `9c987ae2f4fa7d031e0aa81104a5ab3bf9b954da21418de2b58bebe2a4fac26a`;
- driver: `/tmp/opencode/fracture2d_d048_warning_u3e3.py`, SHA256
  `bf3619bef8bdcb6f35fa71a7b43e93793a9d994d515e8612d2810369ff3a3f27`.

The limited unsnapped archive is retained at
`/tmp/opencode/fracture2d_d048_warning_u3e3_unsnapped.npz` with SHA256
`60e2d93f9294de9d9fe1d8bc7633fd9e69493576462fd35ba2659f682ef5ab75`.
It is valid archive evidence but not an approved load curve because of its
three-ulp duplicate terminal solve.

Final verification after the endpoint fix passed all 179 CPU-discovery tests
with three GPU-only skips, all 64 focused GPU mechanics/continuation tests, the
shared operator self-check, changed-module compilation, and
`git diff --check`. The reaction/warning and final-damage preview is
`/tmp/opencode/fracture2d_d048_warning_u3e3.png`.

Exact resume point: choose and authorize a farther staged load endpoint or
switch to Stage 6 fixed-horizon refinement. Do not continue beyond `3e-3`,
claim peak/crack growth, select a production horizon, or begin PINN work
implicitly.

#### 2026-08-23 complete `u_final` run — authorized

The user chose to defer Stage 6 refinement and run one complete fresh-zero
warning continuation to the fracture2d default
`examples/fracture2d/configs/solver_base.py:loading.u_final=0.0075`. Use
`delta=0.048`, the existing nominal `du=1e-4` schedule and cutback settings,
`JaxCPUReplayMechanicsAdapter`, CPU-authoritative replay, and the unchanged
warning gates. The graph, `ExplicitVolumeSparseD` operator, visibility, and
support arrays remain fixed throughout loading: crack growth is represented by
damage/history and degraded mechanics, not by remeshing or dynamic bond
deletion. No clipping or projection is allowed.

The audit must explicitly inspect the damage operator residual and fields as
damage localizes/spreads, fixed graph/support invariants, damage/history range
and healing, mechanics CG/residual/conservation, reaction/work/energy behavior,
warning masks/statuses, and complete archive reconstruction. This is a single
diagnostic run, not production-horizon, peak/crack-growth, refinement, or PINN
approval. Stop at `0.0075` or any hard failure; preserve the exact archive and
summary.

Exact resume point: run the fresh-zero `u_final=0.0075` driver and independently
audit the crack-growth regime before making any further decision.

The first full execution completed its load solves but failed during final
result replay because duplicate residual checks used inconsistent float64
reconstruction tolerances. The runtime operator CSR uses JAX-generated cubic
coefficients while metadata replay independently rebuilds them with NumPy; the
coefficient difference is about `4.55e-12`. The stricter duplicate residual
comparison was replaced by the existing finite, dtype-exact, 512-eps
scale-aware authoritative comparison. Focused continuation verification then
passed all 57 tests with two GPU-only skips.

The rerun retained an atomic per-accepted-step raw checkpoint before another
final replay-only fixed-history comparison rejected a `1.421e-13` reconstruction
defect against `1.137e-13`. No validated archive was written. The raw file is
explicitly `validated=False` and contains 75 loads through exact `0.0075`, with
39 `strict_passed` and 36 `warning_accepted` step labels. It is suitable for
field inspection only, not archive approval.

Raw inspection found a reaction peak `606.7496464` at `u_bar=0.0055`, followed
by a drop to `165.4786331` at `0.0075`. Final damage is
`[6.4375e-6,0.9974146]`, but the maximum is at top-boundary particle
`(0.96,1.0)`: 38/51 top particles have `d>0.5`, their mean damage is `0.7388`,
and the largest non-top value `0.9225` is still near the top at
`(0.8319,0.9840)`. The notch-tip particle `(0.5,0.5)` has damage `0.5872`; on
the centerline, `d>0.1` extends only to about `x=0.600` and no point has
`d>0.7`. Healing occurs at `0.0056`, `0.0057`, and `0.0058`, with minimum raw
increment about `-0.00397`. Final history maximum is `2.89584e4`, and the
reconstructed final damage residual infinity norm is `0.06084`.

Plots:

- `/tmp/opencode/fracture2d_ufinal_raw_overview.png`;
- `/tmp/opencode/fracture2d_ufinal_raw_damage_evolution.png`.

These fields do not support a claim of clean notch-driven crack growth. Exact
resume point: investigate truncated-support/top-boundary damage, hard-loading
degradation, healing, and fixed-operator residual behavior before runner
refactoring. Do not validate the raw archive, hide healing, or interpret the
post-peak drop as approved fracture propagation.

Direct comparison with
`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz`
uses identical particle coordinates and equivalent triangle connectivity. Up to
`u=0.0055`, nonlocal/FEM reaction ratios remain `0.961`--`0.968`, and both
damage maxima remain at the notch tip (`0.5424` nonlocal versus `0.5054` FEM at
`0.0055`). At `0.0056`, the paths diverge: nonlocal top damage reaches `0.9707`
on 13 top particles above `0.5`, while FEM top damage remains below `0.0801`
and its maximum remains `0.5505` at the notch tip. At `0.0057`, nonlocal
reaction has fallen to `236.9` versus FEM `642.5`; by `0.0058`, FEM damage grows
at the notch while nonlocal damage is dominated by the top boundary.

The mechanics hard constraints match exactly in both models: bottom `uy=0`,
top `uy=u_bar`, and lower-left point `ux=0`; all other horizontal top/bottom
DOFs are free. Damage is not hard-constrained in either model. FEM obtains
homogeneous natural zero flux from its weak form and empty damage-BC list. The
nonlocal damage solve includes every particle with its material-only truncated
graph family; it has no exterior collar or missing-volume correction.

The strongest current causal evidence is mechanics/history concentration rather
than a wrong damage solve. Nonlocal top history rises from maximum `76.1` at
`0.0055` to `3875.5` at `0.0056` and `28958.4` at `0.0075`; FEM top-adjacent
history stays about `4.8`--`6.0`, while its large history remains notch-driven.
An independent exact fixed-history solve with the final nonlocal history
reproduces top `d_max=0.9974` and tip `d=0.5872`; a uniform-history nonlocal
solve is constant to `2e-15` despite boundary truncation. Therefore damage
truncation alone does not explain the top crack, although its top family mass is
about half the deep-interior mean and may affect boundary smoothing.

Exact resume point: freeze the last pre-transition state and separate the
feedback loop with three diagnostics: mechanics/Amor at the next load before a
damage update; nonlocal damage solves driven by FEM versus nonlocal history;
and top/tip support, strain, energy, and history comparisons. Do not change
hard masks or impose a damage Dirichlet condition before this discrimination.

The frozen `0.0055 -> 0.0056` diagnostic is complete. It freezes the raw
committed damage/history, performs one authoritative NumPy/SciPy
mechanics-before solve at `0.0056`, computes Amor `H_used` before any damage
update, maps the FEM DG0 history to particles by area mass lumping with exact
mass/volume checks, and solves the identical `d048` fixed-history damage system
for both histories. No repository physics, graph, or BC was changed.

The mechanics-before solve is healthy: reaction `617.7818` versus FEM
`637.5547`, free residual `2.494e-4` below its `2.949e-4` CG limit, and force/
moment defects near `1e-13`. Its top `H_used` maximum is only `78.90`, versus
mapped FEM `5.07`; the raw warning-committed `0.0056` state instead stores top
history `3875.50`. The exact damage solve driven by mechanics-before `H_used`
gives top `d_max=0.4145` and tip `d=0.5523`. The same operator driven by mapped
FEM history gives top `d_max=0.08116`, essentially FEM's `0.08008`, while tip
damage is `0.7186` versus FEM `0.5505`.

Both fixed-history solves have strong residual infinity norms below `7e-13`.
The raw warning target has residual infinity norm `142.64`. Therefore neither
the initial hard mechanics solve nor the fixed damage operator directly creates
the top `d=0.9707` state. The runaway develops across the warning attempt's
damage-degraded mechanics sweeps and is committed after exhausting the budget
without convergence. Top truncation remains a possible amplification mechanism:
its median BA support is 8 versus 13--14 at the tip/front, and median damage
degree is 11 versus 16/31, but all shape tensors remain rank two with modest
median condition `1.62`.

Artifacts:

- `/tmp/opencode/fracture2d_fixed_history_transfer_summary.json`;
- `/tmp/opencode/fracture2d_fixed_history_transfer_fields.png`;
- `/tmp/opencode/fracture2d_fixed_history_transfer_profiles.png`;
- `/tmp/opencode/fracture2d_fixed_history_transfer_metrics.png`;
- `/tmp/opencode/fracture2d_fixed_history_transfer_diagnostic.py`.

Exact resume point: instrument the exact 20 sweeps of the single
`0.0055 -> 0.0056` warning attempt and record top/tip damage, mechanics strain,
Amor driver, `H_used/H_next`, reaction, and residual at every sweep. This is
required to locate the feedback onset before testing any alternate nonlocal
boundary realization or refactoring the runner.

The exact 20-sweep trace is complete and reproduces the raw warning endpoint:
reaction differs by `2.10e-8` relative, damage by `8.39e-8` in infinity norm,
and history by `2.90e-4` in infinity norm. The mechanics, force, moment, range,
and history-monotonicity checks remain valid, but displacement, damage,
history, reaction, and `H_next` damage-residual gates fail in every sweep;
healing also fails from sweep 7 onward.

The feedback onset is ordered rather than instantaneous. The top mechanics-
after principal tensile strain and `H_next` first exceed the notch-tip values in
sweep 8, top mechanics-before driver and `H_used` exceed them in sweep 9, and
top candidate damage exceeds tip damage in sweep 10. From sweep 10 to 20, top
`H_used` grows from `161.63` to `3298.87` (`20.41x`), while tip `H_used` grows
only from `132.62` to `139.57` (`1.052x`). Reaction falls from `607.71` after
sweep 10 to `504.88` after sweep 20. The final top/tip candidate-damage ratio is
`1.648`, top/tip `H_used` ratio is `23.635`, and top/tip after-mechanics
principal tensile strain ratio is `4.909`.

Every fixed-history solve remains internally exact: the maximum strong residual
with its own `H_used` is about `3.64e-12`. The gate intentionally evaluates the
same damage candidate with mechanics-after `H_next`; that residual grows from
`4.668` in sweep 1 to `142.644` in sweep 20. The warning budget therefore does
not approximate a converging staggered fixed point at this transition. It
commits an accelerating top-boundary feedback state after bypassing six gates.

Trace artifacts:

- `/tmp/opencode/fracture2d_warning_transition_sweeps.json`;
- `/tmp/opencode/fracture2d_warning_transition_sweeps.csv`;
- `/tmp/opencode/fracture2d_warning_transition_compact.csv`;
- `/tmp/opencode/fracture2d_warning_transition_trace.png`;
- `/tmp/opencode/fracture2d_warning_transition_strain.png`;
- `/tmp/opencode/fracture2d_warning_transition_gates.png`;
- `/tmp/opencode/fracture2d_warning_transition_trace.py`.

Verification for the external diagnostic: 76 focused tests passed with 2 GPU
skips, artifact audit passed, and `git diff --check` passed.

Exact resume point: the implementation diagnosis is complete. Before further
model changes, select a literature-backed nonlocal Dirichlet/free-surface
realization to test against the current particle-only hard-boundary treatment.
Candidate studies may include a volumetric constraint/collar or an explicitly
derived support correction, but no choice is authorized yet. Do not merely
increase stagger sweeps, impose top `d=0`, or accept the warning curve.

A source audit found no published correction that can be inserted directly
into the current complete project-owned BA energy/residual. Chen--Spencer
defines the bond-associated kinematics and shape tensor but not this
target-inclusive discrete energy with hard particle-only boundary constraints.
Behzadinasab--Foster--Bazilevs is the closest 2D correspondence alternative:
it uses separate kinematic and stress-divergence neighborhoods plus an exterior
fictitious layer carrying prescribed displacement or zero boundary stress.
However, it is an RK/GMLS strong-form formulation; adopting it requires a new
residual, energy/work audit, tangent, and reaction definition rather than a
surface-factor patch.

Prudhomme--Diehl's odd collar and variable-horizon equations are explicit only
for a 1D linearized bond-based model. Scabbia--Zaccariotto--Galvanetto's Taylor
ghost nodes and partial-volume quadrature are for ordinary state-based
peridynamics. Neither source directly justifies transplanting those equations
into the current 2D BA energy. The currently accessible sources also do not
justify an extra scalar/tensor surface correction beyond the existing BA shape
tensor. Mechanics free-surface treatment must remain distinct from the AT2
damage natural-boundary treatment.

Exact resume point: user/model decision required. Either (a) implement a
separate mechanics-only Behzadinasab-style counterfactual as a new operator and
freeze damage/history at the transition, with full energy/reaction validation;
or (b) stop this BA branch at its validated pre-transition range. Do not present
the counterfactual as a correction to the current operator.

The remaining archive replay failure has been corrected without changing any
physics or acceptance tolerance. Strong AT2 residual reconstruction is
cancellation-prone, so comparing stored and independently rebuilt residuals at a
unit output scale rejected a real `1.4210854715202004e-13` float64 ordering
defect. `_require_residual_reconstruction_close` now uses the componentwise
roundoff bound

$$
512\epsilon_{64}\left(
\left|\frac{G_c}{\ell_0}d\right|
+\left|G_c\ell_0D_\delta d\right|
+|2Hd|+|-2H|
\right).
$$

The bound is applied only to redundant committed/fixed strong and
volume-residual reconstruction. Generic mechanics replay, gate tolerances,
dtype/shape/finite checks, and meaningful mismatch rejection are unchanged. A
regression reaches the actual `ContinuationResult` replay path with the observed
defect and rejects a perturbation ten times the local term-scaled bound. Focused
continuation verification passes 58 tests; the broader delegated focused run
passes 65 tests with 2 GPU skips; compileall and `git diff --check` pass.

The requested Behzadinasab Part I/II implementation design was then audited and
is a no-go under the repository no-guessed-math rule. The sources do not define
all data required by this exact BVP: same-node component-wise mixed conditions
(`u_y` essential while `u_x` is free/natural), the lower-left point anchor in a
collar, unconstrained displacement components through an essential collar,
phase-field degradation on fictitious nodes/bonds, a conservative global
discrete energy or symmetric tangent, a work-conjugate support reaction, or
exact rounded-slot collar coordinates/volumes/visibility. The papers' examples
use full benchmark collar data and centroid/element-area particles, unlike the
current Gmsh-vertex/barycentric-volume cloud.

Exact resume point: do not implement the fictitious-layer counterfactual until
those modeling choices are explicitly derived and approved. The safe next
options are to stop this BA branch at its validated pre-transition range, or
open a separate formulation task whose first acceptance target is an
independently derived mixed-component collar weak/strong form and patch test.
The invalid post-transition warning curve should not be rerun merely to produce
a formally valid archive.

### Stage 6: Refinement and handoff gate

- At fixed `delta`, refine particle spacing and report operator, reaction, and
  field convergence.
- Separately reduce `delta` while maintaining adequate `delta/h`; compare the
  local-limit trend with FEM without claiming pointwise agreement at free
  surfaces.
- Keep the selected boundary-particle masks fixed; vary visibility only in a
  declared sensitivity study and vary any selected stabilization parameter
  according to the frozen study plan. Do not introduce a collar implicitly.
- Start nonlocal PINN work only if the direct baseline has stable fixed
  subproblems, traceable conservation/energy diagnostics, and reproducible
  archive/refinement results.

## Planned file layout

- `examples/fracture2d/nonlocal_baseline.py`: direct solver and archive/plots.
- Temporary pilot values belong outside the repository; no production
  `configs/nonlocal_baseline.py` is created by Stage 5.
- `src/pinn/operators.py`: only general weighted/geometry-aware operator
  capabilities that the later PINN should share.
- `tests/test_fracture2d_bond_associated_algebra.py`: self-contained corrected-support and
  boundary-particle hard-constraint float64 algebra regression check; it is not
  a production solver.
- Focused tests named in Stages 1-5, including
  `tests/test_fracture2d_nonlocal_continuation.py`.
- `examples/fracture2d/README.md`: frozen formulation, usage, and validation
  contract after the model decisions are resolved.

## Verification commands

Use `.venv` only for direct nonlocal/JAX work:

```bash
source .venv/bin/activate
python -m unittest \
  tests/test_sparsed_fracture.py \
  tests/test_elasticd.py \
  tests/test_nonlocal_baseline.py \
  tests/test_fracture_losses.py \
  tests/test_fracture2d_geometry.py \
  tests/test_fracture2d_bond_associated_algebra.py \
  tests/test_fracture2d_nonlocal_geometry.py \
  tests/test_scalar_operator_2d.py \
  tests/test_fracture2d_nonlocal_mechanics.py \
  tests/test_fracture2d_nonlocal_mechanics_jax.py \
  tests/test_fracture2d_nonlocal_baseline.py \
  tests/test_fracture2d_nonlocal_continuation.py
python src/pinn/operators.py
python benchmarks/verify_equivalence.py
```

Add and run the new focused suites stage by stage; do not wait for the coupled
solver to test them. Before completion run the full repository suite,
`python -m compileall` on changed modules, and `git diff --check`.

## Status

Planning and the repository-local and external primary-reference reviews are
complete. The stable-correspondence/nonlocal-AT2 route is selected. The Stage 0
BA target-endpoint omission is corrected from Chen-Spencer's bond-associated
support construction and passes the persistent float64 rank, target-sensitivity,
patch, rigid-mode, energy-gradient, JVP, symmetry, and spectrum checks on the
regular manufactured cloud. The boundary BVP is now frozen as material-only
boundary-particle hard constraints with direct DOF elimination/affine lifting;
the exterior fictitious collar and interior material grip remain historical
rejected alternatives. Sol has reviewed and approved this Stage 0 specification
to proceed to Stage 1 geometry/quadrature implementation. This approval does not
replace the required rounded-slot, nonuniform-volume, boundary-mask, and finite-
cloud spectrum checks. Stage 1 geometry/quadrature infrastructure is now
implemented. Stages 2--4 now provide the weighted scalar operator, intact and
fixed-damage mechanics, fixed-history damage, and Amor APIs. Stage 5 strict
continuation is implemented, passes the manufactured-cloud controller suite,
and has independent implementation-review approval. Its rounded-slot
pre-localization pilot is approved. CPU warm starts, vectorized block-Jacobi,
and the CPU-replayed JAX Stage 5 adapter are validated. Its full rounded-slot
pre-localization rerun is independently approved, but peak/crack growth remains
unrun and no production horizon has been selected.
The optional local-FEM monolithic/quasi-Newton review was completed and
deferred.

## Repository-local reference review

The supplied papers support three distinct nonlocal routes. They must not be
combined as though they were one discretization:

1. Haghighat et al. (2021) uses PDDO moment-fitted derivative formulas. Section
   3, Eqs. (3.1)-(3.6), and Appendix Eqs. (A.1)-(A.7) define point-specific
   first- and second-derivative weights for arbitrary families. This supports a
   strong-form derivative-approximation route and boundary-specific moment
   fitting, but it supplies no phase-field evolution, nonlocal energy pairing,
   correspondence mechanics, stabilization, or volume-constraint treatment.
2. Yu and Zhou (2023) uses ordinary state-based peridynamics, not
   correspondence mechanics. Section 2, Eqs. (1)-(9), defines the OSB-PD force
   state and isotropic thermomechanical strain energy. Section 3,
   Eqs. (10)-(19), derives total-potential-energy minimization. Section 4,
   Eqs. (23)-(30), uses one-point quadrature, per-point volumes, surface and
   partial-volume corrections, and a bond coefficient that removes interactions
   crossing an initial sharp crack. It supports an OSB-PD sharp-crack route and
   traction-free natural behavior, but not diffuse AT2 evolution,
   damage-degraded correspondence, history, or correspondence stabilization.
3. Sinha et al. (2026) uses stochastic gradient estimates. Section 3.2,
   Eq. (13), and Section 4.2, Eqs. (20a)-(20b), define local moment-inverse
   estimates for both displacement and damage gradients. Section 3.1,
   Eqs. (7)-(12), puts those gradients into a phase-field energy with an
   irreversibility penalty. The implementation uses one Gauss point per
   triangular element, near-crack size `l/4`, coarse size up to `10*l`, and
   requires influence domains not to cross a crack, stated as `r < l` with
   truncation when needed. This is the only supplied complete pure-nonlocal
   phase-field route, but it is an SGE energy model, not correspondence
   peridynamics, and it uses direct damage irreversibility rather than the
   selected history-field hybrid model.

The local phase-field papers establish useful but local choices:

- Kristensen and Martinez-Paneda (2020), Section 2.1 Eq. (3), defines AT2;
  Section 3.1, Eqs. (13)-(17), defines history irreversibility, the Amor split,
  and the hybrid use of full elastic energy for mechanics; Section 3.2,
  Eqs. (18)-(25), gives the plane-strain FEM residuals and tangents. It does not
  define any nonlocal mechanics, collar, quadrature, visibility, or reaction.
- Martinez-Paneda and Hirshikesh (2020), Eqs. (1)-(10), documents the same
  local hybrid/history FEniCS formulation and natural zero-flux phase-field
  boundary condition. Its reaction is a local boundary traction integral and
  cannot be copied as the nonlocal constrained reaction without a derivation.
- Goswami et al. (2020), Section 2, Eqs. (1)-(13), uses a local spectral-split
  phase-field energy and history field; Section 4, Eqs. (29)-(32), uses
  elementwise Gauss-Legendre quadrature and incremental transfer learning. It
  supports energy quadrature and load-history concepts, but not a nonlocal
  operator or correspondence model.
- Manav et al. (2024), Section 2, Eqs. (1)-(18), gives local AT1/AT2 energy and
  direct irreversibility/penalty formulations. Section 3.2.3, Eq. (25), uses
  FEM shape-function gradients and elementwise Gauss quadrature after finding
  autodifferentiated gradients unreliable. It remains a local-gradient model
  and does not support the planned direct nonlocal operator.

Consequences for the pre-draft blocking model decisions:

- **Particle quadrature:** supplied support exists for explicit `V_j` and
  moment tensors, but not for the proposed Gmsh-nodal barycentric-lumped cloud.
  That quadrature remains an independently validated discretization choice.
- **Notch interactions:** material-domain truncation is common to all nonlocal
  routes. Explicit segment-crossing removal is directly supported only for a
  sharp crack in Yu and Zhou, while Sinha et al. requires influence domains not
  to cross an evolving diffuse crack. Neither result directly decides visibility
  around this task's finite geometric slot for correspondence mechanics.
- **Damage-aware mechanics:** no supplied paper derives degradation of a
  correspondence energy and the resulting pairwise/discrete force. Sinha et
  al. supplies a complete alternative based on degradation of an SGE strain
  energy; adopting it would change the selected mechanics route.
- **Correspondence stability:** no supplied primary source defines an
  hourglass/zero-energy stabilization for the planned damage-aware
  correspondence model. Yu and Zhou uses OSB-PD, and Sinha et al. uses SGE, so
  neither can justify a correspondence penalty.
- **Nonlocal Dirichlet treatment:** Yu and Zhou uses fictitious boundary layers
  and boundary penalties, whereas Sinha et al. uses a hard neural ansatz.
  Neither defines top/bottom volume constraints for a direct particle solve or
  the task's point-like horizontal rigid-mode constraint.
- **Reaction:** none of the supplied nonlocal papers defines the
  work-conjugate reaction for displacement-controlled volume constraints. It
  must be derived from the selected constrained discrete energy and checked by
  an energy derivative.
- **History:** the local hybrid references support a maximum positive-energy
  history, while the complete supplied nonlocal SGE model instead uses an
  irreversibility penalty. A particle history projection remains unsupported.

The supplied collection does not contain one primary source covering the
planned damage-aware correspondence route. The Stage 0 draft below therefore
uses separate, explicitly labeled foundations: Chen-Spencer bond-associated
kinematics, a project-owned energy partition and degradation, the local
Amor-history AT2 law, and a project-derived constrained reaction. None of these
project-owned combinations is treated as a published equation.

## External primary-reference review

The external search separates the intact variational correspondence basis,
stable correspondence variants, nonlocal boundary methods, and published
phase-field/peridynamic fracture models. No source covers all four in the form
required by this task.

### Intact correspondence energy and force

- Silling, Epton, Weckner, Xu, and Askari (2007), "Peridynamic States and
  Constitutive Modeling," DOI `10.1007/s10659-007-9125-1`, is the foundational
  non-ordinary state-based correspondence source. Section 7, Eq. (28), gives
  the two-endpoint state-based equation of motion. Section 10, Eq. (61), defines
  an elastic material by the Frechet derivative `T=grad W`; Eqs. (63)-(68)
  establish the associated energy balance. Section 18, Eqs. (134)-(142), maps a
  deformation state to an approximate deformation gradient, evaluates a
  classical hyperelastic energy/stress, and derives the force state from that
  energy. Eqs. (145)-(146) treat damage by excluding binary broken bonds from
  the influence function. The paper therefore supports intact variational
  correspondence and endpoint assembly, but not diffuse phase-field
  degradation.
- A direct variational extension that preserves the local comparator would
  start from a discrete energy of the form

$$
E_h(u,d)=\sum_i V_i g(d_i)\,\psi(F_i(u)),
$$

  and define every unconstrained force by `f_i=-partial E_h/partial u_i`.
  Variation then includes the contributions from energies centered at both
  endpoints; it does not justify inserting an independently chosen arithmetic
  average of `g(d_i)` and `g(d_j)` into a pair force. This extension is an
  independently derived project model, not an equation supplied by Silling or
  by a located phase-field/peridynamic paper. The Stage 0 draft now writes out
  the discrete derivative, momentum balance, stiffness symmetry, and finite-
  difference checks; implementation remains pending those checks and Sol review.

### Correspondence stability

- Silling (2017), "Stability of peridynamic correspondence material models and
  their particle discretizations," DOI `10.1016/j.cma.2017.03.043`, establishes
  that conventional correspondence and its particle discretization admit
  zero-energy modes and develops a stability criterion. It diagnoses the
  problem; it does not supply the selected phase-field coupling.
- Chen and Spencer (2019), "Peridynamic Bond-Associated Correspondence Model:
  Stability and Convergence Properties," DOI `10.1002/nme.5973`, is the closest
  accessible energy-based stable alternative. Section 3.1, Eqs. (16)-(17),
  defines bond-associated deformation gradients and shape tensors. Section
  3.2, Eqs. (18)-(22), distributes strain energy to bond-associated
  neighborhoods and derives the corresponding force state; Eq. (23) uses the
  two-endpoint state assembly. Section 4, Eqs. (24)-(28), shows why the
  conventional symmetric mode has zero energy and the bond-associated model
  does not. It contains no diffuse phase-field degradation, and Eq. (19) calls
  its volume-fraction energy distribution an approximation.
- Behzadinasab, Trask, and Bazilevs, Part I, DOI
  `10.1007/s42102-020-00040-z`, shows in Section 2 that higher-order RK/GMLS
  quadrature alone does not remove instability. Section 2.4, Eqs. (13)-(18),
  defines a parameter-free bond-associated stress-divergence correction. The
  numerical studies show robust convergence on nonuniform clouds and free
  surfaces. This is strong-form support for a stable operator, not a derivation
  of one global damage-degraded discrete energy.
- Stabilized penalty-force models such as Li, Hao, and Zhen (2018), DOI
  `10.1016/j.cma.2018.05.002`, remain alternatives but introduce a model and
  parameter that must be calibrated before fracture. They are not preferred
  over a parameter-free bond-associated route for this baseline.

### Published phase-field/peridynamic models

- Roy, Pathrikar, Deepu, and Roy (2017), "Peridynamics damage model through
  phase field theory," DOI `10.1016/j.ijmecsci.2017.04.016`, follows a
  constitutive-correspondence route and derives a dissipative Ginzburg-Landau
  phase evolution from Hamilton's principle. Its abstract explicitly states
  that the phase field supplies a bond-breaking criterion and allows the body
  to physically separate. It is therefore a regularized-damage-to-discrete-
  fracture model, not the frozen history-driven diffuse AT2 comparator.
- Bie, Ren, Yan, and Chen (2023), "The unified nonlocal peridynamics-based
  phase-field damage theory," DOI `10.1016/j.tafmec.2023.103980`, is the closest
  published combined model found. The full text is closed, so no equation from
  it is treated as independently verified here. The source's reference list and
  Dorduncu et al. (2024), DOI `10.1016/j.compstruc.2024.107395`, Section 6.9,
  report that it introduces a new deformation gradient, shape matrix, and force
  state to remove the correspondence zero-energy mode, then derives a nonlocal
  phase-field damage constitutive model and rational bond-breaking criterion.
  Adopting it would change both the correspondence operator and the fracture
  semantics; it cannot be cited for an unchanged `g(d)`-degraded hybrid AT2
  energy without inspecting and selecting its full formulation.
- Ren, Rabczuk, and Zhuang (2025), "Variational damage model: A new paradigm
  for fractures," DOI `10.1007/s11709-025-1144-0`, further confirms that this
  research line is not AT2. Section 4, Eqs. (39)-(52), eliminates an independent
  damage solve, omits the damage-gradient term, and makes damage an explicit
  function of the maximum positive energy. It is a useful history-based
  alternative but a different regularization and fracture energy.

### Boundary constraints and reaction

- Behzadinasab, Foster, and Bazilevs, Part II, DOI
  `10.1007/s42102-020-00039-6`, applies essential data on a fictitious layer of
  thickness equal to the horizon. Section 4, Eqs. (8)-(9), uses separate
  kinematic and stress-divergence neighborhoods so natural-boundary nodes do
  not enter deformation-gradient evaluation. This directly supports a
  horizon-thick collar and a correspondence boundary patch-test program, but it
  does not prescribe this task's top/bottom collar partition or point-like
  horizontal anchor.
- Zhou and Du (2010), DOI `10.1137/090781267`, and Mengesha and Du (2014), DOI
  `10.1017/S0308210512001436`, establish well-posed nonlocal displacement and
  Dirichlet-type volume constraints for linear bond-based models. D'Elia, Tian,
  and Yu (2020), DOI `10.1137/19M1266617`, Eqs. (6)-(16), formalizes Dirichlet
  and Neumann volume constraints for nonlocal diffusion and converts local
  surface data through a local surrogate. These references support the volume-
  constraint concept, but none directly derives the exact 2D stable-
  correspondence collars required here.
- Prudhomme and Diehl (2020), DOI `10.1016/j.cma.2020.113391`, analyzes an
  extended horizon-thick layer and a variable-horizon method for a 1D
  bond-based bar. Section 7 explicitly leaves extension to 2D/3D and
  state-based models as future work, so it cannot be used as direct validation
  of this task's boundary treatment.
- No located paper defines the work-conjugate reaction for this task's
  displacement-controlled particle constraints. For a selected discrete energy
  and an affine constrained displacement parameterization, the reaction must be
  derived as the derivative of the reduced energy with respect to imposed top
  displacement, then checked against the signed sum of constrained residual
  forces and an external-work finite difference. This remains an independent
  discrete derivation.

### Recommendation and rejected combinations

The best route for preserving comparability with the local FEM is not to adopt
Roy or Bie. It is to write and independently check a project-owned discrete
variational model that combines the frozen Amor-history AT2 energy with a
parameter-free bond-associated correspondence energy, and obtains both forces
and reaction only by differentiation of that same discrete energy. The Stage 0
draft below resolves the proposed energy partition, damage placement, force,
reaction, and validation conditions. The coupling remains novel and is pending
independent implementation checks and Sol review.

If using one published combined peridynamic fracture model is more important
than retaining the local comparator's physics, Bie et al. is the leading source,
subject to obtaining and reviewing its full text. That selection must explicitly
accept its modified correspondence kinematics and phase-field-driven bond
failure. Roy is a less suitable dynamic, dissipative predecessor. The supplied
SGE formulation remains the most complete already-available diffuse nonlocal
energy route, but it also changes the mechanics and irreversibility model.

The following combinations are rejected:

- conventional Silling correspondence plus an ad-hoc averaged endpoint damage
  factor, because no located source or checked energy variation supports it and
  conventional correspondence retains zero modes;
- a bond-associated strong-form divergence combined with an unrelated nodal
  energy, because force/energy consistency would be lost;
- Roy/Bie bond failure combined silently with diffuse AT2 degradation, because
  that double-counts damage and changes complete-separation behavior;
- direct reuse of bond-based or scalar-diffusion boundary formulas as proof for
  a 2D non-ordinary correspondence collar.

## Full-text Bie PD-PFM review

The full paper is now available at `refs/Bie-PD-PFM-fracture.pdf`:
Y. Bie, H. Ren, H. Yan, and J. Chen, "The unified nonlocal
peridynamics-based phase-field damage theory," *Theoretical and Applied
Fracture Mechanics* 126 (2023) 103980, DOI
`10.1016/j.tafmec.2023.103980`.

It is a genuine variational nonlocal phase-field/peridynamic model, but not a
drop-in nonlocal form of the current hybrid AT2 reference:

- Eqs. (11)-(13) add residual energy for the non-affine state
  `Z=Y-F_tilde X`. Eqs. (22)-(23) define a modified fitted 2D strain measure and
  shape matrix. Eqs. (34)-(35) derive a parameter-free residual force intended
  to suppress conventional correspondence zero modes.
- Eqs. (46)-(50) use Wu cohesive crack density with
  `alpha(d)=2d-d^2`, a nonlocal phase-gradient fit, and a second residual
  energy. This is not AT2.
- Eqs. (70)-(77) use rational degradation, tensile strength, a selected
  softening law, and Rankine or modified-von-Mises effective stress. Eq. (102)
  uses history threshold `ft^2/(2E)`. These differ from the current quadratic
  degradation and Amor positive-energy history.
- Eqs. (81)-(110) and Algorithms 1-2 provide endpoint residuals,
  double-state tangents, dual-horizon particle assembly, and coupled BFGS. This
  is the most detailed published direct-solver route among the candidates.
- Quadrilateral cells become center particles with explicit cell volumes. The
  examples generally use `delta=2h` and about 16 family points.
- All fracture examples and printed 2D coefficients are plane stress, not the
  current plane strain problem.

The full text leaves critical implementation ambiguities:

- Eq. (46), Appendix Eq. (A.3), and Eqs. (60), (64), and (98) do not give one
  unambiguous coefficient/derivative chain for the phase-gradient energy.
- The displayed constitutive model continuously degrades the center-point force
  state through Eq. (73), `T_tilde=omega(d)T`. Despite prose about bond
  breakage, no binary bond variable, deletion threshold, endpoint rule, or
  post-deletion shape-matrix update is printed.
- The claim that only tensile stress is degraded conflicts with scalar
  degradation of the complete printed force state; tension is explicit in the
  damage driver, not in a split force equation.
- Shape-matrix conditioning, nonlocal collars, phase natural boundaries,
  rounded-slot visibility, reaction, convergence tolerances, failed-step
  history rollback, and damage bounds are not specified.

Bie is therefore viable only as a separately named cohesive PD-PFM after
resolving the phase-flow ambiguity. A plane-strain rounded-slot implementation
would require additional derivation and cannot be described as discretization
error relative to the current hybrid AT2 FEM.

## SGE model clarification

The SGE source is R. Sinha, K. H. Sudeep, A. Kaithavalappil, and S. Sarkar,
"Fracture analysis of nonlocally homogenised solids via stochastic gradient
estimates embedded in physics-informed neural networks," *Journal of the
Mechanics and Physics of Solids* 208 (2026) 106458, DOI
`10.1016/j.jmps.2025.106458`.

SGE means **stochastic gradient estimator**, not stochastic-gradient
optimization and not strain-gradient elasticity. Section 3.2, Eq. (13), uses a
centered neighborhood regression:

$$
\widehat G(f,X)=
\left[\int_{\Omega_X}(f-\bar f)(Y-\bar Y)^T\,dV\right]
\left[\int_{\Omega_X}(Y-\bar Y)(Y-\bar Y)^T\,dV\right]^{-1}.
$$

At fixed finite radius `r`, strain and crack-gradient energy at `X` depend on
values throughout `Omega_X`, so the model is genuinely finite-range nonlocal.
It is not peridynamics: it has no deformation state, force state, bond energy,
or endpoint force balance.

It is genuinely phase-field fracture. The paper inserts SGE estimates of both
`grad(u)` and `grad(d)` into a diffuse Griffith energy with AT1 or AT2. It
degrades positive split energy while retaining compressive volumetric energy,
and imposes irreversibility with a penalty on healing. These choices differ
from the current hybrid full-stress/history AT2 model.

An energy-consistent direct implementation would require the discrete adjoint
`G_r^*`:

$$
E_r^*\sigma_r=0,
$$

$$
\omega'(d)\psi_+
+\frac{G_c}{c_w\ell}\alpha'(d)
+\frac{2G_c\ell}{c_w}G_r^*G_r d
+\gamma\langle d-d_{n-1}\rangle_-=0.
$$

The paper does not print or assemble this direct adjoint system. Its residual
diagnostics are local-looking and do not fully match the nonlocal penalized
energy. Quadrature, moment conditioning, evolving crack-side visibility,
direct constraints, and reaction are also incomplete. SGE is consequently the
closest route to a plane-strain nonlocal AT1/AT2 PINN, but it is neither a ready
direct solver nor correspondence peridynamics.

## Updated route comparison

| Criterion | Project hybrid AT2 | Bie PD-PFM | SGE energy |
|---|---|---|---|
| FEM physics fidelity | Highest | Low | Medium |
| Published end-to-end model | No; coupling is new | Strongest | Energy/PINN only |
| Nonlocality | Stable correspondence + nonlocal AT2 | Unified PD correspondence | Moment-fit gradient |
| Plane strain | Can be derived | Not supplied | Used in paper |
| Fracture law | Hybrid Amor-history AT2 | Cohesive strength/softening | Split AT1/AT2 + penalty |
| Zero-mode treatment | BA support; finite-cloud proof pending | Explicit residual energy | Not analyzed |
| Direct assembly | Project-derived energy gradient | Most detailed | Adjoint system missing |
| Later PINN suitability | Research-heavy | Possible | Best demonstrated |
| Main risk | Novel damage coupling and graph conditioning | Formula/scope mismatch | Direct-adjoint ambiguity |

## Review Gate

Sol review approves the Stage 0 equations and the boundary-particle hard-
constraint contract for Stage 1. The BA target-endpoint support, independent
explicit residual, free-block SPD, free residual, top/bottom/anchor reactions,
and reduced-energy work checks pass on the regular cloud. Stage 1 must preserve
Gmsh physical tags or construct deterministic masks from explicit boundary
entities, then validate barycentric volumes, segment visibility, BA rank/
conditioning, and the constrained finite-cloud spectrum on the rounded-slot
cloud. Failure of any check reopens Stage 0 before mechanics implementation.

## Status

Stage 1 geometry/quadrature infrastructure is implemented and the production
visibility gate now uses the polygonized Gmsh material domain. The
`examples/fracture2d/nonlocal_geometry.py` module is independent of FEniCS/local
FEM and stops at particle data, quadrature, boundary masks, and the reciprocal
visible graph. `mesh.geo` declares the unique lower-left anchor as a physical
point. The production loader requires all declared Gmsh physical names, exact
physical-line coverage of the triangle boundary, and one open
`notch_upper`/`notch_tip`/`notch_lower` chain; it does not infer masks from
coordinate extrema or accept the analytic circle as a visibility fallback.
Focused synthetic tests cover tags, volumes/moments, deterministic ordering,
polygon crossing/tangent/closure visibility, retained tip chords and material
boundaries, reciprocity, invalid tags, boundary coverage, internal lines, and
duplicate/inconsistent tags. The final-audit regressions additionally cover the
zero-input effective machine floor on a 20-chord polygon, the shallow-crossing
reproducer (explicit ambiguity rather than `True`), endpoint reversal, and
scaled-copy invariance.

## Superseded analytic-circle Stage 1 smoke results — 2026-08-18

The following diagnostic was produced before Sol froze mesh-conforming
visibility. Its counts are retained only as history and must not be used as
production graph metrics.

The default rounded-slot geometry was regenerated outside the accepted FEM
archive with `gmsh` from `fenicsenv` into
`/tmp/opencode/rounded_slot_stage1.msh`. The Stage 1 loader was run in the same
environment with the explicit diagnostic settings
`horizon=0.05` and `visibility_tolerance=1e-12`; the horizon is not a frozen
production mechanics/damage choice.

- vertices: `3395`; triangles: `6284`;
- barycentric volume sum: `0.9762509374813683` (difference
  `1.86e-11` from the frozen measured value `0.9762509375`);
- first moments: `(0.4941241109732570, 0.4881254687406841)`;
- bottom/top particle mask counts: `51` / `51`; unique anchor particle:
  `0` in deterministic particle order;
- candidate graph: `39233` undirected pairs / `78466` directed edges;
  visibility removed `1493` pairs / `2986` directed edges; retained
  `37740` pairs / `75480` directed edges;
- retained visible neighbor count min/mean/max:
  `7 / 22.232695139911634 / 56`.

No local FEM archive or solver mesh pipeline was modified. The first moments
are recorded as mesh-quadrature diagnostics; the exact continuum slot area is
not substituted for the measured Gmsh triangle area. The unchanged
triangle-only meshio→XDMF conversion also completed successfully with `6284`
triangle cells, preserving local FEM compatibility while keeping Stage 1 on
the original tagged `.msh` path.

## Stage 1 mesh-conforming implementation and smoke — 2026-08-18

The production graph now constructs one ordered slot polygon from the tagged
`notch_upper`, `notch_tip`, and `notch_lower` line-cell chain, adds the absent
left opening closure, and classifies each candidate segment by strict open
polygon-interior subintervals. Tagged tip chords and slot-face edges remain in
the graph. A positive-overlap segment on the unmeshed opening closure is
removed. The analytic `RoundedSlot` argument and analytic production helper
path were removed from `build_particle_cloud` and `load_gmsh_particle_cloud`;
callers pass only the mesh, explicit horizon, boundary tags, and absolute
visibility tolerance.

The fresh tagged mesh was generated with `.venv/bin/gmsh` at
`/tmp/opencode/fracture2d_stage1_mesh_conforming.msh` and loaded in
`fenicsenv` with the diagnostic `horizon=0.05`; the horizon is not a production
choice. The public `ParticleCloud.visibility_tolerance` records the caller
input. The predicate uses the declared effective floor
`tau_eff=max(tau_input,64*eps*L_geom)`, with `L_geom` formed from polygon,
opening, and candidate-segment lengths; this floor is numerical, not physical.

- vertices: `3395`; triangles: `6284`;
- barycentric volume sum: `0.9762509374813683`; first moments:
  `(0.4941241109732570, 0.4881254687406841)`;
- bottom/top particle masks: `51` / `51`; unique anchor particle: `0`;
At both explicit input tolerances, the graph and tagged-edge counts were:

| input `visibility_tolerance` | recorded input | candidate pairs | removed pairs | retained pairs | visible slot cells (`upper/tip/lower`) |
|---:|---:|---:|---:|---:|---:|
| `0` | `0` | `39233` | `1455` | `37778` | `119/20/119` of `119/20/119` |
| `1e-12` | `1e-12` | `39233` | `1455` | `37778` | `119/20/119` of `119/20/119` |

The retained visible neighbor count min/mean/max is
`7 / 22.255081001472753 / 56` for both runs. The area, first moments, masks,
and anchor are unchanged from the values above.

The focused synthetic suite proves retention of a polygonal `notch_tip` cap
chord, slot faces, a tangent, and a material boundary; removal of an interior
crossing chord and the absent opening segment; and loud failure for missing
boundary coverage, internal physical lines, and duplicate/inconsistent tags.

Verification for this implementation passed:

- `source .venv/bin/activate && python -m unittest -v
  tests.test_fracture2d_bond_associated_algebra tests.test_fracture2d_nonlocal_geometry` —
  `12` tests passed;
- `source .venv/bin/activate && python -m unittest discover -s tests -p
  'test*.py' -v` — `77` tests passed;
- `source .venv/bin/activate && python -m compileall -q
  examples/fracture2d/nonlocal_geometry.py
  tests/test_fracture2d_nonlocal_geometry.py` — passed;
- `git diff --check` and untracked-file whitespace checks — passed.

The geometry implementation pass itself made no BA acceptance claim. The
independent validation below closes that remaining Stage 1 gate.

## Stage 1 BA and constrained-spectrum validation — 2026-08-18

An independent matrix-free validation regenerated the current tagged mesh and
used the final mesh-conforming graph. For every directed edge it constructed
the exact target-inclusive support from Section 0.3 and the declared
`J_delta(r) V_k` weights. Rank failure was diagnosed by
`s_min <= 100*eps_float64*s_max`; this is a numerical diagnostic, not a selected
physical threshold.

| `delta` | visible directed edges | support min/median/q95/max | rank failures | condition median/q95/max | affine max error |
|---:|---:|---:|---:|---:|---:|
| `0.032` | `30258` | `2 / 4 / 22 / 30` | `0` | `1.15417 / 4.01798 / 49.6235` | `7.33e-15` |
| `0.040` | `59024` | `3 / 9 / 28 / 39` | `0` | `1.27642 / 2.95465 / 79.5975` | `1.39e-14` |
| `0.048` | `72934` | `5 / 13 / 35 / 47` | `0` | `1.23037 / 2.92226 / 18.6509` | `3.69e-15` |

Every target endpoint weight and deformation-gradient sensitivity was formally
positive. Some edges lying within floating-point ulps of the strict horizon had
very small target terms; no kernel floor, edge deletion, or fallback was added.
This remains a cutoff-conditioning diagnostic for later horizon/refinement
studies.

At diagnostic `delta=0.048`, the intact plane-strain operator used `E=1`,
`nu=0.3`, `d=0`, and `k_res=1e-8`. The matrix-free Hessian action followed the
Section 0.5 energy derivative. Exact boundary elimination constrained 103 of
6790 DOFs, leaving 6687 free DOFs. Translation actions were exactly zero;
rotation energy was `9.62e-39` with action norm `1.08e-19`; five random
bilinear symmetry checks had maximum relative defect `2.14e-14`; an independent
endpoint loop agreed with the vectorized action to `5.21e-16`; and the energy
directional finite-difference relative error was `2.54e-13`.

Two `scipy.sparse.linalg.eigsh` settings were compared without forming the dense
global matrix. The stricter run used `tol=1e-11`, `ncv=80`, and `maxiter=6000`:

- constrained minimum eigenvalue: `2.8440031721752207e-05`, absolute Ritz
  residual `7.2202e-16`, residual divided by the eigenvalue `2.5388e-11`;
- constrained maximum eigenvalue: `1.3314397965182772`, absolute Ritz residual
  `1.1972e-13`, residual divided by the eigenvalue `8.9918e-14`;
- extremal condition estimate: `46815.69308869415`.

The looser independent run agreed to relative `3.03e-12` for the minimum and
`5.00e-16` for the maximum. The positive converged minimum closes the finite-
cloud constrained-spectrum gate for this diagnostic graph. It does not select
`delta=0.048` as the common mechanics/damage horizon.

## Next action

Begin Stage 2 with the explicit-volume weighted scalar operator. Verify its
constant nullspace, volume self-adjoint relation, non-negative energy,
fixed-history SPD system, SciPy export, and manufactured full-square refinement
before evaluating rounded-slot full/boundary-excluded errors. Keep every tested
horizon explicit and do not infer the production common horizon from the Stage
1 diagnostics.

## Session handoff — 2026-08-18

- Changed this task's Stage 0 equations from deficient common-neighbor support
  to target-inclusive support, propagated the endpoint weight through shape,
  moment, force, and tangent definitions, and added the source/ownership audit.
- Added an independent explicit mechanics residual path to
  `tests/test_fracture2d_bond_associated_algebra.py`: it recomputes BA \(\boldsymbol F\),
  strain, \(\boldsymbol P\), and endpoint assembly directly, then drives the
  energy-gradient FD and residual-based Hessian JVP checks. It also asserts
  residual/Hessian consistency, PSD tolerance, and the three-mode spectral gap.
- Froze Section 0.2 as the material-only boundary-particle hard-constraint BVP;
  added exact bottom/top/anchor masks, DOF elimination/affine lifting, free-block
  SPD, free residual, separate reactions, and reduced-energy reaction checks.
- Updated `docs/PROGRESS.md` and appended the support correction, explicit
  residual, and boundary hard-constraint decisions to `docs/DECISIONS.md`. No
  production solver or existing FEM code was changed.
- Verified with `source .venv/bin/activate && python
  tests/test_fracture2d_bond_associated_algebra.py`: all checks passed; the focused Stage 0
  unittest passed, full discovery passed 66 tests, the changed test compiled,
  and `git diff --check` passed.
- At that handoff, the remaining gate was Sol review of the exact Gmsh
  physical-tag/deterministic-mask preservation contract. That gate is addressed
  by the Stage 1 implementation and smoke results below; the current resume
  point is the rounded-slot BA support/spectrum check.

## Session handoff — 2026-08-18 Stage 1

- Added `examples/fracture2d/nonlocal_geometry.py` with explicit Gmsh physical
  tag parsing, deterministic particle/triangle ordering, barycentric nodal
  areas, first moments, reciprocal strict-horizon graph construction, and
analytic rounded-slot open-segment visibility. This visibility implementation
is superseded by the mesh-conforming decision below and must be replaced before
Stage 1 acceptance.
- Added the `anchor` physical point to `examples/fracture2d/mesh.geo` without
  changing the local FEM conversion or weak form.
- Added `tests/test_fracture2d_nonlocal_geometry.py`; the focused six-test
  suite passes in `.venv` without meshio or any personal output directory.
- Regenerated and smoke-tested the default mesh in `fenicsenv`; the metrics are
  recorded above. The module has not entered weighted scalar, mechanics, or
  coupled-fracture code.
- Verification: the focused geometry/original/Stage 0 command passed `12`
  tests; full unittest discovery passed `72` tests; `compileall` passed for the
  changed Python modules; and `git diff --check` passed.
- Exact resume point: review the tag/visibility contract and run the Stage 1
  bond-associated rank/spectrum checks before implementing any operator.

## Independent Stage 1 numerical exploration — 2026-08-18

This is a read-only Sol-reviewer exploration. It did not edit the nonlocal
module, tests, local FEM solver, or solver configuration. The primary mesh
snapshot was `/root/autodl-tmp/data/fracture2d/local_fem/rounded_slot.msh`,
which was generated before the working-tree `anchor` physical point was added;
it has the same 3395-node/6284-triangle geometry as a fresh temporary mesh.
The fresh temporary mesh `/tmp/opencode/fracture2d_with_anchor.msh` has the
additional `vertex` block and `field_data['anchor']=[9,0]`. The configured
archive `.msh` must therefore be regenerated before a loader that requires the
tagged vertex can consume it.

### Meshio and physical-group contract

For meshio 5.3.5, `field_data[name]` is `[physical_tag, topological_dimension]`;
`cell_data['gmsh:physical'][block]` aligns with `cells[block]`, while
`cell_data_dict['gmsh:physical'][cell_type]` concatenates the blocks of one
cell type. `gmsh:geometrical` retains the underlying curve/surface entity ID.
`cell_sets` entries are still block-local cell indices and are not particle-node
indices. The current fresh mesh has physical line blocks for bottom/right/top,
left-upper, notch-upper, the two notch-tip arcs, notch-lower, and left-lower;
the same physical notch-tip tag covers both arc blocks.

The original mesh had no physical point, so its lower-left anchor was only
recoverable as the unique intersection of the bottom and left-lower endpoint at
`(0,0)`. The fresh mesh's explicit point tag removes that ambiguity. The
left outer boundary remains the union of `left_upper` and `left_lower`, not one
physical group. Boundary masks must be made by selecting tagged line cells and
unique-ing their endpoints; shared corner nodes are expected to occur in two
masks. Do not assume cell-block ordering, curve orientation, or disjoint
masks. A generic importer must also fail or explicitly remap duplicate point
coordinates before volume/graph construction; this mesh has no exact or
12-decimal duplicate among its 3395 used points.

The local FEM `_generate_mesh` path intentionally writes an XDMF containing
only points and triangles, so it drops `field_data` and all physical tags. The
direct path must read the tagged `.msh` (or a separately preserved tag artifact),
not infer Stage 1 masks from that triangle-only XDMF.

### Quadrature and geometry numbers

With `V_i=(1/3)\sum_{e\ni i}A_e`, all 6284 triangle areas and all 3395 nodal
volumes were positive. The independent float/long-double accumulation gave

$$
\sum_i V_i=0.97625093748136825,
\qquad
\sum_iV_i\boldsymbol X_i
=(0.4941241109732579,\;0.4881254687406843),
$$

and centroid `(0.5061445700098887, 0.5000000000000002)`. The nodal-volume
range was `[2.971843321027019e-6, 4.728163472600397e-4]`; the median was
`3.464030180192565e-4`. The fresh tagged mesh reproduces these values.

The exact CAD slot has rectangle area `0.022848` and semicircle area
`0.0009047786842338604`, hence exact material area
`0.9762472213157661` and exact material first moments
`(0.4941222853463047, 0.4881236106578831)`. Gmsh has 20 straight chords on
the semicircular cap. Its polygon cap area is
`0.0009010625186317567`, giving the measured polygon material area
`0.9762509374813683`, exactly the project value `0.9762509375` to the
`1.86e-11` decimal-rounding difference. Therefore an area test against the
polygonized mesh can reasonably use `5e-12`–`1e-10` absolute tolerance,
depending on whether the expected value is computed or decimal-serialized;
the `3.716165602e-6` CAD-versus-polygon discrepancy must be reported as mesh
geometry error, not hidden in a test tolerance. First moments should be checked
against the same polygon geometry.

### Visibility degeneracies

The independent segment predicate partitions the open parameter interval at
line/circle intersections and tests strict interior on each open subinterval.
The expected results were: rectangle/cap crossing = removed; tangent to the
cap at the tip = retained; a segment on either straight slot face = retained;
an endpoint on a face followed into the void = removed; a chord near the tip
that enters the cap = removed. A tolerance should be explicit and scale as
`tau_geom=max(tau_input, C*eps*L_geom)` with a declared `C` and
`L_geom=max(Lx,Ly,delta,r)`, plus a parameter tolerance derived from
`tau_geom/|end-start|`. Strict open-interval tests are needed so a tangent or
face segment is not removed merely by a `<=` predicate; near-zero circle
discriminants should be classified as an explicit ambiguous case or evaluated
with an adaptive/extended-precision predicate.

There are two material-domain choices that must not be conflated:

1. The literal analytic `int(Omega_void)` rule retains a segment on the
   `x=0` opening line and uses the exact circle. It removes all 20 Gmsh cap
   boundary chords, because each straight chord lies infinitesimally inside
   the exact circle.
2. A polygonized-mesh/material-closure rule retains those 20 cap boundary
   edges. A conservative opening treatment can also remove a segment running
   through the absent `x=0` opening. The current working-tree analytic helper
   uses a positive clearance and the latter conservative opening behavior.

On the actual mesh, analytic-CAD visibility classified `20/20` notch-tip
boundary edges as crossing, while polygonized visibility classified `0/20` as
crossing; all 119 upper and 119 lower straight slot edges were retained by both.
This is a high-impact formulation/conditioning sensitivity, not a floating
point bug, and needs an explicit Sol decision before mechanics.

### Strict-horizon graph study

The following uses the current analytic helper's conservative clearance,
strict `r<delta`, and the fresh tagged mesh. `h=0.02` is only the median unique
triangle-edge scale (the nearest-neighbor median is `0.019944701821247655`);
near the slot the nearest-neighbor scale is `0.004`. No horizon is frozen.

| `delta` | `delta/h` | candidate pairs | removed pairs | retained pairs | family min/mean/max |
|---:|---:|---:|---:|---:|---:|
| 0.020 | 1.0 | 7,910 | 126 | 7,784 | 0 / 4.586 / 18 |
| 0.024 | 1.2 | 12,630 | 166 | 12,464 | 3 / 7.343 / 25 |
| 0.032 | 1.6 | 15,372 | 272 | 15,100 | 3 / 8.895 / 32 |
| 0.040 | 2.0 | 29,887 | 410 | 29,477 | 5 / 17.365 / 46 |
| 0.048 | 2.4 | 37,036 | 607 | 36,429 | 7 / 21.460 / 54 |
| 0.060 | 3.0 | 65,154 | 4,629 | 60,525 | 11 / 35.655 / 74 |
| 0.080 | 4.0 | 110,924 | 10,388 | 100,536 | 18 / 59.226 / 110 |

For the external-boundary union, family min/mean/max at `delta=0.032,
0.040,0.048` were respectively `3/8.186/27`, `5/13.182/34`, and
`7/17.004/41`; for the 21 tip particles they were `5/7.476/16`,
`8/10.429/22`, and `10/13.810/27`. At `delta=0.020`, 20 external particles
had zero family. Under the literal analytic rule that retains the opening-line
 segment, the `0.060` and `0.080` removed counts are smaller by 6 and 36
pairs, and the BA rank failures below reappear. This confirms that large
`delta/h` is not automatically safer near the opening.

### BA rank/conditioning and moments

For the current analytic-helper graph, target-inclusive support
`{j} union {k:(i,k),(j,k) in E}` was used with the declared `J_delta V_k`
weights. Rank screening used `lambda_min <= 1e-12 lambda_max` only as an
engineering diagnostic, not as a frozen acceptance threshold:

| `delta` | support min/median/q95/max | rank failures | condition median/q95/max |
|---:|---:|---:|---:|
| 0.020 | 1 / 3 / 12 / 17 | 4,305 | infinite |
| 0.024 | 1 / 3 / 15 / 20 | 6 | infinite; finite max `1.33e5` |
| 0.032 | 2 / 4 / 22 / 30 | 0 | `1.154 / 4.023 / 2.274e3` |
| 0.040 | 3 / 9 / 28 / 39 | 0 | `1.277 / 2.950 / 52.6` |
| 0.048 | 5 / 13 / 35 / 47 | 0 | `1.230 / 2.921 / 18.65` |
| 0.060 | 7 / 20 / 47 / 64 | 0 | `1.220 / 2.575 / 14.53` |
| 0.080 | 12 / 34 / 68 / 99 | 0 | `1.186 / 2.573 / 9.78` |

The worst finite support at `delta=0.032` under analytic-CAD visibility had
source `(0.51277254,0.52012133)`, target arc node
`(0.48341641,0.52282536)`, and condition `2.273778e3`. The polygonized
visibility alternative retains the cap chords and changes the corresponding
condition maxima to `49.62`, `79.60`, and `18.65` at `delta=0.032,0.040,0.048`.
All target-inclusive affine patch errors for the polygonized graph were at most
`4.2e-15` for these horizons. Target-term trace fractions are nonzero but can
be as small as `~1e-28` on bonds approaching the strict horizon cutoff, so
target inclusion is necessary but is not a uniform conditioning guarantee.

The continuum kernel has `[J]=L^-4`, `[J V]=L^-2`, and
`J V xi tensor xi` dimensionless. `D_delta` is reciprocal in the volume inner
product, `V_i D_ij=V_j D_ji=-V_i V_j J_ij`; constants are exactly in its
nullspace, but raw discrete first/second moments are not continuum-exact on
this nonuniform cloud. For polygon visibility, deep particles (distance
`>delta` from outer and slot boundaries) had second-moment Frobenius-error
median/q95 `0.183/0.202`, `0.163/0.172`, and `0.0640/0.0719` at
`delta=0.032,0.040,0.048`; boundary/slot particles had errors around
`1.2`–`1.7`. The normalized mass `m_i delta^2/20` was only median
`0.545`, `0.665`, and `0.768` for those deep sets. These are diagnostics of
particle quadrature, not permission to add an unapproved correction.

### Sparse constrained-spectrum exploration

The hard masks give 51 bottom and 51 top particles, 103 constrained DOFs
(bottom/top vertical plus one anchor horizontal), and `6687` free DOFs. A
matrix-free energy-Hessian action was independently evaluated at `E=1`,
`nu=0.3`, `d=0` for the polygon graph at `delta=0.032`. It gave translation
energies/residuals exactly zero in float64, rotation energy `1.16e-32` and
relative residual `5.65e-19`, and a random bilinear symmetry defect
`2.25e-15`. The structural free scalar nonzero pattern was only about
`2.62e5` entries at `delta=0.032`, `6.22e5` at `0.040`, and `8.04e5` at
`0.048`, versus a dense `6687^2` matrix.

The reproducible acceptance route is a `scipy.sparse.linalg.LinearOperator`
whose `matvec` applies each BA support and its Euclidean adjoint, wrapped by
the hard-constraint free-DOF injection; use `eigsh(A,k=4,which='SA')` and a
separate largest-eigenvalue call, or LOBPCG with a verified positive Jacobi/
block preconditioner. Record `info`, Ritz residuals
`||A v-lambda v||/||v||`, symmetry defect, finite values, the three
unconstrained rigid modes, positive constrained minimum, first non-rigid gap,
and a condition estimate. A 100-step matrix-free/LobPCG screening run was not
converged to the proposed `1e-3` residual gate (returned low estimates around
`1.33e-3,1.39e-3,4.75e-3` with residuals `2.23e-3,1.94e-3,4.25e-3`), so these
are not acceptance eigenvalues. The rounded-slot constrained spectrum remains
an open gate; do not replace it with a dense 6790-by-6790 construction.

### Independent findings for executor/Sol review

- Preserve the tagged `.msh`, regenerate the configured archive after the new
  physical anchor point, and validate all line cells cover the material mesh
  boundary; do not rely on the tag-dropping XDMF conversion.
- Keep `r<delta` strict, reciprocal, self-edge-free, and volume-weighted;
  retain explicit `V_i`, `m_i`, `eta_ij`, and target endpoint terms. Fail on
  zero family mass, rank deficiency, nonpositive volumes, duplicate particles,
  or untagged/unknown boundary entities.
- Explicitly decide analytic-CAD versus polygonized-mesh visibility and the
  treatment of a segment through the left opening. The two choices change
  tip boundary edges, graph counts, and the worst BA condition by two orders
  of magnitude; neither should be silently relabeled as roundoff.
- Do not freeze a production horizon from this study. The defensible
  exploratory grid is `delta/h in {1.0,1.2,1.6,2.0,2.4,3.0,4.0}` with
  `h=0.02` plus an explicit near-slot `h=0.004` report. The current rank-safe
  window is conditional on the visibility rule and still requires the
  converged constrained spectrum.

## Sol Stage 1 visibility decision — 2026-08-18

Use the polygonized material domain represented by the retained Gmsh triangles
and tagged boundary edges for the production Stage 1 graph. Tagged notch-tip
chords belong to the material closure and must remain visible. The open segment
between the two left slot lips has no material triangles and is not a tagged
material boundary, so a bond running through that opening is removed. The exact
CAD circle remains only a future, explicitly labeled sensitivity study.

This decision aligns graph topology with the geometry used by barycentric
quadrature and physical masks. It supersedes the analytic-CAD production rule
recorded in the Stage 1 handoff and reopens the focused Stage 1 tests and smoke
metrics. The implementation below proves that tagged line cells cover the
complete material mesh boundary and that the slot topology is one open chain.

The mesh-conforming graph implementation, tip-chord/opening/boundary-coverage
regressions, and fresh-mesh graph smoke are complete. Exact resume point:
run the rounded-slot BA rank/conditioning and constrained-spectrum gate, then
start Stage 2 only if those geometry-dependent checks pass. Do not claim the
constrained spectrum until its eigenvalue residuals and acceptance checks are
actually established.

## Stage 2 explicit-volume weighted scalar operator — 2026-08-18

Implemented `pinn.operators.ExplicitVolumeSparseD` without changing
`SparseD`. The constructor requires `coordinates`, `volumes`,
`directed_edges`, `delta`, and `kernel`; it does not know about Gmsh or
fracture geometry. It normalizes directed edges lexicographically and validates
floating finite unique coordinates, positive finite volumes, matching
coordinate/volume dtype, integer reciprocal self-edge-free in-range edges,
finite positive horizon, strict `0 < r_ij < delta`, registered kernels, and
positive outgoing family mass at every node. Matching float32 and float64 are
general operator modes; float64 construction fails with a JAX 64-bit-mode
diagnostic rather than downcasting. Vector and matrix matvecs accept only
`(N,)` and `(N, T)`; fixed diagonal/volumes/edges and SciPy CSR export are
exposed. The JAX matvec uses precomputed edge differences, which preserves the
constant nullspace in finite-precision accumulation while remaining equivalent
to the stored diagonal/off-diagonal matrix.

The fracture2d direct baseline is frozen to float64 for coordinates, barycentric
volumes, graph arrays, serialized operator data, and operator inputs, with JAX
x64 enabled before operator construction. Strict `r_ij < delta` graph
membership is dtype-specific: a future float32 PINN must rebuild and revalidate
its graph in float32, not cast the float64 graph and not add a cutoff tolerance
or kernel floor.

The focused `tests/test_scalar_operator_2d.py` covers nonuniform volumes and a
reciprocal graph, exact entries/signs, constants, SciPy/JAX vector and matrix
agreement, scale-aware large-offset comparisons, JIT matvecs, volume
self-adjointness, the directed pair-energy identity and random non-negative
forms, the volume-multiplied fixed-history SPD system and residual,
float32/float64/JAX-x64 behavior, matching-dtype failures, shape failures, and
the important invalid-input cases. The large-offset/large-value synthetic CSR
versus JAX comparison had relative norm error `1.1517555018434153e-16`. It also
covers a full-square triangular nodal cloud with barycentric volumes.
Rounded-slot diagnostics are not part of
this persistent unittest suite.

At `n=41`, `delta=0.1`, the float64 constant error was zero; the affine-x and
affine-y interior maximum errors were `1.42108547e-14` and `2.17603713e-14`;
the `x^2`, `y^2`, and `xy` interior MAEs were
`5.88789955e-3`, `5.88789955e-3`, and `1.7856589e-14`. The full-domain `x^2`,
`y^2`, and `xy` MAEs were `2.571727`, `2.571727`, and `2.430874`; these include
the finite-horizon boundary region and are not local-consistency acceptance
gates. The fixed-`delta=0.1` `x^2` interior MAE sequence for `n=41,61,81` was
`[5.88789955e-3, 7.49028372e-4, 4.20822903e-4]`. The coupled finite diagnostic
errors were `[5.88789955e-3,5.88789955e-3,5.88789955e-3]`; this is not a
convergence claim.

### Reproducible independent rounded-slot diagnostic workflow

The rounded-cloud evidence is generated outside unittest discovery through the
two mutually exclusive environments. The following workflow deliberately
serializes only temporary explicit arrays; it does not add a binary fixture or
install packages across environments.

1. In `fenicsenv`, generate the tagged mesh and extract one cloud per explicit
   horizon. The mesh must retain the physical `anchor`, all named boundary
   groups, and the tagged line-cell topology:

   ```bash
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate fenicsenv
   gmsh -2 examples/fracture2d/mesh.geo \
     -setnumber lc_far 0.02 -setnumber lc_near 0.004 \
     -o /tmp/opencode/fracture2d_stage2_float64.msh
   ```

   In that same environment, run the following extraction (the recorded run
   used meshio 5.3.5 and Gmsh 4.15.2):

   ```bash
   python - <<'PY'
   import numpy as np
   from examples.fracture2d.nonlocal_geometry import (
       ROUNDED_SLOT_TAGS, load_gmsh_particle_cloud,
   )

   mesh = "/tmp/opencode/fracture2d_stage2_float64.msh"
   clouds = {}
   base = None
   for delta, key in ((0.032, "d032"), (0.040, "d040"), (0.048, "d048")):
       cloud = load_gmsh_particle_cloud(
           mesh, horizon=delta, boundary_tags=ROUNDED_SLOT_TAGS,
           visibility_tolerance=0.0,
       )
       if base is None:
           base = cloud
       else:
           np.testing.assert_array_equal(base.coordinates, cloud.coordinates)
           np.testing.assert_array_equal(base.triangles, cloud.triangles)
           np.testing.assert_array_equal(base.volumes, cloud.volumes)
       clouds[key] = cloud.visible_edges
   np.savez(
       "/tmp/opencode/stage2_cloud_float64.npz",
       coordinates=base.coordinates, volumes=base.volumes,
       triangles=base.triangles, d032=clouds["d032"],
       d040=clouds["d040"], d048=clouds["d048"],
   )
   PY
   ```

   The loader returns float64 cloud geometry and rejects non-float64
   direct-cloud data. The temporary `.msh` and `.npz` are evidence artifacts,
   not repository fixtures.

2. In `.venv`, enable x64 before constructing the operator, load those explicit
   arrays, and run the constants, `M D` symmetry, pair-energy, fixed-history
   SPD/residual, deep-boundary, and manufactured checks:

   ```bash
   source .venv/bin/activate
   python - <<'PY'
   import jax
   jax.config.update("jax_enable_x64", True)
   # Load the temporary arrays and instantiate/check all three operators here.
   PY
   ```

   The x64 update and operator construction must occur in the same Python
   process. Construct `ExplicitVolumeSparseD(coordinates, volumes,
   visible_edges, delta, "cubic")` for each serialized graph. Compute boundary
   distance from triangle edges with incidence one only; the absent left opening
   closure is not a material boundary edge. Do not cast the graph, add a cutoff
   tolerance, or add a kernel floor between these steps.

### Fresh rounded-slot Stage 2 diagnostic

The diagnostic used the fresh tagged mesh
`/tmp/opencode/fracture2d_stage2_float64.msh`, regenerated clouds at each
explicit horizon in `fenicsenv`, and then constructed `ExplicitVolumeSparseD`
from `/tmp/opencode/stage2_cloud_float64.npz` in the JAX `.venv` with 64-bit
mode enabled.
The cloud had `3395` particles and
`6284` triangles, barycentric area `0.9762509374813683`, first moments
`(0.4941241109732579, 0.4881254687406843)`, and `504` polygonized material
boundary edges. Boundary distance used only those triangle edges with
incidence one; the absent left opening closure was not included as a material
boundary edge. No local-consistency threshold or production-horizon selection
was applied.

| `delta` | directed edges | family min/mean/max | deep points | constant max | max `MD` symmetry defect | random energy / pair energy | fixed-history residual | SPD eigenvalue range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.032` | `30258` | `3 / 8.91251841 / 32` | `2422` | `0` | `1.11e-16` | `9.9685957443e3 / 9.9685957443e3` | `1.67e-15` | `[1.2759850607e-3, 3.9885154804e-1]` |
| `0.040` | `59024` | `5 / 17.3855670 / 46` | `2278` | `0` | `1.11e-16` | `7.7690190407e3 / 7.7690190407e3` | `2.10e-15` | `[9.3651545107e-4, 3.1836227931e-1]` |
| `0.048` | `72934` | `7 / 21.4827688 / 54` | `2189` | `0` | `5.55e-17` | `6.1875630741e3 / 6.1875630741e3` | `2.46e-15` | `[7.5868385936e-4, 2.5446286904e-1]` |

The rounded-cloud manufactured errors are recorded as full-domain MAE / deep
domain MAE; deep maxima were also retained by the run:

| `delta` | field | full MAE | deep MAE | deep max |
|---:|---|---:|---:|---:|
| `0.032` | constant | `0` | `0` | `0` |
| `0.032` | affine-x | `5.74302631` | `8.50574165e-1` | `1.60181893e1` |
| `0.032` | affine-y | `1.14972921e1` | `1.03483826` | `2.91652812e1` |
| `0.032` | `x^2` | `3.96252220` | `8.40285130e-1` | `2.29765733e1` |
| `0.032` | `y^2` | `1.15434719e1` | `1.06944191` | `4.10801551e1` |
| `0.032` | `xy` | `6.09818874` | `6.68435860e-1` | `1.40033618e1` |
| `0.040` | constant | `0` | `0` | `0` |
| `0.040` | affine-x | `4.69434343` | `3.61765682e-1` | `6.06836265` |
| `0.040` | affine-y | `9.31948102` | `4.65076991e-1` | `1.38967998e1` |
| `0.040` | `x^2` | `3.33876781` | `4.11092219e-1` | `1.03654274e1` |
| `0.040` | `y^2` | `9.41631888` | `4.98859230e-1` | `1.56316405e1` |
| `0.040` | `xy` | `5.06530415` | `2.91593417e-1` | `6.35929774` |
| `0.048` | constant | `0` | `0` | `0` |
| `0.048` | affine-x | `4.23515729` | `1.84850402e-1` | `3.56528137` |
| `0.048` | affine-y | `8.27376157` | `2.23413827e-1` | `4.59729054` |
| `0.048` | `x^2` | `3.10916726` | `1.94504797e-1` | `4.23929026` |
| `0.048` | `y^2` | `8.33681242` | `2.28386191e-1` | `5.21752682` |
| `0.048` | `xy` | `4.62784313` | `1.44680626e-1` | `2.89178432` |

The large rounded-cloud full/deep errors are retained as quadrature and
finite-horizon diagnostics, not hidden by a correction or accepted as a local
consistency threshold. They are independent review evidence, not persistent
unittest coverage. Mechanics and coupled fracture were not started.

## Stage 2 review status and next action

The Stage 2 implementation blockers are resolved: the fracture2d direct path is
float64/x64, mixed coordinate/volume dtypes fail, rounded-cloud evidence is
explicitly independent of unittest discovery, and the scale-aware CSR/JAX
regression passes. Sol approved Stage 2 after final review. Matmul values now
must match the operator dtype, manual `ParticleCloud` construction rejects
non-integer connectivity/edge/count arrays, and the float64 manufactured test
restores the prior global JAX x64 setting. The next implementation stage is the
separately derived intact bond-associated mechanics path; do not begin damage
coupling or a PINN before that mechanics path passes its own gates.

## Stage 2 verification record — 2026-08-18

- `source .venv/bin/activate && python -m unittest -v
  tests.test_sparsed_fracture tests.test_fracture2d_nonlocal_geometry
  tests.test_fracture2d_bond_associated_algebra tests.test_scalar_operator_2d`: `40` tests
  passed.
- `source .venv/bin/activate && python -m unittest discover -s tests -p
  'test*.py' -v`: `96` tests passed.
- `source .venv/bin/activate && python src/pinn/operators.py`: passed all
  existing Dense/Sparse/Conv self-checks.
- `source .venv/bin/activate && python -m compileall -q
  src/pinn/operators.py tests/test_scalar_operator_2d.py
  examples/fracture2d/nonlocal_geometry.py
  tests/test_fracture2d_nonlocal_geometry.py tests/test_fracture2d_bond_associated_algebra.py`:
  passed.
- `git diff --check`: passed.
- `source .venv/bin/activate && python benchmarks/verify_equivalence.py` was
  run in both the available GPU and `JAX_PLATFORMS=cpu` modes. The existing
  ConvD-vs-SparseD GradNorm-update case failed its `2e-5` gate
  (GPU loss/gradient/weight relative deviations `8.52e-6/2.35e-5/4.58e-4`;
  CPU `1.65e-6/1.33e-5/1.16e-4`), while the regular ConvD case and both
  tensor-split cases passed. This benchmark contains no Stage 2 changes and
  is the remaining unrelated verification risk; no tolerance or existing
  `SparseD` semantics were changed to hide it.

Final verification after the Sol review fixes passed `96` tests, the CPU
operator self-check, changed-module `compileall`, and `git diff --check`.
Exact resume point: begin only the separately derived Stage 3 intact bond-
associated mechanics work using the approved float64 cloud and graph. Mechanics
production code and coupled fracture remain intentionally untouched in this
session.

## Stage 3 implementation start — 2026-08-21

Stage 3 will add an independent NumPy/SciPy float64 direct mechanics module at
`examples/fracture2d/nonlocal_mechanics.py`. It will consume explicit
coordinates, barycentric volumes, the reciprocal directed graph, and tag-derived
hard-boundary masks. It will not modify `ElasticD`, reuse the local FEM weak
form, introduce JAX mechanics, or implement damaged coupling.

The production API must expose target-inclusive support/shape diagnostics,
edge and eta-averaged particle strain/stress, the declared intact mechanics
energy, its explicit residual, the analytic tangent action, internal-force sign,
and a matrix-free hard-constraint solve with separate top/bottom/anchor
reactions. Acceptance requires persistent manufactured patch, derivative,
conservation, constrained solve, reaction-work, and interior Navier-Cauchy
tests, followed by independent fresh rounded-slot diagnostics.

## Stage 3 implementation — 2026-08-21

Implemented the independent intact mechanics path in
`examples/fracture2d/nonlocal_mechanics.py` and its persistent regression suite
in `tests/test_fracture2d_nonlocal_mechanics.py`. The constructor requires
`coordinates`, `volumes`, `directed_edges`, `delta`, `youngs_modulus`,
`poisson_ratio`, `k_res`, and `rank_rtol`; geometry and volumes are exact
float64, the graph is integer/reciprocal/self-edge-free, and every edge is
checked with strict `0 < r_ij < delta`. Directed edges are copied and
lexicographically normalized. Positive volumes, finite material parameters,
the plane-strain denominator, positive residual stiffness, positive rank
tolerance, positive family mass, and rank/conditioning are validated before
shape inversion. No pseudo-inverse, clipping, kernel floor, stabilization, or
fallback is present.

The fixed target-inclusive supports are padded with a boolean mask and retain
the target as their first valid entry. Fixed beta, xi, inverse shape tensors,
eta, center mass, support sizes/ranks/conditions, target beta, target
sensitivity, and the linear displacement-to-gradient coefficient are retained
as read-only arrays. `energy`, `residual`, `internal_force`, `tangent_action`,
and `evaluate` follow Sections 0.3--0.5. The residual is the explicit endpoint
assembly from `P K^{-T} xi`; the tangent action independently computes
`delta F` and the plane-strain constitutive map. `evaluate` returns edge fields
and source-centered eta averages. Stress fields are explicitly the undegraded
constitutive `P`; separate effective intact stress fields expose
`(1+k_res)P`.

`HardBoundaryConstraints` is immutable and is built only from explicit boolean
bottom/top masks and an integer bottom anchor. It validates mask shape,
non-overlap, nonempty boundary sets, anchor membership, and the constrained/free
DOF partition. `solve` uses direct free-index injection, SciPy
`LinearOperator`, and explicit `cg_rtol`, `cg_atol`, and `cg_maxiter`. It
records CG iterations/info, fails on nonconvergence or nonfinite values, and
returns the full response, free residual norms, energy, residual-gradient
top/bottom/anchor reactions, total force, and total moment. The exact
particle-local block-Jacobi preconditioner is assembled from the same tangent
blocks only for CG convergence; it is not a mechanics stabilization or model
correction. No dense global tangent is formed.

### Persistent Stage 3 metrics

The regular nonuniform-volume `7 x 7` manufactured cloud has `49` particles,
`452` directed edges, and target-inclusive support sizes
`3 / 6 / 7 / 7` (minimum/median/q95/maximum). Every support has rank `2`; shape
condition numbers range from `1.0053842895598344` to
`6.144654759360297` (median `1.4389744313667103`, q95
`6.08583218452475`). Target sensitivity ranges from
`1.1031901550156374e-4` to `0.8259019215841329`. The affine deformation
gradient maximum error is `3.122502256758253e-16`, and source-centered eta
averaging reproduces the affine strain/stress patch.

The two translations have exactly zero energy/residual in the persistent
float64 run. The infinitesimal-rotation energy and residual norm are
`1.4330211500529734e-38` and `1.936673712366743e-19`. The independently
assembled dense test tangent has symmetry defect
`9.445455593430594e-17`, minimum eigenvalue
`-5.760149503161629e-15`, exactly three modes below the test zero threshold,
first non-rigid eigenvalue `0.2023221469589245`, and maximum eigenvalue
`7.642893494549784`. The energy directional FD relative error is
`9.930906875581638e-12`; explicit-residual JVP versus tangent-action relative
error is `5.172104667904982e-12`; and residual versus tangent-action at `u` is
`1.838384805702437e-15`.

For the hard-boundary solve at `u_bar=0.08`, `15` DOFs are constrained and
`83` are free, including `13` retained non-anchor top/bottom horizontal DOFs.
With `cg_rtol=1e-11`, `cg_atol=1e-13`, and `cg_maxiter=3000`, CG returned
`info=0` after `47` iterations. The free residual norm and relative norm are
`3.6479729317319144e-12` and `5.6791810936492335e-12`. The separate reactions
are
`R_T=0.20299444025114266`,
`R_B=-0.2029944402515526`, and
`R_A=-3.445836010192854e-13`; total force is
`[-3.86988178e-19, 4.16333634e-17]` and total moment is
`8.326672684688674e-17`. Independently re-solving the dense reduced test
problem at `u_bar +/- 1e-5` gives a centered reduced-energy reaction relative
error of `4.391927400783492e-12`.

The full `41 x 41` square quadratic diagnostic uses `h=0.025`, explicit
`delta=0.08`, and the strict mask `distance_to_boundary > 2*delta` (`729`
particles). It has `56312` directed edges and support sizes
`9 / 19 / 29 / 29`. For `u=(x^2,0)`, the correct deep plane-strain density is

$$
-\operatorname{div}((1+k_{\mathrm{res}})\boldsymbol\sigma)
=\left(-(1+k_{\mathrm{res}})(2\lambda+4\mu),0\right).
$$

The observed deep relative error has mean
`2.6507386377422355e-14` and maximum
`8.104881812255356e-14`; the maximum transverse residual density is
`8.300975822651345e-14`. This is a manufactured interior diagnostic, not a
production horizon choice.

### Independent fresh rounded-slot evidence

The fresh tagged mesh was generated in `fenicsenv` and serialized only to the
temporary arrays `/tmp/opencode/fracture2d_stage3_float64.msh` and
`/tmp/opencode/stage3_cloud_float64.npz`. The extraction used the retained
physical tags and `load_gmsh_particle_cloud` at each explicit horizon
`0.032`, `0.040`, and `0.048`, with `visibility_tolerance=0.0`; it confirmed
`3395` particles, `6284` triangles, area
`0.9762509374813683`, first moments
`(0.494124110973257, 0.4881254687406841)`, `51` bottom particles, `51` top
particles, and anchor particle `0`.

The following table is independent rounded-cloud evidence, not persistent
unittest coverage and not a horizon selection. It uses `E=1`, `nu=0.3`,
`k_res=1e-8`, `rank_rtol=1e-12`, `u_bar=0.01`, and explicit
`cg_rtol=1e-7`, `cg_atol=1e-10`, `cg_maxiter=3000`.

| `delta` | directed edges | support min/median/q95/max | rank failures | condition median/q95/max | affine max error | CG iterations | free residual / relative | `R_T` | `R_B` | `R_A` | `|R_T+R_B|` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.032` | `30258` | `2 / 4 / 22 / 30` | `0` | `1.15417461 / 4.01797777 / 49.62348215` | `8.54871729e-15` | `457` | `6.74651757e-9 / 9.79557929e-8` | `6.27671063e-3` | `-6.27671540e-3` | `1.73024055e-11` | `4.76237410e-9` |
| `0.040` | `59024` | `3 / 9 / 28 / 39` | `0` | `1.27642235 / 2.95464776 / 79.59747879` | `1.34336980e-14` | `284` | `3.90903129e-9 / 9.44647102e-8` | `6.21426218e-3` | `-6.21425920e-3` | `-3.56998564e-10` | `2.98053689e-9` |
| `0.048` | `72934` | `5 / 13 / 35 / 47` | `0` | `1.23037253 / 2.92226181 / 18.65090919` | `3.55271368e-15` | `243` | `2.66852145e-9 / 9.45558186e-8` | `6.14925052e-3` | `-6.14924839e-3` | `-1.84606422e-10` | `2.13100857e-9` |

The `delta=0.048` independent reaction-energy check uses fresh solves at
`u_bar +/- 1e-5`. The center reaction is
`0.006149250519480712`, the centered energy FD is
`0.006149250519674583`, and the relative error is
`3.152757565030081e-11`; the plus/minus free residual norms are
`2.6829692443719682e-9` and `2.654497533954103e-9`. The rounded diagnostic
therefore verifies the reaction sign/work relation at the declared CG
tolerance without hiding convergence.

The prior independent Stage 1 action was loaded from
`/tmp/opencode/stage1_ba_validation.py` and compared on five fresh random
float64 vectors at `delta=0.048`. Production `tangent_action` relative errors
were
`[2.429743213625704e-16, 2.4324632652446855e-16,
2.4076881618182998e-16, 2.4133231700068427e-16,
2.464284102796781e-16]`. This independently matches the prior action, so the
previous matrix-free spectrum evidence may be reused: constrained minimum
`2.8440031721752207e-05`, maximum `1.3314397965182772`, condition estimate
`46815.69308869415`, minimum Ritz residual relative to the eigenvalue
`2.5388e-11`, and maximum Ritz residual relative to the eigenvalue
`8.9918e-14`. No production horizon is selected by these diagnostics.

### Exact diagnostic workflow

The fresh mesh and temporary cloud were produced with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fenicsenv
gmsh -2 examples/fracture2d/mesh.geo \
  -setnumber lc_far 0.02 -setnumber lc_near 0.004 \
  -o /tmp/opencode/fracture2d_stage3_float64.msh
```

In the same `fenicsenv` process, the loader was called for all three explicit
horizons and saved float64 `coordinates`, `volumes`, `triangles`, tagged
`bottom`/`top` masks, `anchor_particle`, and `d032`/`d040`/`d048` directed
graphs to `/tmp/opencode/stage3_cloud_float64.npz`. In `.venv`, the Stage 3
constructor consumed those arrays without casting; each horizon was solved at
the explicit CG settings above, and `delta=0.048` was independently re-solved
at `u_bar +/- 1e-5`. The prior-action comparison used the temporary
`stage1_ba_validation.py` module only; no binary mesh or cloud fixture was
added to the repository.

## Stage 3 approval and next action — 2026-08-21

The focused Stage 0/Stage 1/Stage 2/Stage 3 command

```bash
.venv/bin/python -m unittest -v \
  tests.test_fracture2d_bond_associated_algebra \
  tests.test_fracture2d_nonlocal_geometry \
  tests.test_scalar_operator_2d \
  tests.test_fracture2d_nonlocal_mechanics
```

passed all `41` tests. Full unittest discovery with
`.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` passed all
`106` tests. Changed-module compilation passed with

```bash
.venv/bin/python -m compileall -q \
  examples/fracture2d/nonlocal_mechanics.py \
  tests/test_fracture2d_nonlocal_mechanics.py
```

The requested CPU shared-operator self-check
`JAX_PLATFORMS=cpu .venv/bin/python src/pinn/operators.py` passed all existing
Dense/Sparse/Conv checks. `git diff --check` passed. The unrelated CPU
benchmark command
`JAX_PLATFORMS=cpu .venv/bin/python benchmarks/verify_equivalence.py` still
fails only its pre-existing ConvD-vs-SparseD GradNorm update case (loss
`1.65e-6`, gradient `1.33e-5`, weights `1.16e-4`); the regular ConvD case and
both tensor-split cases pass. No benchmark or shared operator code was altered.

Sol's final review found no remaining issues and approved Stage 3. The final
fixes restored the exact Gmsh command, removed the hidden machine-floor addition
from the explicit CG residual gate, assembled the reaction-work test tangent
from the independent reference residual, verified the solver-only block-Jacobi
against dense diagonal blocks, tested forced CG nonconvergence, and corrected
the documented coordinate/volume shapes. No-floor rounded reruns satisfy the
requested `cg_rtol=1e-7` criterion at all three diagnostic horizons.

The exact resume point is Stage 4 fixed fracture subproblems: keep this intact
mechanics implementation unchanged, add fixed-`d=0`/fixed-history/fixed-damage
direct solves separately, and do not select a common mechanics/damage horizon
or start PINN work from the Stage 3 diagnostic grid.

## Stage 4 implementation — 2026-08-21

Implemented the fixed fracture subproblems without changing the intact Stage 3
public path or selecting a production horizon.
`FixedDamageBondAssociatedMechanics(intact_mechanics, damage)` shares the
intact instance's read-only BA geometry/support/constitutive arrays, but uses
`g_i=(1-d_i)^2+k_res`, `V_i*g_i*eta_ij`, a separate tangent cache, and locked
fixed state. It inherits the same deformation gradient, energy residual,
tangent action, source-centered fields, hard-constraint CG solve, and reaction
diagnostics. Fixed damage is exact float64, finite, and shape-checked without
clipping; its extrema and out-of-range counts are exposed. Fixed `d=0` has exact
array and constrained-solve compatibility with the intact instance.

`examples/fracture2d/nonlocal_baseline.py` now provides
`solve_fixed_history_damage`, which validates an `ExplicitVolumeSparseD`, forms
the volume-multiplied dimensional AT2 system with `D=operator.to_scipy()` and
`M=diag(V)`, uses a SciPy sparse direct solve, reports strong/Euclidean
residuals and system symmetry, and returns crack-local/history/graph/total
energies plus damage-range diagnostics. The directed graph energy uses the
frozen `G_c*l0/4` coefficient and no clipping. Before solving, it independently
reconstructs the frozen 2D cubic coefficients and gates CSR structure,
constants, and `M D = D.T M`. The same module provides the exact 3D
plane-strain Amor edge driver, normalized source-only particle averaging, and
pure committed-to-trial maximum history update.

The new persistent suite `tests/test_fracture2d_nonlocal_baseline.py` uses
independent reference graph, mechanics, Amor, and energy calculations. The
focused Stage 0--4 command below passed all `48` tests on the manufactured
clouds:

```bash
.venv/bin/python -m unittest -v \
  tests.test_fracture2d_bond_associated_algebra \
  tests.test_fracture2d_nonlocal_geometry \
  tests.test_scalar_operator_2d \
  tests.test_fracture2d_nonlocal_mechanics \
  tests.test_fracture2d_nonlocal_baseline
```

### Fresh rounded-slot Stage 4 evidence

The existing fresh float64 cloud at
`/tmp/opencode/stage3_cloud_float64.npz` was consumed in `.venv` with
`E=210e3`, `nu=0.3`, `Gc=2.7`, `l0=0.024`, `k_res=1e-8`,
`rank_rtol=1e-12`, `cg_rtol=1e-7`, `cg_atol=1e-10`, and
`cg_maxiter=3000`. The temporary diagnostic and arrays are
`/tmp/opencode/stage4_rounded_diagnostic.py` and
`/tmp/opencode/stage4_rounded_diagnostic.npz`. They are reproducible evidence,
not repository fixtures or a production-horizon selection.

At loads `1e-4`, `5e-4`, and `1e-3`, fixed-`d=0` mechanics was compared with
the warning-mode point-anchor FEM archive:

| `delta` | nonlocal/FEM top reaction | nonlocal/FEM displacement norm | nonlocal/FEM mean `sigma_yy` |
|---:|---:|---:|---:|
| `0.032` | `0.9885 / 0.9896 / 0.9930` | `0.99957 / 0.99956 / 0.99954` | `0.9874 / 0.9885 / 0.9919` |
| `0.040` | `0.9787 / 0.9797 / 0.9831` | `0.99837 / 0.99837 / 0.99835` | `0.9776 / 0.9787 / 0.9820` |
| `0.048` | `0.9684 / 0.9695 / 0.9728` | `0.99704 / 0.99703 / 0.99701` | `0.9674 / 0.9684 / 0.9717` |

All trends are positive, monotone, and nearly linear. The FEM damage maxima at
these loads are `1.11e-4`, `2.77e-3`, and `1.11e-2`, so this is a trend check
against the accepted working archive, not an exact intact-state comparison.

The fixed-`d=0` displacement, residual, reaction, and energy at `delta=0.048`
exactly match the intact view. Using the load-`1e-3` Amor driver as fixed
history gave damage ranges `[3.39e-7, 0.01361]`,
`[2.70e-7, 0.01314]`, and `[2.44e-7, 0.01234]`; no value was outside `[0,1]`.
Strong residual infinity norms were at most `7.2e-15`, Euclidean residual
infinity norms at most `1.2e-18`, and system symmetry relative defects at most
`5.4e-17`. The corresponding fixed-damage solves met the declared CG gate;
force norms were at most `6.7e-15` and moment magnitudes at most `7.3e-15`.
At `delta=0.048`, top reaction was `128.541412134934`, the reduced-energy
derivative was `128.541412134815`, and relative work error was `9.24e-13`.

Independent Sol reviews first identified and then verified fixes for the 2D
cubic/CSR gate, Amor eta and strain validation, test x64 isolation, independent
damaged-tangent reference, shared-data fixed view, and state rebinding/deletion
guards. The final review found no remaining issue and approved Stage 4 for
progression to Stage 5. No common production horizon was selected.

Additional verification completed in this session:

- `.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` passed all
  `113` repository tests.
- `JAX_PLATFORMS=cpu .venv/bin/python src/pinn/operators.py` passed the shared
  operator self-check.
- Changed-module `compileall` passed for the Stage 3/4 mechanics and baseline
  modules, both focused mechanics test modules, and the corrected x64-isolated
  `test_sparsed_fracture.py`.
- `git diff --check` passed.

The final CPU full-suite rerun still passed all `113` tests, and a same-process
isolation check confirmed the Stage 4 and `SparseD` float64 tests restore
`jax_enable_x64` to its incoming value.

Stage 5 strict staggered continuation is implemented and independently approved
through the temporary rounded-slot pre-localization pilot at `u_bar=1e-3`. The
exact resume point is to select CPU/runtime/restart work or an explicitly staged
diagnostic extension toward peak/crack growth. Do not promote the Stage 1--4
diagnostic horizons, begin a PINN, or interpret the warning-mode FEM archive as
a strict converged fracture target.
