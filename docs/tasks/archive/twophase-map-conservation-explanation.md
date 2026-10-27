# Explain inverse-cubic coordinates and conservation

Goal: explain how the current two-phase u=phi+phi^3 coordinate was constructed,
its optimization rationale/generality/limits, and what conservative advection,
nonlocal pair diffusion and exact mass projection mean in the actual code.

Acceptance: trace explanation to current source and recorded ablations;
independently check algebra/zero-mode identities, distinguish coordinate mapping
from residual reweighting and operator conservation from finite-loss PINN mass.
No training, implementation/default/reference or Git history changes.

Context: current twophase2d prescribes velocity and uses mixed CH, inverse-cubic
phase map, original residual, spectral nodal flux and grid-mean projection.
SOAP issue stays deferred; preceding precipitate-local task is complete.

Status: complete (2026-10-27).
Blocker: none.
Next action: none. No production/default/reference/training/Git operation.

## Actual construction and limits

For F=(phi^2-1)^2/4, F'=phi^3-phi is not invertible (0 and +/-1 all give0),
and F''=3phi^2-1 is negative near0. Choose Fc=F+phi^2-1/4=phi^4/4+phi^2/2;
Fc''=1+3phi^2>=1, so z=Fc'=phi+phi^3 is globally monotone/surjective on R.
The inverse is implemented analytically with sinh/asinh. This is a particular
simple convexification choice, not a PDE-mandated unique map, not the full
chemical potential q, and not fluid velocity.

For the provisional point field, g'(z)=1/(1+3phi^2). At phi0 this is1 and at
phi+/-1 it is1/4; strict tanh sensitivity vanishes on approach to the pure
phases. The inverse-cubic map covers all real phi, including actual reference
excursions local[-1.09837,1.06254], nonlocal[-1.06899,1.04804]. It gives a mild
state-dependent sensitivity profile without an artificial tanh+leak coefficient.
No proof that the target profile is smoother, the neural loss is convex, the
condition number improves or training/L2 always improves. Convexification of
one local energy is not a convex training problem or convex-split time scheme.

Actual initial handling: z0=phi0+phi0^3; provisional psi=phi0+
g(z0+(t/T)Ntheta)-g(z0). This specifies IC only, no future interface/reference.
The final physical field additionally undergoes the mass projection below;
therefore the complete Jacobian is not the scalar g' alone. For uniform-grid
projection it is P diag(g'), P=I-11^T/N, with a constrained zero-mean tangent.
The original mixed residual is evaluated on the physical field. No division
by g' or inverse-map residual norm is enabled. Nonlocal Lg(z) must be evaluated
as declared; no local chain rule Lg=g' Lz is assumed.

For smooth weakly convex scalar F with F''>=-K on the domain, choose a>K,
Fc=F+a phi^2/2 and z=F'(phi)+a phi. Fc''>0 gives a monotone coordinate;
global coverage requires the appropriate domain/coercivity assumptions.
Vector gradients of strongly convex energies are a related construction, but
coupled constraints, physical bounds, units/scales, inverse evaluation and
finite-network optimization still need analysis. Not every model should use
an unbounded map or this cubic polynomial. The historical matched additive/
conservative vs cubic/conservative pooled phi3.235% vs2.843% is one modest
ablation, not proof of a stable benefit given known same-seed run variability.

## Three separate conservation mechanisms

On a periodic source-free domain, continuum CH is
phi_t+div(v phi)=M Delta q and q=phi^3-phi-epsilon^2 Delta phi. Div(v)=0
makes advective and flux forms equivalent continuously. In discrete form,
take positive L approximately -Delta and C_h phi=D_x(v_x phi)+D_y(v_y phi):
r=phi_dot+C_h phi+M Lq. Both Fourier derivative multipliers have zero constant
mode, so 1^T C_h phi=0 for every nodal flux. Symmetric reciprocal nonlocal
pair weights give 1^T Lq=0 by pairwise cancellation. Thus the semidiscrete
equation r=0 preserves grid mass. This concerns residual spatial operators,
not a time-marching algorithm used to train the network.

Correct point AD derivatives need not have zero summed nodal quadrature. The
simple unresolved periodic sin(2pi N x) has zero nodal values but sampled exact
slope2pi N at every x_i=i/N. A nodal derivative of the zero values is zero.
This illustrates resolution/quadrature mismatch, not an error in AD or a unique
cause of the actual trained checkpoints. Conservative operators preserve summed
flux even with aliasing; that alone does not remove spatial aliasing error.

At finite PINN residual, m_dot=(1/N)sum(r_i), so conservative operators alone
do not force exact mass. The production physical-grid wrapper subtracts
mean(psi-phi0): phi=psi-[mean(psi)-mean(phi0)]. This is the minimal Euclidean
affine projection onto the prescribed mean. All physical loss/time-JVP/eval/
checkpoint replay uses this wrapper, not a postprocessing-only correction.
Conservation to roundoff is imposed, not independently learned. For nonuniform
quadrature replace means by weighted means; open flux/source/nonconserved
models must use the derived time-dependent balance rather than frozen IC mass.
Grid phase mass does not guarantee exact zero-contour area or conserved energy.
For the prescribed vortex, external stirring can supply phase free energy.

## Independent checks and artifact

twophase-map-conservation-check.py reads only existing model inputs/reference,
uses NumPy/SciPy algebra and creates the JSON/figure alongside this record.
Inverse error4.44e-16, inverse-derivative finite-difference error1.00e-10,
convex split error5.55e-16; Fourier flux mean6.66e-16, finite-horizon diffusion
mean2.13e-14, mass projection error3.47e-17. The unresolved-wave illustration
has nodal values<=4.31e-14, correct sampled AD-equivalent slope mean402.12386,
nodal Fourier slope mean~2.52e-29. These are algebra/zero-mode checks, not new
training or conditioning results. The mapping/sensitivity figure is viewed.

Source: examples/twophase2d/network.py (inverse/map/IC), fields.py (projection),
advection.py (nodal flux), losses.py (original residual/time JVP), operators.py
(symmetric pair Laplacian); evidence twophase-operator-preconditioning.md.
Current preset remains a prescribed-velocity CH pilot, not solved coupled NS.

Verification: algebra checks, figure inspection, local links, git diff --check.
Memory changed: this archive, docs/PROGRESS.md and docs/DECISIONS.md. No runtime
code, case README/default, reference, external checkpoint, dependency or history
change; no GPU training required. SOAP issue remains deferred.
