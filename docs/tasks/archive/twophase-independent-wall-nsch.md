# Independent-wall NS--CH feasibility analysis

## Goal and acceptance

Assess the user's independent upper/lower inward-wall parameterization with no
separate eccentricity parameter. Explain emergent width/centerline, velocity
response, complete two-way NS--CH scope and practical difficulty. Recommend a
manageable first configuration and validation route without implementing new PDEs.

## Context

The last two tasks delivered a geometry recommendation and a clearly schematic
channel drawing only. Current twophase2d remains periodic prescribed-flow CH;
nonperiodic first derivatives/BCs are not implemented. User now explicitly wants
NS, so do not silently substitute Stokes or prescribe a future velocity field.
Only geometry varies; material properties, initial droplet and driving law stay
fixed across samples. No current-turn Git authorization or training request.

Read latest progress/last five decisions, relevant numerical/JAX/development
guidance and source/archived scope. Existing earlier literature and solver
evidence is background, not a proof of NO convergence.

## Status and conclusion

Complete as an analysis, not an implementation. Endorse independent inward wall
profiles without an explicit eccentricity parameter. Honor full unsteady NS--CH
with two-way phase-to-momentum feedback. Do not replace it silently with the
previously proposed Stokes or fixed-velocity transport pilot.

## Independently checked geometry and flux identities

For a smooth common shape b(x) between0and1, zero near both channel ends:

$$
y_t=H-a_t b(x),\qquad y_b=a_b b(x),
$$

$$
h=H-(a_t+a_b)b(x),\qquad
c=H/2+(a_b-a_t)b(x)/2.
$$

Thus the amplitude sum controls narrowing, and the difference generates the
centerline displacement. Independent shapes/lengths/centers remain possible;
start with independent amplitudes and optionally a common transition length
to keep the geometry distribution manageable. Inlet/outlet geometry and the
common initial-droplet region stay unchanged. Enforce positive resolved gaps
and bounded geometric regularity without changing physical interface thickness
from sample to sample.

For a stationary2Ddomain, define Q(x,t)=integral_yb^yt ux dy. Leibniz plus
div(u)=0 gives Q_x=-uy(yt)+uy(yb)+ux(yt)*yt'-ux(yb)*yb'. Impermeable stationary
walls imply uy=ux*ywall' on each wall (no-slip is sufficient), so Q_x=0.
Consequently mean axial velocity is Q(t)/h(x). This is a cross-section identity,
not a lubrication approximation or an analytic full NS solution. Fixed-pressure
driving would instead allow the total Q to vary with geometry; recommend one
fixed inlet-flow law for interpretable comparisons.

## Coupled model and difficulty assessment

Propose the matched-property dimensional reference model as a traceable starting
point, not a selected runtime config:

$$
\rho(\partial_t\mathbf u+\mathbf u\cdot\nabla\mathbf u)
=-\nabla p+\eta\Delta\mathbf u-\phi\nabla\mu,
\qquad\nabla\cdot\mathbf u=0,
$$

$$
\partial_t\phi+\mathbf u\cdot\nabla\phi=M\Delta\mu,
\qquad \mu=A(\phi^3-\phi)-\kappa\Delta\phi.
$$

All rho/eta/M/A/kappa are positive constants. mu is the variational derivative of
integral[A*(phi^2-1)^2/4+kappa*|grad(phi)|^2/2]. The capillary work cancels phase
advection work up to boundary energy fluxes; no monotone-energy claim is made for
driven open channels. The alternate force mu*grad(phi) differs by grad(phi*mu),
so pressure/traction conventions must be transformed consistently if that form
is chosen. No copied quartic coefficients from MoFEM are used: its displayed
potential derivative has a coefficient inconsistency (W/4 versus4W), and this
analysis instead uses the independently differentiated A potential above.

Potential network outputs are(phi,mu,ux,uy,p), yielding phase transport,
chemical relation, two momentum components and incompressibility residuals,
in addition to explicit IC/BC handling. Shared backbone need not have5times
the parameter count, but PDE backpropagation, scale balance and wall/interface
resolution increase cost. Geometry point encoding and shared-time preparation
can carry over. No current benchmark justifies a runtime/memory/L2 estimate.

Hardest new work is consistent nonperiodic gradient/divergence and wall treatment,
pressure/incompressibility/capillary balance, and coupled physical optimization.
Periodic FFT and the periodic nonlocal-gradient control cannot be transplanted.
A scalar regional pair Laplacian is not a velocity Dirichlet/no-slip operator.
A pragmatic proposed first hybrid uses time JVP, first-order AD and separately
validated nonlocal second derivatives, with compatible pressure/traction and
phase wall conditions. Discrete energy/force balance is not automatic in a hybrid.

Recommend2D, fixed mild Reynolds/capillary regime, initially matched properties,
smooth moderate constrictions and a resolved carrier film, before breakup or
extreme property ratios. A common ramp from zero inlet flow plus zero initial
velocity avoids introducing independently random velocity data and initial
no-slip/inlet incompatibility. None of these numerical values or BC details is
silently selected. Validate manufactured/known flow and one fixed-geometry
coupled local/native reference and PINN before the geometry-conditioned NO.
Open-domain mass needs boundary-flux balance rather than a copied closed-grid
constant-mass projection. Specify pressure reference once via the chosen outlet
condition or gauge; do not add an inconsistent duplicate prescription.

## Primary sources and limits

- Jacqmin1999, DOI10.1006/jcph.1999.6332:
  https://www.sciencedirect.com/science/article/pii/S0021999199963325 .
  Publisher search abstract confirms incompressible phase-field NS, capillary
  coupling and discrete interface/kinetic energy exchange. Search rendering
  drops a gradient symbol; no equation is copied from that lossy excerpt.
- Abels/Garcke/Gruen, https://arxiv.org/abs/1104.1336 . Opened primary abstract
  confirms thermodynamically consistent differing-density models. It supports
  treating high density contrasts as a distinct constitutive/flux extension,
  not casually inserting rho(phi) into the matched-property equation above.
- MoFEM official NS--CH documentation:
  https://mofem.eng.gla.ac.uk/phase_field_doc.html . Opened full page strong-form
  section confirms momentum/divergence/phase/chemical blocks and separate
  phase/no-flux/wetting/velocity boundary work. Its unrelated free-surface slip
  multiplier setup is not adopted for the proposed no-slip channel.
- Trégouët et al.2018,
  https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.3.053603 .
  Opened primary abstract supports resolving flow stresses for transient droplet
  deformation in a constriction. No unseen full text or numerical parameter is
  used. An attempted arXiv HTML link for2110.11912v4 returned404 and is not used.

## Verification, diff and task-end

Analytical identities and current-source capability checks, primary literature
review, documentation inspection and `git diff --check` only. No numerical code,
config/default/dependency changes; no tests/training/reference solves/Git action.
Append decision, archive this self-contained task and update progress. Earlier
diagram/HTML changes remain intact. Changed memory files are this archived task,
`docs/DECISIONS.md` and `docs/PROGRESS.md`.

No blocker, required resume or queued experiment. Future implementation must
choose explicit scales, compatible IC/inlet/outlet/wall data and independently
validate the nonperiodic operators and complete coupled reference.
