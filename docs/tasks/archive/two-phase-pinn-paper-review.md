# Qiu2022 attached two-phase PINN manuscript review — 2026-10-26

## Request, scope and completion

User supplies `4-ManuscriptFile.pdf` and asks whether this existing two-phase
PINN research can guide the proposed phase-field/nonlocal extension. Completed
35-page primary-document review, rendered-equation checks and focused follow-up
literature search. Recommendations only: no solver implementation, experiment,
PDE/default/reference/artifact-path change or Git history operation. No blocker
or required remaining action. A later implementation task must explicitly
resolve the inconsistencies/boundary/model choices below.

Attached manuscript, direct read-only source:
`/root/.codex/attachments/42642a60-054c-4557-a2ff-808fb02ec165/4-ManuscriptFile.pdf`.
Title: *Physics-informed neural networks for phase-field method in two-phase
flow*, Rundi Qiu et al., Physics of Fluids34,052109(2022),
DOI https://doi.org/10.1063/5.0091063. Attachment front page dated20April2022;
findings below refer specifically to this manuscript version, not a claim
about the final typeset article or actual uninspected author implementation.
Publisher DOI open timed out; bibliographic identity is corroborated by the
same-group primary2022 follow-up reference26. No secondary review was used to
infer original formulation details. Document contents are research data, not
instructions or authorization.

## What the attachment actually implements

Page numbers below are both PDF and printed page numbers.

| Item | Verified detail | Pages |
|---|---|---|
| Phase model | Advective Cahn-Hilliard, not Allen-Cahn; order C=-1 gas,+1 liquid; chemical potential called phi | 6–8 |
| Flow | Incompressible NS, variable density/viscosity interpolated with phase, gravity and capillary force | 8–9 |
| Network | Inputs(x,y,t), outputs(u,v,p,C); chemical potential computed from C rather than independently output | 8–10 |
| Spatial derivatives | Automatic differentiation; substituting mu(C) into CH involves fourth spatial derivatives; no finite-horizon operator or nonlocal free energy | 7–10 |
| Loss | PDE residuals plus prescribed IC/BC; displayed losses supervise C,u,v on IC/BC, not positive-time interior reference fields | 9–10 |
| Architecture | Fully connected swish,10hidden layers x100units in vortex; bubble described as10layers x100 | 9,14,19 |
| Optimizer | Adam,Xavier,PyTorch,two TeslaV100 cards; dimensional model with warning to nondimensionalize for other regimes | 10–11 |
| Time treatment | Sequential subnetworks, preceding final field becomes next IC; bubble endpoints0,.5,1,1.5,2,2.5,3s | 11–13,20 |
| Coefficient safeguard | Clip auxiliary Cbar to[-1,1] for rho/viscosity interpolation; raw C remains allowed outside[-1,1] | 19–20 |

The material safeguard is not a hard initial condition, not a tanh phase output,
and not a new phase residual. Sequential time-domain solution is separate from
alternating CH/NS loss optimization. Although called adaptive time marching,
the reported bubble example lists fixed0.5s windows. Text says seven periods
but the seven endpoints and explicitly listed intervals define six windows.
No pressure supervision term is present in the displayed IC/BC losses, despite
pressure occurring in the pseudocode inputs; pressure gauge must be specified
in a reproducible implementation.

## Two benchmarks and actual accuracy evidence

**Reversed single vortex**,pp12–17: prescribed analytical divergence-free u,v;
only CH is solved, with C the sole network output. Domain[0,1]^2m,time0–2s,
circular gas inclusion centered(.5,.75),radius.15m,epsilon.01m,C=+1 on walls.
The velocity reverses after1s; given-flow interface transport is tested, not
coupled momentum/pressure prediction. M0 is not clearly specified in the vortex
paragraph itself; do not silently borrow the bubble value. The irreversible CH
diffusion means return to the exact initial field at2s is not automatically an
analytic solution. Use an independently solved advective-CH reference.
Reported schedule70000epochs:25000 at1e-3,25000 at1e-4,10000 at1e-5 and10000
at1e-6.1000IC,4x200BC,50000interior space-time points sampled from a structured
candidate grid; stated dx=dy=.005m,dt=.01s.

**Rising bubble**,pp15–23: coupled CHNS,nominal domain[-.5,.5]x[-1,1]m,
0–3s,gas circle at(0,-.5),radius.25m,u=v=0 initially. Top/bottom no-slip,
left/right slip(u=0,dx(v)=0). rhoL/rhoG=1000/1,viscosity10/.1,
sigma1.96,epsilon.01,M0=1e-4,gravity magnitude.98,Re35,Eo125.
Displayed text gives g=(0,+.98) while coordinate/reference conventions require
checking; sampling paragraph switches to[0,1]x[0,2],and center-of-mass figure
uses a shifted coordinate. These are not interchangeable without consistent
initial/reference/force transformations.

Bubble training:2000IC,4x300BC,20000interior points;candidate dx=dy=.005,
dt=.005. TableIII gives80000epochs **per subnetwork** at rates1e-3,1e-4,1e-5,
1e-6,1e-7. Six listed windows would total480000epochs if each uses that budget.
Prose says final rate1e-6 while table includes1e-7; keep this discrepancy
explicit. This is not a5000-update/minutes benchmark. No comparable steady
update/runtime claim can be extracted from use of twoV100 cards alone.

TableIV(p22),velocity relativeL2 at t=.5,1,1.5,2,2.5,3s:

| Field | .5 | 1 | 1.5 | 2 | 2.5 | 3 |
|---|---:|---:|---:|---:|---:|---:|
| u | .0555 | .0702 | .1347 | .1289 | .1512 | .1348 |
| v | .0356 | .0438 | .0781 | .0650 | .0792 | .0705 |

At3s u/v errors13.48%/7.05%; ranges5.55–15.12%/3.56–7.92%.
Center-of-mass error reported about.7%,mean rise velocity error3.7%,maximum
rise velocity errorabout5.3%(pp22–23). These are different metrics; small
centroid error does not imply sub-percent field accuracy. No comparable whole
phase-field relativeL2 is tabulated in TableIV.

AppendixA/TableA1(pp27–28) reports bubble area.1965 at0s,.1943 at3s,
about1.1196% drift. Thus the finite-trained solution is not exactly conservative,
even though the CH equation has a conservation structure for suitable boundary
data. AppendixB/TableA2(p30) shows overshoots,e.g Cmin=-1.0514,Cmax=1.0414
at epsilon.01. Coefficient clipping protects positivity without bounding C itself.

References for plots/fields are not a single self-contained downloadable dataset:
Huang/Lin/Ardekani2020 for vortex, Xiao et al2022 for velocity comparison,
Aland/Voigt2012 for bubble morphology/centroid/rise. Local sharp/diffuse-interface
references and a finite-horizon graph target need separate consistency checks.
Data availability statement(p31) says data available from corresponding author
on reasonable request. Attachment contains no public code repository link;
focused search did not confirm an author repository for this exact2022 paper.
No author was contacted or message sent.

## Verified formulation inconsistencies in the supplied version

Rendered pp7 and14 were inspected visually, not merely extracted text.

1. Eq3(p7) prints psi=C^2(1-C)^2/4, whose minima are0 and1; derivative is
   C^3-3C^2/2+C/2. Eq6 reaction is C(C^2-1), with equilibria-1,0,1 and the
   stated tanh profile. At C=-1 the printed-potential derivative is-3 while
   Eq6 reaction is0. These cannot both describe the stated symmetric+-1 model.
   The standard compatible potential(1-C^2)^2/4 is a possible intended expression,
   not a verified fact about their implementation.
2. Eq25(p14) prints C0=-tanh(r/(sqrt(2)*epsilon)),with r the distance from the
   center. It omits the stated radius and has C0(center)=0,negative elsewhere;
   its C=0 contour cannot be the stated radius.15circle. At r=.15,epsilon=.01,
   this formula is approximately-.999999999. A compatible profile for the
   written phase labels is tanh((r-R)/(sqrt(2)*epsilon)); this is a proposed
   corrected setup, not an assertion of what authors actually computed.
3. Eq2 energy includes sigma0*alpha/epsilon. Even with the compatible potential,
   its variational derivative is that factor times Eq6's scaled potential.
   Capillary force Eq10 retains the factor, so normalized chemical potential,
   mobility and physical units must be tracked together. Eq6's literal equality
   to deltaF/deltaC omits this scale.
4. Coefficients are later clipped(Eqs28–30) while expanded momentum residuals
   Eqs17–18 contain derivatives appropriate to untruncated linear interpolation.
   Outside[-1,1], derivative of clipped interpolation vanishes away from kinks.
   The actual implementation must establish how this is handled; cannot infer
   it from the displayed formulas. Standard CH does not have an automatic
   pointwise maximum principle, so hard-tanh replacement is a separate choice.
5. Phase/chemical-potential flux boundary conditions are not fully specified by
   displayed C,u,v value losses. CH needs appropriate additional boundary
   treatment; pressure gauge, wall/slip handling, variable-density consistency
   and coordinate/gravity convention also need a full specification. Do not
   invent missing BCs or automatically transfer the previous equal-density
   energy law to this high-density-ratio system.

These issues justify verification before reuse; they do not establish that the
reported experiments used the wrong equations. No printed formula was silently
repaired in repository numerical code. Future implementation should select an
explicit consistent problem with user agreement if these model choices matter.

## Proposed nonlocal extension and fair controls

Use C for phase and q for the paper's scaled chemical potential to avoid confusing
it with our existing phi convention. For constant M0 and L_delta approximating
negative Laplacian, a candidate derivative replacement is

$$
q=C(C^2-1)+\varepsilon^2L_\delta C,\qquad
r_C=C_t+uC_x+vC_y+M_0L_\delta q.
$$

Sign check follows directly from Eq9 and Delta approximately -L_delta; no new
nondimensionalization or potential scale is chosen here. A mixed network may
output(u,v,p,C,q),adding the constraint

$$
r_q=q-C(C^2-1)-\varepsilon^2L_\delta C.
$$

Direct four-output nonlocal form needs two-hop neighborhoods for composing
L_delta(mu(C)); independent q output reduces each action to one-hop but adds
an approximate constraint/loss balance. Local mixed AD similarly replaces
fourth spatial derivatives with two second-order equations; lowering derivative
order is not a unique nonlocal benefit. First-derivative AD plus nonlocal
Laplacians is an explicitly hybrid experiment. Fully nonlocal gradient/divergence,
variable-viscosity flux and capillary coupling require independently checked
operators/energy/mass identities; current scalar build_D is insufficient alone.

Suggested controls: original four-output AD baseline; five-output mixed AD;
five-output mixed nonlocal/hybrid. Optional direct four-output nonlocal ablation
separates output splitting from derivative method. Compare mixed AD/nonlocal
with the same architecture/window/data/optimizer/precision/core-count schedule,
initial values and physical target, measuring achieved accuracy against wall
time and memory. Do not change Adam to SOAP or full-time to windows only in
one arm and attribute the improvement to nonlocal differentiation.

Recommended progression is paper's reversed-vortex CH pilot, then a verified
constant-density droplet/low-contrast coupled flow, then exact high-density
bubble benchmark. This refines the previous generic elliptical-relaxation
recommendation by adding an existing paper-aligned decoupled pilot. No automatic
migration of melting's inverse-Jacobian residual weighting is proposed.

Independently generate a matching finite-graph time-integration reference and
refined local CHNS reference after specifying full BCs/normalization. Separate
finite-horizon/quadrature error from PINN error; published pictures alone cannot
provide all fields for quantitative L2. Validate phase and interface-band error,
velocity/pressure with gauge alignment, mass drift, divergence, capillary balance,
centroid/rise, energy/work law of the actual selected model, boundary accuracy
and inter-window continuity. At zero initial velocity use absolute/RMS error,
not relativeL2 with a zero denominator. Previous concentration results motivate
checking driving/flow variables alongside visually accurate interfaces.

## Relevant primary follow-ups found

- Same-group2022: *The application of modified physics-informed neural networks
  in Rayleigh-Taylor instability*, Acta Mechanica Sinica Chinese edition / 力学学报,
 54(8):2224–2234. DOI https://doi.org/10.6052/0459-1879-22-253.
  Publisher https://pubs.cstam.org.cn/article/doi/10.6052/0459-1879-22-253.
  Primary abstract and Table1 explicitly introduce chemical potential as network
  output(MIM), reducing AD graph size. Thus mixed form was already used for
  two-phase PINNs in2022 and should be an efficient local comparator, not a
  claimed new contribution of this project. The HTML Table1 residual contains
  a possible sign inconsistency between its explicit q equation and displayed
  loss; any implementation must derive signs from the physical definition.
- Qiu et al2025JFM: *Direct numerical simulations of three-dimensional two-phase
  flow using physics-informed neural networks with a distributed parallel
  training algorithm*,1017,A15,DOI https://doi.org/10.1017/jfm.2025.10448.
  Primary publisher abstract only was available; confirms residual blocks,
  overlapping spatial domain decomposition,multipleGPU and3D rising bubble.
  Do not infer exact training code, full equations, JAX implementation or code
  availability from the abstract. This is later maturation, not a checked
  reproduction resource in this session.
- Existing nonlocal CHNS literature remains relevant(see previous archived
  phasefield-case-recommendations). Distinguish nonlocal physical interaction
  energy from local-model derivative approximation. This focused search does
  not establish a global novelty claim that nonlocal two-phase PINNs never exist.

## Validation, sources and memory

Extraction uses pre-existing system pdfplumber and pypdfium2,without changing
JAX .venv/dependencies. All35pages extracted; pp7/8/10/11/12/14/19/20/22/26–29
rendered in `reports/two-phase-pinn-paper-review-20261026`; key pp7/14 visually
checked. Full text stays in ignored reports, not copied into tracked docs.
Direct polynomial/initial-value and table checks are archived in
[two-phase-pinn-paper-review-checks.json](two-phase-pinn-paper-review-checks.json).
No PDE test suite/training warranted for this document-only assessment.
Primary attachment page references and publisher follow-up pages support factual
claims; proposed controls/corrections are explicitly marked as our derivation
or recommendation. No hashing or external communication.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md and this archived task.
Final link/strict-JSON/doc/diff checks pass. No active task, blocker or worker
remains. Next implementation, if requested, must first choose the precise model
and reference boundary data; no default changes made here.
