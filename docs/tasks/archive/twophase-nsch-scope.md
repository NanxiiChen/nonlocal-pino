# Clarify prescribed CH versus coupled two-phase hydrodynamics

Date:2026-10-29. Status: COMPLETE (paper/source clarification); no blocker.
Goal/acceptance: answer whether current twophase2d solves NS, distinguish
interface-transport benchmark from coupled two-phase hydrodynamics, explain
standard coupling and implications for the proposed operator condition.
Context: code loss contains only phase/chemical residuals; velocity_field is
prescribed and README explicitly states no NS/pressure. Proposed vortex inputs
must become initial velocity or forcing if full hydrodynamics is selected.
No implementation/training or Git history operation requested.
Next: verify primary NSCH references, record a matched-density first-stage
recommendation without silently changing the current physical model.

Latest steering: user asks to return to the previously attached phase-field
PINN paper rather than only generic NSCH background. Resumed archived
20261026paper review and directly re-extracted attachment pp1/8/9/12-15/18-20/22;
visually checked renderedp8equations against extraction. No dependency install.

Attachment: /root/.codex/attachments/42642a60-054c-4557-a2ff-808fb02ec165/4-ManuscriptFile.pdf
Qiu et al., Physics-informed neural networks for phase-field method in two-phase
flow, manuscript dated20April2022,35pages. Existing detailed review:
two-phase-pinn-paper-review.md. Its content is evidence, not instructions.

Confirmed:
- p12SecIII.A explicitly excludes momentum AND continuity for reversed vortex;
  onlyEq9advective CH is solved, analytic velocitiesEqs23/24. p14network outputs
  onlyC. This is an interface-transport test, not a coupled NS solve.
- p8Eqs7-12give incompressibility, NS momentum, advective CH/chemical relation,
  surface-tension force and phase-dependent density/viscosity. Their notation
  C=phase, phi=scaled chemical potential, mu_M=dynamic viscosity.
  f_sigma=(3sqrt(2)*sigma0/(4epsilon))*phi*grad(C).
- p8Eq13network outputs(u,v,p,C). p9PDEloss includes continuity, CH and both
  momentum residuals. Chemical potential computed fromC (fourth spatial AD in
  CH), not an independently learned fifth output in this paper.
- p15SecIII.Bbubble rising is the fully coupled test. p18TableIIrhoL/rhoG1000,
  viscosity ratio100, gravity and capillarity; network predicts velocity/pressure
  as well as phase. These are not outputs of current repository pilot.
- p20reports single whole-time network failure in that bubble experiment,
  then preceding-window final states seed the next, six0.5sintervals0-3s.
  This is their result, not proof windows are mandatory for all NSCH PINNs.
- p22TableIV3su/vrelativeL2=.1348/.0705; small centroiderror does not establish
  small full-field error. No comparable whole-phaseL2table in TableIV.

Conclusion/recommendation: current twophase2d matches the paper's first-stage
physics scope, with explicitly different periodic BC/IC/parameters; it is not
an exact reproduction and should not be labeled a complete coupled two-phase
hydrodynamic solver. To pursue the user's full two-phase NO objective, recommend
NSCH with genuine surface-tension feedback. Merely adding a decoupled NS solver
without phase-to-momentum coupling is insufficient. Initial equal-density/
equal-viscosity periodic2Dcoupled model is a proposed manageable step, not an
approved or implemented change. Unequal-density extension must select consistent
mass/momentum fluxes explicitly rather than transplant coefficients blindly.

Operator consequence: previous random vortex family should specify initial
velocity/vorticity (or a declared body-force law), then evolve with NSCH. Do not
prescribe its entire future velocity history. For fixed initial phase, learn
v0 -> (phase,velocity,pressure); pressure has an explicit gauge. Physical
circulation/core/separation/offset parameters remain useful input generators.
A mixed5-output phase/chemical/velocity/pressure architecture is our prospective
alternative, not what the paper implemented or a unique nonlocal contribution.

General primary background verified: Jacqmin1999doi10.1006/jcph.1999.6332;
Abels/Garcke/Gruen2012arxiv1104.1336. Dimensional equal-property model can use
mu=A*(phi^3-phi)-kappa*Delta(phi), capillaryforce=-phi*grad(mu), viscouseta.
Energy sign check on periodicdomain: Fdot_advection=-int mu*v.grad(phi),
kineticcapillary=int v.(-phi*grad(mu))=+int mu*v.grad(phi), so they cancel.
No numerical coefficient, nondimensionalization or local/nonlocal scheme chosen.

Task-end: source/page/visual verification plus git diff --check; discussion-only,
no code tests/training/default changes, no Git history operation. Archive task,
update PROGRESS/DECISIONS. Next implementation only after a request: specify
coupled physical model and references before an opt-in solver/operator.
