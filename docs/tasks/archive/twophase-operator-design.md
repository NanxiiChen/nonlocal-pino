# Two-phase transport operator conditioning design

Date:2026-10-29. Status: COMPLETE (design recommendation only); no blocker.

Goal: compare random initial interfaces versus random prescribed velocity fields
for a useful and visually clear physics-only neural operator extension.
Acceptance: inspect actual current PDE/constraints and conditional architecture;
recommend a scoped input family, training/reference/metric plan, and distinguish
prescribed transport from coupled NSCH. No implementation/training requested.

Context: two prior operators already condition on random initial interfaces.
Current twophase2d has prescribed periodic divergence-free reversed vortex,
CH mixed phase/chemical fields, FFT conservative first derivatives, nonlocal
Laplacians, time JVP and exact grid phase mass. It does not solve NS/pressure.
Current direct phase output is default; inverse cubic is optional, no latent
tanh residual default. Preserve existing defaults/checkpoints/artifact paths.

Next: compare routes and propose low-frequency divergence-free spatial velocity
families via streamfunction with fixed known time envelope. Check primary
operator literature for precedent; record recommendation without numerical claims.
No Git history operation authorized in this turn.

Findings/recommendation:
- Prefer fixed initial droplet + random low-frequency spatial velocity structure
  first: complements existing random-IC melting/electropolish operators and
  isolates effects of flow on interface deformation. Random IC is a useful
  later control/extension; neither input type is intrinsically more of an NO.
- Suggested family: periodic streamfunction psi(x,y;xi), velocity
  v=a(t)(d_y psi,-d_x psi), fixed known a(t), initially retaining existing vortex
  reversal. Mixed derivatives commute, so continuum divergence is zero. Use
  several low Fourier modes/continuous coefficients with bounded speed/strain;
  varying only a global speed/translation is too narrow for the intended demo.
  This is a proposed finite-band continuous family, not arbitrary-flow coverage.
- Encoder input can be two sampled spatial velocity components; queries remain
  x,y,t only, with all samples sharing query sets. Fixed a(t) removes ambiguity
  about future forcing; arbitrary temporal histories need additional input.
  Reuse conditional patch encoder/shared spatial preparation/time JVP; later
  joint IC+velocity conditioning can use three image channels. Periodic query
  representation must be maintained explicitly in the adaptation.
- Preserve CH mixed two-field physics, direct phase output, exact grid IC/mass
  wrapper and only two PDE losses. No duplicated mass loss, automatic tanh or
  new residual preconditioner. Solver has prescribed transport, not NS/pressure.
  Full-grid FFT flux and mass projection mean prior core+halo sampling cannot
  be copied unchanged. Start from existing full periodic quadrature/grid.
- Online random conditions; fixed independently held-out64-100samples. Validate
  local-reference grid/time errors and native/local bias before attributing NO
  errors. Measure pooled/terminal AND peak-deformation phi error, interface-band
  error/contours, chemical error and mass; reversal can obscure errors if only
  final time is shown. Do not impose monotone energy under prescribed strain.
- Future true coupled NSCH: parameterize initial velocity/forcing rather than
  treating the full prescribed velocity history as an unknown NS solution.
  No NSCH formulation or present physical model changed here.

Primary background inspected: PINO https://arxiv.org/abs/2111.03794 (physics-only
operator feasibility); PINTO https://arxiv.org/abs/2412.09009 (cross-attention,
physics-only condition-dependent operators). These support the general route,
not an accuracy guarantee for this proposed CH velocity family. Own proposal
and current code findings are distinguished from literature claims.

Validation/task-end: inspected README, problem/fields/losses/advection and
recommended config chain; source explicitly confirms prescribed velocity and
full-grid conservative FFT/projection. Documentation-only, no code tests or
training needed. Git diff whitespace check required. No figures/binary files,
new numerical presets or Git history operations. Changed memory: PROGRESS,
DECISIONS, this archived record. Next action only if implementation requested:
freeze explicit velocity distribution/bounds and reference gates, then build
an opt-in operator through train.py, preserving old single-IVP defaults.
