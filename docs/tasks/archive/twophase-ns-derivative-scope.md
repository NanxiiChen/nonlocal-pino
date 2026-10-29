# Clarify prescribed-flow scope and derivative methods

Goal: answer whether the current two-phase pilot solves NS and how its first
derivatives are evaluated, using current source rather than its informal name.

Acceptance: distinguish given divergence-free velocity from solved NS;
identify time JVP, conservative Fourier first derivatives in the recommended
preset, legacy AD advection and nonlocal/local-AD diffusion. Correct stale
scope wording without changing any equations, training or defaults.

Context: current conserved preset inherits spectral_flux, phase_mass and the
original mixed CH residual. Network outputs phi/q; velocity_field is an
analytic prescribed constant or reversed vortex. There is no momentum,
pressure or continuity residual. User asks a conceptual implementation question.

Status: complete (2026-10-27).
Blocker: none.
Next action: none. Full NS coupling remains outside the implemented pilot.

Findings: losses.make_residual_fn uses a time JVP of fields.predict_fields;
D@fields supplies negative Laplacians for phi/q, with AD Hessian traces in the
local control. Advection delegates to make_flux_divergence when spectral_flux
is selected, otherwise a vmapped jax.grad of phase. Spectral flux forms phi*v,
transforms it, multiplies by ikx/iky and sums before inverse FFT. This is a
local divergence discretization despite global Fourier evaluation. Config
train_vortex_conserved inherits spectral_flux from train_vortex_consistent.
velocity_field supplies the analytic velocity; network returns only phi/q.
No solved velocity, pressure, momentum or incompressibility residual exists.
Thus prior ~3% phase results do not validate a coupled CHNS solver.

Changes: README Spatial controls now distinguishes legacy AD advection from
current conservative Fourier flux; losses.py module docstring has matching
scope. Executable code, equations, configs, data, models and training unchanged.
Source branches and documentation were inspected; local links and diff
whitespace are the proportionate checks. No runtime tests, compilation,
training, dependency or Git history operation. SOAP remains deferred.

Task-end memory: PROGRESS.md, DECISIONS.md and this archived record.
