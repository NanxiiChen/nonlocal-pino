# Record and defer FFT boundary limitations

Goal: record the periodic-boundary limitation of the current FFT first spatial
derivative, defer further work and retain the existing recommended strategy.

Acceptance: standalone deferred issue linked from the case README and progress;
state current FFT/nonlocal/time choices and that both current spatial operators
use periodic geometry. No boundary redesign or resumed nonlocal experiments.

Status: complete (2026-10-27).
Blocker: none.
Next action: none. The boundary research itself is explicitly deferred; resume
only on the user's request.

Changes: docs/issues/twophase-fft-periodic-boundary.md records the limitation,
current derivative choices and explicit deferral. README Spatial controls links
the issue and distinguishes network BC constraints from operator assumptions.
DECISIONS and PROGRESS record the choice and the deferred issue. Both the
current Fourier and nonlocal implementations remain periodic, so no claim of
existing arbitrary-boundary support is made.

Inspection confirms train_vortex_consistent sets advection=spectral_flux and
train_vortex_conserved inherits it. No runtime/config/default/reference/model
or Git history changes; no training, tests, compilation or further numerical
investigation is needed for this documentation-only task. All90local links in
five affected documents, new issue/task whitespace and git diff --check pass.

Task-end memory: PROGRESS.md, DECISIONS.md, the deferred issue and this archived
record. Case README is synchronized; no active implementation task remains.
