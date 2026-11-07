# Particle-corrosion time horizon and kinetic coefficients

Date: 2026-11-06. Status: complete.

## Goal and acceptance

Explain the published default's actual simulation time, kinetic coefficients,
and dimensional interpretation to the user. Verify the resolved configuration,
saved run and implemented phase/concentration equations. Distinguish changing
the time horizon from changing phase mobility; do not change defaults or launch
training. No Git history operation is authorized this turn.

## Context and next action

The current task dashboard has no active task after the particle publication.
Default train_default imports train_online through train_wide/train_base. The
case is explicitly dimensionless, with t_max1,kappa2,n_ch80 and fixed matrix
properties; geometry alone is conditioned. Check provenance of these groups
and whether a dimensional time/mobility calibration exists before giving units.

Blocker: none. No pending action, training or numerical edits.

## Verified explanation

The resolved default is exactly equal to the saved online cold20000 run02
configuration. Its physical section and time endpoint also equal the reused8
local FEM reference configuration. Time spans0..1 dimensionlessly;8fresh times
per update are collocation samples, not eight time-integration steps.

The phase equation multiplies its full AC driving force by kappa2. Internal
coefficients n_ac1=24.3181818182,n_ac2=4,n_ac3=0.00222222222222 therefore become
48.6363636364,8,0.00444444444444 after that multiplier. Concentration transport
has n_ch80, unaffected by kappa. Matrix coefficients are uniform across space
and samples; ceramics are excluded domains with impermeable walls.

Inherited dimensional-group convention is L0*w*T0=4,
sqrt(w/(2*alpha))*Lc=30 and D*T0/Lc^2=80. Actual matrix mobility is2L0 in that
convention. No explicit dimensional T0,Lc,L0 calibration is assigned in this
pilot, so endpoint1 is not demonstrably1second and kappa2 is not a dimensional
mobility value. The README already identifies this as a controlled dimensionless
case rather than the source paper's calibrated material/time units.

For a possible more advanced corrosion stage, extending the horizon while
keeping coefficients fixed continues the same dynamics. Increasing only kappa
changes phase-relaxation/transport competition and is not an exact time
rescaling of the coupled system. A future1.5or2 horizon would require fresh
reference/training validation; none was selected or launched here.

Verification: code/config/reference comparison passed; numeric coefficients
evaluated from the actual default; numerical code unchanged. Documentation-only
diff/link checks passed. No commit/push authorized or performed this turn.
