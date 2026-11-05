# Grain migration: quasistatic mechanics and loading history

## Goal
Explain whether the proposed mechanically driven grain migration problem is
quasistatic or time dependent, and whether load increments are required.

## Acceptance criteria
- Distinguish mechanical inertia neglect from time-dependent grain kinetics.
- State when a fixed-microstructure elastic load can be applied in one solve.
- Explain phase time integration under constant load and evolving equilibrium.
- Distinguish load path, time discretization, nonlinear iteration and PINO time
  windows; no numerical run or model selection is authorized by this question.

## Context
The proposed pilot is small-strain elastic polycrystal plus Allen-Cahn grain
evolution. No actual material, kinetics or loading history has been selected.

## Status
Complete (2026-11-05). Primary model references and independent conceptual audit
confirm the equilibrium/kinetics and loading/discretization distinctions.

## Blocker
None for the conceptual distinction.

## Next action
Await physical-model/loading selection; no numerical implementation requested.

## Conclusions

- Neglecting inertia and body forces gives div(sigma)=0 at every time; the
  grain order parameters obey time-dependent Allen-Cahn kinetics. Time maps to
  real seconds only after kinetic parameters and units are established.
- Fixed-microstructure small-strain linear elasticity permits a direct target
  load solve (with adequate displacement constraints). The phase need not be
  equilibrated at that instant.
- Direct load-and-hold is an idealization after fast mechanical adjustment,
  before appreciable grain migration. A physically finite loading duration
  long compared with mechanical response but short compared with migration
  supports it. Real impact/stress waves are outside quasistatic mechanics.
- After loading, phase evolution changes stiffness distribution and stresses;
  initial stresses cannot be frozen. Strain control and stress control are
  different experiments. Initial displacement must be consistent with the
  imposed boundary loading and microstructure, not arbitrarily set to zero.
- Slower ramps allow migration during loading and can yield different states
  at the same endpoint strain. Load increments are not identical to physical
  time steps, nonlinear solver iterations or training windows.
- Classical solvers step time and enforce coupled equilibrium; a space-time
  PINO can represent the whole interval without mandatory time windows. This
  does not remove the kinetic time derivative or guarantee convergence.
- One energy minimization instead targets an equilibrium; it cannot substitute
  for finite-time migration and may miss dynamically accessible/metastable
  states. A fixed short smooth ramp plus hold is a possible first study, not
  an implemented/default setting or mandatory physical protocol.

## Primary references

- https://mooseframework.inl.gov/modules/phase_field/Grain_Growth_Model.html
  (Allen-Cahn grain evolution and transient implementation).
- https://mooseframework.inl.gov/modules/phase_field/Elastic_Driving_Force_Grain_Growth.html
  (orientation-dependent elasticity coupled to phase driving).

## Completion verification

Documentation-only changes: PROGRESS, DECISIONS and this archived explanation.
Inspect the record, check local links and git diff whitespace. No code tests,
numerical solve, training or Git history operation.
