# Random smooth-wall NS–CH operator pilot

## Goal and acceptance

User accepts the random-wall design and authorizes a local commit of the current
implementation BEFORE new numerical changes, followed by an actual attempt.
Preserve existing results/defaults. Implement independently parameterized smooth
upper/lower stationary walls with fixed IC, flux, materials and physical BCs.
Require shared NumPy/JAX wall values/slopes, explicit bounded prior, fixed-shape
conditioning and rolling inputs, focused backward/geometry/operator checks, and
a few independent trajectories before making geometry-diversity claims.

## Context

- [Accepted design](archive/channel-random-wall-design.md).
- [Integral control](archive/channel-integral-nonlocal-control.md): phi2.07%,
  u2.69%,p3.81% on one stronger single-bump development geometry. Integral
  derivatives, AD streamfunction curl, continuous timeJVP40, SOAP, fixed scales.
- Generic two-stream point encoder already receives wall coordinates/normals/type
  and domain coordinates/fixed phi0; retain this architecture initially.
- Current dev=b2dff4b has accumulated validated loss/operator/default/design work.
  Save source and compact records only; generated figures/data/weights external.
- This turn explicitly authorizes commit; no separate push authorization inferred.

## Status / blocker / next action

Active. Startup memory/policy read and dirty tree reviewed. No blocker.
Next: finish source/content checks and create local checkpoint before changing
geometry code. Then qualify a smooth coefficient family, independent reference
trajectories, and a bounded physics-only neural-operator training pilot.

## Validation and results

Pending. Old expensive checks remain valid for the checkpoint; use proportionate
syntax/config/whitespace checks rather than rerunning unchanged GPU experiments.

## Session changes / exact resume

Task start only. No new numerical implementation, solve, training or cleanup yet.
