# Publish heterogeneous corrosion and preview ceramic geometries

Date: 2026-11-06. Status: completed publication and geometry preview.

## Goal and acceptance

User requests a polished heterogeneous_corrosion2d README, a setup schematic
and representative results, followed by commit/push. Afterwards begin the
geometry-conditioned corrosion extension with randomly placed, smooth irregular
aggregate-like ceramic particles and show several sampled cases. Honor this
order. No new training is needed before reviewing the initial geometry samples.

## Context

Successful nonlocal width96/Adam10000 fixed endpoint: mean local-FV phi1.2269%,
native phi0.8649%, four held-out maps. Existing paired AD and failed SOAP
controls remain documented; no numerical default change is requested.
At task start all earlier source/tests/tasks were uncommitted on main, remote
origin https://github.com/NanxiiChen/nonlocal-pinn.git. User now authorizes
commit and push. Follow dev -> commit -> main -> regular merge dev -> push main.

Figure conventions: repository assets/pino and assets/pino-setups, direct-path
provenance, small curated images; raw arrays/checkpoints remain external.
User cites examples/twophase2d/README.md; also retain the existing four PINO
figure style. Preliminary geometry physics: fixed impermeable particles,
neutral phase walls, KKS phase/q zero flux, open pit reservoir. Do not copy
closed-CH total-mass projection. No claims of solved new geometry dynamics.

## Plan and next action

1. Curate documented method/results and deterministic setup/evolution figures.
2. Validate source changes, links and artifact sizes; publish current results.
3. Only after publication generate smooth irregular-particle inputs, validate
   separation/containment and show representative samples.
4. Leave an exact implementation resume point if full PDE training is deferred
   pending the displayed geometry choice.

## Publication validation

README reorganized into problem, material input, KKS output, representative
strategy/results, reproduction and explicit convergence TODO. Mild CLI default
and selected stronger preset are distinguished; source/defaults unchanged.
Setup SVG uses three prescribed material maps and the same pit, without solved
fields. Evolution PNG uses all four held-out materials at actual saved times
0/.2/.5/.8/1; local FV contours and errors agree with the completed common-target
report. New PNG173611bytes, SVG54036bytes. All five PNGs2051624bytes (<2MiB),
all five SVGs253804bytes (<256KiB). Existing four figures are unchanged.

Both figures visually inspected.107local navigation/image links, all new Python
ASTs, JSON manifests, SVG XML and PNG integrity checks pass; git diff --check
passes. Source tests remain the previously passed11physics+7AD+3training cases;
no numerical code changed during documentation rendering. Read-only scope
audit finds no credentials, sibling dependencies, weights or raw field payloads.
Pending linked lightweight research records are included to avoid broken memory
links. Remote main still3418acf before publication. No blocker or cleanup.

Published: dev commit b173c0b, regular main merge c12b8c3. Push succeeded and
remote refs/heads/main verified as c12b8c33af83579bed7050c6b9115d11289f3718.
No raw fields, weights or logs were included. Returned to dev after publication.

## Geometry preview

After publication, added `examples/particle_corrosion2d`: explicit geometry
configuration, positive low-mode radial/ellipse sampling, analytic point
membership and outward particle normals, and a CPU preview command. Fixed six
particles, equivalent radius0.045..0.075, aspect1..1.45, shape modes3/4/5;
wall/pair/pit clearances0.025/0.035/0.03 and sampled minimum curvature radius0.015.
At least two particle boundaries must be within0.22of the initial pit interface;
this is a declared input prior, not selection using a future trajectory.

Clearance uses dense512/1024contours minus analytic arc-covering error bounds;
nested particles are rejected. Independent4096/8192area quadratures agree.
Curvature is checked at both resolutions, not certified by interval arithmetic.
Three focused CPU tests pass in0.322s (membership/clearance/area, analytic
normal/curvature versus finite differences, invalid-input/overlap rejection).

The initial preview call was interrupted before creating its output. On resume,
no old jobs/files remained. Actual GPU is RTX5090,32607MiB, idle; the new preview
uses CPU only. Generated four fresh accepted samples in2.26s, runtime provenance
seed737898905; no fixed training seed added. Direct outputs:

- `/root/autodl-tmp/data/particle_corrosion2d/20261106-geometry-preview-01/geometry_samples.png`
  (2074x1623,250865bytes; visually inspected).
- Same directory `manifest.json` (24231bytes), all exact particle coefficients,
  configuration, validation and zero PDE/training counters.

Area fractions11.26/10.92/10.26/9.75% in sample order (range9.75..11.26%).
Minimum certified pair/wall/pit gaps across
the quartet0.035892/0.047604/0.078293; minimum sampled curvature radius0.015158.
These are geometry inputs only, no solved corrosion evolution. Current material
pilot's diffuse thickness, mesh and horizon are not qualified for the particles.
The new source/README/tests and registry remain local on dev, after the requested
publication; no second commit/push was requested or performed.

Independent review confirms analytic normals/curvature and conservative
clearances; a positive-gap nested-circle example is correctly rejected. AST,
JSON, PNG and README links pass. Publication/preview deliverables complete.
The user subsequently explicitly activates the full particle-corrosion PINO
goal (1% target;3--5%fallback after repeated attempts) and local-AD comparison.
Continue in `docs/tasks/particle-corrosion-pino.md`; no trained result exists yet.
