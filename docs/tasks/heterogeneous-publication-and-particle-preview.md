# Publish heterogeneous corrosion and preview ceramic geometries

Date: 2026-11-06. Status: active, publication stage.

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
All earlier source/tests/tasks are presently uncommitted on main, remote
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

Next: commit on dev, regular merge/push main; only then begin geometry previews.
