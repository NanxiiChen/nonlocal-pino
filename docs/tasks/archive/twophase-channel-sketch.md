# Constricted-channel concept sketch

## Goal and acceptance

Show the proposed smooth contraction--expansion microchannel spatially, with
throat width, transition length and wall asymmetry represented. Keep the initial
droplet and driving fixed. Distinguish illustrative later droplet positions from
computed solutions; do not fabricate a velocity/pressure field or breakup result.
Deliver a concise in-conversation visual, not another detailed explanation.

## Context and status

Previous discussion complete in `archive/twophase-geometry-design.md`; no new
PDE/BC implementation is authorized by this request. Read current progress, last
five decisions and full visualize instructions. Use a self-contained responsive
HTML fragment with authored spatial SVG, no external resources or controls.
Reusable source lives in `docs/tasks/twophase-channel-sketch.html.in`; display
copy uses the authorized thread visualization directory.

Complete. The compact diagram shows a fixed initial circle, equal-area schematic
later ellipse/circle, throat width and transition length. Three lower examples
show wider/narrower/off-center throats with common straight inlet/outlet and
the same initial circle. No velocity/pressure field or solution accuracy is shown.

## Deliverable

- Source: `docs/tasks/twophase-channel-sketch.html.in` (6934bytes).
- Display fragment:
  `/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/twophase-channel-sketch.html`.
- Compact browser receipt: `docs/tasks/twophase-channel-sketch-validation.json`.
- Preview wrapper and six small screenshots remain in the thread visualization
  directory; no generated media enters the repository index.

## Verification

Source and display copy are byte-equal; fragment parser/wrapper accepts the
literal self-contained markup. Existing isolated Chromium/Playwright checks
light/dark themes at800/390/320px viewport widths. Four diagrams and four initial
droplets render, no horizontal or SVG-text overflow, no SVG label collisions,
and no JavaScript errors. Schematic ellipse/circle areas agree to roundoff.
Desktop/light and mobile/dark screenshots inspected visually.

Inspection initially found crowded labels at320px. Compact labels now read
拉伸示意/恢复示意 and a stronger collision check passes. First preview refresh
was refused because the generated wrapper already existed; rerendering that
task-owned wrapper with explicit --force removes the stale preview and all six
checks pass again. No physical threshold or training result is involved.

`git diff --check` passes. No runtime/config/PDE/default change, training, solver,
dependency installation, external artifact cleanup or Git-history operation.

## Diff-level changes and task-end

Add the small diagram source and browser receipt. Update `docs/PROGRESS.md`,
append the illustration decision to `docs/DECISIONS.md`, and archive this task.
Earlier uncommitted diagram/HTML work remains intact. No blocker or required
resume; no numerical jobs are queued.
