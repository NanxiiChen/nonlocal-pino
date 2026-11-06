# Publish the ceramic-particle corrosion example and figures

Date: 2026-11-06 (Asia/Shanghai). Status: content verified; authorized Git publication pending.

## Goal and authorization

User requests recording the modest full-grid/core-halo difference, updating this
example README with a setup schematic and multi-sample field evolution figure,
then commit and push. Git authorization is explicit this turn. Follow repository
workflow: commit on dev, regular merge into main, push only origin/main. Do not
delete external artifacts, launch training or change numerical defaults.

## Context and selection

Current default remains rolling geometry/core-halo width192/Adam20000. Reused8
localFEM phase1.9827%/3.1209%,concentration2.3964%/3.7405% pooled/terminal.
Full-grid comparison1.7451%/2.8658%,c2.1093%/3.4481%,total1065.8vs959.0s.
The user judges the difference modest. Record these numbers without claiming
equivalence or universal ranking. Curate the default rolling model's saved
evolution on the first four of the same8comparison geometries, with actual
reference interfaces and snapshot times. Keep all other examples' images intact.

Default run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-geometry-spatial-w192-cold20000-02.
Default evaluation: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-w192-reused8-final-02.
Full comparison: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-full-grid-vs-core-halo-w192-01.
New setup SVG contains accepted known geometry and BCs only, clearly not solved
future fields. Use the existing code-native setup style and portable parameters.
One PNG per case under1MiB,all6PNG under2.5MiB;6setupSVG under320KiB combined.
No raw fields,weights,large galleries or logs in Git.

## Plan / exact resume

1. Add portable setup SVG and its parameter/provenance record; regenerate only
   the particle default evolution PNG from saved fields, preserving other images.
2. Update example README,assets docs and overview, maintain concise comparisons
   and exact config/results; review image quality, links and file sizes.
3. Inspect the full related pending example code/docs/test diff. Reuse the already
   completed numerical tests/audits; run proportionate renderer/artifact/diff checks.
4. Archive this record/update dashboard, commit dev, merge main and push main;
   verify remote main matches the published local commit.

Blocker: none. Next: commit the reviewed related changes on dev, regular merge
into main and push origin/main. Verify the remote, then archive this task and
publish the compact completion record through the same workflow.

Actual existing5SVG total253804bytes, leaving only8340bytes under the former
256KiB cap. Explicitly raise the six-figure cap to320KiB while preserving all
five prior SVG bytes and all PNG caps; do not simplify physics solely to meet
an obsolete figure-count budget. Source figures remain compact vector assets.

## Delivered content and validation

- Rewrote the example README around the actual problem, random ceramic prior,
  KKS mapping, five losses, rolling default, usage and one compact comparison
  table. Full-grid lowers pooled/terminal phase error by0.24/0.26percentage
  points with11.13%greater total runtime; preserve the current default.
- New setup SVG uses the first three accepted reused8 geometry records, copied
  as portable parameters. Revalidated all three and marked only initial data
  and boundary conditions, distinguishing inert ceramic from electrolyte.
- Replaced the curated particle evolution PNG with the default rolling model's
  first four reused8 layouts and five actual snapshots0,.2,.5,.8,1. Original
  conforming FEM triangles and reference interface contours are retained.
  Recomputed all eight saved-field phase metrics and verified config/source
  provenance against the selected run and independent reference manifests.
- New PNG230575bytes and SVG38812bytes, combined269387bytes. All six PNGs total
  2282199bytes and all six SVGs292616bytes. Existing five PNGs/five SVGs and
  their two manifests' case objects remain exactly equal to Git HEAD.
- Both renderer ASTs, all image/XML/JSON checks,53local README links and
  artifact inventory11cases/518registered directories pass. All six setup
  SVGs independently rerender to identical bytes; visual previews checked.
- All33example Python source files still match the completed full-grid run's
  source snapshot byte-for-byte. Reuse its already passed16focused+26legacy
  tests and production input/graph audits; no unchanged costly numerical suite
  or training was repeated for these documentation/renderer changes.
- Related pre-existing particle implementation/configuration/tests and task
  records are included in this authorized publication. Root/asset/development
  docs, provenance index, two renderers and compact assets are synchronized.
  No checkpoint, log, raw numerical arrays or large gallery is added to Git.

The first ad-hoc image check assumed every historical provenance record had a
pixel_dimensions field; one older record omits it. The corrected check verifies
all PNGs and validates recorded dimensions where present. Final check passes;
no image or scientific source was changed to accommodate the check.
