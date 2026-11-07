# Publish the six-particle corrosion example

Publish the accepted width-256, 20,000-update result with its overview row,
training recipe, example README, curated figures and reproducibility sources.
The user authorized a commit and push in this turn. The release follows the
local dev branch and a regular merge into main before pushing origin/main.

The accepted study and artifact provenance are recorded in the
[physical optimization archive](particle-corrosion-physical-optimization.md).
The working tree contains the related physical studies, configuration presets,
regressions and cross-reference unit documentation. Include these so the
overview, documented preset and historical controls can be read and reproduced
together.

## Publication checks

All 107 particle-corrosion CPU regression tests passed in 364 seconds. The
explicit width-256 preset matches the saved run configuration. All 79 frozen
source files match the repository, and the link, image-size and staged diff
checks passed. The selected PNG and SVG remain the curated presentation of the
20,000-update endpoint. Checkpoints and reference fields use the recorded
external artifact paths.

The root README puts the case in the results and training strategy tables and
links its run guide. The publication also includes the pending particle model
code, presets, tests, prior study records and unit documentation that accompany
the example.

The example was committed on dev as af2800d and merged into main as e0b2793.
An independent remote lookup confirmed e0b2793 on origin/main. A documentation
follow-up corrects the relative study link after archiving this record.
