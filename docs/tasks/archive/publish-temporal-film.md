# Publish the temporal FiLM clarification

Goal: commit and push the reviewed diagram and explanation and synchronize
the technical report on GitHub Pages.

Acceptance: `main` contains the changes, the Pages workflow succeeds for the
merge, and the live SVG, homepage and architecture page match the build.

Context: [local validation](temporal-film-diagram.md). The prepared
website sources show all three decoder hidden layers and the linear head.

Changes: release the updated SVG, overview embed and architecture explanation
with the existing validation records. The site continues to use the existing
Pages workflow after merging dev into main.

Validation: the nine-page build succeeds. The output SVG matches its source
and has both temporal decoder panels. Both pages embed the updated dimensions,
and the architecture page includes rendered math. `git diff --check` passes.
The remote acceptance check compares these three built files to the served
files after the workflow succeeds.
