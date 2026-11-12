# Publish the architecture/training master figure

Authorization: user explicitly requests commit and push of the accepted master
figure. Scope: its PNG/PDF/SVG, renderer/specification/validation, root README,
asset guide, website placements/build/Pages trigger and relevant task memory.
Other numerical experiments and the separate cost–error publication stay local.

Workflow: commit on dev, regular merge to main, push only main. Use a clean
temporary main worktree for the merge if local research hunks overlap these
documentation files; preserve every unrelated change without stash/reset.

Acceptance: inspect the staged-only tree; validate its build, links and asset
identity; push; identify the Pages run for the exact merge commit; verify the
served architecture files and HTML against the inspected build. Existing
font/visual/source checks are in `artifacts/pino_master_figure_validation.json`.

Staged validation:17selected files; no numerical implementation/config/test
changes. Exact staged tree exported to `/tmp/pino-master-release-s9vhwze7/snapshot`.
Nine-page build,12new Markdown destinations, approved asset identity,
Python/JSON syntax and `git diff --cached --check` pass. Reuse prior font,
visual, desktop/mobile checks since the presented artwork is unchanged.

Status: ready to commit. Blocker:none. Next: commit on dev, merge in a clean
main worktree, push origin/main and verify the exact deployment and live files.
