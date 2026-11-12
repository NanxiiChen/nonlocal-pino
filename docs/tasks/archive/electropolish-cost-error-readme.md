# Add the approved cost–error figure to the example README

Goal: embed the user-approved three-curve Electropolish figure in its example
README, with a portable PDF link and source provenance. The numerical results
and table already exist; use the verified final exports without redrawing.

Acceptance: one PNG/PDF pair in `assets/pino-studies/`, byte-identical to the
approved external files; valid relative README links, concise metric caption,
source/style/validation paths recorded; no numerical or default changes.

Context: completed study is
`docs/tasks/archive/electropolish-three-curve-cost-error.md`; source spec is
`artifacts/electropolish_cost_error_three_curves.json`. Twelve final10k models,
64 common local-FD test inputs each, three particle counts and four delta/h
ratios. Current turn authorizes README assets, not Git/Pages publication.

Status: complete. Copied `assets/pino-studies/electropolish2d_cost_error.png`
(94,046bytes) and `.pdf` (24,239bytes) directly from the approved exports.
The example README now embeds the PNG and links the PDF and copy provenance;
`assets/README.md` documents the pair and regeneration command. New
`artifacts/electropolish_cost_error_readme.json` records direct source and
repository paths, exact byte sizes, renderer and existing validation records.

Validation: both copies byte-identical,118,285bytes total; JSON/source paths
and every relative link in the edited sections pass. `git diff --check` passes.
Reused the original figure's font/visual/numerical checks; no redraw or code
execution is needed for this exact copy. No numerical/default/Git/Pages change.

Memory updated: PROGRESS, DECISIONS and this archived task.
Blocker:none. Next:no required work; commit/publication only if requested.
