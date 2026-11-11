# Relax continuous-field colormap prescriptions

## Goal and acceptance criteria

User requests removal of fixed continuous-field/error cmap prescriptions from
`docs/PLOTTING_GUIDELINES.md`. Each example can choose an appropriate cmap;
keep NPG for categorical plots and preserve basic comparison consistency.
No redraw, font change, code behavior or Git operation is requested.

## Context

The completed plotting-guidelines task introduced fixed NPG-derived continuous
color nodes. This request explicitly supersedes that choice. Previous data,
metrics, typography and export requirements remain unchanged.

## Changes and validation

Removed the continuous-field/error subsection and fixed gradient-node table.
Restricted the overview and section title to categorical NPG colors; moved only
basic comparable-field color-limit consistency to the layout section.
Checked the edited text, removal of stale prescriptions, and local links.
`git diff --check` passes. Documentation only; no behavior tests or redraw.
Updated memory: `docs/DECISIONS.md`, `docs/PROGRESS.md`, and this archived task.

## Status

Complete.

## Blocker

None.

## Next action

None pending. Each example's future plotting work chooses its own field cmap.
