# Publish approved material setup figures

## Scope

The user explicitly authorizes committing and pushing the approved setup
figures, README updates and website changes, and synchronizing GitHub Pages.
Publish the five final PNG/PDF pairs, rendering sources/specifications/receipts,
documentation, and the previously approved ivory ceramic main-figure styling.
Keep the unrelated horizon/thickness experiment code and README research
sections local. No new numerical work is required.

Use the repository workflow: commit on dev, regular merge to main, push only
main. The fixed-particle horizon-ratio investigation remains the active research
task; this is explicitly authorized concurrent publication work.

## Acceptance and status

Acceptance: review the staged-only tree and its links, verify the static build,
commit and push, confirm the Pages run for that exact main commit succeeds,
then compare the served HTML and PNG/PDF files with the inspected local build.
The previous local publication receipt already verifies embedded TrueType
fonts, editable 7 pt setup text, byte identity and the 1.39 MiB setup budget.

The exact staged tree was exported to an isolated directory and built with
`npm --prefix <snapshot>/website run build`. All nine pages, 255 local HTML
links and 58 new local Markdown links pass. All 20 setup/main PDF/PNG exports
match their approved sources and byte budgets (setup 1,458,907 bytes; main
8,317,921 bytes). New JSON and Python syntax checks pass. The 66 staged files
contain no numerical implementation, test or configuration changes; the two
README research sections and numerical memory updates remain unstaged.

## Publication and verification

Complete. Content commit `79d49ab` was regularly merged to main as
`3778e3df4c1c828adae0a0785ced6d98499b7d33` and pushed to origin/main.
Because local experimental README/memory hunks overlap the presentation
files, the regular merge was performed in a temporary clean worktree. That
worktree was removed and the original checkout returned to main; all local
research edits were preserved without stashing or resetting.

[Pages run 43163004937](https://github.com/NanxiiChen/nonlocal-pino/actions/runs/43163004937)
completed successfully, including build and deploy. At 2026-11-12 05:42:20 UTC,
all nine live HTML pages and twenty PNG/PDF assets returned HTTP 200 and matched
the inspected staged build byte-for-byte. This includes all five new setup
previews/PDFs and the ivory ceramic main figure. The live site is
[Nonlocal PINO](https://nanxiichen.github.io/nonlocal-pino/).

The compact receipt `artifacts/pino_setup_pages_validation.json` records the
deployment commit/run, all direct URLs, byte counts and local release checks.
No subsequent website content changes are required. Archive and publish this
verification record as a documentation-only follow-up; it does not trigger the
path-filtered Pages workflow.

Status: complete. Blocker: none. Next: none for publication; return to the
separate numerical task when requested. Memory updated: this archived task,
PROGRESS, DECISIONS and the local active numerical task's concurrent-work note.
