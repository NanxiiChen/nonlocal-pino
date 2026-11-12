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

Pending: review the staged-only tree and its links, verify the static build,
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

Status: local release checks passed; awaiting commit/push and live verification.
Blocker: none. Next: regular dev-to-main publication, then inspect Pages and
served files. No deployment success is claimed yet.
