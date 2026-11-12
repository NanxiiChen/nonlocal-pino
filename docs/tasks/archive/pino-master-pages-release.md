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

Status: complete. Content committed on dev as`fd7ed85`, regularly merged and
pushed to main as`f88709cafd4b194e9d18c10b40e201382b3e7f19`.
Pages run43181750462 succeeded. At2026-11-12T08:19:20UTC all9liveHTMLpages
and3architectureSVG/PNG/PDFassets returned200and matched the inspected release
build byte-for-byte. Receipt:`artifacts/pino_master_pages_validation.json`.
Fifty unrelated files are preserved byte-for-byte. Publication source changes
are limited to the17inspected files; no numerical/default changes are included.

The exact merge used `/tmp/pino-master-release-s9vhwze7/main`, leaving the
original development checkout and overlapping research hunks intact. Publish
this verification as a documentation-only follow-up, then remove the clean
worktree and return the original checkout to main. No Pages rebuild is needed
for verification-only files. Blocker:none. Next:the newly user-authorized
unified rolling core/halo investigation. Memory:PROGRESS,DECISIONS,this archive.
