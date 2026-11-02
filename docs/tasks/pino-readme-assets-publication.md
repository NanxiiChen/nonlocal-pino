# Centralize and publish PINO README assets

## Goal and acceptance

User requests one dedicated image directory and explicit commit/push. Move the
four necessary README PNGs to root assets/pino/<case>.png, update relative
README references, figure generator and metadata. Preserve exact image bytes,
old research artifacts and all numerical settings. Publish the complete prior
README work and current relocation via devcommit -> main regular merge ->
origin/mainpush; keep dev local. No training or reference solves.

## Context

Previous four-PINO README task is complete but not committed. Current dev is
10d9910; main/originmain0dc54bc. Existing uncommitted docs, metadata, script and
four1.79MiB PNGs are all this requested publication scope. Longer20000channel
training remains deferred, exact final LR/restart choice undecided.

## Status / blocker / next action

Implementation and prepublication checks complete. Four PNGs moved without any
byte change to assets/pino/<case>.png;1,878,517bytes total. Root/example README,
assets policy,development guidance,generator,artifact index and both current
provenance receipts agree. Old empty per-example asset directories removed.
42links/anchors,PNG signatures,JSON/helper AST,445external directories and diff
checks pass. No numerical settings/data/model/default changes or new runs.

Next stage only documentation/helper/metadata and four curated PNGs; run indexed
content/size checks. Commit on dev, regularmerge main and push origin/main. Then
archive this task with actual publication receipts in a small follow-up commit
through the same workflow. No blocker. Deferred20000study remains unlaunched.

## Changed memory

docs/PROGRESS.md,docs/DECISIONS.md,this task and its validation JSON; previous
README task's archive now points to this follow-up. Complete prior README work
and its compact figure manifest/helper are included in the publication.
