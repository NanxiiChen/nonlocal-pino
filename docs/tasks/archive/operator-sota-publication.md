# Publish operator code with verified defaults and lean artifacts

Date: 2026-10-29. Status: implementation/verification complete; publication via
the authorized dev -> main -> origin/main workflow. No numerical work queued.

User explicitly authorizes commit and push of code/docs, asks to ensure defaults
are SOTA and only necessary information enters Git. Scope: current3Dmelting NO
and2Delectropolish NO plus outstanding validated FEM/render code/docs. Keep
single-IVP entry defaults; recommended train_operator.py must select measured
best available operator recipe. No new training or numerical operator tuning.

Audit: dev85f1821, mainea37ef6/originmain.1640tracked files~90.00MiB,189PNG
53.67MiB,19NPZ15.60MiB,14SVG1.90MiB,12EQX1.34MiB.153untracked files14.44MiB,
mostly generated PNG/SVG/GIF/video/PDF. Git objects loose184.8MiB+pack27.5MiB;
normal untracking will not shrink old history; no history rewriting/forcepush.
All generated tracked images are docs/tasks/archive; no runtime test fixture
there is required. Keep small migration NPZs at docs/migration unchanged.

Plan/acceptance:
- Promote electropolish operator entry from early finite16/B2/2000/localIC to
  validated onlineB8/5000Adam/image-only/ordinaryFP32 via exact saved config
  comparison. Preserve old finite/online/single/causal/local control meanings.
  Retain melting onlineB16/5000Adam/hardIC/transformed PDE best compromise;
  localFEM/late/cost evidence prevents selecting solely native pooled error.
- Add explicit generated-artifact ignore policy, cancel tracking docs/task
  media/raw arrays/checkpoints/compiler dumps without deleting local files.
  Preserve an external copy and direct-path manifest; provide restoration for
  existing local report links/archived diagnostics. Track code/docs/configs/
  compact metrics and provenance, not repeated bulky rendered variants.
- Verify changed config behavior/focused tests, source compatibility, relevant
  syntax/JSON/links/diff/staged size. No repeat unchanged costly training/FEM.
- Task-end record and user-authorized devcommit -> main regularmerge -> push
  originmain; verify remote equality, no push dev. Do not rewrite history.

Delivered:
- Added3Dconditional operator, random ellipsoid geometry/data/reference/loss/
  trainer, aligned localFEM rewrite, tests, stable example/root documentation,
  and prior run/FEM/render evidence. Single-IVP defaults are preserved.
- Canonical electropolish NO now onlineB8/5000Adam/image-only; explicit legacy
  base preserves exact old finite/online/single/causal/local configurations.
  Melting NO remains onlineB16/5000Adam/hardIC/latent PDE. Both canonical NO
  recipes match saved selected runs except run-name/runtime-seed provenance.
  Independent graph and local reference scores retain distinct run identities.
- Removed248 generated files (76,227,702bytes,72.70MiB) from Git tracking only;
  excluded27new generated files (13,725,346bytes,13.09MiB). All275files remain
  at original paths and byte-identical external backups,89,953,048bytes total.
  Source/config/metrics/provenance/regeneration scripts and tiny migration NPZ
  fixtures stay tracked. New ignore patterns and content guard prevent growth.
- Added artifacts/local_files.json and scripts/restore_local_artifacts.py.
  All275existing files verified; one actual PNG copy to a fresh/tmpdestination
  succeeded. No file hashes. Historical Git blobs remain; no history rewrite.

Verification:
-27focused CPU tests pass in234.643s: electropolish finite/online/causal/local,
  melting3Doperator, canonical training dispatch. Prior numerical/FEM validations
  are retained, not needlessly rerun in this publication task.
- Exact config comparison passes for two NO defaults and five historical
  presets; script and JSON receipt adjacent to this record.
-419registered artifact directories across8cases exist.
-39changed/new Python ASTs,48strict JSON records and2JSONL streams parse;
  no new binary files. One render diagnostic log EOF normalized only.
- Current index1529files/18.74MiB versus1640files/90.00MiB initially, with no
  tracked generated report payload or individual file over1MiB. Numbers include
  this task's textual audit records and may vary by a few bytes after task-end.
- Git diff whitespace checks/content guard are final publication gates.

Publication receipt: the dev commit, regular merge on main and remote main
reference equality are authoritative in Git history and the final user report;
no self-referential commit ID is embedded here. Push only main. If publishing
fails, retain this record and resume at the failing Git operation without
retraining or repeating backup/untracking. No further optimization is queued.

Changed memory: docs/PROGRESS.md, docs/DECISIONS.md, this archived task;
configuration/artifact audit JSONs and verification scripts/logs are adjacent.
