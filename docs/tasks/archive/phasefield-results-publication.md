# Publish accumulated phase-field results

## Goal and authorization

User explicitly requests commit and push of the accepted current results on
2026-10-29. Include accumulated confined-CH and electropolish operator code,
focused tests, example/project documentation and archived validation records
since dev728e5b9. Preserve external checkpoints/data/logs at existing paths.
Use dev commit -> main regular merge -> push origin main only; keep dev local.

## Acceptance and publication record

All repository content is ready for the user-authorized publication. Project
memory and all modified/new file groups were inspected. Starting branch dev;
origin/main fetched, local main tracks it. No
numerical changes or new training are needed. Existing completed task reports
retain actual numerical checks and known accuracy limits.

Pre-publication checks pass: AST syntax for155Python files, strictJSON260files
and strictJSONL100files. New content totals~52.2MB with largest file10.8MB;
no checkpoint/environment/data-root relocation. Staged scope covers823files.
Staged whitespace initially caught generated SVG/stdout line endings and one
extra EOF blank line;20files were normalized only in whitespace. Python AST,
SVG tree/attributes and whitespace-free text equality independently pass.
No numerical code/values were changed by this cleanup.

Delivery follows dev commit -> main regular merge -> push origin main only.
The authoritative publication receipt is the resulting Git history and equality
of local main/origin main/remote main refs; no self-referential commit ID is
embedded in this file. The final response reports the actual commit and push
result after those operations. On an external push failure, retain local commits
and resume with a normal main push; never force-push or rewrite history.

## Verification scope

Electropolish final matched cold starts, all64field metrics, refined4targets,
76reference BC/range/time gates and source/input parity already pass. Confined
CH selected30-window trajectory/default dispatch and historical controls retain
their archived verification. Do not repeat unchanged expensive training/suites
for this publication-only task. Memory files: PROGRESS, DECISIONS and this task.
