# Direct file-path loading without file hashes

Goal: answer where hashing was used and remove executable hashing/dependencies
from melting training, evaluation, research pilots and archived replay scripts.
User preference: load artifacts directly from explicit file paths, without
requiring checksum manifests or expected file hashes.

Acceptance: no executable file hashing in repository Python; source snapshots
list file paths; summaries/reference metadata record paths; archived replay
scripts load checkpoints/config/reference by path; current best checkpoint can
load from a copy containing just config.json and model_final.eqx. Preserve
physics/grid/shape/time compatibility checks and existing numerical evidence.

Context: three runtime modules record hashes (train/evaluation/accuracy), two
archived Python scripts record/check them, static archive/migration/run manifests
retain historical hashes. Actual NPZ/Equinox loading already uses paths.

Status: completed. Blocker: none. Next: use explicit checkpoint/reference paths
for future loading; historical metadata fields are ignored.

## Changes

- examples/icemelting3d/train.py: no file hashing; source snapshots produce
  source/files.json (plain relative file paths), and summary.json records
  model_path=model_final.eqx relative to the run directory.
- examples/icemelting3d/evaluation.py: reference metadata records its explicit
  resolved path, with existing physical/grid/shape/time checks preserved.
- examples/icemelting3d/accuracy.py: research reports record reference_path.
- docs/tasks/archive/icemelting-production-latent.py: direct deserialization
  replaces checksum gates; no source-manifest requirement, no dependence on
  evaluation.json for reference location; initial models compare directly as
  Equinox trees. Outputs list checkpoint/source/reference paths.
- docs/tasks/archive/icemelting-baseline-metric-audit.py: checkpoint/reference
  paths retained and fingerprint fields removed from newly generated reports.

Updated example README and docs/DEVELOPMENT.md to document path loading and the
user's preference for no file hashing. Added explanatory loading notes to the
production and baseline archived task reports. Existing run source snapshots,
checkpoints, event files, and static migration/rename/numerical records were not
rewritten; their prior checksum fields have no loading role.

## Verification and task end

Two focused existing IO/numerical tests passed (5.38 s CPU): monitor/reference
physical-time mapping, padded full-grid inference, mismatch rejection, exact IC
and checkpoint serialization. No PDE, optimizer, sampling or model weights changed.

Copied only config.json and model_final.eqx from the sota run into a temporary
directory, rebuilt the template from that config and directly deserialized the
model. Inference at x=(0.65,0.5,0.5), t*=1 returned finite phi=0.01013101; no other
run file/manifest was available or required. Source snapshot paths all resolved
and all29 copied Python files compared byte-for-byte with their source files.
All five changed Python files parsed; repository Python scan found no remaining
file-hash APIs or checksum requirements; git diff --check passed. No retraining,
checkpoint edits or Git history operation was needed.

Memory files updated: docs/PROGRESS.md, docs/DECISIONS.md and this archived task.
