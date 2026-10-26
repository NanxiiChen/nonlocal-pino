# Melting run directory tags

Goal: mark the currently best audited production cold start with a sota suffix
and rename the previous SOTA directory to describe its historical formulation.

Acceptance: both directory renames succeed without overwriting; all original
file bytes and checkpoint hashes are preserved; live paths in archived evidence
and replay scripts resolve; exactly one melting log directory carries sota.

Context: at physical4 s/full128^3 nodes, the bounded IC-latent run has graph
terminal L2=3.018079939e-5 and analytic L2=0.001028267242, better than the other
production cold start under both metrics. User explicitly authorizes changing
these external artifact directory names; migration preservation rules do not
require another approval for this requested operation.

Status: completed. Blocker: none. Next: use the current paths below; no training
or checkpoint conversion is needed.

## Result

| Original directory | Current directory | Files verified |
| --- | --- | ---: |
| SOTA | legacy-raw-phase-soft-ic | 91 |
| 20261024-ic-latent-coldstart-01 | 20261024-ic-latent-coldstart-01-sota | 48 |

Both are direct children of /root/autodl-tmp/tf-logs/icemelting3d. Exactly one
directory now contains the sota tag. Promotion uses the audited physical4 s
terminal full-grid phase error, not a claim across different horizons or priors.
The promoted model's final checkpoint SHA256 remains
6acc3bd91cad7d75996b3fcf69379049dc42a23ebaab71dfed7740dbb9305665.

## Integrity and provenance

All139 files have identical sizes and SHA256 before/after the directory renames.
The [manifest](icemelting-run-tags.json) records every relative path and hash.
No source snapshot, model, saved config, TensorBoard event, or summary was edited.
Original run IDs/config.log.run_name values remain historical identifiers;
archived production metadata adds original_run_dir and points run_dir at the
current location. Historical SOTA keys in numerical reports still identify the
old baseline, with current checkpoint paths and explicit rename notes.

## Task end

Changed artifacts/index.yaml to include the renamed paths, both new cold starts,
the current sota metric, and old/new mappings. Updated current paths in the
diagnosis, metric-audit and production JSON reports; updated the two replay
scripts' directory lookups and added location notes to their archived reports.
Updated docs/migration/README.md to record the user-authorized naming exception.
Memory changes: docs/PROGRESS.md, docs/DECISIONS.md and this archived task/manifest.

Verification: repeated all139 file hashes; all indexed icemelting directories,
archived checkpoint paths and final model hashes resolve and match; both replay
scripts parse; exactly one sota-tagged run exists; git diff --check passed.
No numerical tests or retraining were needed for the path-only operation.
No Git history operations were performed.
