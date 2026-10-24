# Project split validation — 2026-10-24

Status: complete. Independent project: /root/nonlocal-pinn. Source: dev@61352f1.

- 4 owned CPU test modules: 36 executed tests PASS,
  0 skipped in this environment; zero failures.
- 6 fixed-input arrays exactly/bitwise equal to the original
  environment; 14 configuration dictionaries exactly equal.
- Package imports succeed from /tmp with Python isolated mode; original pinn
  and sibling project namespaces are absent. CLI help works for every case
  without training. uv frozen/offline dependency audit and pip check PASS.
- All53 source locked versions retained. Gmsh4.15.2 is now an explicit optional
  mesh extra, matching the baseline's installed version.
- Artifact-root inventory validates; existing data/logs/checkpoints untouched.
- Original147 source/config/test files remain hash-identical. Git bundle
  recovery to61352f1 and all189 archived payload hashes verified.
- No numerical model change, full simulation, new GPU benchmark, Git history
  rewrite, migration commit, merge or push. New repositories have no remote.

## Evidence

- validation-summary.json, test-results.json and local test-output/*.log
- parity-results.json, *fixed-input.npz, *.configs.json
- source-map.json and ast-parity.json
- cli-checks.json, environment-bootstrap.json, path-injections.json
- recovery-verification.json

## Recovery and limits

Use /root/nonlocal-KKS and its original environment to abandon the split or
continue historical runs. /root/nonlocal-KKS-split-backup-20261024/README.md
describes isolated recovery. Source, ignored assets and external artifacts
remain intact. Package layout has a new provenance identity; historical
in-place resume still uses the original checkout. No identity check was
relaxed. Any future cross-layout checkpoint import requires a new run and
explicit state/provenance validation.

Network downloads timed out; the independent environments were bootstrapped
from exact baseline dependency payload copies and standard local editable
installs. They now pass uv sync --frozen --offline --extra mesh --inexact.
Fresh setup on another host needs package access and the existing external
inputs (or deliberately prepared equivalents); large data is outside Git.

Final diff/whitespace audit PASS, including new files through a temporary
index; actual staging area unchanged. See final-diff-check.json.
