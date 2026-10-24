# Migration and recovery — 2026-10-24

Source: /root/nonlocal-KKS, dev61352f103e6beadaa2960f81f8379af28d9e7658.
User approved two independent projects and explicitly required rollback.
[source-map.json](source-map.json) records original and migrated file hashes.
The original repository and all its ignored local assets remain at the original
path; cloned repositories intentionally do not duplicate ignored experiments.
The preserved asset inventory is in the backup directory.

## Recovery

The original numerical source and .venv are kept usable. To abandon the new
layout, return to /root/nonlocal-KKS and activate its .venv (or fenicsenv for
FEM). Data/log paths are unchanged, so no reset, clean, deletion, stash or
metadata edits are required. Keep new outputs separate from old baselines.

/root/nonlocal-KKS-split-backup-20261024 contains a verified history.bundle,
working-tree.tar.gz, source hashes, refs/status/diff and local asset inventory.
For isolated recovery, clone the bundle into a NEW directory and extract the
worktree archive there, following the backup README. Do not overwrite an
occupied checkout. Original dependencies and external data are not inside the
archive; their original locations remain available.

## Behavioral boundaries

No PDE, kernel normalization, boundary conditions, history/irreversibility,
precision, loss, seed, optimizer or stopping parameters change. Old presets
keep their external paths and default choices. The new package layout changes
run provenance: old identity equality checks remain strict and historical
in-place resume should use the original checkout. An explicit checkpoint
import, if later needed, must create a new run with parent-run provenance and
verify network/optimizer/replay/controller/physical state on copies.

The fracture loading code now snapshots the whole fracture_pinn package
recursively with relative paths, since its implementation spans directories;
basename-only snapshots would lose modules and collide on __init__/config names.
This operational provenance change does not change numerical updates.

## Git and environments

History is copied without hardlinks. The original push remote is removed in
the new projects. No migration commits, merges or pushes have been made.
Each project has its own .venv and lockfile; no source/sibling path is injected.
FEniCS remains in fenicsenv and runs separately with this project's src on
PYTHONPATH. Original mixed history is available via git show 61352f1:<path>.

## Environment bootstrap on this host

The inherited mirror and wheel CDN timed out. Runtime lock versions were
verified unchanged after selecting the official PyPI index. To keep the
validated numerical dependencies, independent copies of the baseline installed
package payloads were placed in each new .venv, the old editable project and
source-path injection removed, and each new project installed locally with a
standard Hatchling editable wheel. No dependency directory is symlinked to
the old environment. See environment-bootstrap.json and path-injections.json.
The environments also retain the baseline's optional gmsh installation.
Normal fresh setup remains uv sync --frozen when package downloads are
available. The baseline itself was never changed or upgraded.

Mesh generation additionally uses the explicit `mesh` extra:
`uv sync --frozen --extra mesh` (Gmsh4.15.2, matching the baseline).
