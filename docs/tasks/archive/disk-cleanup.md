# Conservative disk cleanup

Date: 2026-10-30. Status: COMPLETE. No Git history authorization.

## Goal and acceptance

User authorizes identifying and cleaning genuinely unused or redundant disk
contents, and asks whether capacity expansion is still needed. Recover useful
space without losing numerical results, reproducibility, working environments,
source or recovery assets. Inventory first; record exact cleanup scope and
before/after filesystem space. Preserve external artifact paths.

## Context and protected contents

Previous assessment: confined-ch-density-capacity.md. Root overlay is
approximately137GiB total/133GiB used/4.6GiB available. No GPU training is active.
New refined geometry bank needs additional storage. Protect all project source,
Git metadata/history, installed .venv/Conda environments, checkpoints/optimizer
states, reference solutions, experiment logs/plots/receipts, inherited sibling
repository and /root/nonlocal-KKS-split-backup-20261024. Read migration/README.md;
do not relocate or remove existing data/tf-logs. No process termination.

## Plan and verification

Inspect actual allocated blocks by directory and identify download/package
caches or confirmed abandoned temporary material. Prefer package-manager dry
runs for supported cleanup. Do not treat a large/old file as evidence it is
unused. Do not traverse symlinks or delete active sockets/session data. Record
approved-by-user-scope deletion candidates before mutation. Recheck filesystem
free space, environment usability and protected artifacts after cleanup.

## Completed cleanup and measured recovery

Filesystem available space increases4.5285415649GiB->14.3337097168GiB, a measured
net recovery9.8051681519GiB. Final used fraction89.50%. Record exact byte counts
and protected paths in docs/tasks/disk-cleanup-receipt.json. Logical cache
sizes are not physical-recovery claims; installed hardlinks retain shared blocks.

Executed sequentially through normal package-manager commands:

1. uv cache clean --cache-dir /root/.cache/uv:53510files removed, reported9.9GiB
   logical estimate. No force/in-use-lock override. Cache symlink audit across
   three project .venvs, miniconda3 and managed uv installs found zero cache
   targets. Installed environments remain intact.
2. /root/miniconda3/bin/python -m pip cache purge:288download-cache files removed.
   There were no locally built wheels in that cache.
3. npm cache clean --force --cache=/root/.npm:content download cache removed;
   installed tools and576MiB _npx environments preserved. npm requires --force
   for this cache operation; it was not an override of uv's usage checks.

Package caches alone produce measured net recovery8.6805801392GiB. Supported
command semantics checked in [uv](https://docs.astral.sh/uv/concepts/cache/),
[pip](https://pip.pypa.io/en/stable/cli/pip_cache/) and
[npm](https://docs.npmjs.com/cli/v11/commands/npm-cache/) primary documentation.
Future installations may need to download cleared packages again; dependency
versions/installed payloads were not changed.

Additional exact deletion: nine128MiB shared_mem_cuda_pool files under
/tmp/ompi.autodl-pro-7775bcfa7d82.0/jf.0,1.125GiB allocated total. All were older
than3600hours. Before and immediately before removal, all readable /proc maps
and file descriptors showed no references/no live libmpi process; permission
denials0. Exact regular-file path/device/inode/size/mtime identity rechecked.
Only these nine files were unlinked; session directories and other files remain.
No ompi-clean, process termination or generalized /tmp deletion.

## Preserved and verification

All data/tf-logs (about73GiB), source/Git metadata, installed environments and
compiler caches retained. No checkpoint/optimizer/reference/log/figure removed.
Trash2.2GiB contains historical experiments and is preserved; OpenCode history,
IDE runtimes, sibling repositories and migration backup also preserved. Conda
tarball/index/log dry-run had zero candidates, so no Conda cleanup was executed.

26critical protected files retain exact device/inode/size/mtime metadata,
including selected30k model/optimizer/config/predictions, bank references/
manifest, environment configs and migration backup files. Reference514/518NPZ
arrays remain readable/finite with positive cell measures. Project CPU JAX/
Equinox/Optax imports and a small autodiff check PASS; versions JAX/JAXlib0.10.1,
Equinox0.13.8,Optax0.2.8,NumPy2.4.6. Separate fenicsenv dolfin2019.1.0 import PASS.
No concurrent JAX GPU/FEniCS MPI work or environment modifications.

## Capacity recommendation and task-end

Current-mesh width128 and small refinement checks have enough storage now.
A whole additional h=.030/512geometry bank is estimated about10-12GiB, not yet
constructed or measured. That would leave little room for new checkpoints;
another30-50GiB is a useful planning allowance for full refinement/multiple
controls, rather than deleting unique numerical evidence. No expansion required
for the immediate small pilots and no run queued.

Changed only task/receipt/DECISIONS/PROGRESS documentation in repository; cache
and nine exact orphan-pool cleanup outside it. Archived this task. JSON parsing,
protected-artifact checks, environment smoke checks and git diff --check PASS.
No Git commit/push/history action. Status: complete. Blocker: none. Next action:
none required for cleanup; future full-bank storage expansion is optional.
