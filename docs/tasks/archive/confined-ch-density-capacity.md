# Density, network capacity and resource assessment

Date: 2026-10-30. Status: COMPLETE (assessment only). No Git history authorization.

## Goal and acceptance

Answer whether the current geometry-conditioned confined CH operator should
use more spatial points and a larger network, and whether another GPU is
needed. Base recommendations on measured results and explicit scaling of this
implementation. Distinguish estimates from measurements. No new training,
geometry bank, physical formulation or default change is requested here.

## Context

Completed optimization: confined-ch-fourier-optimization.md.
Selected run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-30k-retry-01.
Native graph/BDF reference only; local FEM remains deferred. Width96/522145
parameters, B16,16positive times plus t0,512training inputs/8reserved development
inputs, FP64 ordinary SOAP. h=.035, delta=.11, epsilon=.06. Actual point counts
1129..1351, mean1229.626953125 across training geometries. Same points at all times
within each geometry; individual geometries have different meshes.

## Current evidence

Matched20k width64->96 improves pooled native phase15.3301%->11.3836%; terminal
16.4691%->12.2805%. Width96/30k further reaches9.6609/10.5131%; training16monitors
5.9243/6.3024%. Width128 was CPU model/state preflighted (905089parameters)
but has not been GPU-trained or memory-benchmarked. Capacity increases have
evidence of benefit; a strict capacity floor or future benefit is unproven.

System nvidia-smi reports NVIDIA GeForce RTX3090,49152MiB total; idle1MiB/0%.
Saved final JAX audit: peak16876689664bytes, live9330428928bytes,
allocator limit38183976960bytes. Peak active memory is distinct from allocator
pool size. The full bank includes520geometries padded to1351; its FP64 dense
residual metric alone has520*1351^2*8=7592836160bytes before small alignment
overhead (largest logged allocation7592836352bytes). This dense bank scales
quadratically with points, independently of model parameter count.

## Recommendation and limits

Prioritize a width96->128 control at unchanged h=.035,512input geometries,
Fourier bandwidth/depth, B16/Nt16, SOAP and physical/loss scheme. Parameter
count increases522145->905089; GPU accuracy/time/memory remain unmeasured for
width128. Existing width64->96 gains support this trial, not a guarantee of
improvement. Continued30k gains also mean a strict capacity floor is unproven.

Independently evaluate spatial refinement on1-2identical geometries first:
h=.035->.030, preserving delta=.11, epsilon=.06 and fixed initial function.
For comparable2D meshes N scales approximately h^-2: mean1230->1674 (+36%).
Dense metric storage scales Nmax^2:7.07->13.10GiB (+85%) for520padded geometries.
These are estimates; actual Gmsh points, graph degrees, workspace, peak GPU
memory and reference differences require measurements. More aggressive h=.025
would give about2410points/geometry and27.17GiB for the dense metric bank alone.

Compare original and refined native trajectories on common physical queries
before constructing a full refined bank. Current9.66% is NN-versus-native error
on the same discretization; it does not itself demonstrate mesh insufficiency.
At fixed delta, refinement tests spatial quadrature/geometry convergence, not
the finite-horizon bias against a local PDE. Keep these questions separate and
report native accuracy alongside a common-query comparison when meshes change.
No local FEM study is newly required or launched here.

Current measured active peak15.72GiB is well below35.56GiB JAX allocator limit
and48GiB system-reported capacity. No present evidence requires another GPU
for a width128 or modest refinement pilot. Actual short-run peak must precede
promising a full denser/larger run. Do not infer available memory from allocator
pool reservations or just model/state arrays. Current dense bank dominates
storage compared with the approximately50MB width128model+SOAPstate.
The distinction between preallocated pool and demand is also documented in
[JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
(checked2026-10-30); numerical memory values above come from local saved audits.

Disk free is approximately4.6GiB (df -h); the existing512bank is already about
6.08GiB. A separate complete refined bank requires more storage. No files were
deleted or artifact paths moved, and no new bank/training was launched.

## Task-end changes, blocker, next action

Completed read-only evidence/scaling assessment. Added compact receipt
docs/tasks/confined-ch-density-capacity.json and this task; updated DECISIONS
and PROGRESS. No numerical code/config/default changes. JSON validation and
git diff --check pass. No Git history action. Assessment blocker: none.
Possible future run: width128 at current mesh and small independent h=.030
refinement check; larger-card need is not established, full-bank storage must
be arranged. No follow-up queued or incomplete experiment promised.
