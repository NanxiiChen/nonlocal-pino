# Reversed-vortex deformation pilot

Date: 2026-10-26. Status: completed. Blocker: none.

## Goal and acceptance

User asks to try the shape-changing prescribed vortex discussed after the simple
transport pilot. Keep the constant-flow control and its historical artifacts.
Implement a clearly identified periodic variant of the supplied Qiu2022 vortex,
not a claim of exact paper reproduction or coupled NS. Independently refine the
reference in time and space, verify mass and visible deformation, then compare
same-initialization5000-update mixed local AD/nonlocal training, plot fields at
0/0.5/1/1.5/2, and report errors at maximum deformation and return, mass and timing.
Training failure is a measured result, not grounds to hide a run or tune seeds.

## Context and choices

Existing pilot: examples/twophase2d; base config/reference stay constant flow.
Paper Eq23/24 prescribes divergence-free reversed vortex, U0=2,L0=1,T0=2.
Printed IC omits radius; wall chemical-flux closure/mobility are not fully specified.
Use explicit periodic rounded-droplet IC and mobility, no implicit paper correction.
No evolved-interface prior, no velocity/pressure network, no residual transformation.
Prescribed strain can increase CH free energy; do not enforce energy monotonicity.
Seed42 is common run provenance. No Git history operation authorized this turn.
Existing uncommitted pilot and old data/log paths must be preserved.

## Status and exact next action

Start: mandatory memory/development/numerical/JAX guidance read; active task None.
Current: create explicit vortex preset and analytical velocity union, extend independent
reference to variable conservative advection, add focused derivative/conservation
checks, generate and inspect reference before serial GPU training.

## Reference and pre-training validation

Seven focused tests pass (19.355s), covering legacy controls, divergence, reversal,
periodicity, manufactured variable-advection residual and conservative reference.
A coarse test exposed an empty-offset graph reference on h>=delta; it now fails
explicitly before division. Production reference uses h<delta and was unaffected.
Five independent reference solves finish: graph64 dt2e-4/1e-4, local128 dt2e-4/1e-4,
local256 dt1e-4. Maximum-snapshot time differences3.6154e-5(graph)/2.5429e-5(local),
local128->256 spatial1.9917e-7. Phase-mass drift<=2.22e-16. All pass tolerance0.002.
Graph/local terminal difference0.0116849, pooled0.0128047, kept separate.
Reference visualization confirms strong crescent deformation at t1 and near-circular
return at2. Local return-to-IC L2=0.0671209 (diffusion continues); this is not a
reference failure. Midpoint-to-IC L2=0.679164. Five-time reference image and raw
geometry JSON are in reports/twophase-vortex-pilot. Direct reference path is
/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz (adjacent check JSON).

Current: serial predeclared5000-update nonlocal then local_ad launcher running;
source frozen for the cohort. Exact resume: inspect reports/twophase-vortex-pilot/
manifest.json and logs, wait for both exits, then CPU evaluate saved final models,
verify initial/config/source parity, plot common-target and matched-target fields.


## Completed paired controls and independent replay

Both seed42 cold starts finish5000 updates. Common initial model bytes match;
configs differ only nl.method/log.run_name. Train/loss/problem/network/operator/
config numerical sources stay identical to both snapshots throughout the cohort.
The original constant-flow default and serialized config/reference still load.

| Final5000 metric | Nonlocal | Mixed local AD |
|---|---:|---:|
| t1 phi L2, own target | 0.0596832 | 0.0746330 |
| t2 phi L2, own target | 0.0841024 | 0.1018614 |
| t1 phi L2, common local target | 0.0547927 | 0.0746330 |
| t2 phi L2, common local target | 0.0788882 | 0.1018614 |
| pooled own-target phi L2 | 0.0604626 | 0.0753610 |
| t2 interface-band phi L2, own target | 0.3426634 | 0.3735551 |
| max soft phase-area relative drift | 0.0337063 | 0.0289911 |
| trainer seconds (JIT/eval/saves included) | 76.8487 | 136.5356 |
| first compiled update seconds | 51.7305 | 76.7804 |
| mean update after100, seconds | 0.00329971 | 0.01030749 |
| subprocess seconds | 82.6320 | 142.1338 |

Both learn the visible stretch/reversal/near-circular return. Nonlocal has lower
field error in this paired run and3.12x shorter compiled updates,1.78x shorter
trainer wall time; its phase-area conservation is worse. No all-metric dominance,
statistical robustness or universal nonlocal superiority is established. Interface
band errors remain large; their relative L2 measures phi values, not a contour
position error. Soft area means integral(1-phi)/2, not binary threshold area.
Unclipped phase extrema/reference excursions are preserved in figures. This is
a route demonstration, not a high-accuracy result or new tuned default.

Initial CPU reload check at3e-6 fails by one nonlocal phase value3.03984e-6.
Independent frozen GPU replay exactly matches all stored phi/q values for both
models; CPU FP32 maxima<=3.30061e-6, phase-relative differences<=4.08e-7.
Promoted CPU FP64 inference provides a further precision control (max<=7.87e-6).
Evaluator now explicitly records max-absolute/relative differences and uses5e-6
FP32 absolute replay tolerance; FP64 unchanged1e-10. Only evaluation validation
changed after the finished cohort. No saved fields were changed. Re-run CPU audit
passes and reproduces both terminal metrics. Full diagnostics are archived.

## Diff-level changes

- Root README/navigation and example README describe the new periodic variant;
  DEVELOPMENT documents the explicit velocity union and shared entry.
- train_vortex.py explicitly sets T2,U2,epsilon0.02,M1e-4,periodic IC,64x64,
  delta0.04,4x64 network and independently refined reference path.
- problem.py validates constant-vector/structured-vortex velocity and supplies its
  analytic field. Existing serialized identities remain exact.
- losses.py uses pointwise velocity times AD first derivatives. train.py logs
  each saved time's relative L2. Both diffusion methods retain mixed second order.
- solver.py accepts --config module and uses time-dependent conservative Fourier
  advection in the explicit BDF2 part; fixed-flow splitting stays unchanged.
  Actual-grid h<delta and finite positive kernel moment are checked early.
- evaluate.py adds common-local/interface errors by time and explicit CPU replay
  differences. Two focused tests cover divergence/reversal/manufactured residual
  and reference conservation/invalid unresolved graph.
- Artifact index registers2 new runs/reference/results; base/default untouched.

## Reproduction, artifacts and completion

Run commands are in example README. Launcher, plotting, verification and replay
roundoff scripts are archived next to this task; each discovers the repository
root independently. Raw reports/source are in reports/twophase-vortex-pilot.
New runs are /root/autodl-tmp/tf-logs/twophase2d/20261026-periodic-vortex-{nonlocal,local_ad}.
Each has config, source, initial/final/1000-update checkpoints, loss and per-time
L2 histories, predictions and summary. Load these direct paths, without hashes.

Archived figures: [common-local comparison](twophase-vortex-pilot-common-reference-fields.png),
[reference evolution](twophase-vortex-pilot-reference-evolution.png),
[nonlocal reference/prediction/error](twophase-vortex-pilot-nonlocal-phi.png),
[local AD reference/prediction/error](twophase-vortex-pilot-local_ad-phi.png),
[convergence](twophase-vortex-pilot-convergence.png).
Raw [CPU audit](twophase-vortex-pilot-audit.json),
[verification](twophase-vortex-pilot-verification.json),
[reference refinement](twophase-vortex-pilot-reference.json),
[CPU](twophase-vortex-pilot-replay-roundoff-cpu.json)/
[GPU](twophase-vortex-pilot-replay-roundoff-gpu.json) replay diagnostics.

Next action: none required. A future accuracy task can test causal time windows
and interface-aware resolution/loss controls; no further training or full NS has
been started. All launched processes exit0; no GPU work remains. No Git history
operations this turn. Complete verification summary is in the adjacent validation
JSON. Memory updated: PROGRESS, DECISIONS, DEVELOPMENT and this archived task.

Final completion gate: all archive artifacts are nonempty; strict JSON, changed
Python/script ASTs, local documentation links,7-test log and complete two-worker
manifest pass. Registry check passes6cases/321directories. git diff --check passes,
branch is dev and GPU process list is empty. Current modifications also include
the prior uncommitted constant-flow pilot; no unrelated changes were reverted.
