# Local FEM versus native nonlocal melting references

Date: 2026-10-29. Status: COMPLETE; all workers idle; no further solve queued.

## User objective and accepted scope

Compare local FEM and native nonlocal references before deciding whether the
nonlocal operator needs optimization. User requests rewriting the old FEM,
aligning parameters/interface width, and initially evaluating one previously
randomly chosen held-out input5; inputs9/2 remain deferred. They subsequently
accept the current FEM accuracy, stop further refinement, and explicitly
prioritize whole-field phi rather than the final tiny remaining solid.
No retraining, operator/default/checkpoint change or Git history authorization.

## Result and decision

On common24^3diagnostic nodes, native versus local pooled phi relative L2 is
**0.147448%**, terminal **0.166971%**, maximum snapshot **0.199291%**. On the
128^2central slice: pooled **0.357479%**, terminal/max **0.535060%**. These are
small for the current percent-level whole-field objective. Continue the native
reference for routine monitoring; retain this local FEM check as independent
physical validation. No operator optimization is queued. This single-input
check does not establish a uniform bound over random ellipsoids.

| t* | Native/local 3D phi L2 (%) | Native/local slice phi L2 (%) |
|---|---:|---:|
| 0 | 0.006980 | 0.010254 |
| .25 | 0.093010 | 0.149473 |
| .50 | 0.175433 | 0.320961 |
| .75 | 0.199291 | 0.454791 |
| 1 | 0.166971 | 0.535060 |

Both final5000checkpoints were freshly deserialized on GPU, without updates:

| Model | Pooled phi/native (%) | Pooled phi/local (%) | Terminal phi/local (%) |
|---|---:|---:|---:|
| Adam | 0.238264 | 0.147860 | 0.167968 |
| SOAP | 0.070881 | 0.186141 | 0.280074 |

Reference choice can reverse this very fine model ordering, despite small
absolute field gaps. Squared-error decomposition verifies the fitting/reference
cross term, with pooled cosines Adam-.8062/SOAP+.3772; relative norms do not add.
Native/local terminal solid L2=3.89388%, diagnostic solid-fraction gap4.29795%
are retained in JSON but explicitly not the current optimization target.
No full-grid/continuous norm, localAD speed, optimizer robustness or global
accuracy claim. Pooled errors concatenate five snapshots with equal weights.

## Matching local reference and accuracy

Canonical solver.py/solver_base.py were rewritten: full unit cube, shifted/
rotated exact Euclidean ellipsoid SDF, phi=tanh(SDF/(sqrt2*.0225)), ice+1,
water-1, fixed epsilon=.0225 and saved physical/nondimensional metadata.
PDE phi_t=N1*Delta(phi)-N2*(phi^3-phi)-N3*(1-phi^2), with coefficients
N1=.00005, N2=.0987654321, N3=7.856742. Natural homogeneous Neumann all6faces;
consistent CG2 tetrahedral mass/stiffness and degree8 exact polynomial reaction
quadrature. Standard IMEX BDF2 weak reaction extrapolation, BE/explicit startup,
constant matrices, CG/Jacobi rtol1e-11/atol1e-13. Independent implicit-midpoint
option retained. No nonlocal/training residual imported.

Accepted field: N96,7,189,057DOFs, dt*=.00125,800steps to t*=1=5seconds;
3166.45s internal MPI12wall. One job at a time under actual12CPU cgroup quota.
JAXGPU and FEniCSMPI were never run concurrently. Every NPZ retains
validated_reference=false; aggregate validation records acceptance separately.

Spatial32->48 and48->64 failed original phase gate;64->96 passed maxphase
.0377924% (3D.0271825%), maxsolid.2136203%. Time.005->.0025 failed; .0025->.00125
has max3Dphase.0294622%, slicephase.0737336%, maxsolid.547736%. The original
.05%phase/1%solid gate therefore still fails for slicephase. User explicitly
accepts this precision; validation.status=accepted_by_user and
strict_refinement_passed=false. Do not relabel the strict check as passed.
A conditional.000625job was started, then identity-checked and stopped per
user steering after~3min; no fine result was saved. All logs/source/stop
receipt retained. No new solve or additional sample is needed for this task.
Native reference dt.005/.0025 max difference1.1755e-7 is negligible here.

Interior operator audit gives six-neighbor quadratic moment ratio.6858548,
D|x|^2=-4.11513 versus target-6. This identifies a discrete consistency issue,
but is not itself a31%field error; measured field gap above is the relevant
current criterion. No normalization or training arithmetic was changed.

## Diff-level changes and verification

- Replaced examples/icemelting3d/solver.py and configs/solver_base.py; moved only
  JAX imports into JAX-specific ellipsoid helpers, preserving all math.
- Added tests/test_melting_fem.py; updated exampleREADME and DEVELOPMENT;
  registered references/results in artifacts/index.yaml.
- Five final independent FEM tests pass on MPI2: fixed coefficients/IC,
  weak diffusion/reaction, distributed queries/owned DOFs, homogeneous-ODE
  second-order time convergence, and independent heat residual/mass/energy.
  .venv correctly skips the five FEniCS-only tests.
- Existing strict CPU FP32geometry test passes. GPU default-matmul probe fails
  at2.85e-4; highest-matmul probe exceeds its CPU3e-7 threshold slightly at
  3.3319e-7. Raw failures retained, no tolerance changed. Direct old training
  source/current math ASTs and actual GPUoutputs are identical (maximum gap0),
  so this is not introduced by import refactoring. Training uses highest.
- Fresh Adam/SOAP checkpoint replay max gaps1.43e-6/1.28e-6; all parameters
  finite and raw checkpoint bytes unchanged. No file hashes.
- Independent native dataset equals both replay target arrays; RMS recalculation
  agrees with pooled metric; true FE-integrated solid fractions finite/decreasing.
- Failed N24midpoint range/interpolation, early affinity-limited N48probe,
  test-harness API/UFL-zero failures, strict time failures and canceled fine
  job are retained, not selected as validated targets. Direct Python MPI avoids
  the observed python-mpi4py affinity trap. No further expensive tests needed.

## Direct artifacts and reproduction

Local: /root/autodl-tmp/data/icemelting3d/operator_local_fem_v1/case005-imex-n96-dt0p00125.npz
Native: /root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1/condition_005.npz
Adam: /root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01/model_final.eqx
SOAP: /root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-soap-01/model_final.eqx
Fresh outputs: local artifact root/checkpoint_replay/{adam,soap}-005.npz.
Per-resolution frozen source/controls: local artifact root/source/.

[Acceptance and original checks](melting-local-fem-validation.json),
[all metrics](melting-local-fem-evaluation.json),
[reference fields](melting-local-fem-reference-fields.png),
[model fields](melting-local-fem-fields.png),
[refinement/errors](melting-local-fem-refinement-errors.png),
[checkpoint replay](melting-local-fem-checkpoint-replay.json),
[independent audit](melting-local-fem-metric-audit.json),
[geometry parity](melting-local-fem-geometry-audit.json),
[operator moment](melting-local-fem-operator-moment.json).

All melting-local-fem-* scripts/logs/receipts are archived alongside this file.
Read-only report regeneration, from repo root in.venv:
python -m docs.tasks.archive.melting-local-fem-evaluate
For a new FEM output use the README explicit solver CLI and a new output path;
do not overwrite existing fields. Checkpoint replay intentionally requires a
new output directory if rerun. The archived finish.sh is a canceled historical
one-shot pipeline, not the current entry point. Next action: none; expand
samples or optimize operators only if the user requests it.


## Closeout verification

Nine changed/task Python modules parse; affected strictJSON/artifact index,
acceptance status, real file paths and three PNG signatures checked. Final
`git diff --check` passes; no new raw-log trailing spaces remain in the repo.
Two GPU test logs had line-end spaces only: byte-original logs retained under
local artifact root/logs, with [normalization receipt](melting-local-fem-log-provenance.json).
All raw numerical/test content retained; no tolerance or test weakened.
Project memory updated: PROGRESS.md, DECISIONS.md and this archived task.
CPU/FEM/GPU workers idle; no commit, push, merge or further queued work.
