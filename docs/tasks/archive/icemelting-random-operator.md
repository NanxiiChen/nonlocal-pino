# Random-ellipsoid 3D melting operator

## Goal and acceptance

User requests random initial ellipsoids for melting, a3Dcondition encoder and
actual GPU-memory validation. Extend canonical train.py opt-in, retain original
single-sphere defaults/checkpoints and physical PDE. Use continuous online
center/semiaxis/orientation draws, volumetric input only in encoder and shared
coordinate/time queries. Preserve IC-only hard lifting and stable latent graph
residual, no future-shape/speed prior or positive-time training labels.

Acceptance: independently check ellipsoid Euclidean SDF including interior axis/
center/sphere cases; preserve2Dconditional behavior/checkpoints;3Dsplit/timeJVP/
parameter-gradient checks; verify actual128-cube six-neighbor graph/reference;
measure synchronized training memory/cost for B8/B16, choose explicit safe
configuration, cold start and report physical-field errors on held-out shapes.
No Git history operation is authorized in this new task. Work on local dev.

## Context and decisions

Original melting:128^3unit-cube grid, cubic delta.01, eps*.0225, M.1,lambda5,
Lc100,Tc5, original ic_latent default. Only natural in-domain pair closure.
Do not reuse sphere analytic contraction or octant reference for rotated/shifted
ellipsoids. RTX3090 reports49152MiB and~1MiB idle at task start.

Initial distribution: independent semiaxes[.24,.36], center[.45,.55]^3, Haar
SO(3)rotation from normalized Gaussian quaternion; bounds guarantee containment.
Encode1x32x32x32phi0volume into4x4x4patches (512tokens); planned width48,
heads4, encoder2, point decoder3, coordinate-only local_dim0. Start core256,
Nt8, Adam1e-3/B8; measure B16 rather than guessing it fits. Pure spatial cross-
attention is shared across times. Keep all shape/config/range choices explicit.

Ellipsoid-distance derivation follows Lagrange multiplier nearest-point root,
including smallest-axis singular branch; reference David Eberly,
https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf.
Implement independently, validate with constrained SciPy minimization and sphere
identity; no vertical/radial gap mislabeled as SDF. Source PDE/reference and
nonlocal normalization remain unchanged. Global physical phiL2 can hide small
solid structures; also report solid-indicator and interface/volume diagnostics.

## Status / blocker / next action

COMPLETE. Canonical5000Adam/B16cold start exits0,80000distinct online shapes,
8held-out tests. Full process300.574s/trainer257.167s, first update108.238s,
warm update22.428ms/sample3.411ms. Peakactive1429767424bytes(1.332GiB), pool4GiB.
All10reference time checks,4new numerical checks,2shared2Dregressions, legacy
source/output/checkpoint parity, final-source bytes, stream, independent metric
and GPUreplay checks pass. No blocker/active worker/queued optimization. Archive
this task and task artifacts; single-IVP defaults/old checkpoints remain intact.
No Git commit/merge/push authorized or performed in this task.

## Implementation / checks so far

Shared conditional module accepts2D(CYX) or3D(CZYX),2Dparameter leaves/key order/
positional arithmetic retained. New3Duses XYZquery frequencies and per-axis
sine/cosine token positions, width divisible by6. Ellipsoid sampler/Euclidean
SDF, operator preset/schema/physical loss/trainer/reference and train.pydispatch
implemented. Actual128^3graph check:2097152nodes/12484608directed edges, degree6,
constant weight11062.15234375 versus independent11062.15167785. Pool32/core256
gives1788padded queries/1532edges. HardIC/no redundantIC loss, latent metric and
coordinate-only decoder queries preserved.

Three focused CPU tests PASS36.8s: sphere identity, inside/axis/center and
rotated ellipsoids versus multi-start constrained SciPy closest-point solves;
3Dpaired/split,timeJVP/hardIC, independent physical/latent residual identity and
encoder-parameter finite differences; independently derived reference stencil
against actual graph. Initial tests exposed tiny principal-axis coordinate
roundoff after rotation: direct tiny-root division was inaccurate at fixed
bisection resolution. Surface-constraint reconstruction of the smallest-axis
projection norm is an algebraically equivalent stable evaluation, validated at
unchanged tolerances. Original failed stdout retained; graph-test API typo
corrected from call to matrix multiplication, no operator change.

B8canonical GPU memory probe running, session75232, run
/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-memory-b8-01.
Legacy2Dsource/initial arrays/output/checkpoint check is session94511. Next
finish B8probe, measure B16serially, then CPU full-cube10held-out references
(2monitor+8test), and fresh GPU training after reference processes exit.

## Measured memory and current resume point

B8probe produces finite8updates: JAXpeak671281920bytes(.625GiB), pool2GiB,
warm update14.811ms/sample2.567ms, firstupdate111.230s/setup11.070s. Original
dispatch returned a dict to absl, yielding CLIexit1after valid COMPLETE; change
to call+returnNone. B16canonical probe then exits0: peak1428841728bytes(1.331GiB),
pool4GiB, warm update21.759ms/sample3.294ms, firstupdate113.217s/setup10.215s.
These are7warm updates per probe, include backward+Adam, not full-run speed.
Choose B16for new operator preset; single-IVP defaults untouched. Device runtime
allocation limit38183976960bytes (~35.6GiB) despite driver reporting48GiB.

Legacy2Dcomparison PASS53equal initial arrays/zero raw-output gap on small model
and successful original full trained2Dcheckpoint load. FP32rotated/near-axis
SDF versus float64 PASS3e-7absolute gate in0.493s. Shared2Dsplit/timeJVP/physical
residual/parameter-gradient regression running session26483; raw
melting-operator-2d-regression.txt. No numerical2Dchange.

CPU reference generation session10875, rawmelting-operator-reference.txt,
dataset /root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1.10identities
declared first in manifest. First128^3full-cube reference passes coarse/fine
time gate8.805e-8, phase in[-1,1]. Later refs reuse CPUcompiled integrators.
Wait for exit0/10files, then cold GPU5000/B16through canonical train.py. No
GPU/FEniCS worker active now. Main required next actions: finishrefs, train,
independent field/condition/checkpoint/source audit, plots/docs/index/archive.


## Reference completion / cold-start handoff

Reference session10875exits0;10full-cube RK4 coarse/fine pairs complete. Max
relative field time discrepancy1.421699211e-7 versus explicit1e-5gate; phase
finite/in[-1,1]. All4new numerical tests plus2shared2Dchecks and legacy parity
pass. Reference wall per input50..54s is preparation cost, separate from train.
Cold wrapper74011runs canonical train.py with unchanged selected B16config,
5000Adam1e-3,Nt8/core256/pool32. Input volumes32^3/patch4^3/512tokens/width48.
No CPU/GPUjob overlap during this measured training; no future-field labels.
Monitor2every250updates, final8test only. Scripts plotting/verification are
ready; verifier expects actual80000distinct parameter draws and final config/
checkpoint/source/field parity. Final phase metrics use24^3held-out diagnostic
nodes, slices128^2; do NOT label these full128^3PINNerror or continuum accuracy.


## Final held-out physical results

Eight final test inputs were declared before reference solving/training. No
positive-time training labels or checkpoint selection. Native direct references
integrate full128^3cubes; PINNmetrics use24^3fixed diagnostic nodes, five times
0,.25,.5,.75,1, plus128^2central z slices. These are not full128^3model errors,
local-PDE spatial convergence, arbitrary-topology or training-seed stability.

|Mean over eight inputs|Pooled|At5s|
|---|---:|---:|
|phi relativeL2, diagnostic3Dnodes|0.226648%|0.092213%|
|solid-indicator relativeL2|0.642494%|4.501659%|
|phi relativeL2,128^2slice|0.420117%|0.286081%|

Terminal solid-fraction relative error mean1.213330%, max3.345292%; absolute
mean1.22694e-5cube fraction. Maximum terminal solid-field L2 is10.593553%, so
small late bodies remain materially less accurate than whole-domain phiL2.
Mean interface absolutephiRMSE .010876. FrozenIC baseline pooledphi45.6221%
confirms future dynamics are learned. Training remains one global network for
all inputs/times with exact prescribedIC, not one model per ellipsoid/time.

Fullrun source/config: same selectedB16setting as probe. Physical parameters,
128^3/degree6graph and original latent residual preserved. Default single-PINN
SOAP preset is unchanged; new operator explicitly uses Adam. Dataset generation
is serialCPU (~50..54s/shape) and exits before training; no reference/FEMjob
competes with measured GPUtraining. B8probe's post-completion CLIreturn issue
is documented above and corrected before B16/full canonical runs.

## Final verification and changed files

Four numerical tests pass: exact sphere, ellipsoid constrained-SciPy/interior/
axis/rotation checks, FP32distance agreement,3Dsplit/timeJVP/hardIC/physical-
latent/parametergradient checks, and independent stencil (geometry checks are
combined in one test). Existing2Dsplit/timeJVP/condition response and physical
parametergradient tests pass. Saved2Dcheckpoint loads;53small-model initial
arrays/rawoutputs exactly match archived source. This validates2Dcompatibility
without changing old files or adding checkpoint migrations.

Independent final verification confirms all80000online parameters distinct,
within explicit center/axis/quaternion ranges, disjoint from10held-out inputs;
all actual times valid. All10reference initialfields/time gates, saved metrics,
physical ranges, aggregate8means and current training-source bytes pass. Fresh
GPUcheckpoint replay on two predeclared cases has maxgap9.56e-7(<2e-5gate).
HardIC difference from independent FP64geometry max5.22e-6(<1e-5gate). Independent
NumPy solid/interface/volume/slice/frozen-IC metric replay also passes. Three
scientific figures were generated and visually checked.

Code changes: shared conditional.pyadds dimension-aware patch/position/query
handling while retaining legacy2Darithmetic; new ellipsoids.py, operator_data/
losses/reference/training.py and configs/train_operator.py; canonical train.py
opt-in dispatch; tests/test_melting_operator.py. No shared operator or optimizer
math changes. Project README updated first; caseREADME/DEVELOPMENT document
schema/physics/memory/errors; artifact index records direct data/run paths.

Memory files changed: PROGRESS.md, DECISIONS.md, this archived task. Reproducible
scripts/raw failures/tests/probes/metrics/configs/figures are alongside it.
No hashes, sibling dependencies, external artifact moves or Git history edits.

## Artifacts and use

Dataset:
/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1.
Run:
/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01.
Checkpoint: that run/model_final.eqx, with its saved config through
operator_data.build_network. Memory runs use corresponding memory-b8/b16-01
names in the same root. Existing outputs are protected from overwrite.

[Cold-start script](melting-operator-coldstart.py),
[independent verification](melting-operator-verify.py),
[verification results](melting-operator-verify.json),
[field audit](melting-operator-field-audit.json),
[plot script](melting-operator-plots.py),
[process cost](melting-operator-process.json).
[Convergence](melting-operator-convergence.png),
[phase fields](melting-operator-fields.png),
[prescribed initial shapes](melting-operator-initial-shapes.png).
Copy records retain original run-name prefixes; full sampling/checkpoints/
predictions remain at explicit external paths. No queued training or exact
resume required; future accuracy/distribution expansion is a separate request.

Final /task-end checks PASS: changed-source/script AST, strictJSON/JSONL,
new-text whitespace, PNG signatures/visual inspection, README/task/archive
links, explicit artifact paths, copied record/current trained-source bytes,
activeNone/localdev and git diff --check.38task artifacts archived; only one
failed-test stdout needed whitespace-only trailing-space normalization, with
nonwhitespace equality preserved. GPU idle1MiB/0%; all workers exited. No
unchanged expensive suite rerun or Git history action. Memory files updated:
PROGRESS.md, DECISIONS.md, this archived task.
