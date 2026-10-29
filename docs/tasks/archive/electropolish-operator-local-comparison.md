# Electropolish operator: matched local AD/nonlocal cold starts

## Goal and acceptance

User authorizes full cold-start timing/accuracy comparison on the ORIGINAL
random six-mode interfaces, including IC/classical-Neumann incompatibility.
Do not flatten endpoint slopes or alter the physical initial condition.
Two serial canonical train.py runs, identical initialization, actual40000
online profiles, Nt4/core256/pool32/IC1024, Adam1e-3/5000updates, ordinaryFP32,
image-only conditioning arch.local_input=zero. Local uses AD time/space,
shares encoder/time-independent spatial work, omits graph halo, and requires
an explicit sixth loss for phi_x=0,q_x/c_le=0 at x=0,1 and positive times.
Retain original five physical losses/scales and inverse-L1 gradient balance.
Report this necessary boundary/objective difference, not only derivative cost.

Final checkpoints, no test selection. Both evaluated on frozen64-interface
graph bank plus independently time-checked local FD targets; report reference
spatial uncertainty and early side-boundary errors separately. Validation:
manufactured local derivatives/residuals/Neumann/parameter gradients; exact
initial/profile/time/core/IC parity; trained checkpoint replay and NumPy L2.
No optimizer/seed sweep, initial-distance change, default promotion or Git action.

## Context

Parent archives: electropolish-operator-training.md and
electropolish-operator-test64.md. Existing best image-only online run315.8s,
64graph testphi pooled/terminal4.506/3.502%. Microbenchmark PDE+parametergrad
NL7.62/AD6.99ms excludes boundary/IC/balancing and cannot settle full training.
Graph closure is not exact pointwise classical Neumann at finite horizon.
Preserve that distinction and reject claims of better stability from one pair.

Frozen test root:
/root/autodl-tmp/data/electropolish2d/operator_test64_20261028.
New runs will use independent names in the existing tf-logs root.
Independent local FD mirror matrix already exists in corrosion2d/local_reference.py;
reuse its checked rectangular operator, not its corrosion boundary/spec loader.
Local BC is top phi/c=0, bottom phi/c=1, sides phi_n=q_n=0.
The graph and local reference prescribe the same initial field at common nodes.

## Status

COMPLETE. Both serial5000-update cold starts, common64graph/local evaluations,
predeclared201/401refinements, all76time-tolerance checks, exact training parity
and independent final field/aggregate/source verification pass. No blocker,
queued training, paused worker or required next action. Keep the opt-in local
control, original ICs, existing defaults and all historical artifacts. No Git
history change. All timings and limitations below refer to this one pair.

## Session changes and validation

### Matched final64-input results

Primary target is a common independent local FD101 reference for all64
original unseen interfaces. Compute a physical field relative L2 separately
for each input and then average. Pooled selects21uniform snapshots on0..5s;
terminal is5s. No positive-time labels or test-based checkpoint selection.

|Mean physical relative L2|Nonlocal|Local AD|
|---|---:|---:|
|phi pooled|4.427359%|8.594609%|
|phi terminal|3.175534%|5.497616%|
|c pooled|5.222193%|9.834575%|
|c terminal|3.779149%|6.431075%|
|Early side phi|4.218719%|12.222698%|
|Early side interface-band phi|19.469291%|47.432525%|

Common graph64phi pooled4.391909/8.634674%, terminal3.041180/5.508388%.
Thus the ranking is not a same-target advantage. Nonlocal pooledphi is lower
on64of64commonlocal inputs, terminal62of64. PooledphiP90/max againstlocal101:
NL6.5239/9.8178%, AD12.2641/16.7394%. These are test-input variability at one
matched initialization/stream, not seed stability or causal attribution.

Early means t*=.001,.003,.01,.025,.05 (not t0). Side width.02 withDirichlet
corners excluded; interface band .1<referencephi<.9. Early interiorphi4.0389/
7.9604%. Initial globalphi3.3708/5.5363%, side4.2208/10.3830% NL/AD. Measured
early point-normal RMS phi2.02231/.0245894, q/c_le .415444/.00834011: local
classical Neumann loss is active and much tighter, while physical field error
is worse at this budget. Do not state pair closure is exact pointwise Neumann,
or that these measurements uniquely establish IC/BC competition as the cause.

|Time|Nonlocal|Local AD|
|---|---:|---:|
|Full process cold start|368.349s|693.032s|
|Trainer|357.001s|681.429s|
|Setup|7.775s|7.450s|
|First update compile+call|223.353s|533.873s|
|Warm synchronized mean update|16.708ms|19.351ms|
|Warm nonrefresh median|14.742ms|16.915ms|
|Warm balancing-refresh median|41.653ms|53.317ms|

Cold ratio1.8814 is dominated by compilation; warm means differ15.8%.
First update includes a balancing refresh. Warm means also include refreshes;
sampling/evaluation are excluded from update timings but included in trainer/
process totals. Reference preparation and64finalevaluation are excluded from
training. Reference workers were stopped during both training measurements.
Dataset paths stay unchanged; copied configs/histories/timing/provenance live
beside this task, checkpoints/sampling/full predictions stay in their run roots.

### Implementation and focused verification

Local six-loss/AD/preset dispatch and synchronized timing/sampling provenance
implemented. Analytic phase/q Hessians, AC/CH, normalized normals, masked core,
true IC targets and parameter gradient tests pass; invalid configs reject.
Independent Jacobian/existing-solver agreement passes on101grid. Underresolved
11/51fixtures hit unchanged concentration range gate (1.00336/1.0000946);
preserve raw failures.101finite-difference Jacobian fixture initially suffered
subtractive cancellation at h1e-7; h1e-5 passes unchanged tolerances. Production
reference equations/range gates remain unchanged. Original five-loss nonlocal
parameter-gradient regression passes33.4s. Four CPU workers generate64local101
and predeclared201/401/native extra-time references at
/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029.
Original-case101index42c reaches1.00001907 and trips the inherited c range
heuristic. Analytic/RHS counterexample proves no total-c0..1maximum principle;
new helper now validates finite c and invariant phi range, records extrema,
without clipping or changing equations/tolerances. Legacy reference gates/data
stay unchanged. Preserve original process/stdout; explicit --resume validates
completed references and solves missing ones after that worker group exits.
Four original CPU workers889876/889879/889880/889881 were SIGSTOP during both
GPU cold-start timings, then resumed; all have exited. This deliberate pause
invalidates reference wall times as an algorithm-speed comparison, but ensures
reference work did not compete with either measured GPU training.
Nonlocal5000run20261029-electro-operator-match-nonlocal-01 COMPLETE exit0:
trainer357.001s/process368.349s/setup7.775s; firstupdatecompile223.353s,
warmupdate16.708ms mean/nonrefresh14.742ms median/refresh41.653ms median.
Fourgraph monitors meanphi pooled/terminal5.2856/3.6285%; separate from64test.
Local5000run20261029-electro-operator-match-local-01 COMPLETE exit0:
trainer681.429s/process693.032s/setup7.450s; firstupdatecompile533.873s,
warmupdate19.351ms mean/nonrefresh16.915ms median/refresh53.317ms median.
Initial checkpoint bytes and initial monitor values exactly match nonlocal.
Full parity audit PASS73initial arrays/40000actual profiles/times/core/IC/BC/
common source bytes; configdelta only method,6thbias/scale,runname.
/usr/bin/time absent (launch exit127before any training); raw
error retained, Python coldstart wrapper records full subprocess elapsed time.
Cosine eigenmode/constant-nullspace FD check passes0.19s; concentration no-max
counterexample passes0.185s. No updated source changed after the training snapshot.
Both64evaluations COMPLETE, saved run/evaluation64.{json,npz}; status complete. No GPU worker now active. Common graph64
NLphi pooled/terminal4.3919/3.0412%, AD8.6347/5.5084%; commonlocalFD10164
NLphi4.4274/3.1755%, AD8.5946/5.4976%; NLc5.2222/3.7791%, AD9.8346/6.4311%.
Early side-strip phi4.2187/12.2227%, interface-band19.4693/47.4325% NL/AD;
interior4.0389/7.9604%. Monitor replay gaps pass2e-5 gate (NL2.65e-6).
Scripts evaluate base64onGPU once; explicit finalize action adds refined4targets
to saved fields onCPU (no second inference). Plot stage base/all is explicit.

Old parent889670/session53770 exited1 after draining its pool, retaining the
original42range exception;3FD401 and all4native101 outputs completed. Manual
101/201index42 solves both exit0 with the new phase-only range gate (same
coarse/fine tolerances). Second explicit resume completes401index42 (fine641.17s, total resume1122.54s);
all76outputs now pass, max time discrepancy2.5829333e-8. No duplicate writes.
Independent base verification passes all64metrics, aggregate statistics,
input-image/grid parity, current training-source bytes and75reference gates.
Convergence/test64/representative-field PNG/SVG figures exist; field plot uses
predeclared interface0, not the most accurate test input. Finalization, all four PNG/SVG scientific figures, independent64and refined4
field/aggregate/normals/current-source checks and all76reference gates PASS.
Reference total-c range0..1.00031175 is recorded without clipping; the independently
verified absence of a unit-interval maximum principle explains why this alone
is not a reference failure.
Reference wall timings include deliberate pauses and are not algorithm-speed
measurements. GPU full-training timings had no active reference workers.

## Refinement and physical boundary interpretation

The four PREDECLARED refinement inputs are0,21,42,63. Independent localFD101-to-
201 pooledphi differences0.302..0.334%;201-to-4010.068..0.076%, terminal0.089..
0.093%. These differences assess only these four inputs; they are not a64-case
continuum-error bound. Nativegraph-to-local401 pooledphi0.56..1.01%, earlyside
1.10..1.49%: finite-horizon boundary/operator differences remain measurable.

|Mean on four declared inputs against localFD401|Nonlocal|Local AD|
|---|---:|---:|
|phi pooled|5.065223%|8.773307%|
|phi terminal|2.944937%|4.886266%|
|c pooled|5.911663%|10.026445%|
|c terminal|3.502884%|5.771259%|
|Early sidephi|3.828842%|10.336398%|
|Early side interfacephi|18.665325%|42.773035%|

This preserves the64-case ranking on refined targets. A field-error advantage
under this budget is supported; exact pointwise Neumann enforcement, stable
training-seed superiority, or a uniquely isolated boundary mechanism is not.

## Diff-level changes and session end

- New `examples/electropolish2d/operator_local.py`: shared encoding, directional
  coordinate Hessians, physical AC/CH and normalized positive-time normals;
  original five losses plus explicit sixth side-wall term.
- New `configs/train_operator_local.py`; `operator_data.py` validates explicit
  method/zero-query/noncausal/reference/six-bias/Neumann requirements.
- `operator_training.py`: local dispatch, matched padded free cores, shared
  sampling records, source snapshots and synchronized timing. Existing sparse/
  causal/legacy arithmetic and parameter shapes retained.
- New independent `operator_reference.py` with explicit times/analyticJac,
  finite-c/phase-range checks; focused `test_electropolish_operator_local.py`.
  No shared neural-network/operator/optimizer implementation changed.
- Example README and DEVELOPMENT document API/losses/physical scope; artifact
  index registers direct run/dataset/checkpoint/results paths and figures.
- PROGRESS/DECISIONS and this archived report record completion. Config/history/
  timing/memory/selection copies use `20261029-electro-operator-match-` names.

Focused checks: analytic residual/Hessian/normal/mask/encoder-parameter gradient,
invalid local config rejection, independent Jacobian/old-solver101agreement,
FD cosine/constant identity and total-c no-maximum-principle counterexample all
pass. Existing nonlocal residual/parameter-gradient regression passes. Earlier
underresolved fixtures and Jacobian cancellation failures remain in raw stdout,
with causes/corrections above; no tolerance relaxation or silent fallback.
Both canonical cold starts exit0; exact73initial arrays/40000actual profiles/
time/core/IC/BC/common source parity passes. Final64and refined4norms, reference
level differences, aggregate statistics, saved evaluation inputs, physical
ranges and current training-source bytes independently replay. All76reference
BC/range/time gates pass. Training process times exclude reference preparation;
reference raw times containing deliberate worker pauses are not benchmarks.

Scripts: [cold starts](electro-operator-coldstart.py),
[references](electro-operator-local-references.py),
[evaluation/parity](electro-operator-local-evaluate.py),
[plots](electro-operator-local-plots.py),
[independent verification](electro-operator-local-verify.py).
Results: [nonlocal64](electro-operator-sparse-evaluation64.json),
[local64](electro-operator-local_ad-evaluation64.json),
[parity](electro-operator-local-parity.json),
[all verification](electro-operator-local-verification-all.json),
[reference summary](electro-operator-local-reference-summary.json).
Figures: [convergence](electro-operator-local-convergence.png),
[64test CDF](electro-operator-local-test64.png),
[boundary](electro-operator-local-boundary.png),
[phase fields](electro-operator-local-fields.png); SVG copies alongside PNGs.

Training runs/checkpoints/full predictions remain at their existing direct paths:
`/root/autodl-tmp/tf-logs/electropolish2d/20261029-electro-operator-match-nonlocal-01`
and `.../20261029-electro-operator-match-local-01`.
Independent local dataset:
`/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029`.
Old common graph bank:
`/root/autodl-tmp/data/electropolish2d/operator_test64_20261028`.
No hashes or sibling imports are used. Checkpoints replay with saved configs.

For reproduction from repository root, reference script takes `--resume` to
validate existing files. Evaluation actions `nonlocal`/`local` refuse overwriting
base results; `finalize` adds refined targets to saved predictions once.
Plot `all` and verify `all` operate on complete saved results. After archiving,
use script paths under `docs/tasks/archive/`; outputs go alongside scripts.
An earlier automatic approval usage-limit failure prevented a memory edit;
no unsafe-action rejection occurred. An interrupted first resume produced no
401/interface42file; second explicit resume reused75outputs and completed only
that missing reference. Original failures/resume stdout remain archived.

Final session verification includes strict JSON/JSONL, changed-module AST,
PNG/SVG/direct-path/archive-link/index checks and `git diff --check`. No unrelated
expensive suite repeated. All workers complete; no optimization is queued.
