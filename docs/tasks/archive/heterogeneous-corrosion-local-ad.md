# Heterogeneous corrosion: matched optimizer and local-AD comparison

Date: 2026-11-05 to 2026-11-06. Status: complete.

## Goal and accepted scope

Compare the successful stronger material-conditioned nonlocal PINO with local
spatial AD, using one independent local reference for both accuracy and cost.
User then requests nonlocal+SOAP first, followed by AD, and permits concurrent
experiments when memory allows. Complete fixed10000-update runs, preserve all
controls, and separate uncontended update timings from overlapping cold costs.
No commit, push, deletion or default promotion was requested or performed.

## Method

Same width96 network,323522learnable parameters, B8/Nt8,1024IC points,9 mouth
nodes,256core points,32pools, online13coefficient material inputs, timeJVP,
ordinaryFP32/highestmatmul, original KKS PDE and softIC/mouth losses. All initial
model arrays, actual10000material/time draws and core/IC/spatial arrays match
exactly. Matching optimizers also share the byte-identical fresh initial state;
SOAP from an Adam control initializes its own fresh state. No final-model
warmstart, evolved labels or reference-based checkpoint selection.

Nonlocal uses complete halos with its graph no-exterior-edge closure. Local
AD evaluates only padded core points for phase/q Laplacians and adds explicit
zero-normal-gradient phase/q MSE on248 open midpoint wall nodes (length3.09).
Both use stable q=cle*(raw_liquid-1). AD queries remain independent, sharing the
material encoder and using directional JVPs without dense point Jacobians.
The added two boundary losses enter inverse-gradient balancing: this is a
complete-method comparison, not an isolated interior-operator ablation.

Adam settings stay unchanged: LR1e-3, continuous.5/2000decay, floor1e-5,clip10.
SOAP uses explicit b1=.95,b2=.9,frequency10,maxdim256,no1D. One declared SOAP
LR3e-4 control changes only the initial rate, with an explicit replay exception.

## Complete results

All percentages are means of four per-material relative L2 values, over the
same11reference times and3969nodes. Fixed10000 endpoints:

| Method | LR | phi pooled | phi terminal | c pooled | c terminal |
|---|---:|---:|---:|---:|---:|
| Nonlocal Adam |1e-3 |1.226924% |2.086134% |1.486678% |2.550962% |
| Local AD Adam |1e-3 |0.872104% |1.516444% |1.064200% |1.883251% |
| Nonlocal SOAP |1e-3 |94.933637% |100.000000% |94.606767% |99.998258% |
| Local AD SOAP |1e-3 |94.933671% |100.000000% |94.608109% |99.999731% |
| Nonlocal SOAP |3e-4 |22.666845% |28.548872% |19.524692% |27.606257% |

Local AD improves three of four materials at the fixed endpoint; sample2
slightly worsens. At9000 both Adam methods are very close. One paired
initialization/four-map cohort does not establish a cross-seed ranking.
Nonlocal errors include its finite-horizon discrepancy from the local target.

| Adam resources, RTX6000D | Nonlocal | Local AD |
|---|---:|---:|
| Mean warm update, all refreshes included |14.59766ms |17.92140ms |
| First update including compilation |176.0037s |438.3071s |
| Observed complete process |380.2590s |673.4435s |
| Peak active GPU allocation |3.9314GiB |3.0194GiB |

Adam warm windows have no overlap with the tracked other GPU experiments.
The AD cold startup does overlap the tail of the SOAP controls; total is an
observed process cost, not a strictly isolated compilation benchmark. SOAP
whole-run timings contain overlap and are not used for speed ranking.
Nonlocal SOAP1e-3 uncontended updates2..1000 average23.62119ms. Memory allocation
pools20%/30% are separate from actual active peaks. All owned jobs have ended.

## Negative-control diagnosis

Independent CPU evaluation of the SOAP1e-3 model at5000 confirms a narrow
wrong initial transient, then FP32tanh saturation. On15points of heldout0,
phi at.001 is below.00023 and at.01/.1/1 is exact0; physical time gradients
also vanish, although raw derivatives do not. This explains exact0 AC loss
without correct fields. Initial phi is approximately fitted only at t0.

The3e-4 run avoids full saturation but bulk phi is.8105-.8120, with q/cle
about-.985. It lies near the homogeneous non-pure reaction root
phi*=(1-3*N1*q*dc/N2)/2, about.809 at its actual q; reference bulk phi exceeds
.997. This supports a wrong stationary branch, not dynamic stability or a
full residual certificate. Do not infer a general SOAP defect. Existing Adam
presets remain intact; no residual/IC intervention or further sweep is queued.

## Direct artifact paths

Common local reference (independent FV, not FEM):
`/root/autodl-tmp/data/heterogeneous_corrosion2d/20261105-activation-stronger-reference-01/local_reference.npz`.

Run root: `/root/autodl-tmp/tf-logs/heterogeneous_corrosion2d`.
All entries below are direct children of that root:

- `20261105-activation-stronger-w96-cold10000-01`: original nonlocal Adam.
- `20261106-activation-stronger-w96-soap-cold10000-01`: nonlocal SOAP1e-3;
  independent saturation script/JSON in `saturation_diagnostic/`.
- `20261106-activation-stronger-w96-local-soap-cold10000-01`: local SOAP1e-3.
- `20261106-activation-stronger-w96-soap-lr3e4-cold10000-01`: lower-LR SOAP.
- `20261106-activation-stronger-w96-local-adam-cold10000-01`: local Adam.
- `20261106-local-reference-comparison-01`: all55saved-checkpoint local-FV
  evaluations, `summary.json`, `metrics.json`, full `convergence.png` and
  `final_interfaces.png`. The `adam_comparison/` child has focused figures
  regenerated from these same records and saved predictions, with no extra
  model evaluations.

Each new run has a sibling launch receipt/log. They record concurrency and
source commands. The compact independently checked provenance/timing record is
[heterogeneous-corrosion-control-audit.json](heterogeneous-corrosion-control-audit.json).

## Code and validation

- `train.py`: explicit nonlocal/local dispatch, original initial-state replay,
  strict per-update draw checks, local-reference evaluation and accurate runtime
  metadata; no shared operator or optimizer-library changes.
- `local_ad.py`, `boundaries.py`: checked physical coordinate derivatives,
  time chain rule, wall quadrature and7physical losses.
- `replay.py`: exact config/array checks, explicitly limited optional LR change.
- `losses.py`: declared loss-count support, unchanged5loss arithmetic.
- Explicit SOAP/local-Adam/local-SOAP/lower-LR presets preserve historical configs.
- `compare.py`: common local target, every published checkpoint, complete
  negative controls and figure generation. README and artifact index updated.

Seven focused AD/boundary cases pass: analytic Laplacians/normals, actual-net
point Hessian comparison, no query coupling, encoder parameter-gradient finite
check, independent loss/BC reductions and invalid-geometry rejection. Three
optimizer/replay integration cases pass for Adam5/SOAP5/SOAP7 and strict LR
exception. Prior11physics/reference tests remain valid; unchanged expensive
suites were not repeated. All four full runs, final dense-grid field evaluations,
actual input/state/target audits, AST/JSON and diff checks pass.

## Resume point

No required work remains. Optional future work: more paired initializations,
early-transient/IC treatment for SOAP, or a separately validated lax.map
GradNorm compilation optimization. None is implemented or queued. Preserve
all failures and successful controls; no cleanup or history action authorized.
