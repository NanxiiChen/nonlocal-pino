# Extreme interface thickness and nonlocal horizon

## Goal

Extend the completed sphere-melting comparison to epsilon*.00375/.001875/.0009375,
testing whether nonlocal offers better accuracy or numerical stability than
localAD under a matched finite budget. Separately study the effect of changing
nonlocal horizon and spatial resolution as the interface thins. Do not select
settings or interpret failures to favor a desired winner.

## Acceptance

- Qualified common local reference resolving the very thin interface, with
  explicit spatial/time accuracy checks and any symmetry/domain assumptions.
- Matched cold architecture, initialization, actual query streams, constraints,
  optimizer and10000update schedule for each derivative comparison.
- Separate horizon, grid resolution and kernel moment normalization effects.
  Preserve original graph baseline and production defaults.
- Record finite-budget optimization, nonfinite failure, interface geometry and
  dense field errors separately. A bounded/smooth wrong field is not stability.
- Independent derivative identities and proportionate focused regressions before
  launching new cases; serial GPU ownership and explicit artifact paths.

## Context

[Prior study](thin-interface-derivative-study.md) completed16cold10k
trials. At epsilon*.0075, nonlocal/localAD pooled phase0.8783/0.03057%, interface
band10.26/0.350%, with dense quadrature confirming ranking. Current128^3six-axis
graph usesdelta*.01,h=1/127,delta/h1.27 and secondmoment0.6858547757.
Nonlocal/native error is not zero, so native/local baseline is not an exact
irreducible floor. Local errors approach the old FD512reference resolution.
User notes a faster relative growth of local error and requests much thinner
cases plus a separate horizon study. Relative growth from a tiny reference-scale
baseline does not itself show impending localADinstability.

Physical M=.1,lambda5,R035,Lc100,Tc5 and cubeNeumann remain the intended problem.
Old paired recipe uses hardIC latentMLP64x4,tanh residual,SOAP1e-3,decayfactor.5/
2000steps,min1e-5,K256/Nt16 and ordinaryFP32. Domain grids, horizon, output
coordinate scaling and loss preconditioning are distinct possible confounders.
No Git actions or production-default changes are authorized.

## Implemented experiment matrix

The opt-in `configs/train_extreme_interface.py` dispatches through `train.py`
to `extreme_interface.py`, keeping old presets unchanged. Shared raw model,
SOAP and loss match the previous study; explicit finite-loss/gradient/norm/
model/optimizer checks prevent a clipped overflowing norm from appearing stable.
Odd grids are rejected because they include the non-smooth distance-function
center in local AD. Last finite states and numerical failures are preserved.

There are32distinct runs, allcold10000with two paired runtime seeds42/43:

| Group | Settings | New runs |
|---|---|---:|
| Thickness | epsilon.0075/.00375/.001875/.0009375; original128^3,delta.01; nonlocal/local |16|
| Horizon | epsilon.001875,128^3;delta.01/.015/.02;raw/moment; reuse originaldelta.01raw |10|
| Joint resolution | epsilon.001875,512^3coordinates,delta1.27/511;raw/moment/local |6|

`thin_interface_operators.py` enumerates complete symmetric cubic offsets,
omits outside-cube edges and uses fixed-shape core/halo coordinates. Raw/moment
normalization is explicit. The three128^3horizons have6/26/80neighbors and raw
secondmoments.68585/1.04740/.97483. Globalmomentnormalization preserves symmetric
weights; no boundary-row scaling. Refinement is jointly changingh anddelta,
not a pure horizon effect; its localADcontrol shares that grid's actual draws.

## Reference qualification

Independent `extreme_reference.py` solves the derived radial latentPDE with
SciPyBDF and an analytic tridiagonal Jacobian. It does not prescribe front motion.
8193/16385rnodes,outerradius.9,rtol1e-9/atol1e-10/maxstep.0025. Everywidth passes
spatial,timehalf+tightertolerance,andouterradius1.1sensitivity checks.
Newdata: `/root/autodl-tmp/tf-logs/melting-extreme-references-20261111-01`.
Fourwidths took73.5sCPU/I/O,total235MB. Worstthinspace pooled9.98e-9/interface
3.24e-7; time/outerwholefield below1e-10. Referencekind explicitly remains a
far-field radial surrogate, not an exact cube Neumann reduction.

Common evaluation uses256Fibonacciangles and4096radialmidpoints with
rmax(omega)=.5/max(abs(omega)) and weightsr^2drdOmega over the actualcube.
1,048,576queries and9times, including~6800interfacequeries/time at the minimum
epsilon. Weights are not renormalized; theirvolume.9994236records quadrature
error. Physicalfields/rays and profiles are stored outsideGit; no labels enter
training. The planned finalthin audit uses512angles/8192radialmidpoints.

`docs/tasks/melting_extreme_cube_check.py` independently comparesnewthin.00375
against cube512/768and768dt/2; all predeclaredgates pass. Cube/radialpooled
0.06830%->0.03074%,ratio.45006; cube768timehalf8.66e-14. Radialphysicalnormal
flux on cube faces is at most1.96e-10at.0075and decreases to1.62e-95atminimum.
The comparison supports the far-field approximation but does not turn it into
an exact geometry equivalence. Receipt:
`/root/autodl-tmp/tf-logs/melting-extreme-cube-check-20261111-01/comparison.json`.
Oldcube384/512errors against radialalso reduce consistently withsecondorder.

No new native graph solution is supplied: extreme latent graph integration
would need independent stiffness/refinement checks. Report no new native error
floor. In purephase graphquotients can grow~exp(2h/(sqrt2epsilon)); failures may
therefore reflect the transformed residual/discretization, not AD itself.

## Verification and resources

Fiveoperator tests pass (allrows/boundaries/fullSparseD6/26/80neighborparity,
moments,symmetry/PSD,fixedshapes and exactlegacy6arrayparity). Fiveindependent
radialtests pass (Jacobian,Neumannconvergence,cubequadrature,IC/refinement).
Sixtrainingtests pass (old/newrawloss+gradientparity, referencecompatibility,
latentnonfinitechecks,lastfinite model/state preservation,overflowing gradient
norm rejection,oddgrid rejection). Source syntax andgit diff --check pass.

The80neighborrawprobe completed40updates on the currentRTX4090/49140MiB:
11.27mswarmstep,1.233GiBactualactivepeak (allocatorreserved~26GiB),41.3sloop
includingcompile/evaluation. This qualifies capacity, not accuracy. Artifacts:
`/root/autodl-tmp/tf-logs/icemelting3d/melting-extreme-probe80-20261111-01`.
NoFEMwork ran; no data was deleted and noGit action was taken.

## Final status and results

Complete: all32 cold10000-update training arms and the dense evaluation passed.
The queue ran from2026-11-11T07:36:47.533539+00:00 to08:40:03.744897+00:00,
63.27minutes including child startup, compilation, evaluation and analysis.
No owned GPU job remains; production defaults are unchanged. This is a
single-sphere PINN derivative study, not the random-ellipsoid PINO test set.

Direct artifact root:
`/root/autodl-tmp/tf-logs/melting-extreme-interface-20261111-01`.
Its `journal.json`, `results.json`, `dense-evaluation.json`, immutable `source/`
and per-run direct paths retain the complete evidence. No hashes are used.

### Thickness controls

Mean positive-time whole-field relative L2 across two paired seeds:

| epsilon* | Nonlocal pooled | Local AD pooled | Nonlocal terminal | Local AD terminal |
|---|---:|---:|---:|---:|
|0.0075|0.90310%|0.01056%|0.90242%|0.02947%|
|0.00375|51.28271%|51.48623%|33.75277%|32.70109%|
|0.001875|61.11809%|69.71064%|57.98993%|57.07851%|
|0.0009375|65.76458%|73.33678%|77.04331%|70.07576%|

Both methods fail to learn accurate front motion at the three new thin widths.
All32 runs retain finite monitored loss, gradients, gradient norms, models and
optimizer states; no numerical-failure status occurs. For each new width and
both seeds, nonlocal terminal error exceeds local AD terminal error. Therefore
its lower pooled error at the smallest widths is not evidence of an accurate
stability advantage. Interface-band errors are approximately190%. The
minimum-width first-seed final radius is about.331nonlocal/.312local versus
reference.099, starting from.35. Zero-crossing diagnostics find no missing or
multiple crossings in any run, which does not imply correct interface motion.

The epsilon.00375 runs are still slowly improving at10k; the smallest-width
local error is effectively on a plateau. These are finite-budget results,
not convergence proofs or a universal comparison of differentiation methods.

### Horizon, moment normalization and joint refinement

At epsilon.001875, two-seed mean positive-time whole-field L2:

| Coordinate grid | Horizon | Raw weights | Unit interior second moment |
|---|---:|---:|---:|
|128 cubed|.01|61.11809%|60.10019%|
|128 cubed|.015|65.31272%|65.42257%|
|128 cubed|.02|65.16208%|65.08658%|
|512 cubed|1.27/511|69.12277%|68.75145%|

Matched local AD gives69.71064%on128coordinates and69.69635%on512coordinates.
Neither a larger horizon, moment normalization nor joint h/delta refinement
recovers accurate evolution under the declared optimization recipe. The512
case uses random core points on the coordinate grid, not a full512-cubed
training batch. No pure grid-only or pure horizon-shrink effect is claimed.

Warm update times on the same RTX4090 are about3.2ms for6neighbors,
5.2ms for26neighbors,11.2ms for80neighbors, and3.4--3.5ms for local AD.
These exclude compilation and checkpoint evaluation; whole-queue walltime is
reported above. Increasing horizon is materially slower in this implementation.

### Dense evaluation and independent audit

The four minimum-width checkpoints were reevaluated with512angles times8192
radial midpoints:4,194,304cube queries, four times the primary evaluation count.
The quadrature volume is1.0001527641; weights remain unrenormalized. All four
checkpoint replays on the original evaluation coordinates are bit-identical
(actual maximum discrepancy0), below the predeclared tolerance8e-5. Dense
pooled scores change by-.02362 to-.02611percentage points; terminal scores
change by-.02441 to-.02729points. This does not explain the large field errors.
Dense minimum-width means: nonlocal/local pooled65.74072/73.31068%, terminal
77.01627/70.05078%. Both primary and dense metrics are retained.

The owner validates all actual configurations, full update budgets, case
identities and saved-field norms. All22pair records cover10000updates and pass
cold-model, optimizer and actual-input comparisons. A separate CPU audit
recomputes the last eight seed43horizon/refinement saved-field metric sets,
checks eight seed43cold-state/actual-sampling pairs, and verifies all13source
snapshot files against current numerical source bytes. All pass. This adds to
the independent first-seed and minimum-width interim audits; no GPU retraining
or reference solve was needed for the final report.

### Output-coordinate diagnosis and possible follow-up

Independent CPU loading of the actual first-seed checkpoints shows a shared
output-amplitude shortfall. At r=.2/.3/.35/.4,t=1, the required reference latent
correction is approximately-23.6/-47.2/-94.4/-188.8as epsilon decreases, but
local models produce approximately-23.6/-28.6/-28.2/-28.6. Their final tanh
trunk/linear-head bounds abs(bias)+sum(abs(weights)) are23.65/28.66/28.30/28.88.
The saved weights cannot produce the required corrections. Unconstrained
output weights could grow: this is an optimization/coordinate-scale diagnosis,
not a theorem about architecture capacity or proof of the sole failure cause.
Nonlocal head bounds are similarly limited, and its minimum-width correction
also changes sign spatially. Graph-quotient stiffness remains another possible
contributor.

Full direct-path diagnostic receipt:
`/root/autodl-tmp/tf-logs/melting-extreme-interface-20261111-01/interim-seed42/latent-scale-diagnostic.json`.

A separate matched control could normalize the learned correction using the
known interface width, equivalently predicting a distance-scale correction
before division by the width. It must apply equally to local and nonlocal
arms, without prescribing a future radius trajectory. No such control is
implemented or launched here; the completed32arm matrix remains unchanged.

### Figures and reproducibility

The artifact root contains `extreme-error-vs-thickness.png`,
`extreme-convergence.png`, `horizon-comparison.png`, and
`refinement-comparison.png`. The report additionally uses
`final-thickness-summary.png`, showing pooled and terminal two-seed means and
ranges directly from `results.json`. The horizon interface panel's auto-scaled
axis magnifies negligible differences around190.66%; do not interpret those
visual differences as meaningful recovery. Primary numerical tables are above.
Earlier `interim-seed42/` and `interim-two-seeds/` artifacts remain preserved.
No generated figure, raw field, model or optimizer state is copied into Git.

The finite owner is `docs/tasks/melting_extreme_interface_study.py`. Its
32-arm mock analysis also passes, including step0failures, missing radius,
None warm timing, all-NaN error bars and the four plots. Exact original launch:

```bash
.venv/bin/python -m docs.tasks.melting_extreme_interface_study queue \
  --output /root/autodl-tmp/tf-logs/melting-extreme-interface-20261111-01 \
  --reference-root /root/autodl-tmp/tf-logs/melting-extreme-references-20261111-01 \
  --cube-check /root/autodl-tmp/tf-logs/melting-extreme-cube-check-20261111-01/comparison.json
```

Do not rerun into the completed directory. No blocker remains for this declared
matrix; output-scale investigations would be a separate task.

## Session-end record

The isolated source/config/operator/reference/owner files and normal train.py
dispatch were implemented in earlier sessions. Sixteen focused CPU tests,
reference-bank/cube qualification, GPU capacity smoke, mock analysis and syntax
checks passed then; unchanged numerical tests were not repeated for this
status-only completion. Final inspection confirms32complete arms, the dense
audit, exact pairing, frozen-source parity and independent field norms.

This final session updates DECISIONS/PROGRESS, archives this task, fixes the
example README research-study link and adds a short negative-result qualification.
It also renders one external summary figure from existing final results.
Numerical source, production presets and running configurations are unchanged.
Documentation links and git diff --check pass. No commit, push, data deletion,
new training, reference solve or environment change was performed.
