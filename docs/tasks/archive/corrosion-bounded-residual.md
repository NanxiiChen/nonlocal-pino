# Bounded corrosion phase-residual experiment — completed 2026-10-25

## Goal and conclusion

Test whether bounded inverse phase-Jacobian weighting improves corrosion2d,
then whether inherited equation pre_scales can be removed. Eight new cold
starts completed5000 updates each. The selected eta0.1/cap11 improves both
terminal physical errors at seed42, but worsens both at validation seed7.
Removing AC pre_scale fails for both original and bounded formulations: phase
L2~0.512 and no terminal phi=0.5 interface. Keep production original-residual,
KKS and scale defaults. The new bounded mode remains an explicit research
option; do not advertise consistent improvement or promote the screened seed.

## Formulation and fixed conditions

$$
r_\eta=\frac{1+\eta}{4\phi(1-\phi)+\eta}\,r_\phi,\qquad\eta>0.
$$

The weight is1 at phi0.5, positive and bounded by(1+eta)/eta. It preserves
physical PDE zeros and includes differentiation of the state-dependent weight.
It multiplies the original AC residual; it is not the exact latent residual
and does not silently regularize that mode. CH, physical KKS free energy,
output mixture, soft IC/BC, graph and reference are unchanged. No hard IC.

All current paired runs use101x51 cubic graph/delta0.02,256 graph cores,
16 times/update,100 fixed graph batches, SOAP/lr0.001, seed42 or7,5000 updates,
50 s horizon and five-component GradNorm. Screen eta1/0.1/0.01 (caps2/11/101)
at AC1e-7/CH1; select eta0.1 using both terminal field errors. Validate that
choice without retuning at seed7. No-scale controls set AC=CH=1; CH was
already1, so the operative change is removing AC's1e-7 factor. All use fresh
identically initialized raw parameters within seed and matched sampling.

The reference remains /root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz.
It was independently integrated in physical fields, aligned to the same graph,
IC and prescribed BCs, with prior temporal-refinement relative L2 error3.64e-9.
This task rechecks metadata/coordinates/time/BCs; no PDE changed and no new
reference/FEM solve was needed. Reference fields are evaluation-only.

## Terminal full-grid results

Phi/c columns are physical50 s relative L2 at exactly5000 completed updates.
The last column is the median of logged terminal-phase errors at updates
4000,4200,...,5000; it is not a substituted checkpoint or an averaged network.
Historical controls are context only. The primary seed42 original is a fresh
current-code cold start: historical source predates the selected-constraint
JVP refactor, and its training trajectory is not bitwise identical. Do not
claim the refactor alone is proven to cause that trajectory difference.

|Run|Terminal phi L2|Terminal c L2|Last1000 terminal phi median|
|---|---:|---:|---:|
|historical original|0.03483970|0.04135163|0.04628786|
|original|0.03983833|0.04721918|0.04771523|
|full inverse|0.56838572|0.54584502|0.56850951|
|cap2|0.05998548|0.07092421|0.04509669|
|cap11|0.01618859|0.01927956|0.03016799|
|cap101|0.56890280|0.54622266|0.56805667|
|original seed7|0.02663608|0.03159772|0.02978132|
|cap11 seed7|0.03661033|0.04342720|0.03631665|
|original no scale|0.51238957|0.51696865|0.51237685|
|cap11 no scale|0.51218778|0.51563369|0.51211305|

Seed42 cap11 terminal errors decrease about59% relative to the fresh original;
its late phase median decreases0.04772->0.03017. At seed7, terminal errors
increase about37%, and the late phase median worsens0.02978->0.03632. The
negative validation is not only a final-update reversal. Two seeds do not
establish a general population effect; the screen itself is exploratory.

## Additional diagnostics and interpretation

- Cap101 has73.62% zero concentration-output derivatives at terminal full grid,
  versus85.25% for the historical full inverse. Both reach wrong solid
  concentration~0.96434 with BC error~0.03566. Cap11 at both seeds and cap2
  show0% zero concentration-output derivatives on this terminal probe. Thus a
  finite but large cap can reproduce saturation; removing saturation alone
  does not guarantee improvement. These fractions are terminal-only here,
  not the pooled-time fractions from the earlier diagnostic.
- Neither selected cap11 seed has more than one vertical phase crossing in a
  column at terminal time. Poor cap101 also lacks multiple vertical crossings;
  do not reuse the earlier extra-band explanation for this new failure.
- Both no-scale arms lose the terminal0.5 contour. Maximum physical BC errors
  are0.78963 original/0.78300 bounded, with poor IC/BC losses. Their recorded
  GradNorm weights for CH/BC/IC hit10000 while AC stays~0.2. The inherited
  norm clipping/caps are not invariant under multiplying one equation by1e7.
  This is consistent with the earlier gradient diagnosis; spatial capping
  cannot remove equation-scale imbalance. No alternative GradNorm rule or
  joint AC/CH preconditioner is implemented or claimed validated here.
- Bounded weighting mitigates extreme pure-phase amplification in one case,
  but does not resolve seed dependence, stiffness or coupled optimization.
  The diagnosis identified a real mechanism, not a sufficient universal cure.

## Verification and artifacts

[Replay/plot script](corrosion-bounded-residual.py),
[machine-readable audit](corrosion-bounded-residual.json),
[screen curves](corrosion-bounded-residual.png),
[second-seed/no-scale curves](corrosion-bounded-residual-validation.png),
[seed42 terminal fields](corrosion-bounded-residual-fields.png),
[explicit validation controller](corrosion-bounded-controls.py),
[controller return codes](corrosion-bounded-controls.json).

The audit declares every model/config/reference path explicitly. All ten saved
runs (eight new plus two historical) pass config/physical metadata checks,
byte-equal initialization within seed,5000-update checks, saved physical-field
L2 replay and independent CPU final-checkpoint prediction. Maximum CPU/GPU
field difference is2.563e-6. Figures were visually inspected. All five
controller subprocess return codes are0. No Git history action or old artifact
rewrite/move occurred; no file-hash loading gate is used.

Nine focused algebra/reference/gradient/SOAP tests passed in23.14 s. After
adding early train validation, the two affected bounded tests passed again
in6.88 s. Direct invalid-eta train rejects before reference access or run-folder
creation. The CLI also rejects nonpositive/nonfinite eta and eta use outside
bounded training. Final diff, schema/path and whitespace checks are recorded
at session end; no full-suite or unchanged expensive reference rerun.

Safe independent audit from repository root, using its JAX environment only:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u docs/tasks/archive/corrosion-bounded-residual.py
```

The controller and standard train CLI refuse existing run directories. For
new training replicates use new explicit names in a copied plan; do not resume
or overwrite these checkpoints. All new paths/source snapshots and physical
metrics are registered under the corrosion bounded experiment in artifacts/index.yaml.

## Diff-level changes, task-end and exact continuation

Added research bounded residual/explicit eta config and CLI, early validation,
summary provenance and two mathematical/gradient tests; documented usage in
corrosion README. Added eight separate run artifacts and this replay/report
bundle; updated artifact index, DECISIONS and PROGRESS. Production train.py,
physical PDEs, default configs, existing data/checkpoints and melting are
unchanged. User's conditional scale removal is not promoted because the
no-scale controls fail. All training/audit processes exited; GPU idle.

No blocker or required remaining action. Future work, if pursued, should
isolate coupled-gradient balancing/late-update variability rather than infer
that a larger pure-phase cap or removal of scales is a sufficient fix. This
experiment is complete; that investigation is a separate task.
