# Corrosion output-constraint and pre-scale ablation — completed2026-10-24

## Goal and acceptance criteria

The user authorizes removal of KKS output constraints or adjustment of
pre_scale where they impair transformed-residual optimization. Preserve the
physical KKS free-energy/PDE, graph,50 s horizon and soft IC/BC. Compare four
fresh5000-step/seed42/SOAP/lr0.001 runs: phase/latent times bounded KKS/direct c,
all at AC scale1e-9 and CH scale1. Include the existing bounded-KKS/AC1e-7
phase/latent pair as the scale controls. Report physical terminal full-grid
phi/c relative L2 and range/BC/gradient diagnostics, with direct artifact paths.

## Context and rationale

Previous paired task is archived as coupled-latent-residual-experiments.md.
Corrosion AC1e-7 original/latent phi errors0.03484/0.56839; concentration
saturation and capped GradNorm proposals were observed in the latent run.
AC1e-9 is an explicit experimental scale motivated by initialization gradient
L1~1990 versus other terms~23 at1e-7. It is not a physical coefficient change
or a validated optimum. Direct concentration is c=raw[1], while phase remains
(1+tanh(raw[0]))/2. This removes the output mixture and concentration bound,
not the KKS physical free energy or the phase map required by the derivation.
Identical raw weights need not produce identical physical fields across maps.

## Status

Complete: four fresh GPU cold starts finished5000 updates, and independent
replay of all six controls passed. Seven focused test methods passed in
21.77 s. Actual network constraint is used consistently by the PDE JVP.
No training/FEM process remains. Console logs are preserved as train.log in
each new run, with temporary controller copies at
/tmp/corrosion-scale-<map>-<form>.log.
Run names are20261024-residual-<phase|latent>-map<kks|direct>-ac1e-09-ch1.0-seed42
under the existing corrosion2d tf-log root. The archived replay script runs
with JAX_PLATFORMS=cpu. No production default changes are justified.

## Blocker

None.

## Next action

None required for this completed ablation. Optional future work must distinguish
removing concentration bounds alone from removing the mixture structure, or
study milder phase preconditioning while retaining a physical-field validation
target. These have not been tested. Baseline reference remains
/root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz.

## Final results

| Output map | AC scale | Residual | Terminal phi L2 | Terminal c L2 |
|---|---|---|---|---|
| KKS |1e-7|original (previous)|0.03483970324|0.04135162630|
| KKS |1e-7|transformed (previous)|0.56838571953|0.54584501961|
| KKS |1e-9|original|0.04212177990|0.04992269261|
| KKS |1e-9|transformed|0.43678754680|0.44956616361|
| direct c |1e-9|original|0.53845845883|0.53685448894|
| direct c |1e-9|transformed|0.53644677044|0.52965017018|

Independent CPU probes of the two new KKS final checkpoints reproduce GPU
fields within2.15e-6. Both have0% exactly-zero concentration output derivatives
on the full grid at times0,0.5,1 (old latent85.2%). Thus removing wrong-endpoint
saturation by scale reduction did not restore low L2. Direct-c original has
c range[-0.0009788,0.9999425], boundary max error0.01467; it also fails to grow
the pit correctly. All entries are exactly5000 completed updates; no earlier
favorable checkpoint is substituted. The direct latent run briefly decreased
to~0.35 near1200 updates, then rose; last1000-update phi medians are0.53931
for direct original and0.53682 for direct latent. KKS1e-9 phi medians are
0.05641/0.43771. The new endpoint comparisons are not isolated late spikes.

## Findings and interpretation

- Smaller AC scaling has a measurable optimization effect: transformed
  phi L2 improves0.56839->0.43679, solid boundary max field error decreases
  0.03566->0.00448 and wrong-endpoint saturation disappears in the probe.
  It does not restore correct dynamics. The transformed field develops extra
  liquid bands (28/101 columns have three vertical interface crossings and
  13/101 have two), unlike the single-pit reference.
- Removing the entire concentration output map does not fix the late field.
  Both direct-c arms remain near the initial pit size, with phi L2~0.54.
  Direct c has unit output derivative, so concentration saturation cannot
  explain that failure. Soft IC/BC remain unchanged.
- KKS is not algebraically incompatible with the phase residual. Its mixture
  simplifies q=c-dc*h(phi)-c_Le to c_L-c_Le; direct c makes that cancellation
  something the network must learn. In CH, the KKS map removes phase from the
  spatial action on q while retaining the time coupling in c_t. This explains
  a structural difference in conditioning, not a proven sole causal account.
- Direct concentration can leave physical ranges: over all saved times of
  the final models, original c spans[-0.0009788,0.9999425] and latent
  [-0.00502235,0.9979184]. Their fractions outside0..1 are0.0213%/0.3522%.
  Reconstructed liquid concentration in the latent arm reaches-0.0051106;
  this is recorded rather than clipped. Boundary max errors are0.01467/0.02350.
- No new setting outperforms prior KKS/AC1e-7 original in either field.
  Keep the original defaults. This is one new scale and one seed on a101x51
  grid, not an exhaustive tuning or generalization result. Direct-c and
  scaling options remain available for explicit experiments.

## Implementation and verification

- Experimental phase-only output map retains phi=(1+tanh(z))/2 and predicts
  c=raw[1]; physical free energy, reference, graph and soft IC/BC are unchanged.
  The runner records the map in arch.use_hard_constraint (research semantics),
  adds explicit positive finite scale overrides and distinct run names,
  logs actual adaptive weights/losses, and monitors concentration ranges.
- PDE field/time derivatives use the selected net.constraint, avoiding a
  hidden KKS remapping when direct concentration is requested. Focused tests
  verify both physical residuals, phase chain identity, unchanged CH,
  finite gradients and direct-c Jacobian=1 even at raw concentration-20.
- All six initial parameter files are byte-identical without hashing. Config
  differences are restricted to residual choice, run name, output-map flag
  and AC scale. Different physical initialization across maps is explicit:
  initial c error at t0 is0.4766 with KKS and0.9601 with direct c. Raw weights,
  time keys, graph pools, SOAP, learning rate, budgets and losses otherwise match.
- Independent CPU checkpoint reload of all six controls reproduces saved GPU
  fields within2.51e-6; terminal physical-field L2 recomputes to1e-12.
  Reference metadata/time/geometry/BC/graph checks pass unchanged; its prior
  temporal refinement error is3.64e-9. No additional FEM solve is needed
  because physical equations, spatial graph and prescribed data are unchanged.
- Final git diff --check passes; four changed Python modules/scripts parse,
  all249 registered external directories exist, and report links/direct
  reference/checkpoint paths resolve. No unarchived active task remains.
  Terminal-field figure layout was visually inspected. No Git history action
  was taken; old artifacts were neither overwritten nor moved.

## Artifacts and diff-level changes

- [Replay script](corrosion-kks-scale-ablation.py),
  [machine-readable report](corrosion-kks-scale-ablation.json),
  [L2 curves](corrosion-kks-scale-ablation.png),
  [terminal fields](corrosion-kks-scale-ablation-fields.png).
- Direct references/checkpoints/results are indexed in artifacts/index.yaml
  under constraint_scale_ablation_20261024. New run names are given above.
- Changed research residual/runner, one focused test file, corrosion README,
  artifact index, decisions/progress/task memory, and replay artifacts.
  Production presets, physical PDEs and existing data/checkpoints are retained.
