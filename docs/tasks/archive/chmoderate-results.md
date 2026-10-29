# Intermediate-difficulty confined CH control — 2026-10-28

User requests a less difficult global-time problem. Change only the prescribed
IC wave list (ten to three low waves) and terminal time1.5 to .75; rebuild time
nodes/snapshots and use a separate newly solved reference. Same ellipse/two
holes, mesh.02, epsilon.02, M.02, beta1, delta.03, IC amplitude.98/scale.55/wall.03.
Config: examples/confined_ch2d/configs/train_moderate.py. One global phase-only
M5/rank24 integral R/t network, exact mu elimination, hard IC/mass, no tanh or
windows, SOAP5000/LR.001+LBFGS500, ordinary FP64, runtime seed42. No target labels
in loss. Run /root/autodl-tmp/tf-logs/confined_ch2d/20261028-chmoderate-m5-01.

Independent reference /root/autodl-tmp/data/confined_ch2d/moderate_multimode3/
reference.npz passes unchanged gates. Max-time FEM mesh/time phase .162673%/
.015437%, mu .185718%/.037107%; mass drift <3.6e-15, sampled energy monotone.
Native/local phase gap2.09428%, mu14.7586%, separate from PINN fitting error.
FEM exited before GPU training. Initial native evolution RMS31.09 vs rich99.66.
Reference native pure area18.75% to42.92%, final/initial relative change60.80%:
this is not a nearly static weak-perturbation case.

| Physical problem / same training strategy | Phase pooled / FEM | Phase final / FEM |
|---|---:|---:|
| Historical ten-wave T1.5 M5 R/t |17.9854%|29.3681%|
| Three-wave T.75 M5 R/t |5.1557%|9.7665%|

Different IVPs, horizons, snapshot counts and denominators: this is exploratory
bundled-difficulty evidence, not a same-problem method improvement, an isolated
IC-versus-time explanation or a stability study. One seed only; old run retained.
Frozen-IC baseline on the NEW case is40.8405% pooled/47.6458% terminal, so the
trained field does learn nontrivial evolution rather than benefiting only from
a short nearly static target.

New native phase pooled4.2891%/terminal8.9375%; native mu45.0646%/156.4975%,
common-FEM mu46.2813%/158.3599%. Final equals best checkpoint at5500. Trainer
188.15s including refinement, excluding reference and audits. Late boundary
patches/grid oscillations persist. Terminal native mu reference RMS.126568 and
error RMS.198076: the large relative error is not solely a vanishing denominator.
Dense151-time phase range[-1.115806,1.173628] versus native reference sampled
[-.971069,1.018692]; dense sampled energy has max increase.00033249 near the end.
Therefore complexity matters but lowering it does not resolve chemical/high-
frequency/energy defects; do not call the global solver solved or promote SOTA.

Independent CPU final/best physical-field replays pass (phi max4.66e-15,
mu3.03e-14); independent SciPy chemical identity max3.28e-15, dense mass drift
8.05e-16. Both5-column field plots and convergence/energy were visually inspected.
The shared phase colour scale clips at +/-1; actual overshoots are reported above.
No production mathematical code changed: new config, audit/plots/raw records,
README/artifact index and memory updates only. Config schema/horizon/seed policy,
strict reference loader, completed full solver/trainer/replays, index paths and
whitespace/diff checks are the proportionate validation; no unchanged suite rerun.
No default, old artifact path or Git history change. All workers finished exit0.

The global accuracy task remains OPEN. If resumed on this moderate benchmark,
first isolate spatial band5 versus3 with the SAME physical case/reference/budget,
report mu/dense energy as well as phi; no reference regeneration is needed.
For causal IC-versus-duration attribution, a ten-wave T.75 retraining is a
separate required matched control, not supplied by this single bundled trial.
