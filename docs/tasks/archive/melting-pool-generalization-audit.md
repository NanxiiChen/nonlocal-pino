# Melting fixed versus rolling diagnosis

## Goal and acceptance

Answer whether the measured pool difference is caused by FP64, the size of
its test cohort or random optimization/evaluation variation. Check actual
saved configs and checkpoint dtypes, input/time/cold-state parity, every test
sample and late training-time histories. Distinguish spatial sampling coverage
from generalization over initial-condition parameters and convergence at a
fixed optimizer budget. Ground each claim in saved records.

## Status

Complete. All four actual final model states contain only FP32floating leaves.
Cold models/Adam and all actual initial-condition/time arrays match within
both pairs. The native reference uses FP64in both arms. Precision mismatch
is excluded as the cause of the measured comparison.

## Measured evidence

The test cohort is8distinct predeclared ellipsoids,ids2..9. Two other inputs
are monitoring cases,notpart of that final test. Each test evaluates5times
and24^3diagnostic spatial nodes,from an independently time-refined128^3native
nonlocal reference. Spatial nodes and times do not increase the number of
independent initial-condition realizations.

Final test pooled/terminal phase errors,all expressed in percent:

| Run seed | Fixed | Rolling |
|---|---:|---:|
|42|0.1631 /0.2218|0.4217 /0.2514|
|43|0.1290 /0.2062|0.5449 /0.5199|

All8pooled phase comparisons favor fixed in each seed,butterminalphase
favors rolling on5of8cases in seed42. The same8inputs are reused,so there
are8testconditions,not16independent cases. The two training seeds use the
same32fixed spatial sets and5000rolling draw sequence. They vary model and
initial-condition/time draws,not the spatial pool realization.

Late monitoring exposes endpoint instability. For seed43rolling,phase pooled
L2on the2monitorinputs is0.11153%at4750and0.47390%at5000. Fixed is0.11923%
and0.12638%at those times. Over saved evaluations4000..5000,the monitor medians
are0.20115%fixed/0.14976%rolling atseed43,and0.09362%fixed/0.10894%rolling
atseed42. Seed42bestmonitorvalues are0.07921%fixed and0.07865%rolling at
different steps. Those scalar monitors are not unsaved eight-case checkpoint
evaluations and cannot replace the final test metrics.

Both modes train on80000fresh initial-condition draws and40000time values.
The fixed pool fixes spatial quadrature only. It covers8174distinct PDE cores
versus958123rolling,at the same256cores/update and complete1792node graph
padding. Fresh spatial samples improve expected quadrature coverage but do
not guarantee a better finite-update neural solution. Increased gradient
variation and fixed-set empirical bias are candidate mechanisms,notestablished
causes. Strong late error swings directly support sensitivity to the chosen
training endpoint. An eight-condition test cannot establish a population ranking.

## Interpretation and completion

Qualify the earlier fixed-pool conclusion as the measured final5000update
checkpoint ranking. Current data do not establish that fixed sampling has a
better converged generalization error. A larger independent IC test bank and
paired stable-tail/multi-initialization study would answer different remaining
questions. None of these follow-up training/reference jobs were started here.
The saved fixed-chunk inference scope and known chunk-size issue remain in
force. This audit changes no training code or selected default.

External direct artifacts are under
/root/autodl-tmp/tf-logs/melting-pool-diagnosis-20261111-01:
precision-and-pairing.json,late-and-per-case.json,late-monitoring.png.
Actual four-checkpoint deserialization,floatdtypes,coldbyteidentity and sampled
IC/time equality pass. Statistics reproduce the saved scalar histories and
per-input test values. The diagnostic plot is visually checked.

## Blocker and next action

None for this diagnosis. Optional broader generalization/stable-tail experiments
are separate future work; there is no running child or solver.

## Task-end

Updated the Melting spatial-comparison qualification and project memory.
No new training or reference solve. git diff --check passes. Memory files are
PROGRESS,DECISIONS,and this archived record.
