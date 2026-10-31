# Longer cold training for the512-domain confined CH operator

Date:2026-10-29. Status: COMPLETE. Blocker: none. No Git history authorization.

Goal: user observes falling validation curves at5000steps and requests a longer
training run from cold initialization. Finish20000Adamsteps on the SAME512input
operator bank and SAME8held-out development geometries, without loading weights.

Context: richer512-01final nativephi4.690/5.454% pooled/terminal, local2FEM
8.976/10.431%, geometry response35.514%, amplitude88.478%, correlation.9362.
The former5000run takes524.416s,43.217mswarm. Operators are cached (~6.11GiB);
no new trajectory/mesh/operator generation is needed. Native/local bias7-9%
remains separate from network fitting. Previous record:
[512 expansion](archive/confined-ch-geometry-richer.md).

Acceptance/plan:
- Add explicit opt-in train_operator_richer512_long.py, changing only iterations
  from5000to20000. Same initialization/runseed42, architecture, physical case,
  sampling and lr(.001*.95**(step/1000), floor1e-5). Do NOT stretch the schedule.
  Fresh Adam state; fixed-budget final checkpoint, no held-out model selection.
- Canonical train.py, independent output directory. Native evalevery500,
  checkpointsevery1000. Record actual80000condition draws. Verify same initial
  parameter arrays and first20000draws/5000samplingsteps against the short run.
- Monitor validation/training-monitorphi/mu convergence. Complete budget even if
  curves fluctuate; report final and tail behavior, not a best-score claim.
- Independent raw-field/native/local evaluator, same4627common-query geometry
  response, CPUfieldreplay and reference/payload/config/source identity checks.
  Plot common short/long convergence and representative held-out fields.
- Update README/artifactindex and memory, proportionate checks/gitdiffcheck,
  archive at task-end. Keep generated arrays/models/media external.

Status/next: create long preset, verify exact configuration difference and
launch the20000step cold run. Keep single-IVP/window and5000controls unchanged.

Implementation: `configs/train_operator_richer512_long.py` changes ONLY iteration
count; exact ConfigDict comparison and validate(c) pass. No production runtime,
PDE/operator/network/optimizer change, so no new numerical unit tests needed.
Canonical20000step run launched as
`/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-richer512-long-01`.
Raw log `/tmp/confined-geometry512-long-training.txt`, process/session12633.
New audit/comparison scripts accept explicit run paths and record cold/prefix
parameter equality, actual condition draws,24fieldCPUreplay, raw/local metrics,
common-query geometry response and final5000step trends. README documents entry.
Exact resume: finish20000steps, run native evaluator then CPUaudit and geometry
response concurrently, comparisonplots, final docs/index/memory/task-end.

At5000steps: full validationphi4.6898620673%,terminal5.4535226213%, matching the
short run within roundoff. Initial network arrays are exactly equal; independent
CPU startup confirmation `/tmp/confined-geometry512-long-initial.txt` exits0.
Common5000metrics differ only~1e-16. Now entering added-budget interval; CPU
checkpoint5000parameter-prefix check is running. No checkpoint/validation-driven
change to the predeclared20000budget or LR schedule.

Checkpoint5000parameter-prefix verification passes: {"checkpoint5000_parameter_difference": {"maximum_absolute": 4.3728909382423353e-14, "relative_l2": 7.283621909367666e-15}, "common_evaluation_maximum_difference": 4.3021142204224816e-16}. Same initial model and numerically identical shared training prefix. Continue20000budget.

Milestone10000: {"step": 10000, "train_phi_pooled": 0.030564500475695297, "test_phi_pooled": 0.03419853932060313, "test_phi_terminal": 0.04023980195623128, "test_mu_pooled": 0.09238455529916553}. Continue predeclared20000steps; no model selection or reference changes.

Milestone15000: {"step": 15000, "train_phi_pooled": 0.022988928775441182, "test_phi_pooled": 0.02672847616044431, "test_phi_terminal": 0.0316209006682131, "test_mu_pooled": 0.07316396276213681}. Continue20000budget; final metrics and tail trend remain pending.

Training COMPLETE20000steps: {"train_phi_pooled": 0.02245030301984203, "test_phi_pooled": 0.024626939186027704, "test_phi_terminal": 0.02837029351830096, "test_mu_pooled": 0.06971053826672483};wall_seconds=1401.5394972786307. Native saved-field evaluator running; next CPUaudit+GPUgeometryresponse, comparison and task-end.

Final task-end — COMPLETE. No blocker, pending jobs, Git action or queued tuning.

Delivered changes:
- New explicit configs/train_operator_richer512_long.py changes only iteration
  budget5000->20000. Canonical train.py cold initializes model/Adam, preserves
 512/8data, hardIC/mass, eliminatedchemical/native residual metric and LR rule.
  Runtime/PDE/operator/network/optimizer sources remain unchanged. README gives
  command, numerical results, tail behavior and direct model/artifact paths.
- New direct-path audit/comparison scripts plus compact receipts. Raw curves and
  physical/reference/error panels stay external; no arrays/media/checkpoints
  added to Git. Artifactindex links long run without replacing old controls.

Validation/results:
- Full20000Adamupdates complete1401.539s,118.654setup/93.890firstupdate,
  41.889mswarm. Nativephi4.690/5.454%->2.463/2.837% pooled/terminal; nativeμ
  11.661->6.971%; training-monitorphi4.314->2.245%,worsttest6.725->3.874%.
- Same2FEMphi8.976/10.431%->8.111/9.473%; unchanged native/local bias7-9%.
  All24raw-fieldchemicalidentity/hardIC/mass/sampleenergychecks pass; no new
  reference solve. CPUcheckpointreplaysmax1.443e-15 against savedGPUfields.
- Initial parameter arrays exactly equal.5000checkpoint parameter maxabs
  4.373e-14/relative7.284e-15; commonmetricsmax4.302e-16. Actual first5000
  condition draws EXACTLY match.80000draws cover512domains123..201times;
  no held-out/source-reserved overlap, no future fields in optimizer payloads.
- Same4627common queries/reference variation: geometry response35.514->19.284%,
  amplitude88.478->93.316%,correlation.9362->.9825. Wholecommonphi2.591%,mean1.122%.
- Last5000validationphi2.333..2.791%, fitted slope-.0585percentagepoints/1000;
  terminal2.733..3.315%,slope-.0732pp/1000. Final20000checkpoint retained;
  tailminimum is diagnostic, not model selection. More optimized, no complete
  convergence/statisticalseed/arbitrarygeometry/continuumaccuracy claim.
- Exact ConfigDictdiff/validation check, actual full training, raw/FEM/postprocess,
  CPUaudit, source-snapshot equality and strictJSON/artifactpaths/diffchecks pass.
  No new numerical unit tests: only configuration and analysis scripts changed.
  Representative512field and full comparison curve visually inspected.

Artifacts (all under /root/autodl-tmp/tf-logs/confined_ch2d/
20261029-confined-geometry-richer512-long-01): model_initial/model_final.eqx,
ckpt_001000..020000.eqx,config.json,training_draws.npz,source/,metrics/loss.jsonl,
evaluation/{metrics.json,fields_512/516/519.png,convergence.png},
geometry_effect/{metrics.json,geometry_spread.png},comparison/{metrics.json,
convergence.png/pdf,physics_loss.png}. Dataset stays at its originalricher512path.
Raw logs /tmp/confined-geometry512-long-{training,initial,prefix,evaluation,
response,audit,comparison}.txt. Archive accompanying source scripts/JSON receipts.

Memory changed: docs/PROGRESS.md,docs/DECISIONS.md and this archived task.
Exact resume: no work required. Further budget/LR/seed/mesh controls are optional
new tasks. Existing5000and windowed single-IVP defaults/results remain intact.
