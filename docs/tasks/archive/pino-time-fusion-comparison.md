# Time-feature fusion on two PINOs

## Goal and acceptance

User extends the active five-case derivative-comparison objective with2–3
short-running operator cases comparing temporal fusion. Choose Electropolish2D
and Melting3D. Preserve the original derivative-control objective and SOTA
defaults. This is a child of `../pino-three-method-suite.md`, not a replacement.

For each case compare spatial-only cross-attention + every-decoder-layer time
FiLM against one affine projection of concatenated spatial/time Fourier features
into cross-attention, followed by unmodulated decoder layers. No phase labels
enter training. Match physics, nonlinear output, hard constraints, learning-rate
schedule/optimizer, batch/time/core draws, horizon and complete-test evaluations.
The shared encoder/attention/spatial projection/MLP initial leaves must match;
the two time modules differ in parameter count and optimizer tree, explicitly
reported rather than claiming identical total capacity/state.

Both use nonlocal spatial operators and time JVP. Keep the shared-time spatial
coordinate organization; joint queries necessarily repeat attention for each
time while retaining one encoder call per condition. Test actual joint affine
feature equivalence, time JVP, parameter gradients and the unchanged legacy
forward path independently before GPU qualification.

Reuse `electro-threeway-20261114-03-nonlocal` and
`melting-threeway-20261114-04-nonlocal` as new same-card cold10k FiLM baselines.
Add one cold10k joint-query arm per case only. All64Electro local-FD tests and
all64Melting native tests are monitored every250updates; retain Melting's
separate accepted localCG2case005. Use final fixed-budget weights. Measure
parameter entries, synchronized warm timing, full-process timing including
all-test evaluation, and active-array peak memory. Keep native/local reference
scope explicit. Final outputs: audited per-case errors, convergence and final
distribution figures, matching README and local website records. No Git or
deployment is authorized now.

## Existing evidence

`electro-temporal-conditioning.md` records an older width64/Nt4/cold5k
Electro comparison on RTX4090. Its results support feasibility, not a current
same-device/two-case conclusion. Keep that evidence unchanged. The reusable
wrapper is `examples/electropolish2d/temporal_query.py`; current spatial-FILM
backbone is `src/nonlocal_pinn/nn/conditional.py`. No modification to live
channel/shared numerical dependencies is needed.

## Completed — 2026-11-14T22:18UTC

Status: complete for both requested temporal controls. The parent five-case
three-method objective remains active for Spinodal. No SOTA/default or Git/
deployment change. Both methods retain nonlocal spatial derivatives and timeJVP,
common physics/optimizer/sampling/backbone, cold10k budgets and full64test
monitoring at all41records. Temporal-module capacities differ explicitly.

| Case / quantity | Decoder FiLM | Joint query |
|---|---:|---:|
| Electro local-FD64 phase |1.9713%|2.0577%|
| Electro concentration |2.3563%|2.4520%|
| Electro channel mean |2.1638%|2.2549%|
| Electro cold time |15.63min|23.65min|
| Electro warm-inclusive median |64.61ms|105.76ms|
| Electro active peak |12.03GiB|32.45GiB|
| Melting native64 phase |0.2860%|0.0935%|
| Melting separate localCG2case005 |0.1928%|0.1302%|
| Melting cold time |6.89min|14.99min|
| Melting warm-inclusive median |15.81ms|60.44ms|
| Melting active peak |1.45GiB|9.36GiB|

Joint queries improve Melting's final accuracy but cost more time and memory.
Electro final errors are close, slightly favoring FiLM in this realization.
Do not infer universal accuracy superiority, cross-seed significance, equal
parameter count or guaranteed inference-speed scaling. Melting retains its
24³query/five-time native metric and separate accepted localcase005with an
unpassed strict-refinement gate. Neither localcase005nor the native64cohort
establishes a64case local-continuum comparison.

Both actual two-case audits pass initial shared-backbone bytes, all actual
sampling/source checks,256saved-field pairs/384channel norms in total and final
fixed-budget selection. Four actual plots are visually inspected; eight PDF/PNG
assets total478,985bytes in the separate2MiBcollection. Exact sources, checkpoints,
configurations, field errors, pairing/capacity/timing and reproduction commands:
`artifacts/pino_time_fusion_figures.json`. The renderer preserves all actual
records, log-LOWESS/whole-trajectory bootstrap intervals and final64log-KDEviolins;
fonts are6/7pt editable embeddedHelvetica with NPGcolors.

README chapters, root overview, asset index and local website match. Nine-page
build/375local links, raw mean/capacity/resource tables and source/repository/
built-site byte identities pass. Receipts:
`artifacts/electropolish2d_time_fusion_publication_validation.json` and
`artifacts/icemelting3d_time_fusion_publication_validation.json`.
No GUIbrowser verification or deployment is claimed. Full runtime journal:
`/root/autodl-tmp/tf-logs/pino-time-fusion-20261114-02/journal.json` is complete.

/task-end: archive this completed child and update all five incoming references;
PROGRESS,DECISIONS,parent task and suite inventory retain the active Spinodal
resume point. Previous and current turns are progress. No more temporal training
or figure work is required. The remaining parent objective is not declared done.

## Historical preparation and execution observations

## Status and next action — recovery19:38UTC

The previous temporal owner exited after the upstream Electro strict-input audit
failed; no joint-query training started. Fresh owner284434/start928607839 waits
on Melting284431/start928607828. Journal: `/root/autodl-tmp/tf-logs/pino-time-fusion-20261114-02/journal.json`.
Baselines are now Electro03 and Melting04. Electro baseline/joint query both
inherit the qualified common-input path and require exact shared witnesses.
Eight pairing/time-fusion regressions passed; all budgets and full64monitoring
remain unchanged. Use the parent inventory and15minute observation deadline;
freeze numerical sources while upstream experiments run. Recovery receipt:
`artifacts/pino_shared_inputs_queue_transition.json`. Complete actual GPU controls,
audits and figures before claiming an accuracy/speed result.

## Earlier qualification and queue history

Implementation and CPU qualification pass; GPU/reference results are pending.
Ten focused tests verify2D/3Djoint-affine equivalence, independent time-JVP finite
differences, no cross-time derivatives, differentiated residual gradients,
hardIC/zerohead and complete64monitoring/figure-loader semantics. Actual preset
initializations have identical common backbone arrays and differ only in the
declared temporal module. Stored entries: Electro320706/264738; Melting85825/71665.
Direct receipt: `artifacts/pino_time_fusion_validation.json`.

The new reference helper preserves10old files and appends56declared unique
parameters (extension seed20261114). A spawned CPU12cube smoke test passes real
RK4/refinement/saving, maximum dt/dt2error7.584e-8. This does not replace the
queued full128cube reference gates. Production bank remains unbuilt until the
preceding Electro owner finishes. No FEniCS runs alongside current JAX GPU work.

Both two-arm and unchanged three-arm rendering pass embedded-font/vector/text
checks on explicitly synthetic fixture data. The initial two-arm layout check
exposed a literal LocalAD label in the PDF edit probe; the probe now handles
both declared comparison label sets. No physical result was inferred from the
fixture, and no selected figure has changed.

Verified waiting owners (2026-11-14):
- Original channel81883/trainer145877 continue unchanged; Electro197352 waits.
- Melting212944/start927258469 waits Electro, prepares64refs, then three10k controls.
- Temporal212945/start927258479 waits Melting, then two full-size joint resource
  probes and two cold10k runs, final64field/input audits and PDF/PNG previews.
- Spinodal212946/start927258489 waits temporal completion before its own reference,
  probes and three50k controls. Shared derivative-preview owner213297/start927261323
  waits independently on each successful derivative audit.

All commands, PIDs/start identities and journals are in
`artifacts/pino_time_fusion_queue_transition.json` and the suite inventory.
Three superseded waiting owners were childless; no live trainer was stopped.
All117snapshot Python sources of the running channel remain byte-identical.

Exact resume: monitor at the requested15minute cadence; on actual stage failure
inspect its direct log and resolve the concrete gate. Do not silently reduce
cohorts/budgets, restart the live channel, run competing GPU trainers or invent
measurements. Once actual audits/figures pass, visually review, then add accepted
results to both example READMEs and local website. No Git/deployment authorization.
The five-case derivative objective remains active; temporal controls are an
additional child task, not completion or replacement of the original work.

## Session changes

Added explicit joint-query presets, the wrapper's encoded-query interface,
Melting test64extension, finite temporal queue/audit, two-arm plotting support,
focused qualification and read-only full-test documentation. Updated comparison
queue dependencies before starting any affected training. Physical equations,
production SOTA defaults, existing checkpoints and selected assets are unchanged.

Final session checks: changed-module syntax, artifact JSON and `git diff --check`
pass. Replacement owners are waiting and childless, each with an exact recorded
predecessor. The updated read-only status helper uses inventory identities and
honors the15minute deadline; it cannot restart or terminate runs. Changed memory
files: `docs/PROGRESS.md`, `docs/DECISIONS.md`, the parent suite task and this
child task. Leave both tasks active; actual comparisons and result promotion
remain the exact outstanding work.
