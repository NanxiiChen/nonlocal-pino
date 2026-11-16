# Electropolish15k library and time-fusion controls

## Goal and authorization

Expanded2026-11-16request: finish the existing32/online pair, cold-start fixed
64/128/256/512at15k, then joint(x,t)query15k. Update Electropolish-only library
mean±SD figure and the temporal cold-wall-time convergence/violin figures,
example/overview README and website; user goal explicitly authorizes commit/push.
Keep Ceramic historical data/TODO but remove the combined library plot from
public display. Reuse the current completed15kFiLM online run as temporal baseline.
No other examples, numerical defaults,20kstudies or new seeds are authorized.


User defers Ceramic stability work to TODO and requests two further Electropolish
runs:32fixed initial profiles and a new continuous-online run, evaluated on the
same64test conditions. Preserve prior fixed64/128/256/512results and defaults.

## Acceptance and controls

- Both cold15ksteps, retaining a10kcheckpoint, with seed42 and identical network/optimizer initialization.
- Fixed32takes the first32entries of the original nested512profile library;
  online generates new profiles at every update from the unchanged distribution.
- Same frozen numerical source as the prior fixed-library sweep; same physical
  parameters, width96/B8,16random times, Adam, rolling256core/halo sampling,
  GradNorm and host_float64_shared prescribed-input computation.
- Same64local-FD test references,21regular evaluation times, monitoring every250
  updates including zero,10kand15k. Final-weight equal-point full-space-time norms
  perphi/cchannel, then arithmetic channel/sample mean; one sampleSD, ddof1.
- Independent final inference/norm checks, shared sampling/initialization checks,
  full cold process timing and explicit hardware provenance. No automatic retry
  or preferential seed/weight selection to make online appear best.

## Context and direct paths

Previous bundle: `/root/autodl-tmp/tf-logs/fixed-condition-library-20261115-01`.
Nested coefficients: `/root/autodl-tmp/data/fixed-condition-library-20261115-01/electropolish`.
Reference inputs: `/root/autodl-tmp/data/electropolish2d/operator_test64_20261028`.
Local FD references: `/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029`.
Prior matched online run: `/root/autodl-tmp/tf-logs/electropolish2d/electro-threeway-20261114-03-nonlocal`.

Current GPU: NVIDIA RTX6000D,85651MiB,UUIDf0c22930-0180-a523-898a-9fef5fec4eb9.
It differs from the previous RTXPRO6000UUID85fe6412-636e-5d3c-d5f0-d1e9523c3faa.
Both new arms use the same current device. Historical times are not a controlled
cross-arm comparison; retain all hardware details. User explicitly prioritizes
final errors, accepts hardware change and authorizes concurrent execution if
memory permits. Prior identical-shape peak12.02GiB per arm is below the current
40%device reservation (~33.5GiB) each; two processes reserve80%combined.
Training is parallel; full-process times are recorded as concurrent, not isolated.
The later expanded authorization above supersedes this original two-arm scope;
Ceramic remains deferred.

## Status and next action

All seven declared cold15ktrainings, final/10kreloads and independent64case
physical-field/source/sampling/initialization audits completed2026-11-16 16:31:52
Asia/Shanghai. Expanded numerical journal iscomplete; figure queue finished
ready_for_visual_reviewat16:32. All threeactualPNGs were inspected: unclipped
readable labels, correct NPGmeans/bands/violins, complete cold-time axes; PDF
font embedding,5–7pteditable text and temporary-copy edit probes pass.

Final fixed32/64/128/256/512channel means16.44550/12.27110/5.45571/2.12162/2.02321%,
SD3.63969/3.12823/2.21589/0.98824/0.71652percentage points. OnlineFiLMmean1.53351%,
SD0.58088; jointmean1.69628%,SD0.60413. FiLM/jointfullcold1283.698062115/
2146.985985462s; medianwarm63.40995/113.63464ms; activepeak12.02522/31.59048GiB.
No training remains; do not restart any owner. Finalweights only, no retuning.

Library and temporal figures are selected locally inassets/pino-library-sizeand
assets/pino-studies, with explicit originalsourceprovenance and new15kdata.
README/root/site chapters updated; Ceramic library graph removed while its
historical statistics/TODOretained. Productiondefaults and main galleries unchanged.
Current9pagebuildpasses. The first attemptedbuildwas invoked from the repository
root and correctly failed missingpackage.json; the website-directorybuildpasses.
Finalsource/asset/link/stagedtree validation remains, then user-authorized
Gitdev→commit→mainregularmerge→pushmainandverifyPages. Keep unrelatedsoftIC/
core-halodirtywork outsidecommit. Archive this task only after deploymentchecks.

Exact selected source paths:
- bundle/library-figures/provenance.json
- bundle/temporal-figures/provenance.json
- bundle/time_fusion_comparison15k.json
- bundle/time_fusion_clock15k.json
- artifacts/electropolish_library15k_results.json
- artifacts/pino_time_fusion_figures.json(Electrocaseonlyreplaced)
- artifacts/pino_time_fusion_wall_clock.json(Electrocaseonlyreplaced)

The temporarydocupdateralreadyran and its textwas subsequently reviewed/polished;
do not rerun it and overwrite final edits. FinalPDF+PNGbudget375662bytesfor6files.
Next: finalvalidator→scopedstage/commit/merge/push→Pageslivebytechecks→memoryarchive.

## Preparation and validation

`docs/tasks/electro_library_extension.py` freezes the prior numerical source,
adds only two experiment configs and controls preparation/launch/final audits.
Numerical model, residual, optimizer and defaults unchanged. Preflight uses the
same with_run_seed entry transformation as actual training: the initial check
correctly rejected comparing unresolved configs with a serialized resolved seed;
the harness was corrected before training, with no seed change or relaxed gate.
Both resolved configs match the prior64arm except run name and fixed32count or
removal of the fixed-library block. Original32prefix byte identity passes.

Online audit was exercised against the completed historical online control with
output capture only (no writes to old run). All64recomputed channel norms match;
mean0.02163787033, sampleSD0.00779638443;80000distinct profile draws and shared
initialization/spatial/time checks pass. Both final audits retain original strict
metric tolerances. Historical coefficient identity is reported separately because
hardware differs. No extra reference solve or training probe.

## Launch verification and exact resume — 2026-11-16 12:49 Asia/Shanghai

OwnerPID7034; concurrent trainers7043(fixed32) and7044(online), both started
12:47:48Asia/Shanghai. Both run directories and actual initial model/optimizer
exist. Byte equality passes between the new arms AND the prior fixed64initial
states; receipt:bundle/initial_pair.json. Each CUDA process reserves34580MiB;
combined69174MiB of85651MiB, with no OOM or reported numerical error.
These are reserved allocations, not measured active peaks.

Both all64initial evaluations at step0 completed and match exactly:
phi0.64474486861,c0.64762158046. First optimizer-step compilation is in progress;
no trained-error claim yet. Next routine observation no earlier than13:04
Asia/Shanghai, unless user asks or a failure is reported. Read existing journal
and each run/local_metrics.jsonl; do not launch duplicate jobs.

/task-end: changedTODO,PROGRESS,DECISIONS,this active task and the new finite
experiment owner/audit helper. Frozen numerical files and public defaults are
unchanged. Actual configuration/prefix validation, historical online audit
rehearsal, cold initialization identity and initial64case evaluation all pass;
git diff whitespace check passes. Training and final automatic inference/audits
remain in progress. No Git operation or website deployment.

## Publication checks — 2026-11-16

The completed eight-run library study and five accepted ranked main galleries
are prepared for the authorized commit/push. Includes the optional fixed-library
samplers, reproducing experiment owners/audits, figures, example/overview READMEs,
website chapters/build manifest, and Ceramic stability TODO. Unrelated stopped
Melting soft-IC and old core/halo experiments remain outside this commit.

Previously recorded15focused library/online tests passed in9.613s; numerical
source is unchanged since that verification. Local nine-page build passes.
Staged build,405local links/anchors,12asset identities and18liveHTML/figure
comparisons passed. Pages43808630409completed build/deploy successfully.
The active task stays open for15kcompletion and final10k/15kaudits.

## Publication complete and exact resume — 2026-11-16

Content commit dev1d3e68a → mainb752981 (regular merge) → origin/main.
The completed study, selected main galleries, optional library controls,
reproduction helpers, READMEs and website are published. Receipt:
`artifacts/fixed_library_pages_publication.json`. No raw arrays/weights/logs added.
Unrelated stopped numerical changes remain untouched; the pending Melting README
section was preserved byte-for-byte while switching branches, outside the commit.

The two15ktrainers remain owned by30224in the15kbundle named above. Latest
routine inspection reached2500updates in each arm:64case channel means
0.1507490992fixed32 and0.07261499565online. These are intermediate observations,
not final model selection or a15kaccuracy claim. Both initial states match the
historical baseline. Read the existing journal before acting; never restart an
owner that is running or already complete. The next work is the scheduled
10k/15kcheckpoint inference and independent audits, then compare the two budgets
before considering any older-library/FiLMrerun or20kextension.

/task-end: publication is complete; leave this numerical task active. Updated
PROGRESS,DECISIONS,TODO,this task and archived fixed-study record. Focused tests,
config/prefix/initial-state audits, staged website build, link/assets and live
Pages checks pass. No numerical defaults changed. Continue coarse15minute
observations when monitoring is requested; no duplicate training or extra runs.

## Status check — 2026-11-16 13:15 Asia/Shanghai

Both trainers and owner are alive, with the expected15kconfigs/process identity.
Latest common inspected evaluation:7000/15000,29records including zero and
64samples per record. No completion weights/audit exist yet. Recent logs have
no traceback/OOM. GPU allocated memory69202/85651MiB; no restart or setting change.

| Input sampling | Step5000mean | Step7000phase | Step7000concentration | Step7000mean |
|---|---:|---:|---:|---:|
| Fixed32 |0.1506783479|0.1362423128|0.1512039052|0.1437231090|
| Continuous online |0.04098861852|0.02863613786|0.03421808125|0.03142710956|

Metrics remain equal-point full-space-time norms against the same64local-FD
references, then channel/sample mean. Online improves substantially; fixed32
improves slowly. No final budget or online-optimality claim. Continue the existing
owner to15kand the queued10k/15kinference/audits. Do not rerun older studies.

/task-end: updatedPROGRESSandthis task only. Read-only process identity, monitor
completeness and recent failure checks pass; no code/default/Git/site change.
Exact resume: read existing bundle journal before monitoring or collecting final
results; retain coarse15minute cadence unless the user asks for status.

## CUDA execution failure and manual recovery — 2026-11-16 13:37 Asia/Shanghai

Failed online process30231exited1; fixed3230230was terminated by the owner, not
by an independent numerical failure. Last online loss log is9900; last valid
common64case norm evaluation9750:

| Input sampling | Phase | Concentration | Channel mean |
|---|---:|---:|---:|
| Fixed32 |0.1360274017|0.1510266080|0.1435270049|
| Online |0.01786847433|0.02129255101|0.01958051267|

No10kmodel/optimizer state exists, hence a cold recovery is necessary. All old
logs/initial weights are preserved. The explicit owner execution mode changes
only launch order and XLA command capture; all173frozen numerical Python files
match the failed source byte-for-byte. Physical config, original32prefix and
64casemonitor preflight pass. Two focused controller tests verify strict serial
execution, evaluation only after both trainings, failure persistence and no
unannounced retry/second trainer after a failure. Production defaults unchanged.

OpenXLA documents disabling capture with an empty operation list:
https://openxla.org/xla/command_buffer_tutorial . This is a recovery candidate,
not proof of the precise original driver/runtime cause. The new run records
mode/flags per child and retains strict final audits. Source runs keep their
original frozen launcher, so old record reproduction remains available.

Reproduction from repository root:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m docs.tasks.electro_library_extension \
  --action prepare \
  --previous /root/autodl-tmp/tf-logs/fixed-condition-library-20261115-01 \
  --bundle /path/to/new/recovery-bundle --prefix new-unique-prefix \
  --iterations 15000 --checkpoint-step 10000 --execution runtime_serial
```

The prepared source must pass its existing preflight before starting one owner.
Current active bundle already passed; do not execute this preparation command
again for an existing directory. Receipt:active bundle/recovery.json.

/task-end: updatedPROGRESS,DECISIONS,this task, the finite experiment owner and
its focused tests. Preflight, source identity, controller tests anddiff checks
pass. Recovery is in background; completion and10k/15kaudits remain pending.
No Git action, deployment, numerical/default changes or extra studies.

Recovery launch verified:owner57842andonline trainer57849are alive. The actual
child environment disables command capture; initial model/optimizer files match
the failed online attempt byte-for-byte. All64step-zero errors match exactly.
The first update is compiling, with no traceback; fixed32is queued after online.
Receipt:recovery bundle/launch_verification.json. Next routine check no earlier
than13:54Asia/Shanghai unless user requests status. No further launch needed.

## Online recovery status — 2026-11-16 13:48 Asia/Shanghai

The existing serial owner57842and online trainer57849are running normally.
Latest inspected evaluation6250/15000:26records including zero, each with64
local-FD test cases. Phase0.02846355776, concentration0.03388734549,
channel/sample mean0.03117545163. At5000the mean was0.04422017925.
This is the new recovery run; do not conflate it with the failed run's9750
mean0.01958051267. Fixed32has not started and remains queued after online.
No final/10kweights yet, no traceback/OOM in inspected online log. Preserve the
current execution settings; no new launch, code/default change or Git operation.

/task-end: updatePROGRESSandthis active task only. Current-run identity,61record
target schedule and64case monitor scope are checked. Next action remains wait
for the existing serial owner and collect10k/15kresults; routine15minute checks
unless user requests otherwise. Final inference/audits have not begun.

## Fixed32 queue confirmation — 2026-11-16 13:50 Asia/Shanghai

The recovery journal contains only active train-online57849; fixed32run directory
is absent. It is queued, not complete or independently failed. Online's latest
64case monitor7750has phase0.02434921125, concentration0.02911141836,
mean0.02673031480. Existing serial owner57842will start fixed32after online.
The original fixed32mean14.3527%was an interrupted9750observation only.
/task-end: updatedPROGRESSandthis task; no code/default/Git or scheduling change.
Exact resume is the current bundle journal; do not start a second fixed32process.

## Online15kfinished; fixed32started — 2026-11-16 14:01 Asia/Shanghai

Recovery online trainer57849exited0after1283.698062115seconds from cold process
launch through exit (including setup/compilation/evaluation/output/teardown).
Its final/10kmodel files and final optimizer state are available. The owner
started fixed32PID70946at13:58:55.789Asia/Shanghai. No CUDA capture failure recurred
in this online15krun. The specific original runtime trigger remains unproven.

| Online step | Phase mean | Concentration mean | Channel/sample mean |
|---|---:|---:|---:|
|10000|0.0176509782613|0.0210776005475|0.0193642894044|
|12500|0.0151127148944|0.0181036271715|0.0166081710329|
|15000|0.0139597082773|0.0167105758218|0.0153351420495|

All61monitor records cover the same ordered64local-FD cases; every finite
per-case two-channel reduction matches the stored channel mean. Final
sampleSD0.00580879023713. The channel mean decreases20.8071015%from10kto15k
within this run. This does not establish superiority to older fixed-library
10kcontrols at equal budget or universal superiority of online sampling.
Initial network/optimizer files of the two recovery arms match byte-for-byte.
Fixed32has completed initial64case evaluation but not yet a post-update record;
its process is alive without traceback. Preserve the existing queue.

/task-end: updatedPROGRESSandthis task only. Read-only completion/monitor/pairing
checks pass; final reloaded10k/15kinference and independent field audits are still
scheduled after fixed32training. No new experiment, configuration/default change,
Git operation or website update. Exact resume: current bundle journal and fixed32
local_metrics.jsonl; never restart the completed online trainer.

## Expanded suite plan — 2026-11-16

New bundle:`/root/autodl-tmp/tf-logs/electro-15k-completion-20261116-01`.
Dependency:recovery bundle02must complete its two trainings and final audits.
Serial order:fixed64,fixed128,fixed256,fixed512,joint_query. Every cold run uses
15k/seed42/checkpoint10k/64test cases every250updates, unchanged frozen numerical
source and runtime_serial flags. Exact sampling/initialization and reloaded
10k/15kfields are audited before plotting. Joint-query shares initial encoder/
attention/pointwise decoder leaves but has a different temporal module and total
parameter count; record that difference instead of claiming identical networks.

Acceptance:all7declared15ktrainers (two reused plus five new) and independent
final checks; Electropolish-only mean±sampleSD and15ktime-fusion graphs using
final weights; editable Helvetica/NPGPDF+PNG, reproduced metrics/provenance;
README/site agree, build/link checks pass; authorizeddev→commit→main→merge→push
and Pages verification. The current32trainer remains untouched.

##15kpublication preparation —2026-11-16

Added finite five-run owner/audits and explicit CPUfigure queue. The result
collector requires final64case15kaudits; it keeps library-size and temporal
completion gates separate. Library graph is Electropolish-only with fixed
32/64/128/256/512and online mean±sampleSD. Temporal graph uses complete cold
parent durations plus calibrated observed counters, original solid LOWESS/95%
trajectory-bootstrap style and final component violins. No new measurements
are invented from step-rate estimates.

Publication helper now distinguishes same-data restyling from explicit
Electropolish10k→15ksupersession; preserved cohort/reference/field/method checks
reject accidental substitutions. Current published figures are untouched until
all new results pass visual review. Figures are draft external artifacts only.

/task-end resume: read expanded journal andfigure_queue; preserve completed
baselines and failed original bundle. Continue the existing finite owners,
collect audits, inspect plots, update docs/assets, then authorizedGit/Pages.
Changed memory:PROGRESS,DECISIONS,andthis active task. No current-turnGitoperation
or deployment yet; defaults and unrelated dirty experiments remain untouched.

## Publication preparation continued —2026-11-16 14:35Asia/Shanghai

Added scripts/publish_electro_library15k.py: selects only the reviewed six-arm
Electropolish15k result (five fixed libraries plus online), verifies every stored
64×2array against its passed final audit, preserves direct source paths and
copies only the selectedPDF/PNGpair. Old two-panel assets/data remain archived;
they will be removed from current README/site display, not erased from history.
New script syntax and diff checks pass. The new selector has not been invoked:
no incomplete figure or result is published.

Previous goal turn is classified progress: numerical baseline audits completed;
new five-arm owner and CPUplot queue are live, and final rendering/publication
guards were implemented. This turn also makes progress on the actual final
publication path. Current parent84522and figure owner105355were verified live
at14:32; fixed64trainer93395is the active child. Next ordinary training check
no earlier than14:47unless a failure or user request arrives. Do not restart.

/task-end: this active task andPROGRESS/DECISIONSretain the exact resume path.
Waiting for completed experiments is not a numerical blocker. Final audit,
figure inspection, README/site updates, authorizedcommit/push andPagesverification
remain required; the goal stays active.

## First expanded arm audited —2026-11-16 14:52Asia/Shanghai

Fixed64completed15k; actual inputs/initial state/source and both10k/15kphysical
field audits pass. Final phase0.11552588107624, concentration0.12989609972196,
mean0.12271099039910, sampleSD0.03128234932514. Cold1269.017328567s.
Fixed128is active as110287since14:46:07. Verified live numerical owner84522
and plot owner105355; no restart, failure or premature figure publication.

Prepared one-time doc updater:/tmp/update_electro15k_docs.py; it refuses to run
before final15kassets/manifests are selected. Review generated language/diff;
script itself is temporary and not a research artifact. Still update build
budget guards, Pages paths, reproduction guide and final receipts explicitly.
Next routine check15:07. This goal turn made concrete publication-preparation
progress and verified the running queue; it is not a blocked condition.
/task-end exact resume: expanded journal→remaining audits→figure_queue→visual
review→select assets→README/site/build/links→authorizedGit/Pages.

## User clarification and current controls —2026-11-16 15:15Asia/Shanghai

User accepts the distinction between training-library count and64test cases,
and requests continuation. Rechecked raw historical audits: old fixed64at10k
mean12.8724%, not2%; old fixed5122.0137%and matched online2.1638%. Current
fixed64improves within its own run from12.4404%at10kto12.2711%at15k.
Fixed128current15kmean5.45571%also improves from its own10k5.70869%, but is
worse than the historical10k2.66125%; the old run used differentGPU/runtime.
Do not present cross-run differences as solely an effect of extending the budget,
or infer overfitting from test errors without training-solution errors.

All completed outcomes remain selected as planned; no retuning/reseed/retry.
Verified parent84522, current256trainer126029and CPUfigure105355live. Routine
nextcheck15:31; continue exact current source and final-weight selection.

## All library endpoints audited; joint control active —2026-11-16 15:57

Library same64cohort collection passes for all6controls. Fixed256mean
0.02121622560455,SD0.00988239625392; fixed512mean0.02023212735479,
SD0.00716522210754. Jointfirst compile underway with nonfatal internal XLA
autotuning timing warnings. Preserve real full cold duration; do not subtract
compile delays or claim internal profiler warnings invalidate parentwallclock.
No GPUOOM observed, no card change needed so far.

Changed website temporal guard from41fixedrecords to explicit10k/15kbudget
plus computed record count;15krequires the approved10k-supersession reason.
Existing9pagebuildpasses. Added newlibrarymanifest Pages trigger. Helvetica,
PyMuPDF1.26.7andpdffontsreadiness pass. Gitdev/main content is identical; actual
commit/push remains pending final figures/docs.

/task-end: previous goal turns made concrete publication-preparation progress
and verified live waiting. Currentturn completes library audit collection and
publication prerequisites. UpdatedPROGRESS,active task;DECISIONSretain current
authorization. Exact resume isjointtrainer158173andthe two bundle journals.

## Joint control progressing —2026-11-16 16:12Asia/Shanghai

Jointtrainer158173is alive at6250/15000; all64local-FD cases in each observed
record. Mean3.47101%is intermediate, not final. CurrentGPUcapacity has sustained
actual joint updates; no need to request a different card. Earlier first-step
autotuning warnings did not stop compilation. Parent84522andfigureowner105355
remain live; continue unchanged. Next routine16:27thenfinal audit/plots.
