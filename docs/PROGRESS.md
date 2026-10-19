# Project Progress

> Short dashboard. Current work is self-contained in the one active task;
> completed experiments and superseded process observations are archived.

## Active task

[Nonlocal PINN history-field learning and architectures](tasks/nonlocal-pinn-architecture.md).
Opened 2026-10-15 after the user closed the fracture2d feasibility stage.

Shear replay result retained for commit, 2026-10-19: user explicitly wants
both choices, default replayOFF. Existing train_base remainsFalse; opt-in
train_replay selects the completed random5 experiment. Commit preset, README
and completed accuracy/time records on dev only. Validate both schemas and
saved-config equality; no numerical changes, repeated full tests, data/events
in Git, new training, merge or push. No-replay snapshot is dev3625b99.

Shear random5 replay COMPLETE, 2026-10-19: all200cold loads through0.0200,
2892outers/371072Adam,8961.17s(149.35min), versus no-replay3026outers/
10619.25s(176.99min).15.61%less time,4.43%fewerouters, not equal accuracy.
PostpeakR<1 delayed0.0143 vsno-replay0.0137/FEM0.0135; finaldL2error36.053%
vs31.812%, finalR0.0788702 vsFEM0.0551178. At0.0120replaydL2error17.087%
vsno-replay35.235%, so effects vary byload. Both neural rates stayed1e-4.
Visualcomparison shows replaypath steeper towardlower-right corner. Saved
threecurve/matched-load damage plots+receipt under shear diagnostics/
replay_random5_comparison_20261019. No old-H retention audit or baseline
replacement yet; no further training queued. Earlier running notes historical.

Shear random5 replay RUNNING, 2026-10-19: owner26889/worker26894, managed
session15729, launch diagnostics/replay_random5_launch_20261019/job.json.
Actual cold q/d/H zero, Adam0/bank0 and identical initial network/optimizer
bytes verified against no-replay. Preset matches saved baseline except replay
and output paths;200loads/bank512/fixed6tasks validated. Events created under
pinn_history_conditioned_replay_random5/events. Initial compilation, no load
endpoint yet at last check. Same physical/stopping/seed settings; no core
edits, new full tests or Git operations. Inspect actual job/log/result next.

Shear replay experiment authorized, 2026-10-19: add an isolated train_replay
preset inheriting the completed no-replay shear config. Reuse validated global
random5 completed-H policy: totaloldweight0.1/every4Adam/last32current-only,
fixed6task shape,bank512. Cold0→0.0200; same physics/rates/stops/full-domain
energy and seed. Preserve original results and matched FEM. Validate config
diff/fresh state, then background run with separate replay TensorBoard root.
No Git operation authorized in this turn.

Shear snapshot authorized for commit on dev, 2026-10-19: include material-edge
shear example, explicit configs, shared mesh BC guards, focused tests and
completed cold-result records. Verification remains4focusedPASS plus UFL and
both200-load runs; changed Python syntax and whitespace checked. Data/events/
figures remain local. No merge/push or new training requested; commit identity
is available from Git history. Active architecture task remains open.

Shear PINN COMPLETE, 2026-10-19: true cold0→0.0200,200field-stable endpoints,
3026outers/388224Adam,10619.25s(176.99min); owner11419/child12667 exited.
All logged rates1e-4. Peak359.833at0.0100 vsFEM359.228at0.0099; first
postpeakR<1 at0.0137 vs0.0135. FinalR0.0544799 vs0.0551178(error1.157%),
damage volume-relativeL2 error31.812%: similar downward-curved topology,
remaining position/width differences. No accuracy-equivalent speedup claim
(FEM6388outers/25.11min uses stricter damage stop). Same-load force/damage
plots and receipt at shear diagnostics/completed_comparison_20261019.
No further training queued; next discuss refinement, not automatic restart.
Earlier running observations below are historical.

Shear PINN verified running, 2026-10-19: existing child12667/owner11419 alive,
84 field-stable endpoints through0.0084, latest5outers,R326.580,dmax0.30520.
Actual state0000 q/d/H are zero; saved config replayOFF, matching edge BCs,
u_final0.0200 and evaluation-only FEM path. No duplicate run or solver change.
Monitor next near-reference initiation0.0101 and spanning0.0135; events under
tf-logs/fracture2d_shear/pinn_history_conditioned/events.

Shear FEM figures delivered, 2026-10-19: full reaction-displacement curve,
five damage snapshots and two-row ux/uy snapshots at0.0099/0.0101/0.0120/
0.0140/0.0200. Stored with reproducible script/CSV/receipt in reference
`figures/overview_20261019/`. Damage0–1; displacement actual per-panel extrema.
Visual check confirms downward-curving crack reaches right edge by0.0140.
No solver changes or new training; existing background PINN continues.

Shear reference COMPLETE, 2026-10-19: all200 loads through0.0200 are
field-stable, 6388outers,1506.59s(25.11min) including original prefix.
Final R0.0551178,dmax0.9999503. Launcher transitioned to fresh replay-OFF
PINN, child12667; reference data remain in data/fracture2d_shear/fem_reference.
Earlier reference-running notes below are historical observations.

Reference-path confirmation, 2026-10-19: shear FEM fields, checkpoints,
figures and result.json already reside in
`/root/autodl-tmp/data/fracture2d_shear/fem_reference/`.
Only event logs reside in `tf-logs/fracture2d_shear/fem_reference/events/`;
no relocation needed. Latest read: 140 completed loads through0.0140,
reference still running; PINN has not started.

Material-only shear IMPLEMENTED/VERIFIED/RUNNING, 2026-10-19. New
`examples/fracture2d_shear/` shares local FEM-energy/conditioned-history core,
top edge ux=U,uy=0; bottom edge ux=uy=0, no extra grips. Original geometry
and tension baselines unchanged. Four focused CPU cases PASS30.077s;
actual UFL stiffness/driver/scalar checks ~1e-15, actual P1/UFL scalar~1e-14,
GPU elastic q relative error1.37e-8 and horizontal reaction4.32422584 match.
Initial reference completed100loads to0.0100 (688outers/346.94s), dmax0.594,
R358.82. Extended to0.0200 with unchanged physical prefix; newbank512 covers
400snapshots. Initial automatically started short PINN had no outer report,
was cancelled/archived; extended PINN will start fresh only after reference.
Active launcher11419/session68686, child11423 currentlyFEM; logs/job/backup
at tf-logs/fracture2d_shear/diagnostics/extended_cold_launch_20261019.
Reference0.0101 grows a downward-kinking crack, converged558outers/R253.62;
completedthrough0.0104, continuing. No full0.0200 or neural-completion claim.
Reference data/fracture2d_shear/fem_reference; futurePINN TensorBoard
tf-logs/fracture2d_shear/pinn_history_conditioned. No Git operations this turn.

Shear update, 2026-10-19: user requests removing added grip regions for this
pure local-FEM case. Use original MATERIAL mesh only: top y=1 ux=U,uy=0;
bottom y=0 ux=uy=0. Remaining exposed faces traction-free; damage natural
zero flux. Supersedes the preceding grip-strip plan. New reference required
because stiffness/BC domain changes. Existing tension assets stay untouched.

Shear BC CONFIRMED, 2026-10-19: top grip ux=U,uy=0; bottom grip ux=uy=0.
Implement `examples/fracture2d_shear/`, shared FEM-energy/conditioned-history
core, replayOFF, separate BC cache/reference/logs. Reference first explores
0→0.0100 in1e-4 steps; bank256 covers200 snapshots. Other accepted neural
settings retained (128inner,1024bootstrap, scaledstop/progress blocks).
Independently validate UFL mechanics/history/scalar damage and horizontal
reaction before directGPU, then freshPINN. No Git operations authorized.

New case requested, 2026-10-19: same rounded-notch geometry with horizontal
top loading, separate `examples/fracture2d_shear/`, matched local FEM direct
reference and history-conditioned replay-OFF PINN. Continue the same active
architecture investigation. Required BC clarification is pending: clamped
top/bottom vertical motion versus vertical freedom with one anchor. Existing
tension cache fixes grip uy and one ux anchor, so it cannot be silently reused
as shear BC data. No shear solve or assumption-dependent implementation yet.
Prepare separate boundary/cache/log/reference identities, validate independent
UFL mechanics and damage, then select loading extent from direct behavior.
No Git operation authorized in this turn. Previous publication completed as
dev6d0c720/main3b11dd3, independently verified on origin/main.

Publication handoff, 2026-10-19: user provisionally retains the random5 replay
result and authorizes commit on dev, regular merge into main, and push ONLY
main. Code/tests/Markdown include replay selection and exact shared-coordinate
AD plus complete work/accuracy/retention records. Logs/data/slides remain
local. The accepted replay-free FEM baseline and nonlocal MLP default are
preserved. Existing9focused+1legacy checks and full60-load GPU validation
remain valid; no numerical code changed during publication. Main is checked
out in `/tmp/nonlocal-kks-publish-main-20261015`; use that clean worktree for
the merge. Delivery commits are identifiable in main/origin/main Git history;
verify their equality after push. No new training or research work queued.

Random5 replay cold run COMPLETE, 2026-10-18, including retention audit; GPU
idle. All60loads to0.0060 fieldstable, 778outers/100480Adam, 2595.99s
(43.27min), versus accepted no-replay930outers/3423.28s(57.05min):16.34%
fewer outers,24.17% less loading/reporting time. Critical0055:377vs482;
prefixthrough0054:317vs295; after0055:84vs153. Both kept1e-4 throughout.
Replay damage L2 errors0055/0060 are7.508%/6.935%, worse than baseline
1.359%/1.130%; equal stop thresholds do not establish equal accuracy. Old-H
audit shows improved retention (replay2.67–7.52% over five fixed-H checks),
but this is combined replay/shared-query implementation, not isolated replay
speedup. Full60step CSV/plots/receipt in diagnostics/
replay_random5_comparison_20261018. No baseline replacement or new training.

Global random5 replay IMPLEMENTED/VERIFIED/RUNNING, 2026-10-18. Worker46793/
session32676, log/tmp/fracture2d_fem_replay_random5.log; run
`runs/pinn_history_fem_replay_random5`, launch receipts in diagnostics/
history_fem_replay_random5_launch_20261018. Actual state0000 q=d=H=0,
Adam0/bank0 and initial network/Adam bytes equal accepted fresh baseline;
config differs only in replay settings/output paths. Current+5 fixed shape,
all completed-history snapshots eligible, no window, no current-load snapshot,
no replacement; total replay0.1 every4Adam, last32current-only. Coordinate
MLP/query normalization shared per update with summed query adjoints and one
coordinate pullback; H encoders/attention remain distinct. Final9 focused
shared/gradient/query/resume cases PASS55.137s plus legacy replay regression
PASS11.869s; earlier9basic/rate cases PASS41.331s. GPU now completed loads
0.0001/0.0002/0.0003 in5/5/6outers (baseline5/5/5), running0.0004 with six
eligible old snapshots and five sampled. First no-old-history endpoint agrees
with baseline: max d difference1.76e-14, reaction difference4.30e-10. Events
and first_endpoints_audit.json saved; no overall speed/quality claim. Preserve accepted
baseline; monitor actual job/results next. No Git operation in this turn.

Replay experiment authorized, 2026-10-18: user clarifies the alternatives are
all recent snapshots versus random samples from ALL past H; do not combine
window and randomness. Implement the latter: five distinct snapshots sampled
from completed loads, total old-task weight0.1, replay every4Adam with last32
current-only updates. Fixed current+5 shape, early shortage masked. Retain
baseline physics/rate/stopping and start an isolated cold trajectory to0.0060
after sampling/gradient/resume checks. Accepted baseline remains untouched.

Replay discussion, 2026-10-18: user wants to begin retaining past-H behavior
and asks global random versus a five-load window. Recommend random sampling
WITHIN the preceding five completed loads for the first isolated comparison,
one old H per replay event and current-H-dominated normalized energy. Keep
all collected snapshots for later global-memory assessment. Window replay
does not guarantee retention outside the window. Proposal only: no replay
flag, sampling policy, LR constraint or training job changed in this turn.

Route audit, 2026-10-18: baseline code was committed on dev as e73301d,
without push/merge. Confirmed accepted route is local P1 direct mechanics +
history-conditioned neural damage trained by local P1 lumped FE energy,
full sampling/sigmoid/Adam, replay training OFF. This accepted optional route
uses no nonlocal spatial damage operator; the nonlocal MLP remains the default
profile. History floor is the preceding load endpoint, frozen within each
damage fit; no explicit nodal irreversibility projection is added. No new
policy, configuration or training change in this explanatory review.

The user ACCEPTED the full FEM-history cold result as a baseline, 2026-10-18.
Canonical directory is now
`/root/autodl-tmp/tf-logs/fracture2d/pinn_history_fem_scaled_stop_baseline/`;
the old `runs/pinn_history_fem_cold_damage_scaled` was relocated. Config and
path-bearing JSON are updated; original metadata and migration verification
are in `diagnostics/fem_history_baseline_promotion_20261018/`. All 60 endpoint
TensorBoard errors remain readable, 240 checkpoint paths and 61 replay store
paths verified; 856 payload files retain original size/mtime, final arrays
exactly equal. User authorizes committing code/tests/Markdown on dev, without
push/merge. No new training; nonlocal MLP remains the default.

Fresh phase-free cold run COMPLETE, 2026-10-18: all 60 loads from zero to
0.0060 passed the configured joint field stop. Root
`pinn_history_fem_scaled_stop_baseline` under tf-logs/fracture2d; launcher
`diagnostics/history_fem_cold_scaled_launch_20261018/job.json` is complete.
930 total outers, 119936 Adam updates, loading/reporting wall 3423.28s
(57.05min). Same-model FE direct and PINN both span at 0.0055; PINN uses
482 outers there versus direct 388. Damage volume-weighted relative L2 error
is 1.359% at 0.0055 and 1.130% at 0.0060; plots show a straight, clean band.
Tail reactions remain imperfect: 1.76914 vs 1.53605 at 0.0055 (15.17%),
1.05458 vs 1.17955 at 0.0060 (10.59%). All rates remained 1e-4: no lowering
occurred, so this supports scaled stopping/continued propagation, not evidence
of adaptive-LR benefit. True fresh initialization verified in launch receipt;
no reference-dependent training/stopping, no manual fracture labels. Existing
8 focused checks remain valid; default nonlocal branch unchanged. No further
training queued. Baseline promotion is now accepted; further accuracy or
generalization experiments await the next user request. Task remains active.

Damage-scaled forwardvalidation COMPLETE,2026-10-18: worker30165 exited,
GPUidle; firstactualstop468(matchesofflineprediction),84extra fromraw384,
Adam132480. Passes466–468 underuserformula/u1e-4/3passes. SyncR1.83023257
vsreference1.53605241 (19.1517%tailerror), damageL2error10.35993%; straight
spanningband confirmedvisually/materialedge d>=.9componentx.4997→1.0.
Accepted768 remainsfiner(R1.54923,error6.663%). Saves3000055outersrelative
to768endpoint, notmatched-qualityspeedup. SingleJIT/hostforcechecksPASS,
restoretraceagreesmetricsto~1e-9/force2.12e-5. Newrootdiagnostics/
fem_energy_damage_scaled_stop_0055_20261018/stop0468. No newcoldqueued;
recommend fullfresh experiment asquality/speedtradeoff, ensure128checks
allowcontinuedpropagation ratherthanold256hardcut. Defaultsunchanged.

Forwardstopvalidation chosen,2026-10-18: resumeprogressdiagnostic raw0055/
outer384, net/Adam121728 andsourcebank110/0054floor. Keep1e-5/128inner,
useuserformula(dmax+.01)*1e-3,u1e-4/3passes, max128extra. SaveFIRST
actualstop andindependentlysyncforce/plotfields beforenewcolddecision.
Preparing diagnostics/fem_energy_damage_scaled_stop_0055_20261018; no
fullcoldlaunch orpublicdefaultchange yet.

Damage-scaled tolerance discussion,2026-10-18: user rejectsmanualpostfracture
phase andproposes(dmax+.01)*1e-3=1e-5+1e-3*dmax. Offline257–768jointstop
withu1e-4/3passes givesfirst468,error10.3599%,stageR1.83076. Capat1e-4
gives695,error6.6962%,stageR1.55452. Early0055outer128alreadydmax.99777
whileR467.77: maxdamage is notspanningdetector; jointu gate matters.
Bothformulas are phase-free tolerancecandidates, rawversion potentially
acceptablegivenuserpriorities; capconservativecomparison, not mandatory.
No newtraining/configchange; fullcold/actualcandidatefields stillunvalidated.
Receipt damage_scaled_tolerance_audit.json inprogressdiagnosticroot.

User accepts current768 crack/force result and questions strictfieldstop,
2026-10-18. Offlineaudit ofactual257–768records supports postfracturecandidate
damage1e-4 withunchangedu/load1e-4/3passes: firstcandidate695,error6.6962%
vsfinal6.6632%. Original1e-5neverpasses; damage5e-4first474error9.1074%.
Relaxingbothlimits1e-3wouldstop259witherror51.38%/stageR325.86, premature.
Auditjson inprogressdiagnosticroot. Candidate695hasno fullsavedfields/net,
reaction1.55452 isstageonly, no finalsynccertification. No actualthresholds
changed orjobslaunched. Record768asuser-acceptedphysicalresult whilekeeping
strictstopfailure; recommendvalidatepostfracture1e-4withoutglobalrelaxation.

Progress-block trial COMPLETE butfieldstopNOTmet,2026-10-18. Former21637
exited/GPUidle. All4blocks/512extra done,total0055outer768,Adam170880.
SyncR at384/512/640/768:211.46693/1.72118/1.56826/1.54923; finalsame-load
reference1.53605 (forceerror0.85795%),damageL2error6.66319%. FirststageR<3
at463, bandvisuallystraight/spanning. Finaldchange7.68852e-4, maxconsecutive
passes2<3, no advancement. Allblockdrops>1% (last1.213%) so1e-5NEVERlowered;
heuristic delayedfinishing afterunload. OneJITtrace/hostforcechecksPASS.
Trialcost1694.98s plus112.66swarmup (~30.13min); source256workadditional.
Exactfinalstate diagnostic/outer0768 raw+sync/net/Adam. Nextcandidate3e-6
finishing fromthisstate; no newjob or globalpolicyadoption yet.

Progress-block trial VERIFIED/RUNNING,2026-10-18. GPU21637/session77941,
log/tmp/fracture2d_fem_progress_blocks_0055.log; diagnostics/
fem_energy_progress_blocks_0055_20261018. Sixsynthetic policychecks andtwo
warmAdam rateequivalence checksPASS (zeroupdatediff, identicalmoments/counts).
Restored0055seedAdam105344/bank110/rate1e-5 and0054floor/hostforce verified;
commonwarmup112.657s. Actualextra8/total264 nowRstage320.828 vsseedR328.097,
fieldL2error50.7810% vs51.7870%,Adam106368. No fieldstop orblockboundaryyet.
Leavejobrunning, next128boundarytotal384; same-load diagnostic only, no
publicsolver/config/Git changes. No new fullcold/completion claim.

Progress-block trial authorized,2026-10-18: restore incompletecold0055 endpoint
Adam105344/bank110 and0054floor. Stay1e-5, inspect every128outers; own syncR
drop>1% keepsrate, plateau lowers3e-6; finalrateplateau or4blocks512extra
endsdiagnostic. Originalfieldstop everyouter, reference evaluationonly. Empirical
localrule, not adoptedglobalpolicy. Preparing separate diagnostics/
fem_energy_progress_blocks_0055_20261018; no loadadvance/sourceoverwrite.

Latestcold audit,2026-10-18: budget128run STOPPED automatically0055outer256,
not converged. Jobfailed18:56:30Shanghai; former16172 gone/GPUidle. Rates
dropped0049after128(1e-4→3e-5),0055after128(3e-5→1e-5), thenfinal128block
exhausted. SyncR328.097 vsreference1.53605, damageL2error51.78697%,dchange
4.90964e-4. Through0054error0.63045%/65outers. Same-horizon0055raw128forces
467.77new/468.94old;256329.24new/316.79old. Bothfar fromunloaded at256;
old3e-5firstR<3 at416. Totalbudget insufficient; earlylowering modestlyslower
in this nonidentical-seed comparison, not solecause. No0060completion orlivejob.
Next distinguish propagation fromfinishing, use128 as observation blocks rather
than assume everyrateblock suffices. No restart or newpolicy implemented.

Freshcold budget128 implementation VERIFIED and GPU RUNNING,2026-10-18.
Worker16172/session54587, log/tmp/fracture2d_fem_cold_budget128.log; root
runs/pinn_history_fem_cold_budget128, launchreceipt diagnostics/
history_fem_cold_budget128_launch_20261018. Verified actualstate0000 has
q=d=H=0,Adam0,bank0, and exactsameNN/Adambytes as originalfreshseed. Config
diff limited tooptionalratecontrol, criticalcap1024→128 andoutputpaths.
16focusedchecks covered:14passinitialsuite plus2fixturecorrected rerunsPASS
(fieldstop had been disabled by inheritedtestfixture). Defaultnonlocal and
fixedFE regressionPASS; dynamicrate oneJITtrace, budget/stop/exactresumePASS.
Rate ladder1e-4→3e-5→1e-5 with128outers perrate, persistacrossloads; same
60loads to0060 and128inner/bootstrap1024. Newinitialrun compiling/bootstrap;
no completedfullpath claim. Monitorjob/results/TB next; noGitoperations.

New cold run authorized,2026-10-18: optional budget-rate ladder1e-4→3e-5→1e-5,
user specifies128outers PERrate (includingfracture),128inner and unchanged
fieldstop; rate/moments persist acrossloads. Fresh0→0060, no warmweights.
Implement/test optionalcheckpointed ratecontrol, preserve fixedrate defaults,
then launch separate pinn_history_fem_cold_budget128. Finalratebudgetfailure
must stop, not silently advance. Prior completed0055diagnostics preserved.

0055 matched finishing COMPLETE,2026-10-18. Identical Adam203904/bank110/q/d
seed and0054Hfloor,128inner, originalthreepass stop. Baseline3e-5 fails32extra
outers;1e-5 and3e-6 bothpass at4extraouters/512updates (passes2–4), final
dchanges4.671e-6/3.308e-6. SyncR1.536913/1.536590 vs1.536052 (errors0.0560%/
0.0350%); damageL2errors6.57715%/6.57744% essentially unchanged. LowerLR
fixes fluctuation, not remainingfieldbias. Root diagnostics/
fem_energy_lr_finish_0055_20261018; worker10757 exited0,GPUidle. CPUequivalence,
oneJITtrace and finalhostforce checksPASS; imagesinspected. No furtherloadqueued,
no source/defaultchange. Candidate1e-5finishing supported locally, not a global
automaticpolicy. Exactnext: explicitnormalendpoint/fork withupdatedlastH before
0056continuation;1024+4=1028totalouters, preservemoments204416/bank110.

2026-10-18 next trial authorized: matched0055 finishing from saved synchronized
budget endpoint net/Adam203904/replay110. Compare fixed3e-5,1e-5,3e-6 for at
most32extraouters each,128inner and original fieldstop; preserve0054Hfloor.
Prepare separate diagnostics/fem_energy_lr_finish_0055_20261018, no advance,
reset or mutation of source. Verify dynamic-LR Adam equivalence before GPU.

Latest terminal audit, 2026-10-18: pinn_history_fem_lr3e5_continuation STOPPED
automatically at0055outer1024 (budget_exhausted); former3427 exited, job.json
failed with explicit budget message. Saved synchronized0055R1.4109715 vs
matched direct1.5360524 (8.143%relativeerror), damagevolumeL2error6.56737%,
dmax.9992177, dchange4.50395e-4>1e-5. Outer512error6.71150%: extra512outers
barely improve fielderror; persistent late fluctuation, not missing crack
propagation. Completed fieldstable endpoints only through0054;0055saved but
unconverged, no0056–0060. No live continuation/restart/defaultchange. Next
candidate is a matched lower-LR finishing test, not more unchanged iterations.

Latest0055 audit, 2026-10-18: worker3427 still running, outer584 stageR1.49635,
dchange4.9957e-4, not yet1e-5 fieldstop. First loggedR<3 atouter416 (R1.73279).
Savedouter512 damage relative volumeL2error6.7115% vs same-load direct;
stageR1.40915, high-damage(d>=.9) component extends x.49970→1.0 near y.5
on existing damage-edge graph. Reference endpointR1.53605/388outers. Crack
has developed across remaining ligament; settling/field accuracy still open.
Latest reaction is in-load/unsynchronized, not a completed endpoint. No changes
to LR/tolerances, no restart; continue existing job toward0060.

Latest verified update, 2026-10-18: low-LR FE-energy continuation completed
0050–0054 with11/13/15/21/66outers, versus matched direct11/13/15/20/63.
Volume-weighted damageL2errors0.5408/0.5435/0.5461/0.5461/0.6210%;0054
force619.85033 vs619.54858 (0.04870%error). GPU3427 remains running0055,
latestouter40: stageR578.1581,dmax0.98614,dchange6.8544e-4; not converged
or mechanically synchronized. Matched direct0055 requires388outers and
finishesR1.53605. Continue unchanged; no breakthrough/speedup claim yet.

2026-10-18 continuation authorized after successful0049finish. Prepare NEW
runs/pinn_history_fem_lr3e5_continuation, retain0→0048prefix and accepted
0049refinement (136totalouters,Adam56704), then load0050→0060 atfixed3e-5.
Rebuildbank98 endpointH/fine_ids forrefined0049, preserveprevious-loadfloor,
verify exact model/Adam/replay roundtrip before GPU. CPUforkverificationPASS:
allprefixfields preserved, model/Adam/bankexact, lastendpointHreplaced, host
forceresidual2.168e-5withinlimit. GPU3427/session63149 RUNNING futureloads,
job/log under newroot and/tmp/fracture2d_fem_continue_lr3e5.log. Verified0050
endpoint COMPLETE in11outers: damageL2error0.54075%, R593.69470 vs593.68024
(forceerror0.002435%); now at0051. FirstAdamcount56832 (56704+128), no reset. Originalfailed
full/diagnostic trials preserved. Same task, no adaptivecontroller/defaultchange.

2026-10-18 matched0049 finishing COMPLETE. Identical failedendpoint NN/Adam
55680/q/d seed and0048Hfloor,128inner, unchanged1e-5/1e-4/threepass stopping.
Baseline1e-4:32extraouters/104.38s stillfails; low3e-5:8extraouters/25.75s
passes (dchange3.065e-6), fieldL2errors.5368%/.5391% respectively. Lowrate
syncR585.15451 vs585.14517(error.001598%),Adam56704. Sharedwarmup108.14s
excluded from armtimings; oneJITtrace and exact Adamupdate/state checksPASS.
Diagnostic root fem_energy_lr_finish_0049_20261018. Worker42674 exited0/GPUidle;
no loadadvance, originalfullrun remains stopped0049. Next integrate validated
finishing into a traceable continuation; no globaladaptivepolicy adopted yet.

2026-10-18 next stage AUTHORIZED: user accepts pilot field appearance and asks
to proceed. Started a separate full0→0060 cold pair (new direct then fresh NN),
retaining the pilot hyperparameters and old saved trials. du1e-4, energy/Adam
1e-4/128inner, bootstrap1024, fieldtol1e-5, unrelaxed local-mechanics stagger,
full sampling/replayoff. Config-only addition; numerical core unchanged.
Owner35936/session17231; direct35987 completed60loads to0060 in316.46s.
NN36601/owner35936 now EXITED: automatic budget stop at0049outer128, not user
cancellation. Saved synchronized0049 endpoint has dL2error1.086%, R585.78965
vs585.14517 (error.1101%), dchange3.218e-4>1e-5. Raw8/64errors.821%/.677%
were smaller: additional updates oscillated rather than steadily improving.
TBtiming3.176s/outer at0049 vs3.194s at0048: iterationcount, not slowerforward,
caused the slowdown (~6.8min for128outers). GPUidle, no automaticrestart.
Exact audit in launcher/budget_0049_audit.json. Next candidate: matched low-LR
finishing trial retaining moments, before furtherloading; not launched yet.
New direct spans0055
with388outers (R1.53605), giving a matched critical-stage comparator.
Roots runs/fem_damage_reference_full and
runs/pinn_history_fem_full below tf-logs/fracture2d. Receipts/logs under
diagnostics/history_fem_full_launch_20261018. Pilot parameter equality checked,
128bankcapacity covers120snapshots. No completed full trajectory claimed yet.
Same active task; monitor job.json and per-run result.json next.

2026-10-18: existing work committed on dev as82c2f3c (no push). Optional
single-geometry conditioned-history neural damage now uses material-P1 shape
gradients and matched lumped-FE energy; local direct mechanics/nonlocal
defaults retained. New implementation uncommitted;16targetedCPUchecks PASS.
GPU fixed-H512updates each at0025/0052/0053:damageL2errors21.54/13.22/10.85%.
Postfracture0055/4096updates:straight spanning band, error34.26%, so not yet
an accuracy/speed improvement. Actual S matches independentcache to4.04e-14.
Both matched direct/neuralcold0→0010 COMPLETE10endpoints. Fresh NN/Adam0/
bank0 verified; final dL2error9.360%, absoluteRMS1.520e-4, forceerror.00746%.
Neural/direct totalouters56/50, loadingwall244.36/37.85s includingJIT/reporting;
no matched-qualityspeedup claim. Former31171/session6589 exited0,GPUidle.
Runs below tf-logs/fracture2d/runs; exact paths/metrics in task. No full0060
queued. Next calibrate inner accuracy/stopping before extension. Same task.
Result figures displayed: same-load0010 damage with tip zoom/actual shared
color range, cold reaction curve, and separately labeled fixed-H0055 damage.
New plots saved in diagnostics/history_fem_energy_20261018/display/.

2026-10-18: user agrees SINGLE geometry first, retaining cross-geometry
architecture potential. Current question is FEM mechanics-vs-damage cost.
Typical hybrid fixed-H AT2: vector elasticity is usually costlier than scalar
damage, especially with crack-induced stiffness contrast, but project-specific
split is not measured consistently. Mechanics has a synchronized solve timer;
damage direct helper lacks matching timing. PINN damage training cost is a
different quantity. No benchmark/new run or code change under this discussion.

Latest user priority: cross-GEOMETRY generalization is the final objective.
Mesh differentiation is compatible: per-geometry FE matrices/quadrature can
vary while one geometry/history-conditioned network shares weights. Proposed
direction: geometry/boundary set encoder + multiscale H encoder + coordinate
decoder, trained by each geometry's FE energy. No fixed-node learned output
vector or node-ID parameterization; distinguish remeshing, unseen shape and
topology tests, plus zero-shot vs fine-tuning. Current history model has
partial geometry information in patch positions/masks, but no validated
cross-geometry training. Discussion only, no new implementation/GPU run.

2026-10-18 discussion: user proposes replacing damage nonlocal derivatives
with shape-function derivatives on the existing mechanics mesh. Most promising
new engineering candidate: neural nodal damage + material-P1 interpolation +
matched lumped-FE energy training; nonlocal energy retained as same-model
control. Current FEM reference already uses S=integral gradN.gradN and lumped
reaction/history. Cached localS496651nnz vs nonlocalD1138075; full neural
forward node count unchanged, so no claimed total speedup. P1 has zero interior
second derivatives: use weak/energy form, not pointwise Laplacian. New local
operator changes finite-horizon damage model; no code/default/run changed.

Latest discussion: user asks for an overall remedy, then the exact energy
functional. Candidate next fixed-H comparison: existing L2 residual vs the
equivalent volume-weighted quadratic damage functional. Checked reciprocal
positive graph/volume symmetry and directional derivative on actual operator
(relative mismatch2.13e-11). This is a proposed training objective only; no
loss/code/default change or new GPU run. It is a frozen-H damage functional,
not a joint energy proof for history-field stagger. Details in active task.

Sparse-probe screening COMPLETED, 2026-10-17 23:44:45Shanghai.
Root `tf-logs/fracture2d/diagnostics/sparse_response_lr_20261017` (autodl-tmp).
All4arms exit0; former7468/17367gone,GPUidle,no queuedcold or extension.
qualification.json eligible=false; newcoldroot absent. Oldcold stayscancelled.

- At0053/512,baseline8 vs event32 bothspan. DamageL2error21.564/10.342%,
  syncR1.23188/1.42092 vsreference1.57096. Wall34.66/31.26min excluding
  warmup:9.80%less fornewarm atfixedhorizon, withbetterfinalfield. Bothstill
  fail1e-4fieldchange at512 (2.676e-4/1.580e-4); no certifiedconvergence.
- Newfirstconfirmedshadowstop409:damageerror17.619%,syncR1.28275, despite
  dchange9.165e-5. Fieldchangepasses do not guarantee targetfieldaccuracy.
  Baselinehasnocandidate. Keepthisfirstcandidate, nooracle replacement.
- Elasticwindows baseline/event85.83/170.48s;0018outers14/39. Thisregression
  andmissingbaselinecandidate blockpredeclaredfullcoldgate. Currentvariant
  useful forcriticalpropagation butnotqualifiedasoverallcoldstrategy.
- Same-sourcebaseline rerun didnotreproduceprior8.77%quality; causeunresolved.
  Exactnext: inspectreproducibility andfinishing/stopadequacy beforeanother
  qualifiedpolicy/fullcold. No newnumericalcode/defaults or Gitoperations.
  NineCPUchecks plusmigrationregressionremainvalid; details in active task.
  Taskincomplete, finalMLPvscompletedadaptivecold two-curveplotstillpending.

Matched cold branches128→512 continuation COMPLETED (2026-10-16).
Root: `/root/autodl-tmp/tf-logs/fracture2d/diagnostics/cold0052_nonmonotone_lr_to512_20261016/`.
Both512exit0, allformer60313/60340/64187exited; GPUidle,no queuedextension.
TwofocusedCPUcontinuationchecksPASS32.098s remainvalid; bothclouds inspected.

- Adaptive/fixed3e-5 BOTHspan. Firstlogged stageR<3 at376/432outers,
  correspondingcumulativewall32.18/30.40min. Feweriterations is not less time.
- Final512adaptive/fixed: syncR1.43456/1.05498 vsreference1.57096;
  damagefieldL2error8.768%/20.631%,ownHerror8.342%/20.694%.
  dRMS5.844e-5/2.100e-3: adaptivepasses1e-4(10consecutive),fixedfails.
  PhysicalAT2.19854/.11718; thismetric alone doesnotrank fieldaccuracy.
- Cumulativefirst128+continuation43.01/35.98min, executed79744/65536updates;
  fixed16.33%lesswalltime but poorerunsettledfinalfield. No matched-quality
  speedupproof. Supports adaptivefinishing quality; trialcost remains open.
- Raw512checkpoints remainin_load/bank105 atfixed.0053. No loadadvance or
  fullcoldcompletion claimed. Finalcomparison.json/closeout.json recordscope.
- User agrees longerhorizon. Restore each arm's OWN rawouter0128 q/d/H,
  net/Adam86272,bank105,controller643/probephase/flags. H_parentchecked
  against actualcold0052H. Add384outers×128updates each, fixedload.0053,
  no fieldstop/loadadvance or policychange, originalcold remainscancelled.
- Checkpoint129/every16; physicalsync/validation at192/256/320/384/448/512.
  Each arm in separate serialprocess; adaptiveguardfailure is explicit and
  fixedarm stillruns independently. No fallback or rate-rule relaxation.
  Track cumulativeprior128+newtime/work, never label diagnostics as fullcold.
- Initial128comparison (completed, twoCPUchecksPASS17.201s) established:
- Adaptive vs fixed3e-5: synchronizedR552.86 vs554.60; both frontx.54386
  (notch tipx.5),same-load damageerror74.63% vs76.42%. Neither spans or
  satisfies fieldstopping. Reference0053R1.57096 is the fullyspanning target.
- Walltime644.35s vs538.09s (10.74 vs8.97min); fixed16.49%less time,
  executed16384 vs19584updates. Saves candidatework without meaningful
  crack-progress improvement within128. OwnHerrors11.66% vs11.52%.
- Matched old/new FIXED3e-5 audit resolves apparent contradiction: at128
  stageR555.72old vs553.60new,frontx.54386395 both. Oldfixedrapidunload
  occurred later:256R365.92,344R1.19. Newtest ended128, so no evidence
  it lost oldfixedrate benefit. Comparestage-to-stage, not synced-to-stage.
  Separate oldwarmADAPTIVE at128R447 used1e-4 too; not fixed3e-5control.
  Receipt matched_horizon_audit.json. User subsequently authorized extension above.
- Bounded result does not establish final breakthroughtime or prove all
  nonmonotone strategies ineffective. No newcontroller/default adopted;
  longer-horizon evaluation has completed as recorded above.
- User authorized multistagger evaluation allowing temporary loss growth,
  then explicitly stopped the prior full cold. Old43941/43968 verified exited;
  `adaptive_cold_launch_20261016/job.json` now cancelled_by_user. Saved52
  endpoints through.0052,lastscalar0053outer264R481.29,latestfullcheckpoint256.
  Stoprequest/verifiedreceipts preserve state. Do not resume or claim0→.0060.
- New comparison restores EXACT actualcold0052 network/Adam/bank/q/d/H:
  count69888,bank104,globalcontrollerouter515. Arm1savedadaptivepolicy;
  arm2fixed3e-5 without lossdescent veto. Fixedarm is a diagnostic, not a new
  adaptivecold replacement. Each128outers×128innerupdates; no early stopping
  or loadadvancement during fixed observation horizon.
- Evaluate synchronizedfields/force/H/AT2,ownHdirecterror,same-load reference
  error and connectedcrackfront at8/32/64/128. Additional reporting mechanics
  do not feed back. Same physics,Hfloor,full-domain L2/Adam/replayOFF;
  reference never used in training or choosingrates.
- Latest cold behavior beforestop: lower-rate choices accompanied slow propagation;
  one128update same-Hloss increase alone does not establish physicalfailure.
  OuterHtarget changes can legitimately raise loss. This is the motivation
  for a multiouter comparison, not an adopted new onlineLRscore.
- Earlierfirst14cost:71vsMLP132outers but557.94svs288.62s; bootstrap and
  heaviermodel/trials matter. Timing is historical,not hardwarecontrolled.
  Saved auditreceipts in oldlauncher; detailed provenance in active task.
- Requested final plot remains ONLYtwo plaincurves:MLP and a completed actual
  adaptivecold. Oldcold was interrupted; do not splice warm diagnosticarms
  into it or label new fixed-rate diagnostic as cold. Finaldeliverablepending.
- TensorBoard6007 continues; oldcold events preserved, newdiagnostic events
  live within this root's adaptive/events and fixed_3e5/events when produced.

Exact resume: completedcomparison/closeout and perarm512raw/validation/figures.
No liveGPUjob or automaticextension; do not signal formerPIDs. Nextresearch
choice is reducingadaptive trialcost and assessing actualfullcold trajectory,
not silentlyresumingcancelled oldrun. Publiccode/defaults unchanged; noGitops.
FinalMLPvscompletedadaptivecold two-curve deliverable remains pending.

## Fracture2d artifact cleanup completed — 2026-10-16

- All 201 fracture2d TensorBoard event files now live below
  `/root/autodl-tmp/tf-logs/fracture2d/`. Accepted directories are
  `pinn_mlp_baseline/` and `pinn_history_conditioned_baseline/`, with their
  own `events/`; trials are under `diagnostics/`, new defaults under `runs/`.
- Current comparison data are `/root/autodl-tmp/data/fracture2d/fem_reference/`
  and `nonlocal_reference/`; prepared inputs are `geometry_assets/`.
  Older models are distinguished in `legacy_references/`.
  Current FEM uses the shared globally fine h=0.004 mesh and volume-grip BCs;
  it is not a demonstrated mesh-converged reference (details in artifact index).
- [Artifact index](../examples/fracture2d/ARTIFACTS.md) records canonical paths.
  Full inventory, metadata backup, migration map and verification receipts are
  in `tf-logs/fracture2d/_layout_migration_20261016/`.
- Verified 77,543 files, 2,183 rewritten JSONs, unchanged accepted fields/curves
  and seven presets' numerical settings. Both PINN checkpoints restore on CPU;
  history buffer restores 106 snapshots. TensorBoard6007 discovers both runs.
  Numerical experiments remain ON HOLD; no Git operations or training started.

## Accepted fracture2d stage

- [Stage closed and archived](tasks/archive/fracture2d-nonlocal-pinn.md),
  2026-10-15. Accepted engineering result uses local P1 direct mechanics,
  nonlocal AT2 PINN damage and maximum-history stagger. Same-model direct
  damage and independent local FEM remain comparators.
- Accepted `pinn_mlp_baseline` saves 0 -> 0.0057, spans by 0.0053; the user
  interrupted 0.0058. It is not a completed 0 -> 0.0060 run. Field accuracy,
  path smoothness, convergence and efficiency remain improvement targets.
- Default: full domain, L1/Adam/sigmoid/FP64, du=1e-4, 128 inner updates,
  outer caps 128/1024, damage-change tolerance 1e-5. No symmetry constraints.
  Optional core+halo keeps a fixed 2048-pattern exact-support padded pool.
- Measured insight: cheaper core+halo iterations needed more outer work;
  matched early-load total time was 3.35 times full. Keep full as default.
- Consolidation and publication complete: final main b395e48 independently
  verified on origin. Four public configs tracked. All _legacy/ directories
  and docs/reports/ are local-only/ignored; 324 withdrawn files preserved.

## Old GPU baseline stopped by user

Saved-config repeat pinn_baseline_gpu_rerun_20261015 is CANCELLED after
through-fracture at explicit user request. Verifiedowner31071/worker31076 exited.
54completed endpoints through0.0054 preserved,partial0.0055outer600 and named
checkpoint512 retained. TensorBoard6007 continues. External stoprequest/verified
receipts and archived stage record exact state; target0.0060wasnotcompleted.
This cancellation is separate from the current short history-operator probe.

## Other retained state

fracture1d remains an accepted engineering baseline. Older fracture2d
nonlocal-mechanics, warning-accepted direct runs and detailed diagnostics are
historical controls, not the public main route; see stage archives.
[DECISIONS](DECISIONS.md) retains append-only rationale. Example READMEs
contain stable formulations and usage.
