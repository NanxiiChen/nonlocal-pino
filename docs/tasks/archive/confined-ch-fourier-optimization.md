# Fourier/architecture optimization of the fixed10mode CH geometry operator

Date:2026-10-30. Status: COMPLETE — requested approximately10% pooledphase target reached. No Git history authorization.

## Live state (latest, supersedes historical stage notes below)

Final result: fixed30000step96width/512geometry operator reaches9.660899%
pooledphase,10.513060%terminalphase,9.188718%over61native times. Requested
approximately10% overallphase aim is reached. Terminal is reported separately;
not allgeometries/quantities are below10%. No strictboth-below10%claim.
Final root /root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-30k-retry-01;
model_final.eqx + adjacentconfig.json are the selected measured checkpoint/template.
Receipt docs/tasks/confined-fourier-final-results.json; fullfinalaudit
 docs/tasks/confined-fourier-wide96-30k-retry-audit.json.

Final validation: all24savedphysicalfields independently replay onCPU within
4.3334e-15 ofGPU. Imported20kmodel/optimizer gaps0, restoredstatefinite; actual
20001..30000geometrydraws matchglobalcommonseed, all512visited258..371times,
noheldoutdraws/no futurefields in optimizer payload. Dense61massdrift1.4382e-16,
energyincrease0. Training16monitor phase5.9243/6.3024%, worst8geometryphase17.2550%,
meanmu21.0559%; nativegraph/BDFonly, localFEMdeferred. Oneinitialization/original8
reuseddevelopment geometries, notfreshfinaltest/multi-seed/continuumvalidation.
Geometryresponseerror21.7096%, amplitude93.2746%, correlation.977160 on4627
commonprobes; networkcapturesconditionalvariation, notonlymeanfield.
Selectedlineage14927.127s (248.79min), laststage4954.843s, warm428.302ms;
lineagecost excludes earlierincompleteattempt. No speedimprovementclaim from
capacitygain alone.

Selectedopt-in coldrecipe: examples/confined_ch2d/configs/train_operator_smallscale10_best.py.
Numericalsettings exactlymatch measured30kconfig after removing onlyruntime
seed/runname andcontinuationparent; no parentlogsrequired. Actualaccuracy/time
was measuredas10k+10k+10k complete-state stages, NOTa separatecold30000run.
Explicitlineagepreset train_operator_smallscale10_512_wide_30k.py remainsloadable;
canonical single-IVPtime-windowdefault andallhistoricalcontrols unchanged.
Width96/4heads/4conditionblocks/3decoder,128space/64timeFourierchannels, timeband4,
B16/Nt16/.2initialPDEfraction, stablepartresolvent, FP64ordinarySOAP, uniformfullT3,
hardIC/mass, eliminatedmu, oneglobalnetwork. KnownRHSfeatureoff, no distancebias.

Completedcontrols:64/512/10k18.945/20.260%;96/512/10k15.329/16.428%;
64/512/20k15.330/16.469%;96/512/20k11.384/12.281%; old64/25k14.378/15.451%.
1024/64/10k20.212/21.787% doesNOTbeatmatched512/64/10k; trainingmonitoralso
worsens15.308->18.299%. Same160kdraws/twiceinputs halvesperinputvisits. No claim
largerbanknecessarilyineffective orsmallertrain/devgap provestransfer. Preserve
negativecontrol. 1024bank129.565s,512newoperators/520oldhardlinks/5816exactarrays,
no newPDE/FEMsolves; peakGPU32.44GB<38.18, noOOMfallback.

Original30kfirstattempt remainsINCOMPLETEatlastlogged26500; nofinalSOAPstate/
summary, causeunknown/noPythontraceback/cgroupOOM. Keepoldroot/logs/checkpoints
and terminationreceipt confined-fourier-wide96-30k-interruption.json. Retryfrom
complete20kstate inNEWroot; no seed/optimizerreset/min-over-retries selection.
Common20k..23kmetrics matchwithin2.4e-13; prefixreceipt records exactscope.
Detachedrunner docs/tasks/confined-wide30k-recovery.py and journal
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-wide30k-retry-process.json
now COMPLETE; trainingexit0, native/geometry/61time/modal/finalCPUaudit allpass.
Runner/GPUchild/monitor exited; no queuedGPU/FEM/follow-up job.

Recovery I/O: everyexisting1000step checkpoint now also saves optimizerstate
and publishes step/model/state marker AFTERclosedatomicpayloadrenames. Lone
model/partialfile isn'tcompletepair. OldcompletedparentAPI/legacy modelnames
unchanged. RealSOAPstep6pair recovery to8matchesuninterruptedfinal(37.736sPASS);
injectedwriteinterruption cannotpublishmarker(.215sPASS). RealGPU21kpair loads
onCPU/allarraysfinite/bothcounters21000 (confined-fourier-retry-checkpoint21000.json).
No PDE/update/schedule/PRNG/architecturechange fromI/O. Newconfigdict equivalence,
Pythoncompile/JSONsyntax/diffcheck pass; unchangedexpensivesuitesnotrerun.

Session/task-end changes: optionalwide/timeband NNmetadata; explicit stablepart
input-only residualmetric andsamplingcontrols; strictfullstate continuations/
LRoption; knowninitialRHS optionalcontrol(defaultoff); validatedincrementalbank
copy/hardlinkreuse; pairedcheckpointI/O; durablefixedbudgetrunner; explicitbest
coldrecipe; native/dense/geometry/replay/distribution/recovery diagnostics.
README/DEVELOPMENT/index/DECISIONS/PROGRESS matchactualresults. Figures/checkpoints/
largepayloads external; no newgeneratedbinary inrepo, no Git history action.

Artifacts: comparison
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-final-capacity-coverage/comparison.png
and selectedrun/evaluation/fields_512.png,fields_516.png,fields_519.png use fixed
first/middle/lastvalidationpositions; smoothingfor displayonly, metricsoriginal
centroidfields/physicalareas. No plot/error-driven sample selection.

Exactresume: none required; requestedapproximately10%overallphase optimization
completed andverified. Optionalfuturework: freshvalidation/multi-seed/localFEM or
matchedlonger1024coverage study require a newscopedtask, notanunfinishedpromise.

## Goal and acceptance

User requests larger Fourier embeddings (at least comparable to backbone width)
and continued general optimization, aiming for about10% phase relative L2.
Use the same fixed small-scale10mode/T3 physical case and512/8geometry bank.
Native references only; FEM remains explicitly deferred. No future training
labels, geometry-specific distance bias, IC parameterization or time windows.
Primary target: eight-development-domain mean phase pooled across5saved times
and terminalT3 both near/below.10, then independent replay/physical/geometry
response checks and denser-time sanity if close. Target is not a guaranteed
outcome. Keep failed controls and report actual accuracy, no seed tuning.

## Baseline

/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-smallscale10-soap-01
SOAP20kB4/FP64ordinary: nativephi.39561365/.39169509, mu.37707634;
last5kphase44.00->39.56%, so slow incomplete convergence rather than a proven
trap/capacity floor. Warm69.817ms, cold2030.369s. Architecturewidth64,heads4,
conditionblocks4,decoder3,spatialfeatures24at scale1,timefeatures4,outputscale.1,
224705trainable scalars. Prior Adam and all previous artifacts preserved.
Archived baseline record: docs/tasks/archive/confined-ch-smallscale10-operator.md.

## First candidate: generic wider/broader Fourier inputs

Keep width64/heads4/depths/outputscale/LR/20k/B4/sampling/physics unchanged.
spatial_features64 ->128sin/cos channels, time_features32 ->64sin/cos channels;
spatial_scale2 broadens spatial input frequencies. The two Fourier counts denote
frequency vectors, not total sin/cos width. This is a feature-count/bandwidth
bundle, not an isolated count-only ablation. Same generic architecture code.
Raise SOAP max_precond_dim128->256 because expanded lifting inputs reach133;
otherwise the old threshold would silently exclude the expanded input axis from
SOAP's intended matrix preconditioning. Other SOAP settings unchanged.
Pure physics, hardIC/mass, chemical elimination and existing residual metric.
Add final optimizer-state serialization only (no update/loss change), allowing
future explicit continuation without discarding the expensive20kstate if needed.
No resume path or old-state fallback is introduced at this stage.

Config: examples/confined_ch2d/configs/train_operator_smallscale10_512_fourier.py
Planned run: /root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-fourier64-t32-s2-soap-01
Dataset: /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1
Expected243649trainable scalars;128fixed spatial frequency scalars. Shared-space
preparation/timeJVP remain. Common runtime42, no config seed or seed selection.

## Plan and exact resume

1.Validate explicit candidate config/shapes/real SOAP state and final-state save.
2.Coldtrain canonical train.py20k, monitor every500, preserve physical data/cache.
3.If insufficient, diagnose current residual/time/spatial fitting and try simple
   generic changes (output scale, temporal conditioning or width96) independently
   where practical. Do not change PDE/horizon/initialdata to meet the target.
4.Finalist replay, native metrics/fields/full convergence/geometryresponse,
   README/index/memory updates. Generated payloads external; no Git action.

Historical first-stage status: candidate configuration/shapes/real SOAP initialization and
optimizer-state round-trip PASS.243649trainable parameters verified. Cold20k
run started, raw /tmp/confined-fourier-train.txt. Exact resume: monitor log and
500stepmetrics; assess against priorSOAP20k39.56%. No FEM/Git action.

## Frozen-checkpoint diagnosis and second control

At2k, geometry512 reference phaseRMS .0831,.07199,.1500,.2955,.4476;
prediction misses growing modes (Tsubspaceerror94.49%). Dense257time CPUprobe
has early/latehalf transformed RMS .0213/.00842; raw .3091/.2057. Frozen weighted
A spectrum gives19growinglinear modes, fastest gamma1.388, exp(3gamma)64.32,
biharmonic residualweight .1857. Linearization is about phi=0, not the spatial
initial field. Nonlinear late evolution is not this infinitesimal model.
Receipt/script: docs/tasks/confined-fourier-probe2000.json and
confined-fourier-probe.py. This motivates early accuracy; it does not establish
sampling as the sole cause. Control improves to9k54.58%; retain full20k.

Optional operator_temporal_samples now dispatches to the already-existing
log-time measure, prepending exactt0, while absent record preserves the original
uniform RNG/expression bitwise. Explicit temporal config validation mirrors the
single-IVP checks. Training loss logs include horizon; physicalT/lifting/metric
stay3. Newconfig train_operator_smallscale10_512_curriculum.py changes only
sample.temporal=.05/log_uniform/.15initial horizon/8kwarmup, with4positive+0nodes.
One global network, no time slabs or previous endpoint, no future labels.
Validation: unittest operator_temporal2/2 PASS, existing temporal2/2 PASS;
pytest unavailable, so repository unittest runner used. git diff --check PASS.

Queued serialized GPU control (launcher session79336) starts only after owned
PID27842 exits and full model/optimizer/summary exist. Next run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-fourier-curriculum-soap-01
log /tmp/confined-curriculum-train.txt. It is not yet started as of9kfirstcontrol.
No FEMjob, Gitaction or duplicate dataset. Exact resume: monitor both logs;
finish first20k, assess curriculum full-horizon metrics after8k; independent
baseline20kCPUprobe currently writing confined-fourier-probe-baseline20000.json.

## Width/bandwidth separation (implemented, not yet measured)

Generic point models now accept optional static time_max_frequency. Missing
record preserves arange(1,K+1), RNG and all parameter leaves; explicit finite
maximum>=1/K>=2 gives K linearly spaced frequencies from1 to maximum. New bounded
preset adds max4 to the32frequency curriculum, without changing its weights,
counts, physical problem or planned cold seed. It is unrun/unclaimed, not queued.
Band tests2/2 PASS: explicit original4band has identical arrays/outputs in BOTH
point families; bounded32timeJVP agrees with finite differences; invalid metadata
rejected. Existing operator16/16 PASS (125.201s). Existing temporal2/2 and new
operator temporal2/2 previously PASS. Old saved checkpoint CPUreplay still to do.
Baseline20kprobe confirms geometry512 growing-subspaceTerror42.85%, not solely
new Fourier instability. First control at17.5k native43.113%/42.828%; stillfalling.
New helper audit permits/reportsonly explicit sample changes; unchangedphysics,
no evolved payload, actualconditiondraws remain mandatory. No defaultpromotion.

## First completed control

Fourier64/time32/scale2/maxSOAP25620k: phasepooled .40817581911461887,
terminal .40677031353337395, mu .39575148449560726; cold2145.233s,warm72.564ms.
No gain over priorSOAP39.56/39.17%. Independent native saved-field evaluation
matches and hardIC/mass<=2.18e-16 plus sampled5time energy gate PASS.
Three representative fieldplots (512,516,519) and all24numeric outputs exist.
Geometry response/curve/replay CPUorchestration active session75149,
log /tmp/confined-fourier-postprocess-rest.txt. An earlier postprocess launcher
was stopped before its missing geometry --dataset argument; completed evaluation
was independently inspected and retained, then remaining stages restarted with
correct explicit arguments. No numerical run restarted or field overwritten.

Global curriculum now running GPU PID34747 (launcher79336), start~03:01UTC;
initialmetrics exactly match firstcontrol, setup119.357s. Firstcompile in progress.
Bounded-time candidate only prepared, not queued or measured. Currentbest remains
priorSOAP20k. Exactresume: finish curriculum, assess fullTafter8k, complete first
CPUaudit, then decide boundedband/scaling/continuation based on actual curves.

## Dense validation and active follow-ups

Firstcontrol audit PASS: CPU24fieldreplay6.412e-15, freshinitgap2.887e-15,
finite restored optimizer, identical80kconditiondraws, all512visits123..201,
noheldoutdraws/futurepayload; source template static-band extension compatible.
Geometryresponse81.33%error/40.67%amplitude/.6195corr (not improved over prior).
Last5kphase45.29->40.82%, slope-.884percentagepoints/1000, stilllearning.
Compactreceipt confined-fourier-uniform-audit.json. Allcontrolprocessesexited.

PureSciPy dense reference on SAME8meshes,61snapshots (no JAX/FEM) completed
156.784s/6workers. All8native timegates pass: maxphi6.578e-8,mu4.805e-7;
existing5anchorsagree<=3.49e-15. External path:
/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation
Script confined-fourier-dense-reference.py; receipt confined-fourier-dense-reference.json.
These validation labels are isolated from the unchanged512training bank/payload.
DenseCPUevaluator confined-fourier-dense-evaluate.py reconstructs only8input
operators, checks IC/physicalmass/energy and compares61times (nooptimizer).
Curriculum3k: H=.4611857, trained-prefixmeanphi19.4586%, full61time134.027%,
terminal120.208%; severe fullinterval extrapolation is not same as prefixerror.
Full metrics in run/diagnostics/dense_003000/metrics.json. First/baseline20k
61time evaluation currently running CPU sessions13646/92191; raw /tmp/confined-
fourier-dense20000.txt and /tmp/confined-baseline-dense20000.txt.

Curriculum GPU PID34747 at6.5k(H~1.71),5timephase79.85/84.62%; continue past8k
before fulltimejudgment. Conditional launcher59833 waits for its successful20k
completion, then starts bounded-time32/max4 control ONLY if bothphase targets
are not yet<=10%. That next coldrun:
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-fourier-bounded-soap-01
log /tmp/confined-bounded-train.txt. The condition is unmet when EITHER primary
metric exceeds.1. No overlappingGPUruns/FEM. Comparison plotting helper
confined-fourier-comparison.py records completed/running status and common
8domain fullTmetrics. Current best remains original39.56%; no defaultpromotion.

## Stable-part candidate validated and now running

New input-only operator_metrics.linear_stable_resolvent independently passes
3node analytic stable/growing spectra, finite-difference CHJacobian with
nonuniform volumes, weightedSPD/constants and invalidgraph checks (3/3tests).
Derivation/meaning is in README and decision log. Full520domain CPUspawn
preflight127.645ssetup/155.373swall PASS, minweight.00203982,
maxconstantmodeerror6.75e-14, finitePDEgradientloss.01246, nonzeroencodergradients,
IC0/mass4.12e-18, NOoptimizerupdates/futurefieldpayload. Receipts
confined-fourier-stable-preflight.json (onecase) and
confined-fourier-stable-bank-preflight.json; reproducible preflight.py retained.
Focused old-loader/invalidconfig regressions2/2 PASS23.54s aftermetricextension.

Candidate train_operator_smallscale10_512_stable.py: same243649weights,
spatial64/scale2, time32/max4, fulluniformtime4+0 (no curriculum),
SOAPmax256/20kB4/FP64ordinary. Newmetric uses6CPUworkers during setup and writes
run/residual_metric.json; source physical bank and cached matrices untouched.
This is a band/norm/sampling bundle, NOT isolated proof of the norm benefit.
The bounded-only waiting launcherPID39124/session59833 was verified idle/nochild
and canceled before any training; its preset remains unrun. Replacement
launcher30890 successfully started stableGPU PID41881 after curriculumfinished.
Stable run /root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-fourier-stable-soap-01
raw /tmp/confined-stable-train.txt, start03:38UTC, setup currentlyinprogress.
No laterGPUjob is queued. No FEM/Gitaction.

## Curriculum completed (new provisional best, pending fullaudit)

20kphasepooled .3541803758517749 /terminal .3573085465242825,
train .3548109810346998, mu .387292149767608;
wall2169.549s,warm72.889ms. Better thanprior39.56%, NOTnear10%yet.
CPU postprocess session56249 (log /tmp/confined-curriculum-postprocess.txt) runs independent nativeevaluation,
geometryresponse, curve,61timevalidation, then audit to
confined-fourier-curriculum-audit.json. Curri3kprefixdiagnostic separate.
Baseline/fullFourier61timephi40.4686/41.6318%, energy monotone all61times,
mass<=4.23e-16. New dense validation isolated from trainingbank.

Exactresume: monitor stablePID41881/launcher30890 (firstmetricaftersetup/compile);
finish curriculumCPUdiagnostics, then assess stablefull20k. If stillslowlyfalling
above10%, consider explicit state-preserving continuation (finaloptimizerfiles
exist) rather than declaring a stationaryfloor. No continuation implementation
or extra training is currentlystarted. Currentbest35.42%, target10%unmet.

## Current measured progress and reusable continuation

Curriculum20k audit complete PASS: CPU24fieldreplay6.398e-15; optimizerfinite,
allactualdraws/8keypayloadcheckspass. Dense61timephi35.8868%, mass/energyphysical
checks pass. FirstFourier and curriculum initial checkpoint bytes EXACTLYequal
(direct file comparison, no hashes); both80kgeometrydraws identical.
Stable-part/bounded-time control at12.5k: phase25.3361/26.9139%, stillfalling,
train22.48%;5000stepdensephi32.4030%, early .05phi24.31%. Better latephasefit
but initialtransient remains difficult. This is a bundled control, not isolated
proof for any one change. Final20k result still pending.

Added strict operator_continuation.py option train.continuation.parent_run into
NEW run with higher TOTALglobaltrain.iterations; numerical configs/optimizer/
sampler/seed must match, both statesrestored, counterscontinue. Metadata/files
validated beforebanksetup; allparentphysicalpredictions replayed beforeupdates.
Direct paths/initialmodel+state/provenance and stage/cumulativewall recorded.
Fresh branch retains initialization/update sequence, no automaticreset/fallback.
Tests PASS: realSOAP/PDE8stepcontinuous vs3+5state serialization (includesrefresh)
25.52s, actualtrain_operator toybankparent3+child5/globaldraws/replay/cumulativewall
33.48s. No FULLbenchmarkcontinuation has started or been queued. Savedfinalopt
states may support continuedslowconvergence rather than waste newcoldwork.

## Input-only initial-scale diagnosis and fourth conditional control

Known initialgraphRHSRMS .907..987 requires CENTEREDraw(0)2.72..2.96 under
linear hardlifting t/T, despite initialphaseRMS~.08. Existing generic
logarithmic_additive coordinate s=log1p(t/.05)/log1p(T/.05), shareddecoder/lifting,
reduces requiredraw(0)to .186..203 (factor14.595), by differentiating hardansatz;
no evolvedfield used. Receipt confined-fourier-initial-time-scaling.json.
Operator optional phase now reuses that coordinate; defaultno-phase retains
exact t/T. PhysicaltimeJVP/phasePDE remains unchanged; mass/offgrid wrappers
sharecoordinate. Newlogtime tests2/2 PASS30.65s: closed-forminitialderivative,
legacybitwiseexpression, exactIC/mass, realqueryconsistency, physicalFDchainrule.
No trainableleaves/modelwidthgrowth. Continuation rejects coordinatechanges.

New train_operator_smallscale10_512_logtime.py changes ONLY phasecoordinate
relative tostablepreset (same243649weights/metric/uniformsamples/SOAP/20kB4).
Conditional launcher85225 waits for stablePID41881 completion, then coldstarts
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-fourier-logtime-soap-01
log /tmp/confined-logtime-train.txt ONLY if either primaryphi metric remains>.1.
No model/state warm transfer across the changedcoordinate, noGPUoverlap/FEM.
Current12.5kstablecontrolstillrunning; fourthnotyetstarted. Comparisonhelper
writes live curve in stable/comparison; status explicitlyrunning, not finalscore.
Exactresume: finishstable20k/native+61time+CPUaudit, monitorconditional logtime;
then consider strictstatepreserving continuation only if targetstillunmet and
curvesjustify. No Gitaction, defaultpromotion, FEMwork or finalaccuracyclaim.

## Third completed control; fourth now running

Stable-part norm + bounded32/max4 + fulluniform20k finished:
phi .22729163786101253 /terminal .24302326762324494, mu .37501864291315445;
wall2225.732s,warm73.172ms. Better than35.42%, stilltargetunmet. Fixed20kfinal,
not minvalidationcheckpoint. CPU postprocess log /tmp/confined-stable-postprocess.txt
runs native/geometryresponse/curve/61time/audit in CPU session93830.
5k61timephi32.403%/terminal33.314%, earlyt.05/.1errors24.3/29.4% highlight the
initialtransient fitting issue. This is not a final5kstrategy recommendation.

Conditional log-time launcher85225 successfully started newGPU PID49039 at
04:15UTC, owned parent48586; no otherGPUtrainingorFEM. Run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-fourier-logtime-soap-01
raw /tmp/confined-logtime-train.txt. SameNNarrays-shapes/metric/fulluniform/SOAP/
20kB4, only case-level timecoordinateandlifting differ. GPUsetup inprogress.
No furtherGPUjob queued after this fourthrun. Strictcontinuation exists/tested
but unused; fullnewrun has no stateimport or counterresetacrossconditions.
Current completedbest22.73/24.30%, targetroughly10%unmet. Do not finalize asgoalmet.

## Latest validation and fifth conditional candidate

Stablefinal CPUaudit PASS3.365e-15all24fields; finiteoptimizer/allactualdraws/
noheldout/noevolvedpayload gates pass. Dense61phi .21872105874161935,
energy monotone/mass<=2.7e-17. Geometryresponseerror.47103464,
amplituderatio.72905342,corr.89818196; meanfielderror.06995881. Worst8domain
pooledphi.41960, so do not imply allconditionsnearthemean. Generic norm improves
conditioning response substantially; most remaining contrast stillunderpredicted.

Fourthlogtimeat11.5kphi26.084/27.361%, comparable tolinearstable. Its5k61time
CPUdiagnostic session5881 writes run/diagnostics/dense_005000. Stillcomplete20k
beforejudgment; no automatic fieldcriterionearlystop. InitialNNcheckpointarrays
matchstable in shape/seed; direct bytecomparison to prove equality stilloptional.

Prepared train_operator_smallscale10_512_unit_output.py inheritslogtime and
changes only existingoutput_scale .1->1. Preflight actualgeometry512 initializer
hasphi[-.2469,.2550],RMS .0831..08348, exactIC/mass8.86e-18, nooptimizerupdate
or evolvedreference. Script/receipt confined-fourier-unit-preflight.py/json.
Conditional launcher3276 will coldtrain20k after PID49039exits iff primarytargets
stillunmet, externalrun/log above. It is currentlywaiting, not training. This is
a generic field-gain check before speculative geometry-specific designs or model
growth; do not claim measuredgain yet. No future queued work after it.

Continuation2tests PASS: lowlevelrealSOAP/PDE25.52s and actualtinytrainer33.48s.
Newoptionallogtime2tests PASS30.65s. Stablemetric3tests rerunPASS after explicit
nonempty-volume/finiteweightedmatrixguards (validation only, same numerics).
Existingnew temporal2+oldtemporal2, bandwidth2 and operator16werealsoPASS;
oldloader/invalidconfig2regressions rerun aftermetricextensionPASS. gitdiffcheckPASS.
No fullbankstatecontinuation or widebodyrun has started; legacywidth96preset
remains untrained. Exactresume is the Live state block at top.

## Completed parameterization controls and lower-noise training

Log-time20k phi .22607401757979578 / .24068864450369726; dense61 .2183644462.
Unitgain+linear20k phi .22815051211793533 / .2426164792238425;
mu .3710296231, dense61 .2201012109. Both conserve mass/decrease energy at61times.
CPU24fieldreplay log/unit<=3.220/4.636e-15; actual draws and finite optimizer
audits pass. Log practical selection rule required BOTH metrics improve>=.01;
neither does, so the simpler linear coordinate is retained for the nextcontrol.
Unitgain also gives no material L2gain; do not infer scale mismatch was the
uniquecause or an effective remedy. Historical fifth-launch notes above are
superseded: actual fifth was unit LINEAR, not log. Both variants remain explicit.

Frozen last-output-weight gradient probe uses16traininginputs,8time draws,
zero optimizer updates and no evolvedfields. AtNt4, geometryvariance .0004976141,
shared-timevariance .0000802075, estimatedB4variance .0002046110. AtNt16,
geometryvariance .0003605366,timevariance .00000197259, estimatedB16variance
.0000245061,8.35x lower. This partial-gradient finite-cohort estimate motivates
a sampling control, not a proof that fullnetworknoise sets the final L2floor.
Receipts confined-gradient-noise-probe.json and confined-gradient-noise-nt16.json.

New initial_residual_fraction=.2 preserves E[L]=.2*l_PDE(0)+.8/T*integral(l_PDE)
when changingNt4->16. t0PDE is not a separate IC loss. Absent record keeps old
arithmetic. Value/fullgradient identity atNt4 and B16/Nt16toybackprop PASS2tests
32.757s. New operator_history helper follows direct continuation parents, merges
global metric/loss counters and cumulative wall time, rejectingcycles/disorder;
focusedtest PASS. Generic trainingreport supports arbitrary final budgets and
the complete global history. No numerical production defaults were promoted.

Sixth coldB16/Nt16 started05:56:31UTC, PID65595/session2764, fixed10k.
Latest4.5k trainphi21.24%, validation24.58/26.00%. Fullphysics/SOAP first-update
memory live9.023/peak16.289GB, below38.184GB limit. OrdinaryFP64, samephysics,
globalNN243649weights/stablemetric/linear coordinate/unitgain/LR/seedprovenance.
No nextGPUjob queued; continue through10k then assess measured slope for strict
state-preserving continuation rather than an arbitrary freshrestart or floorclaim.

## First fullbank continuation and completed10k checks

B16/Nt16 10k fixedfinal: phi18.9452/20.2599%, mu32.5525%,
trainphi15.3082%; wall2786.335s,warm219.249ms. BothprimaryL2drop>4.5pp
overlast5k, satisfying predeclared extension rule. Native61time pooledphi
.1826536630, mu.3201308669, mass<=1.01e-16 and sampledenergy nonincreasing.
Geometry responseerror.40247799997/amplitude.8050256043/corr.9230002594,
meanfielderror.0540561462. Worst validationdomain remains36.15%; not all
geometries are near the mean. Frozen-growing-subspace pooled/terminalerrorenergy
share89.06/89.12%, versus priorstable90.94/90.63%; diagnostic only.
Audit PASS: all24CPUfieldreplays3.886e-15, all160000actualgeometrydraws match
global seed/counters, all512inputs visited268..369times, noheldout/futurepayload,
finiteoptimizer, preserved expectedinitialPDEfraction.2. Compact receipts:
confined-fourier-batch-audit.json, -batch-modal-error.json.

Fullstate continuation PID73970/session10408 now adds10000updates to20kTOTAL.
Parentfinal and childinitial model AND optimizer files are byte-identical by
complete direct reads, nohashing; receipt confined-fourier-batch-state-import.json.
All24physicalfieldimport gates pass beforeupdates;10kmetricspreciselymatch.
Generic history helper is exercised on real fullbank parent/child and plots
updates PLUS cumulative wall time. Use unit-linear B4 as the direct sampling
baseline (same outputgain1); stable B4gain.1 is a separate labeled control.

Input-only CPUcapacity preflight coverswidth64/96/128:243649/522145/905089
trainableweights; model+SOAPstate memory only~13.6/28.8/49.9MB. These exclude
bank/activations/allocator and are not a fullGPUmemory or accuracy benchmark.
No wider backbone has yet trained. Preserve wideFourier128space/64timechannels,
4heads/4conditionblocks/3decoder. Continue measured optimization before claiming
a hard architecturefloor or adding geometry-specific features.

Frozen10k per-geometry residual allocation (CPU session41984, nowCOMPLETE) uses
128positive midpoints plus0 on16monitors/8development domains, zero updates.
DerivedP from inputs only; residual formula is T*(phi_t+M*A*mu). Production
make_loss parity passes. PDEobjective vs physicalphierror correlations:
train Spearman.77647/logPearson.76972; dev.61905/.81429 (smallcohorts,
diagnostic association only). The highest-error developmentgeometry517 has
objective.001285 vs otherdomains' unequal values; the loss has not simply
become insensitive to all difficult fields. This does not prove a unique cause,
establish generalization or justify a new sampling rule by itself. Keep the
currently improving objective and finish20k before architecture/loss changes.
Script/receipt confined-fourier-residual-allocation.py and
confined-fourier-batch-residual-allocation.json. No evolved data enters PDE
inputs/time-JVP or optimizer updates. MainGPU continues independently.

Prepared width-only config train_operator_smallscale10_512_batch_wide.py:
96body/522145weights, othercoldB16settings identical. Input-only geometry512
initialphase/chemicalfield/mass preflight PASS (drift1.51e-17), no evolvedfields
or updates. No wider run queued. Receipt confined-fourier-wide-initial-preflight.json.

Prepared lower-LR config train_operator_smallscale10_512_batch_finetune.py:
parent=current complete20k (not yet finished), total25k, exactstateimport,
lr_init/min scale.3, globalcounter continues. Optionaldeclaredfactor acceptsONLY
this LRchange; absent retains strictbranch; unknownoptions/nonfinite/outofrange/
wrongscaledLR/otheroptimizerchanges fail. Still UNRUN, not queued. Factoryunchanged.
Tests: existing strictstate and actualtinytrainer paths PASS in61.496s combined
run; newLRtest's initial exact.3 expectation failed because Optax scalar schedule
usesfloat32 withint32counter even forfloat64weights. Independently measured
actualratio atcount3=.2999999767133556. Corrected expectation uses that actual
scalar ratio, not a loosened update tolerance. NewLRtest then PASS23.882s:
nonzeroupdates scale at1e-11rtol, alladaptivearrays/counters bit-identical,
serialization and negativeguards pass. No optimizer arithmetic was changed.
Fullaudit JSONmetadata now explicitly serializes nested continuation records;
checked before the first full20kcontinuation audit. Currenttrajectory remains
unchanged through its fixed20k budget; choose the nextcontrol from measured result.

## Incremental1024 input-only bank and continuing capacity control

Mixed-source hardlink/remapping/negativeguards + legacy native-copy tests3/3
PASS .131s (first negative-test expectedtext corrected; guardalreadyworked).
Canonicaltrain.py CPUprepare sixworkers completes129.565s:512newoperators,
496oldoperators+24oldnativefiles reused as hardlinks, no new PDE/FEMsolves.
Root /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1.
Logical12.926GB butonlynew~6GiB allocated; disk5.2GBfree. Fullinput/ref array
identity and physicalpartition audit in progress session25744; nmax/kmax and
GPUallocator estimate mustpass before future1024training. Not yetqueued.

Strictwide96 continuation20kTOTAL queued session21782 after RHS PID103030 exits.
Config train_operator_smallscale10_512_wide_continue.py; exact10000step model/
SOAPstate/globalcounters, noLRchange. Run
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-continue-01
raw /tmp/confined-wide-continue-train.txt, PIDfile /tmp/confined-wide-continue-pid.txt.
Launcher /tmp/confined-wide-continue-launch.py also sequentially performs full
native/61time/geometry/CPU24field/audit postprocess at fixed20k. No thirdGPUjob.

## Training-time status query

User asks duration per complete training. Measuredcold10k/512inputs:64body
2786.335s (46.44min),96body4981.730s (83.03min), includinginputload/derived
metric/JIT/evaluation. Warm219.25/428.82ms respectively. 1024cold64same10k
notstarted; estimate~50min allowinglargerbank startup, notanactualmeasurement.
Currentwidth96continuationPID114111 isfinite through10900of20000; estimated
remaining65-70min at measured428.82ms/update+evaluation. Next1024cold10kqueued
session81308 remainsintact. Combinedtrainingremainingabout2h; independentCPU
postprocessaudit/plots adds~10-20min, canpartlyoverlapGPUwork. Thesearebudget
completionestimates, not a guarantee of reaching10%error. Nojobs/settings/Git
changes. Memoryupdatedtask+PROGRESS; documentationdiffcheckPASS.
