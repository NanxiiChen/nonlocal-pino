# Confined CH width256 cold50k with online rolling geometry inputs

Date:2026-10-31. Status:COMPLETE. No Git history authorization.

## Goal and acceptance

User requests width256, online rolling inputs and a fresh50000-update run on
the replacement GPU. No model, SOAP state or schedule import. Keep FP32ordinary,
B16/Nt16, SOAPb1=.9/b2=.9/cap512, T3/fixed10smallscale modes, physical hardIC/mass,
eliminatedmu and stable resolvent residual. Reuse known input-only1024bank as
the initial pool; continuously generate fresh geometries without PDE/FEM labels.
SixCPUworkers/32replacements per100updates/two-prefetch batches/1536x128padding,
origin0. Width256/Fourier256channels/four64dim heads/four condition blocks and
three temporal decoder layers. Same frozen16historical/8reserved native monitors.

Acceptance: origin0 cold-online validation and recovery test; fresh initial
model/state receipt; first25actual GPUupdates finiteFP32/headroom; paired states
every5000steps and final independentCPU draw/input/state/field audit. One
production compilation, no throwaway full GPU compilation or default promotion.

## Current context and disk

GPU NVIDIA RTX6000D,85651MiB,compute capability12.0, initiallyidle. Previous
width192online runner/trainer are absent after host/GPU switch, stopped32900log/
32500eval: validation7.2507%/terminal7.7399%, last paired32000. It is archived by
user switch, not reported as complete. All online artifacts retained.

Initial diskfree4964708352bytes. User explicitly approved249-file cleanup
proposal in this turn. Deleted only redundant paired intermediates in three
inactive runs, freeing6166249472bytes; measured free11130957824bytes. All13
protected checkpoint groups, initial/final weights, logs, source/configuration
and data retained. Exact lists in confined-ch-width256-online-cleanup-proposal.json
and measured receipt confined-ch-width256-online-cleanup-receipt.json. No bank,
reference, environment, unrelated run or Git file deleted.

## Status / next action

COMPLETE. Fresh50k, independentCPUaudit, allcheckpoint/field checks and requested
physicalgallery delivered. Runner/trainer exited; GPUidle. Exact artifacts and
validation in the finalcompletion section below and compactresults JSON. No
blocker or requiredresume; no newexperiment queued, no defaultpromotion.

## Implementation and launch

Cold online origin0 is now explicit; positive origins without continuation fail.
Initial model and SOAP snapshots are both saved, with fresh zero-counter receipt.
Existing paired/complete-parent continuation semantics remain unchanged. New
config train_operator_online_width256_cold50k.py derives the checked1024known
input preset and sets width256/Fourier256channels, origin0 and checkpoint5000.
No seed field added to the preset; runtime seed42 is recorded for replay.

Focused five online unittest tests PASS71.222s, including canonical cold online
train/save/recover, frozen evaluation fields and old paired continuation. The
first pytest launcher found no installedpytest; standard unittest was used, with
no dependency installation. AST/whitespace checks PASS. Independent11training
function AST comparisons and four metric/precision/network/SOAP source byte
comparisons match the previous online source, excluding only the orchestration
function changed to record cold initialization. Old256architecture/physical/IC/
graph/time-sampling/optimizer config exact; bank/input feeding and I/O explicit.

CPU64actual input gate PASS9.0188s/7.096inputs/s; sixworkers,maxNx1360/degree93,
constantmode<=3.29e-14, both old mesh route comparisons and repeated draw0exact.
Fresh model/SOAP finiteFP32 and integer counters0. No newPDE/FEM trajectories.
JAX0.10.1 FP32GPU GEMM works on RTX6000D; no environment/software changes.

Detached runner9834/trainer9835 launched2026-10-31T07:21:25Z(15:21:25Asia/Shanghai).
Run /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-online-cold50k-01-train.
Journal /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-online-cold50k-01-process.json.
Training log /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-online-cold50k-01-training.log.
Runner docs/tasks/confined_ch_online_cold_runner.py. Launch, CPU and validation
receipts confined-ch-width256-online-{launch,cpu,validation}.json. Single fullstep
production compile; first25realupdates finiteFP32/headroom gate pending.
Initial loading reconstructs per-geometry stable residual metrics onCPU; GPU
allocator reservation is not active tensor usage. Do not report50kcompletion or
new accuracy yet. Automatic CPUaudit after50k writes
docs/tasks/confined-ch-width256-online-audit.json; cold initialization and actual
slot/time/generatedgeometry order are independently checked, then all24saved
fields replayed.

Next action: monitor ONLY this journal/owned9835trainer, verify fresh-zero receipt
and first25updates/online_preflight.json, then keep running to50000and finalaudit.
Do not start a second copy or resume previous192run. Blocker:none.

## Launch-session task-end / exact resume

Checked2026-10-31T07:28Z: setup/initial assessment328.224s, fresh-step0native
validation phi94.22%/terminal96.31% (untrained). Fresh zero-state production
receipt PASS; independent CPUaudit of actual initial snapshot reconstructs SOAP
exactly byte-for-byte, all floating leaves finiteFP32;3579905model parameters.
First32online inputs installed before update1, collect5.062s/GPUreplace0.161s.
Firstupdate compiling, no newtraining L2or25updatepass yet. RTX6000D CUDAbackend
JAX0.10.1 running; compilation generates sm120a kernels. Occasional CUDA
autotuner delay-kernel timing messages are in the raw log; process remains live.
Newhost cgroup has25CPUbudget/98784247808memory limit (not the old12CPUhost).
Do not conflate GPUallocator reservation with active memory.

Diff-level changes: online validation permits explicitcoldorigin0; training
orchestration saves freshinitialSOAP and zero-counter receipt; explicit cold256
preset/checkpoint5000; independent cold-compatible CPUgate/finalaudit and detached
runner; canonical cold/recovery regression. No physical residual, derivative,
network/optimizer kernel or existing numerical default changed. README and
DEVELOPMENT updated, artifact index/currentPROGRESS switched, DECISIONS appended;
old partial task archived.249-file approved cleanup exact list and byte receipt
recorded; no other artifacts removed or Git history operation.

Validation receipts: confined-ch-width256-online-{cpu,validation,launch,initial-
audit,live}.json and cleanup-proposal/cleanup-receipt.json. Five focused tests
PASS, sourceAST/byte parity, independent actualinitialstate, JSON/index/syntax/
single-active-task and git diff --check PASS.

Task remains ACTIVE because production50k/finalaudit continue in background.
Exact resume: read NEW journal above; check owned9835cmdline/status, read
run/online_preflight.json after25actualupdates, then latestcomplete metrics and
5kpairedmodel/SOAP/pool markers. Firstcompile currently pending; no duplicate
launch or secondGPUcompile. On successful50k runner automatically executes
independentCPU actualdraw/time/geometry/state/24field replay. On failure inspect
NEW runner/training log and preserve partial artifacts; never synthesize50k.
No blocker, default promotion or further cleanup authorization.

## Speed follow-up and task-end — 2026-10-31

No training/source/config change or newbenchmark. Actual25production-update
finiteFP32 gate PASS, warmGPUupdate99.426ms, peakactive20.4337GiB versus62.2867GiB
allocatorlimit. Full500..1000wall109.066ms/update including online refresh and
assessment, excluding startup; short interval has no5kcheckpoint yet. Old256/
fixed512bank500..40500=188.373ms, old192online25500..32500 about142ms. Practical
configuration comparisons, not isolatedGPUhardware benchmarks (256oldsmaller
padding/fixedbank,192oldnarrower model). New256is approximately1.73xfaster than
old256 and1.30xold192, despite wider model than the latter.

Latest complete evaluation and exact timing bounds are in
confined-ch-width256-online-speed.json; do not replace a current native score
with a final50kclaim. Initial1000phasevalidation28.7421%/terminal30.0979%;
training still early. Rolling refresh steady direct overhead~3ms/update; no
producer starvation diagnosis from collect/stack timing alone. Update portion
of50k estimated~91min at observed rate; startup/finalaudit extra.

Task-end: PROGRESS/task/live/validation/index and speedreceipt updated, no new
DECISIONS entry required for observational measurement. Documentation/JSON and
git diff --check validated; no tests rerun for status-only work. Exact resume:
keep existing9834runner/9835trainer through50k and automaticCPUaudit; next read
latestmetrics and firstcheckpoint5000. No duplicate process or scheme change.

Extended speed snapshot:500..1500=108.340ms/update, old256188.373ms (1.739x),
old192142.398ms (1.314x); latestlogged1800. Steady18refresh directoverhead3.184ms
perupdate. This supersedes the shorter500..1000window above. Exact live1500L2
and timing in speed/live receipts; unchanged training continues.

## Error-status follow-up — 2026-10-31T07:38:05.081245+00:00

Existingrunner9834/trainer9835verified live/correctcommand; latestlogged3400,
completeevaluation3000. Nativephi historical16monitor20.5250%/
terminal21.4804%; reserved8validation21.3947%/terminal
22.7692%. Mu validation30.1204%/terminal39.7396%.
Validationphi range9.27..39.68%, allcases retained.
Same native eight reused development references, mean of per-geometry weighted
L2 ratios, not a new larger test cohort or localFEM result. Initial early
phi500=33.9004%,1000=28.7421%,2000=23.4719%,3000=21.3947%; still decreasing.
All aggregate means independently reconstructed from8/16percase scores.
No source/config/process/default/Git change; existing50kcontinues.
Task-end: PROGRESS/task/live/index updated; JSON/numeric/diffcheck PASS.
Exactresume: sameNEWjournal; next500step nativeL2 and first5000pairedsnapshot,
then50k/finalCPUaudit. No extra tests or benchmark for observational update.

## Five-version comparison — 2026-10-31

Common cold-budget5000 nativephase validation/terminal: current256rolling
18.2051/19.4672%,192fixed51217.7599/18.6856%,192fixed102417.2262/18.3847%,
256fixed51216.9734/18.2018%. No early accuracy advantage, +1.232percentagepoints
versus old256fixed512. Initialsame8physical reference arrays (points/volumes/
holes/time/phi/mu) verified exactly across relabeled libraries; all aggregate
means reconstructed through every historical eval. Old192online startsat25k,
so no fabricated cold5krow. Latestoldendpoints:192fixed51230k8.9148/9.8017%,
192fixed102426.5k7.9115/8.6213%,256fixed51240.5k8.5276/9.3328%,192rolling32.5k
7.2507/7.7399%. Unequalbudget/devcohort/hardware/padding/library limitations
explicit; no isolated architecture/online causalclaim.

Researchscript docs/tasks/confined_ch_online_comparison.py freezes full complete
metrics/rawpoints, plots early common budget and fullobserved validation curves
(no smoothing/extrapolation,192rolling only25k+). Plot visuallyinspected.
External output run/version_comparison_01/{comparison.png,comparison.pdf,
comparison.json,*-metrics.jsonl}; compact receipt confined-ch-width256-online-
comparison.json. No images inGit and no overwrite of priorcurves.

First actual checkpoint5000marker/model/SOAP/pool allpresent,50refreshes/1600
newinputs; full1024poolidentities reconstructedexactly, originalpool replaced.
Status/live/index updated to latestcomplete snapshot by this session.
Task-end: script executed successfully, reference/aggregate/marker/pool/image
checksPASS; AST/JSON/git diff --checkPASS. No numerical/runtime/default/Git
change, no newbenchmark or testsuite. Memory changed: PROGRESS and this task;
compactlive/comparisonreceipt/artifactindex updated. Next: same9834/9835run
through50000with5kpairedstates and automaticfinalCPUaudit. No duplicate launch.

## Updated same-budget curves — 2026-10-31T07:49:02.415482+00:00

Frozen common8500 whole/terminalphasevalidation: current256rolling14.2692/
15.1579%,192fixed51213.8139/14.8612%,192fixed102413.7297/14.7602%,
256fixed51213.9877/15.0110%. Gap to old256 narrows5k1.2317pp->8.5k0.2815pp,
current still slightly higher. Current8.5k historical16monitor13.9326% versus
validation14.2692%; old25610.7890% versus13.9877%. These are frozen16historical
monitor domains, not an error estimate over the active rollingpool. No causal
generalization/precision/GPU claim from this single reused8developmentcohort.
Old192online starts25k; before that it is the192fixed1024control, so no invented
independent8.5k-online row. Existing50k unchanged.

Comparison script adds matched_budget.png/pdf with two panels for pooled and
terminalvalidation, allfourcoldcurves clipped identically500..8500. Also updates
early/fullhistoryfigure comparison.png/pdf. Raw full histories andcomparison.json
frozen in run/version_comparison_02; v01 retained. No smoothing or extrapolation.
Same8reference arrays/physical config/allhistorical pergeometry aggregate means
verified, clippedsteps exactly500increments, bothfigures visuallyinspected.
Receipt docs/tasks/confined-ch-width256-online-comparison-02.json.

Latest live refresh recorded9400logged/9000evaluated: nativephase
validation13.7100%/terminal14.5173%; live receipt has exactperfieldscope.
Task-end: researchplot script, PROGRESS/task/live/index andcompactcomparison-02
receipt updated. Plotexecution/reference/aggregate/clipping/image/AST/JSON/
git diff --checkPASS. No productioncode, scheme, default, optimizer, process or
Git history change; no testsuite for observationalplot update.
Next: existing9834runner/9835trainer through50k/finalCPUaudit, next10kpaired
state and latestnativeL2. No duplicate launch.

##12k status and matched convergence — 2026-10-31T07:55:00.763714+00:00

Frozen12000validationphi pooled/terminal: current256rolling11.2847/11.9639%,
192fixed51211.9859/12.9625%,192fixed102412.0352/12.9502%,256fixed51212.0952/
13.0870%. Current snapshot lower than allthreefixedcoldcontrols on bothmetrics;
relative reductions versusold2566.7014%/8.5820%. Single seed/reused8native
development geometries, no isolated sampler/capacity/hardware causality or stable
finalgain claimed. Current historical16monitor11.6126%, a frozen olddomainset.

Existingcomparison script executed successfully to version_comparison_03; full
rawhistories/JSON and matchedpooled/terminalPNG/PDF retained externally. Matched
figure visuallyinspected; allreference arrays andaggregatemeans checked. Receipt
confined-ch-width256-online-comparison-03.json; no productioncode/configchange.
Actual10kmarker/model/SOAP/pool allpresent,100refreshes/3200newinputs,1024pool
identities reconstructedexactly. No newweightload orGPUcompile.

Latestlive atrecording12600logged/12500evaluated: nativephi
10.9409%/terminal11.5615%. Existing9834/9835verifiedlivecorrectcommands.
Task-end: PROGRESS/task/live/index/comparison-03 updated. Scriptexecution,
reference/aggregate/image/paired-pool/JSON/git diff --checkPASS. No testsuite
for observational update, no default/process/Git change. Next: preserve50k
budget, monitor latestnativeL2 and15kpairedstate, then automaticfinalCPUaudit.

## Status follow-up — 2026-10-31T08:04:21.059544+00:00

Existing9834/9835 verified live. Latestlogged17700, complete17500:
phasevalidation9.8172%/terminal10.3794%; historical16monitor
9.9006%. Same8native development references, aggregate means reconstructed.
Matchedbudget controls pooled/terminal: 192fixed512 10.16/11.03%; 192fixed1024 9.61/10.40%; 256fixed512 10.14/11.04%.
At17.5kcurrentpooled lower than512controls;192fixed1024pooled9.6123%is
slightlylower thancurrent9.8172%. Terminalcurrent10.3794%versus10.4011%is
essentiallyclose. No causal or stableaccuracy claim. Recentrawhistory (ratios): [(15000, 0.1029791815996883, 0.10880125776230701), (15500, 0.09918261594103053, 0.10526849379632477), (16000, 0.0983754133192869, 0.10396390090883975), (16500, 0.10224101190359418, 0.10907771950581475), (17000, 0.09839131144676176, 0.10481031423878012), (17500, 0.09817197097580463, 0.10379435647396437)]. Preserve16.5krebound.

15kmarker/model/SOAP/pool complete;150refreshes/4800generatedinputs and all1024
active identities independently verified. No additionalweightload/GPUcompile or
benchmark, default/source/config/process/Git change.
Task-end: PROGRESS/task/live/index updated; aggregate/exactmatchedbudget/paired
pool/JSON/git diff --checkPASS. No tests for status-only update. Continue existing
50k and finalCPUaudit; next20kpairedstate/latestnativeL2, no duplicate launch.

## Convergence-plot refresh — 2026-10-31T08:07:10.959321+00:00

User requests latestcurves. Existingcomparison script rerun successfully to
run/version_comparison_04, freezing rawcomplete records. Fourcoldcurves clipped
at common18500; pooled andterminal panels, no smoothing/extrapolation. Same8
physicalnative refs/allaggregate means/exact500stepordering checked; matched
figure visuallyinspected. PNG/PDF external, compactcomparison-04.jsonreceipt.

18.5kphasevalidation pooled/terminal: current256rolling9.3698/9.9838%,192fixed512
10.3428/11.2757%,192fixed10249.3092/10.0494%,256fixed51210.1566/11.0461%. Current
veryclose to192fixed1024, better than512controls; bothcurrent meansbelow10% at
thissnapshot, no stableaccuracy/monotonicconvergence claim.

Latestlive19200logged/19000complete: phi9.1235%/terminal
9.7048%; sameexisting50kprocesses unchanged.
Task-end: PROGRESS/task/live/index/comparison-04 updated. Plotexecution/reference/
aggregate/clipping/image/JSON/git diff --checkPASS; no source/config/process/
default/Git change or testsuite for plot-only work. Next: same50kandfinalCPUaudit,
monitor nativeL2/20kpairedcheckpoint; no duplicate launch.

## Parallel-experiment resource assessment — 2026-10-31T08:16:05.558798+00:00

User asks peakVRAM and whether otherexperiments can run while this continues.
Existingdetachedtraining left unchanged. Savedstep25JAXinternalactivepeak20.4337
GiB, activeatstep25~12.19GiB, pool/limit62.2867GiB. These savedstats are NOT live
all-training peak; no intermediateexport exists beyondstep25, finalsummary
willrecordupdatedlifetimepeak. No debugger/injection/restart added tofetchit.
CurrentNVMLGPUused63.681GiB/free19.369GiB of
83.644GiB; used includesreservedpoolandCUDAcontext. Unused
JAXpoolremainsowned bycurrentprocess and is not automaticallyfree for another.
GPUutilizationsnapshot97%, no timingbenchmark.
Containerhostmemory47.10/92.00GiB, CPUquota25cores;
trainerRSS30279304 kB, ownhostHWM73617752 kB; diskfree9.88GiB.

SmallCPUanalysis/plotting and smallGPUtrials feasible withinactualfreebudget.
For anewJAXprocess propose XLA_PYTHON_CLIENT_PREALLOCATE=false and
XLA_CLIENT_MEM_FRACTION=0.15 (~12.55GiBpool, contextoverheadextra). Installed
jaxlib optionsparser verifies both inCPU-onlyquery; noGPUallocation/newexperiment
launched. GPUcontentionchangeswalltiming, so exclusiveGPUperformancecontrols
shouldwait. Avoid asecondlargegeometrybank load within92GiBhostcap.
AGENTS.md forbids FEniCSMPI/JAXGPU concurrently; pureCPUanalysis is distinct.
Do not alter existingallocator, samplepool or50kbudget.

Latestlive24000logged/24000evaluated, nativephi7.5535%/terminal
8.0566%. ResourceJSON/live/index/PROGRESS/task updated.
Task-end: NVML/cgroup/source/installedoptionsparser/JSON/numeric/diffchecksPASS;
no source/runtime/config/Git change or testsuite. Next: existing50k/finalCPUaudit
continues, monitorlatestL2and25kpairedstate; no newexperiment started.

## Error-status and25k backup — 2026-10-31T08:19:22.873307+00:00

Existing9834/9835 verifiedlive. Latestlogged25800, complete25500: nativephi
validation6.8901%/terminal7.3507%, historical16monitor
7.2576%. Same8reusednative devreferences; percaseaggregates verified.
Exact samebudget results: 192fixed512=9.2304/10.1197%; 192fixed1024=8.0659/8.7835%; 256fixed512=9.0238/9.8594%; 192rolling_from25k=8.5972/9.3325%.
Current continues to improve; no stability/multiseed or causal samplerclaim.

25kmarker/model/SOAP/pool complete;250refreshes/8000newinputidentities, full1024
activepool reconstruction exact. All previoussnapshots retained.
Plot-helper followup: existingcomparison script hardcodeslen(common)==4; at25k
old192online adds a fifth matchingrow. Before nextplot adapt thisexplicitcohort
validation and report runavailability forlaterbudgets; do not drop/interpolate
controls silently. No numericaltraining issue and no plotrerun inthissession.

Task-end: PROGRESS/task/live/index updated; JSON/aggregate/exactbudget/paired
pool/git diff --checkPASS. No source/config/default/process/Git change or
testsuite. Next: same50kandfinalCPUaudit, monitor latestnativeL2/30kpairedstate.

## Post25k convergence refresh — 2026-10-31T08:23:46.875499+00:00

User requests latestcurves. Researchhelper scores_at_step now uses explicit
perrun first/lastcoverage; exactlyone assessment required withininterval,
unavailableoutside reported. Removes hardcodedfourcoldrows, supports fifth25k
continuationstage and fixed1024stopped26.5k without fabrication. Currentendpoint
required. Plotlegend labels endedhistories, same-budgetcurves clipped tocurrent
step, eachobservedstart/end retained. Adds last8kzoom matched_tail.png/pdf.
No production/network/optimizer/physics code or source of activeprocess changed.
Decision appended immediately toDECISIONS.md.

Actualcomparison passes at27000: phasevalidation pooled/terminalcurrent6.7602/
7.2230%,192fixed5128.9759/9.8488%,256fixed5128.9787/9.8194%,192rolling8.5633/
9.2208%. Fixed1024last26.5k7.9115/8.6213% displayed at actualend; no27kvalue.
Same8native devreferences/allhistoricalmeans checked. Fifth online stage starts
25k (its priorstage is fixed1024); no seed/stability/isolatedsamplerclaim.
Matchedfull and19k..27kzoom visuallyinspected; smallrecentrebounds retained.
Outputversion_comparison_05 PNG/PDF/JSON/rawhistories external; compact
confined-ch-width256-online-comparison-05.json. v01..04 retained, no mediaGit.

Focusedrealhistory coverage checks at24k/25k/27k/41kPASS; missinginterior25001
raises rather than silently omitting/interpolating. Researchscript AST, JSON,
reference/aggregate/coverage/figureexecution/image/git diff --checkPASS.
Task-end: script/PROGRESS/DECISIONS/task/live/index/comparison-05 updated;
no testsuite/GPUbenchmark for plot-only work.
Latestlive28200logged/28000evaluated: nativephase6.4713%/terminal
6.9384%. Next: same9834/9835runthrough50k/finalCPUaudit, monitor
latestnativeL2/30kpairedstate, no duplicate launch or numericaldefaultchange.

## Later-training result follow-up — 2026-10-31T08:41:28.182643+00:00

Existing9834/9835live with correctcommands. Updatedcomparison script runs to
version_comparison_06, common37000 onlycurrent256online andold256fixed512 have
exactrecords:5.5436/5.9301% versus8.5813/9.3808%, relative reductions35.3991/
36.7845%. Othercontrols ended26.5/30/32.5k, retainedatobservedends; no invented
37krow. Same8native development references/allhistoricalmeans checked. Recent
29k..37kzoom visuallyinspected, small31.5/35.5/37krebounds retained. No seed/
stability or isolated samplingcausality claim.

35kmarker/model/SOAP/pool complete;350refreshes/11200generatedinputs, all1024
activepoolidentities exact. Scope user requests currentresults; existing50k
continues, no model/config/process change.

Latestlive37800logged/37500evaluated: phi5.3081%/terminal
5.6825%; historical16monitor5.6017%. Samebudget:256 / fixed512=8.4842/9.2857%; 256 / rolling from0 (current)=5.3081/5.6825%.
Last5kfullwall0.109512s/update; remainingupdateestimate22.8min, finalauditextra.
Task-end: PROGRESS/task/live/index/comparison-06 updated. Plotexecution/reference/
aggregate/image/35kpairedpool/JSON/git diff --checkPASS; no testsuite orGPUprobe
for observationalupdate. Next: same50kandautomaticCPUaudit, next40kpairedstate;
no duplicate launch, numericaldefault orGit change.

## Near-budget status — 2026-10-31T08:53:56.197801+00:00

Existing9834/9835verifiedlive. Latestlogged44600, complete44500:
nativephi validation5.4573%/terminal5.8519%, historical16monitor
5.2163%. Last5000nativephi range5.1069..5.4573%; decline
has slowed and oscillations retained. Lowest recordedmean5.1026%
at38500 is a metricrecord, not a saved/best-selected model claim.

Alloldcontrols stopbeforecurrentstep. Lastshared40.5k current5.2263/5.6297%
versusold2568.5276/9.3328%; no synthetic43k+oldvalues.
40kmarker/model/SOAP/pool complete,400refreshes/12800newinputs and1024active
identities exact. Last5kfullwall0.109438s/update, remainingupdateestimate
10.0min, finalCPUauditextra. Summary/auditnotyetproduced,50knotcomplete.

Task-end: PROGRESS/task/live/index updated; aggregates/exactoldsharedbudget/
40kpairedpool/JSON/git diff --checkPASS. No source/config/default/process/Git
change or testsuite. Next: preserve50kbudget, verify45kpairedstate, thenfinal
summary and automaticCPUdraw/state/24fieldaudit. No duplicate launch.

## Training complete, gallery in progress — 2026-10-31T09:06UTC

Runnercomplete09:04:42UTC, no runner/trainerprocess, GPUidle0MiB.50000newupdates
complete; finalnativephase validation4.8329%/terminal5.1935%, historical16monitor
5.0529%; mu validation10.7329%/terminal14.2757%. Wall6135.630s(102.26min),
firstupdate283.280s/setup328.227s; activeGPUlifetimepeak20.4337GiB. CPUauditPASS:
24fieldreplays phi<=7.1526e-7/mu<=3.6583e-6, allactualslot/time/geometry IDs and
freshinitialSOAP/finitefinalstate/final1024poolidentities verified.16000generated
inputs,15994sampled; no newtrainingPDE/FEMsolve. Userrequestsphysicalfields;
keep taskACTIVE until gallery validation/inspection and finalarchive complete.

## Final completion and task-end — 2026-10-31

50000freshupdates complete; runnercomplete09:04:42UTC(17:04:42Asia/Shanghai),
independentCPUauditPASS, allprocesses exited/GPUidle. Finalfixedbudgetphi
validation4.8329%/terminal5.1935%, historical16monitor5.0529%/terminal5.4186%;
muvalidation10.7329%/terminal14.2757%. Cohortphi range1.7241..9.5686%, terminal
1.6130..10.5862%; one individualterminalcase remainsabove10%. These are means
ofpergeometry physicalarea-weighted L2 onfive storedtimes, reused8native
development references, not localFEM/freshblindtest or multiseed result.
Trainingwall102.2605min, setup328.227s, firstupdate283.280s, steadyGPUmean96.877ms.
LifetimeactiveGPUpeak20.4337GiB nowconfirmedthrough50k, ratherthan onlyearly
preflight.16000inputgeometries installed,15994actuallyused;500refreshes, fixed1024
pool, no newtrainingPDE/FEMsolves or futurelabels.

Finalaudit validates actual800000slot/geometrydraws, alltimes/statelessholes/
freshinitialSOAP/finitefinalmodelandstate/finalboundedpool.24CPUfieldreplays
maxphi7.1526e-7/mu3.6583e-6. Independent savedfield massdrift and exactall10paired
checkpoints in compactresultsreceipt; finalmodel/SOAPbyte-equal50kpair. No lower
49.5kdevelopmentmetric model selected. Fullrawconvergence frozen in
version_comparison_final with trueoldendpoints; no invented50kcontrols.

Physicalgallery: lowerror1027(1.7241/1.6130%), uppermedian1028(4.6463/4.6794%),
maximumerror1026(9.5686/10.5862%). Three3x5phasefigures at0/.75/1.5/2.25/3 with
reference/prediction/absoluteerror, sharedscales acrossall8cases, zero-phase
contours, originalmesharea-weighted continuousdisplay andanalyticholemasks. All8
rawreference/geometries/metrics/FP32ICrounding verified; initialmaxgap1.275e-8.
All3figures visuallyinspected. Outputs final_fields_02/{fields_1027,fields_1028,
fields_1026}.{png,pdf} and gallery.json, externalonly. Source researchhelper
docs/tasks/confined_ch_final_gallery.py. Firstattempt rendered final_fields_01
thenJSONfailedon NumPyfloat32tolerance; explicitPythonfloatfix and new02rerunPASS.
Partial01retained, no additionalcleanup or hiddenfallback.

Task-end diff: newresearchgallery script; finalresults/live/validation/index and
PROGRESS/DECISIONS/taskcompletion records. No training/network/optimizer/PDE/
reference/default change, environment install orGit history operation. Real
gallery/convergenceexecution, finalaudit/24replays, rawscore/reference/IC/display
bounds/pairedbytes/pool/mass/image/AST/JSON/git diff --checkPASS. No unnecessary
unchangedexpensivetests rerun. Summary/audit and compactresults directpaths:
run/summary.json, docs/tasks/confined-ch-width256-online-audit.json,
docs/tasks/confined-ch-width256-online-results.json. Alllargeartifacts external.

Acceptance complete. Archive this task, clearactivePROGRESS. No requiredresume,
queuedtraining orautomaticnewexperiment. Defaults remain aspreviouslydeclared;
any furthercohort, localFEM orlongertraining is a separate user-directed task.
