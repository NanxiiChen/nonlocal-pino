# Channel time derivatives and coupled phase accuracy — completed

## Completion — 2026-11-02

The requested physical time JVP/fresh random-time scheme and the nominal
geometry's droplet-shape optimization are complete. No owner/child or GPU job
remains; no later experiment is queued. No default/SOTA promotion, Git/history
operation, cleanup, new FEM solve, evolved training labels or probe-weight
initialization was performed. This is a finite nominal-development validation,
not a multi-seed or unseen-geometry accuracy claim.

Final opt-in configuration:
`examples/channel_nsch2d/configs/train_operator_width128_gmls_time40_jvp_chemical_online.py`.
Canonical entry remains `examples/channel_nsch2d/train.py`. Physical time uses
JVP;40 continuous stratified times are drawn afresh every update and shared
across the two conditions. Each geometry retains its fixed spatial queries;
spatial PDE/BC gradients and Laplacians use degree3/k36 GMLS. Streamfunction
velocity remains the original AD curl. Eight-time memory blocks share the same
parameters and contribute to one clipped SOAP update; these are not time windows.

Only the chemical PDE-row normalizer changes from4 to1 relative to the completed
uniformJVP control. Physical mu output amplitude4 and boundary normalizer4 are
unchanged. This multiplies only chemical MSE by16 and preserves the other eight
terms, physical model/IC/BC, architecture, SOAP, geometry/time streams and budget.
No tanh map, field mask, projection or reference labels are used by this control.

Matched cold5000 results against independent local FEM, one nominal geometry:

| Scheme | Phase pooled L2 | Terminal phase L2 | Velocity L2 | Pressure L2 |
|---|---:|---:|---:|---:|
| UniformBDF40 |5.3677%|7.1787%|4.4834%|9.2103%|
| UniformJVP40 |4.9726%|6.8860%|4.2190%|8.8069%|
| UniformJVP40, chemical row1 |1.1498%|1.2157%|2.7100%|4.1186%|

Terminal area-weighted minority-fraction L2 improves19.9952% ->3.5462%,
arc-weighted mean zero-contour distance10.3223%R0 ->1.7140%R0, IoU80.6076% ->
96.5400%. Across13 saved snapshots minority L2<=4.5595% and contour<=2.1312%R0;
minimum binary IoU92.2226%. Five contour and physical-field panels were visually
inspected: the formerly substantial late flattening is greatly reduced and the
predicted boundary closely follows FEM. Minority L2 rescales phase error and is
not independent shape evidence; actual contours/IoU provide that evidence.

Independent cold/source/input proof and validation:

- Original model and complete initial SOAP state bytes exactly match control;
  initial/final input banks and ALL actual time/condition/geometry arrays match.
  Actual5000 global-key draws,40 continuous strata/update and99 rolling refreshes
  (198 installed geometries) replay successfully.89 actual source/owner byte
  checks passed at launch; no duplicate run or probe-fitted weights imported.
- Training/validate_run/diagnose_phase/droplet_shape/phi_gallery and independent
  continuous-time diagnostics all exited0. Saved-field gap2.8610e-6,
  timeJVP primal gap7.1526e-7, flux error1.9477e-7, wall speed9.3909e-8,
  physical divergence3.7253e-9 pass unchanged numerical gates.
- Full15633-point residual diagnostics at240 independent continuous times on
  nominal and final-pool geometry completed. These are finite canonical residual
  diagnostics, not an exact GMLS mass identity or continuum certificate.
- Actual cold wall1911.763s; warm update0.287009s; active peak8.1018GiB. JAX's
  reserved pool is a different quantity. Geometry-producer total wait0.604s.

Diagnostics support coupled-loss scaling as a useful cause to address, rather
than the time finite difference as the dominant explanation. Raw terminal
constitutive mismatch RMS0.09934 ->0.03147 (53.39% ->16.91% of native-reference
mu RMS); interface ratio56.25% ->17.17%. On the SAME independent240 nominal
probe times, physical chemical RMS0.09376 ->0.02452, while physical CH RMS
0.003629 ->0.004720: not every residual improves together, despite much better
fields. Never compare differently normalized total losses as accuracy.
Auxiliary pooled FEM mu L2 remains16.7885%; terminal area-weighted transverse
velocity L2 improves70.74% ->49.57% (absolute RMS0.005759 ->0.004035). Global
velocity error conceals this small transverse component. These limits are
retained, not presented as a fully converged continuum coupled solution.

Current receipt: [channel-chemical-scale-validation.json](../channel-chemical-scale-validation.json).
Direct run and owner journal paths, full metrics, all stages and diagnostics
are recorded there. Final figures remain outsideGit:

- `/root/autodl-tmp/data/channel_nsch2d/chemical-scale-curve-20261102-03/convergence.png`
- `/root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-chemical-online-width128-01/droplet-shape/contours.png`
- Same run's `phi-fields/phi-fields.png`.

Session-end diff: one read-only convergence/rawchemicalRMS plotting helper;
completed existing finite owner and inspected saved physical fields; read-only
chemical/error-region diagnostics; updated receipt/artifact/decision/dashboard
and stable README links. No production numerical code changed in this final
continuation. New helper AST/JSON/link checks and git diff --check pass; prior
3new/3legacy focused tests remain valid and were not repeated unchanged.

## Historical investigation record

All active/live/pending states below are retained history. The terminal state
and exact completed scope above supersede them; no old queue should resume.


## Goal and acceptance

User explicitly enabled ACTIVE ongoinggoal: “请按照这样的思路持续解决优化，解决液滴形状与参考解不匹配的问题”.
get_goal confirmedactive/noexplicitbudget. Continue finite evidence-driven
optimization beginning with exacttimeJVP and freshcontinuous resampling; do not
mark complete merely from global<10% or a launchedrun. Localizedminority and
contour/physicalshape agreement are required. Oldqueues/FEMschedules remainoff.
No Git/default/cleanup authorization is inferred from the goal.


Latest steering: user requests physicaltimeJVP and freshcontinuous randomtimes
everyupdate. The matched adaptivecausal40-time cold5000 test is COMPLETE with
ALLfiniteaudits: physicaltimeJVP works, exactactualresampling passes, warmtime
improves .5303->.3378s but phase/shape accuracy is similar to adaptiveBDF.
This evidence does not support time-difference error as the main shape cause.
Current broadgoal remains active; see the authoritative continuation below.

CompletedadaptiveBDF5000 and allfiniteaudits: nominalFEM phi6.7303%,terminal
8.8861%,u4.4399%,p8.2753%,terminalminorityq25.4029%,contourmean12.9806%R0.
Uniform40control5k is better for phase/shape(phi5.3677%,terminal7.1787%,
q20.8196%,contour10.7787%R0), slightly worse for velocity/pressure. Preserveboth;
no newdefault/SOTA promotion. The fixedstageschedule failed earlier and remains
retained, not restarted. Reference/operator checks have passed within declared
finite space/time gates; they do not prove continuumaccuracy or operator
population generalization. Physicalfields/contour comparison is generated and
inspected, confirming the lateflattened droplet mismatch is still unresolved.

Preceding training authorization (fulfilled):
Latest user explicitly authorizes a matched cold20k GMLS-first residual PINO
control; previous training is complete, CPU reference refinement retained.
Preserve the physical streamfunction/IC/BC/optimizer/sampler; replace only
residual first derivatives and conservative CH flux, as recorded below.

Preceding scope additionally includes a nominal-geometry direct
CPU nonlocal NS--CH numerical baseline, requested to separate FEM/nonlocal
discretization from network error. The online20k pilot and finite audits are
now complete; preserve its final files. CPU reference refinement remains live.

Preceding scope: launch a NEW finite cold20k geometry-online
rolling NSCH run with the present original-residual width128/linear-log scheme.
Use durable background ownership, verify real optimizer updates and rolling
replacement, preserve finite post-run field/input audits. Earlier bounded
shape/reference diagnosis is complete but shape bias remains unresolved. Old
automatic goal/conditional schedulers/FEM queues remain cleared; this is not an
authorization to restart those queues or promote a default.

Answer how the current channel_nsch2d pilot differentiates each physical field
and why it did not reuse the established regional nonlocal operator. Distinguish
the temporary AD/GMLS implementation choice from a mathematical restriction.
Explain future nonlocal adaptation requirements without changing PDEs, BCs,
defaults or checkpoints just for the explanatory question. Subsequent user
explicitly authorizes a comparison: direct full NS--CH with GradNorm versus
streamfunction. Run matched direct/streamfunction GradNorm controls, retaining
the earlier uniform-weight streamfunction run as reference. Keep the same
derivative scheme to isolate this representation/weighting question.

Historical broader acceptance (automatic goal remains cleared): relative L2 below10% for the coupled channel on a
single geometry, then smooth-geometry operator generalization using
background online generation and rolling pool replacement. The existing16bank
controls are not evidence that either criterion is achieved. No evolved field
labels enter training.
Track phase/droplet integrity, but do not declare the overall goal solved
from phase alone: report phase, velocity and pressure separately and require
the principal physical fields to meet the target before completion. Chemical
potential is an auxiliary field reported separately; its FEM refinement error
is currently outside the independent phi/velocity/pressure acceptance gate.

Latest user explicitly accepts pressure near10% if necessary. Communicated
provisional working limit pressure<=11%, phi/terminalphi/velocity each<10%.
Retain and report the strict<10%flag separately; exact L2 and finite-cohort scope
remain unchanged. This user-directed exception permits online work from the
already independently audited15k pressure10.5979%baseline if needed.

Earlier steering rejected region balancing as the preferred route, requesting
the actual output-coordinate/Jacobian residual transform used in melting.
Isolate map-only versus map+Jacobian before changing other schemes; report the
physical phase/droplet/flow and L2, including any stiffness/range limitation.

## Context

The first coupled pilot is complete but phase accuracy failed. User questions
the move away from the project's nonlocal differentiation objective. Current
network/loss/operator/config and last five decisions were inspected; prior
fully-nonlocal feasibility record distinguishes stable operator pairs from
naive gradient substitutions. Existing PROGRESS had no active task.

## Status / blocker / next action

### Current continuation — independent chemical normalizer cold5000 LIVE

Previousgoalturn PROGRESS: completeduniformJVP/fullaudits and identifiedlocalized
interface/uy/chemicalmismatch. Currentturn PROGRESS: implementedsingleloss-only
control, passedcontracts/resourcequalification and launchedcanonicalcold5000.
GoalACTIVE/unachieved, no blocked/complete claim.

SoleLIVEowner358835/training358893, screenchannel-chemical-jvp40-20261102.
Run /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-chemical-online-width128-01;
journal /root/autodl-tmp/data/channel_nsch2d/20261102-5090-gmls-time40-jvp-chemical-online-width128-01-process.json.
Actualps/screen/journal confirmLIVE. Coldmodel/fullSOAP/initialgeometrybank
EXACTparityPASS atstep0;89savedactualsource/owner bytechecksPASS andallinitial
physicalmetrics EXACTLYmatchcompleteduniformJVP control. Actual optimizer updates observed through logged2000, with rolling refresh.
Latest physical monitor step2000: phi2.645457%, terminal3.411534%,
velocity3.942695%, pressure7.149422%, mu28.773461%.
Same-step control1500: phi6.785216%, terminal10.064399%, velocity5.554422%,
pressure21.562424%, mu43.245714%. Partial nominal-development-geometry evidence;
final saved physical fields/contours and all input/time audits remain pending.
No restart from quiet logs. Read-only physical-curve helper
channel_chemical_scale_curve.py plots actual monitors and de-normalized sampled
chemical RMS only; external chemical-scale-curve-20261102-02/convergence.png.

Preset examples/channel_nsch2d/configs/train_operator_width128_gmls_time40_jvp_chemical_online.py.
Explicitpositive loss.chemical_residual_scale1 overridesONLYchemicalPDErow;
legacychemical_scale4 stillphysicalmuOUTPUTamplitude andabsentfieldrowdefault.
ChemicalMSE16x, other8terms/BCs/architecture/SOAP/GMLS/geometry/IC/ref/Nt40/
eightblocks/5000budget unchanged. No newequation/term/controller/mask/labels/
timewindow/projection. Rejectunusednormalizerwheneverchemicalroweliminated.
Summary/continuous/frozenchemicalRMSdiagnostic recordoutputandresidualroles.
Strictcomparison allowsONLYnewfield/runname versusoriginalsharedscale control.
Sourcecontrol /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-uniform-online-width128-01.

3newcontracts PASS70.769s: coldparameter/primal/other8parts bitwiseparity,
chemicalMSE16x/allparametergradientdecomposition/FD/global40vs8accumulation;
positive/unused/singlefield config; actualsmalloriginal+scaledcoldtraining/
replay/allinputs/timekeys/scalemetadata. Explicitfixturebandwidth.25inBOTH
stages; production/gatesunchanged. 3olduniformJVPregressions PASS54.527s.
8AST/actualproductioncontrolsinglefieldconfig/gitdiff--check PASS; testprocesses
reaped. Subsequent paired-report key rename actual_times_and_geometry_exact was
METADATAONLY, avoiding a false continuous-only claim for genericBDFuse.
No unchangedexpensivechecks repeated; no numericfield/source test invalidated.

Production20resourceprobe COMPLETE/EXIT0:compile339.770223s,warm.28534916s,
activepeak8534905344bytes=7.948GiB,all20gradientsfinite,all5000plannedstrata/
order/20actualglobal-time-key draws and11actualsourcebytes PASS beforemetadata
keyrename. Exactsourcecopies retained externally; no fittedprobeweights imported.
Probe /root/autodl-tmp/data/channel_nsch2d/chemical-scale-production-probe-20261102-01.
Actualreference/construction gates pass; GPUfree and35GiBdiskfreeatlaunch.

Exactresume: followONLYexisting358835/358893 throughremainingfixed5000
andallownedaudits (physicalreplay/hardflow/input/
actualalltimekeys/rollinggeometries/phasebalance/dropletshape/gallery/pairedcold
state/240freshcontinuoustimefullspace). Cold/source/step0proofs alreadycomplete.
Comparephysicalerrors/uy and rawchemicalRMS, NOTnormalizedloss totals across
scales. Preserveallnumericalgates; no fittedprobeweights/duplicate/retryfrom
observationtimeout/newFEM/default/SOTA/Git/cleanup/laterqueue. Currentreceipt
channel-chemical-scale-validation.json. Goalstillactualdropletshape agreement.

Sessiondiff: problem/loss/train/causal/diagnose_time support independentpositive
normalizer+scalemetadata; researchpreset and3focusedtests; diagnosticrowmetadata
usesactualnormalizer; README/DEVELOPMENT/PROGRESS/DECISIONS/task/receipt/index.
No physicalnetwork/output/PDE/BC/optimizer change or legacyarithmetic change.

### Completed uniform JVP and supporting diagnosis (prior continuation)

### Prior continuation — uniform JVP complete; loss-only normalization selected

Previousgoalturn classifiedPROGRESS (qualified/startedactual5000). Currentturn
PROGRESS: completed alltraining/finiteaudits, exactfourwaystreamproof, physical
fields/shape/convergenceinspection and errorlocalization/scalingdiagnoses.
No liveGPU work/laterqueue; owner314769 and allchildren EXITED afterexit0.
Run /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-uniform-online-width128-01.
Journal /root/autodl-tmp/data/channel_nsch2d/20261102-5090-gmls-time40-jvp-uniform-online-width128-01-process.json.
Receipt channel-uniform-time-jvp-validation.json is authoritative/current.

Final5000 phi4.972615%,terminal6.886025%,velocity4.218975%,pressure8.806874%,
chemical27.337925%,pooledminority13.924403%; coldwall1912.181s,warm.28732779s,
activepeak8700285952bytes=8.103GiB. Terminalweightedminority19.995163%,
contourmean/R0=10.322280%,IoU80.607553%; latecontoursstillflattened. UniformBDF
terminalminority20.8196%/contour10.7787%R0: modestgain, NOTsolvedshape goal.
Oneinitialization/reusednominalFEM, notstablemultiseed/populationaccuracy.

ALLownedtraining/validate_run/diagnose_phase/droplet_shape/phi_gallery/
continuous_time_residual exit0. Fieldreplay2.384e-6/JVPprimal9.537e-7,
flux6.72e-8/wall9.39e-8/div1.49e-8 PASSunchangedgates. Allactual5000globaltime
keys/strata/condition/rollinginputs/prefix and finalbank proofs PASS;88actual
source/owner checks passedatlaunch. MatcheduniformBDF auditPASS. Extrauniform/
adaptiveJVP pair exactinitialmodel/fullSOAP and ALL46input/drawfields (including
allcontinuous times, allinitial/finalgeometry arrays), strictconfigs differonly
causal/name. Source docs/tasks/channel_weighting_pair_audit.py; actualresult
run/weighting-pair-validation.json. Do notrerununchangedexpensivechecks.

Savedphysicalfieldsandcontours INSPECTED: t0/1.5close,t3startsflattening,
t4.5/6clearxyaspect mismatch. Figures run/phi-fields/phi-fields.png and
run/droplet-shape/contours.png. Fourwayactual0--5000curve source
channel_time_weighting_curve.py, external
/root/autodl-tmp/data/channel_nsch2d/uniform-time-jvp-curve-20261102-02/.
No smoothing/extrapolation; pooledqcurve is rescalingofsamephaseerror numerator,
not independent contour evidence. PNG/PDF/actualdataJSON outsideGit.

DIAGNOSTIC ONLY, no labels/masks entertraining:
1. Source channel_phase_error_regions.py, externalfrozen-phase-error-regions-
20261102-03/validation.json. Currentthininterface accounts96.1--96.5%of final
phaseerror acrossfourcontrols; core<1%, oldIC-only footprint<.014%. Reject
residualoldblob narrative as mainobservederror; do notchangehardIC onthatclaim.
Newterminalweightedux3.6187%,uy70.7384%; uyreferenceRMS.00814066/error.00575857.
Globalvectoraccuracy hides shape-relevant transverse errors.
2. Extendedchannel_mu_projection_probe.py reports rawconstitutiveRMS/actualmu
size on independently solved directGMLS nativegrid, preservedFP32fields/ops
promotedCPUF64 only. Currentuniform run externaluniform-jvp-mu-scale-20261102-01:
r_muRMS(t0)=.9524(67.33%of native_mu),t3=.05617(29.57%),t6=.09934(53.386%),
interface56.251%. Nativeprojectionconsistency2.831e-7 remains. Prioradaptive
report frozen-mu-projection-scale-20261102-01 gives terminal58.502%/interface60.455%.
Not anewsolver/continuumMuaccuracycertificate or trainedprojectionresult.
3. Code network.py usesloss.chemical_scale as muOUTPUTamplitude; losses.py also
usesit to normalizechemicalrow. Simplysettinglegacychemical_scale1 conflates
outputcoordinate and objective; do notcallthatloss-only preconditioning.

Exactresume / NEXTONECONTROL (notlaunched/queued): addexplicitpositiveoptional
loss.chemical_residual_scale overriding ONLY chemical-row normalization; absent
fieldretainslegacychemical_scale arithmetic/checkpointsemantics. Candidate1.0
versuslegacy4.0 (chemicalMSE16x), keep muOUTPUTamplitude4.0 and allother current
uniformJVP/architecture/IC/BC/scales/SOAP/GMLS/geometry/timekeys/budget identical.
No hardchemicalprojection, regionmask, evolvedlabels, newterm, controller,
timewindow or reference/PDE change. Sourcecontrol is THIScompleteduniformJVP5k.

Implementstrictsinglefieldconfiguration/control matching with clear outputvs
residual metadata. Verifyphysicalprimal/coldstates/allotherloss terms bitwise
unchanged, correctchemicalfactor/allparametergradient/finite differences and
focusedactualsmallcanonicaltraining/replay/input proofs. Validatepositivefield;
rejectunusednormalizer with eliminatedchemicalrow ifneeded. Thenbounded20actual
productiongradient/resourceprobe BEFOREONEcold5000. Preservealloriginalgates,
source/fullSOAP/actualgeometry/time streams, no fittedprobeweights. No newFEM,
default/SOTA/Git/cleanup or laterqueue. GoalACTIVE untilactualshape agreement;
currentcomponent/chemicalevidence is a hypothesis fornexttest, notprovenrootcause.

Sessiondiff:4smallreadonlydiagnostic helpers (timeweightingcurves,errorregions,
chemicalprobeRMS extension,weightingpairaudit), actualexternalplots/reports;
PROGRESS/DECISIONS/thistask/receipt/index updated. No productionmodel/numerical
arithmetic/default change in thiscontinuation. AST/JSON/diff verification only,
no unchangedexpensiveNNtest reruns.

### Previous chemical branch (retained, research-only; no longer current)

### Prior completed JVP and failed chemical research outcome

No liveGPU process or laterqueue. Alltrainingowners and finiteprobes EXITED.
Requestedoriginal-variable adaptiveJVP5000 and allfiniteaudits PASS:
phi6.77595%,terminal9.00161%,u4.36082%,p8.20213%,terminalminority25.68282%,
contour12.93833%R0,IoU74.96570%. ExacttimeJVP/40freshcontinuous draws areworking;
no material shape improvement versus matchedadaptiveBDF. GoalACTIVE/unachieved.
Both inletgate controls failed accuracy and are rejected, allauditscomplete.
Heightphi38.2624%,terminal41.2045%,u32.7372%,p65.0771%; no ownerrestart.

This session integrated explicitinteriorchemical elimination with ALLboundary
traces/physicalqueryfactory/rawAPIguards,8softterms, strictphysics-model-only
initialization and freshcompleteSOAP/pool/counters, canonical replay/diagnostics.
Prototype6contracts PASSED beforeintegration; integrated3 PASS (fullphysicalFD,
40vs8gradientidentity, actualparent3step+child3steptraining/replay/input/controller/
provenance/freshoptimizer/tamper audits), fixture50.964s; oldJVP/height6regressions
PASS81.268s. Firsttinyfixture64pointflowfluxgatefailed1.442e-4>2e-5; BOTHfixture
stages nowspatial_scale.25, production/gatesunchanged. Initial2newcontractpasses
and corrected1fixturepass retained; no expensive unchangedtests repeated.

Production20probe01 COMPLETE/EXIT1:20finiteupdates,compile355.775s/warm.33449s;
ordinaryquerybatch3.815e-6passes, physicalmuJVPprimal2.78614e-5 FAILSunchanged
2e-5gate. CompleteprobeFAIL, no fullstage/modeloptimizer/counters imported.
Actualsources/log/updates/failure.json preserved, no fittedweights exported and
no activepeakmemorypublished. Frozenoriginalparent native15633node13vs3time/
same3timeJVP values arebitwiseequal acrossall5channels; candidate single-time
mapdoesnotimprove that alreadyexact result. No map integrated.

Fullquerycontext frozenparent independentlyreproduces mu-onlyJVPgap3.79039e-5;
phi/u/v/p1.19e-7/3.73e-9/3.49e-10/7.15e-7. A rawstatebarrier doesnotfixit;
finalstatebarrier worsens querymu2.77042e-4/JVPmu4.28064e-5. Allfrozenjobs
exit0with explicitfailedgates, source/report preserved. No barrier/tolerance
change integrated; chemical branch remains explicit research-only with FP32
physicalquery limitation documented. Do notclaimit improves physics/shape or
launch5000with the failedresource qualification. Currentreceipt:
channel-projected-mu-refinement-validation.json (all directpaths/exactevidence).

Exactresume: prepare matched UNIFORMoriginal-variable coldJVP40continuous-time
5000 from completeduniformBDF40 preset/control
/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-gmls-time40-online-width128-01.
This isolates requestedtimeoperator/resampling with the betteruniformloss recipe,
without chemical/head/geometry/optimizer changes. Existingcomparison/proberunner
currently assume adaptivecausalJVP: extendstrictly for uniformmatching, preserving
oldfailclosed/actualcold/fullSOAP/allgeometry/globaltimekey/physicalmetric proofs.
Run focusedsmallcanonicaltraining/replay and boundedproductionresource probe
beforeONEfinitecold5000. No launchedrun/automaticqueue/newFEM/default/SOTA/
Git/cleanup authorization. Same taskremains active until dropletshape agreement.

Diff-levelsessionchanges: newchemical_projection/projected_fields/initialization/
diagnose_projected_phase and explicitresearchpreset; opt-intrain/loss/config/rawAPI/
replay/phase/pairedaudit/finiteowner/probe integration;3focusedtrainingcontracts;
production/frozen diagnostics and receipt; exampleREADME/DEVELOPMENT, this task,
PROGRESS, DECISIONS and artifactindex updated. Originalabsentflag behavior retained.

### Prior height launch and read-only diagnosis (historical; no longer live)

### Historical height launch receipt (superseded by completed state above)

SoleLIVE owner232327/training232333, screenchannel-height-jvp40-20261102.
Run /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-height-online-width128-01.
Externaljournal /root/autodl-tmp/data/channel_nsch2d/20261102-5090-gmls-time40-jvp-height-online-width128-01-process.json.
Currentreceipt channel-stream-height-validation.json; exactpreset
examples/channel_nsch2d/configs/train_operator_width128_gmls_time40_jvp_height_online.py.
Strictstream_lifting normalized_height reusesexistingH. AllotheroriginalJVP
settings/rawmodel/SOAP/actualsampling unchanged; coldpositive-timevelocity
intentionally differs. Actualrawcoldmodel/fullSOAP/initialgeometrybank exact
parity and14source/owner checks PASS. Step0physicalmonitor available; first
fullgradient/500physicalmonitor pending, firststepcompiling. No probe/label-
projected weights imported. Do notlaunch duplicate or resume oldcompletedowner.

Newheight/oldbranch hardIC/inlet/wall/flux/divergence/full9residual parameterFD/
strictschema contracts PASS (5entries51.704s included3legacy+2height; import
cleanup later preventsduplicatedbase discovery,2intendedheightentries checked).
7AST/strictoneflagcomparison pass. Production20update fullgradient/controller
GPUprobe COMPLETE/EXIT0, cold362.567s,warm.336673s,activepeak7.58811GiB;
planned5000timekeys andactual20replay pass. Resourceonly on2fixedinputs, no
accuracy result. AlloldCPUprobe/test sessions reaped, no extraGPUqueue.

COMPLETED failed narrowstream5k owner187439/training187445 EXITED, ALLfinite
training/replay/phase/shape/gallery/full40time/state/input/rawmetric/240continuous
probes PASS. AccuracyFAIL phi38.14215%,terminal41.52662%,u33.56797%,p64.02713%,
chemical212.1328%; terminalweightedq122.9634%,IoU0,meancontour6.51846R0.
Physicalphi gallery inspected: dropletstays atinitialx ratherthanadvecting.
Cold2394.436s,warm.367221s,activepeak8.01311GiB(notisolatedspeedbenchmark).
Receipt channel-stream-lifting-validation.json retains source/data/metrics.
The actualoldtrained source is preserved and comparedto its immutablepreceding
resourceprobe snapshot; laterheightflag changes intentionally differfromroot.
No oldownerrestart/defaultpromotion or accuracybenefit is claimed.

COMPLETED originaladaptiveJVP40 cold5k andALLfiniteaudits:
phi/terminal/u/p6.77595/9.00161/4.36082/8.20213%,terminalweightedq25.68282%,
contour12.93833%R0,IoU74.96570%; stillflat. Warm.337752s,wall2167.892s,
activepeak7.77479GiB. Receipt channel-time-jvp-validation.json. Actualfullcold/
SOAP/geometry/timekey/controller/replay/hardflow/shape/240freshcontinuous
fullspaceprobes pass. Same nominalFEM geometry/oneinitialization, notpopulation.
OriginaluniformBDF40 has betterphase/shape thanadaptiveJVP, retainedexplicitly.

Readonly diagnoses, no productiontrainingweights/labels/PDE change:
- Frozenoriginal/narrow/height gate onactual15633node coldmodel exactlyreproduces
  canonicalphysicaltimeJVP(0gap) forold/current. hthirdat0=0/-48000/-6;
  inletmomentumdefect9.06/93.98/7.09% and GMLSdiv RMS1.16e-5/1.45e-3/2.09e-5.
  Source docs/tasks/channel_stream_envelope_probe.py; external02report inreceipt.
  First01executioncorrected afteractualXLAconstantfold logs, interruption/source
  preserved;02passesarrays explicitly, EXIT0. No timeout-only restart/gate change.
- Full40continuous originalJVP gradientcheck onnominal+finalpool0: sharedcosine
  +.69863, phase/flowgradientnorm1.29589/.60828. No persistentcancellation
  evidence here; do notintroducesplit/staggerarchitectureonthatclaim. First8
  diagnostic wasbiasedearlyandpreserved; proper40diagnostic is authoritative.
  Source docs/tasks/channel_coupling_gradient_probe.py; external02report.
- FROZENphasebasis referenceprojection only: currentphi6.77595% -> unconstrained
  subspace3.36440%,terminal4.73696%,129rank,headnormratio2.5003. Reconstruction
  gap1.55e-15/GPUphasegap4.26e-6 pass. This uses49920evolvedreference rowsONLY
  forlinearprojectiondiagnosis, NOTtraining; NOcoefficients/checkpoint emitted
  or imported. Not a coupledPDE feasibility/prospectiveaccuracy result.
  Source docs/tasks/channel_phase_subspace_probe.py; externalfrozen-phase-subspace
  report01. Minoritynormthereisunweighted, separatefromweightedshapereceipts.

Exactresume: followONLY existing232327/232333 throughactualgradientupdates,
first500physicalmonitor and fixed5000/allownedfiniteaudits againstORIGINALJVP. Same5000actualtimes/geometry/
rawcold/SOAP expected; positive-timecoldphysicalvelocity intentionallychanges.
Complete allfiniteaudits andphysical/localizedshape checks. No newFEM/default/
Git/cleanup/oldqueue. UsergoalACTIVE, original dropletmismatch remainsunachieved.

### Completed adaptiveBDF control and requested physicalfield gallery

Oldowner94022/training94025 EXITED; ALLfinite stagescomplete0, including full240
BDFnode nominal/finalpool0audit. Fixed5000 nominalFEM phi/terminal/u/p=
6.7303/8.8861/4.4399/8.2753%. Cold3192.659s,warm.530308s,activepeak9.83604GiB.
All5000feedback transitions and51globalprefix records pass, maxstategap2.98e-8,
maxweightgap8.185e-8, finalnextstrength.289664. Actualfullmodel/SOAP/initial-final
banks/allcondition/time arrays match uniform; independentfieldmetrics pass.
Unweightedfull240nodal meanPDE(phase,chemical,mx,my): nominal
[.0004866664,.0005035548,.0003894157,.0004164129],trainingpool0
[.0005462897,.0004943775,.0004047816,.0004288637]; independentcanonicalgaps
1.061e-7/4.898e-8. Failure/success histories are not conflated with fixedstage.

Terminalminorityq25.4029%, contourmean/R0=12.9806%, IoU75.4597%; worse than
uniformq20.8196%,contour10.7787%,IoU79.6662%. Allphasecontours remainpresent,
but latePINO droplets are visibly flatter than FEM. Lowerpressure/global<10%
does not solve the usergoal.

Physicalgallery from EXACTsaved5000fields at0,1.5,3,4.5,6:
/root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-adaptive-causal-online-width128-01/user-gallery-20261102-01/.
Helper docs/tasks/channel_causal_field_gallery.py wraps existing plot_fields
for phi/speed/pressure reference/prediction/absoluteerror and existingrawcontour
extraction for FEM/Uniform/Adaptive comparison. Same96x40mappedqueries,reference,
times,geometry and updatebudgets; no newsolver or NNprediction. Spatiallinear
interpolation only for display, no temporalinterpolation. Commonphysicalzoom
halfwidth.352 contains all15contours. Rawendpointmetricgap1.39e-17; figures
inspected, fullphase/error and contourzoom shown to user. PNG/PDF/externalJSON
retained outsideGit; metadataembedded in completedBDFreceipt. Priorpartialfield
request is fulfilled, currentJVP trial has no fields yet.

### Historical fixed-stage causal40 endpoint — regression and failed fulltime gate

Run /root/autodl-tmp/tf-logs/channel_nsch2d/
20261101-5090-gmls-time40-causal-online-width128-01.
Journal /root/autodl-tmp/data/channel_nsch2d/
20261101-5090-gmls-time40-causal-online-width128-01-process.json.
Trainingchild61385 EXIT0; owner61358 has EXITED after its fulltime numericalgate
failed9.590e-6>3e-6. All preceding stages including shape/gallery EXIT0; the
skipped paired proof later passed independently on CPU. Exact failure/log and
completed proof are in channel-causal-time40-validation.json. Do not restart
this owner or substitute this historical endpoint for the currentadaptive run.

Same5000step nominal localFEM L2, uniform40 / causal40:

| Field | Uniform | Causal |
| --- | ---: | ---: |
| Pooled phi | 5.3677% | 22.3064% |
| Terminal phi | 7.1787% | 30.6219% |
| Velocity | 4.4834% | 4.8490% |
| Pressure | 9.2103% | 21.2869% |

Causalstep4000 phi6.1451%/terminal8.3781% still slightly worse than matched
uniform4000 phi5.9099%/terminal7.9216%. Epsilon changes10->100at4001.
Logged final-timeweights .5303/.5400at4000 become .0001056/.00004549at4001,
then zero at4500and5000. At5000weightedtotal.000352 but unweightedphaseMSE
.72597, up from.000653at4000. These are fresh sampledinputs/times, not a fixed
input ablation; nevertheless the persistent collapse and physicalregression
show that small discountedloss is not a fulltime convergence certificate.
A fixed1000update epsilon schedule is too aggressive in this run; do not
promote it or infer all causal training fails. A convergence-gated advancement
is a candidate for later work, not an implemented or proven cure.

Canonical savedfield/hardflow/input audit EXIT0: replay2.384e-6,JVP7.302e-7,
flux1.734e-7,wall9.702e-8,physicaldiv1.863e-9;99refreshes/198freshinputs/16final
bank rebuilt and all5000draws/time strata reconstructed. All55loggedglobal
causal prefixes/schedule/objectives pass; maxweightgap1.015e-7. Initialmodel/
fullSOAP/initialbank actualexactparity passed beforefirstupdate. Completepaired
5000sampling proof and shape/gallery subsequently pass; full240node numerical
crosscheck fails its unchanged3e-6absolutegate. This failedgate is retained.
No inference that hardflow/replay passing establishes PDEsolutionaccuracy.

Trainingwall3209.564s(53.493min),warm.53155s,activepeak10560493056bytes.
Uniformwall3028.287s,warm.47918s; these observed runs are not an isolated timing
benchmark because prior diagnostics overlapped part of the control.

Causal API remains the explicit opt-in coupled_prefix_mse with stopped
pergeometry global40time prefixes, nine existing spatially normalized PDE/BC
terms, commonBC weights and eight-time gradientblocks before oneclip/SOAP.
Schedule(.01,.1,1,10,100)/1000updates was fixed-budget, not the cited paper's
weight-triggered stage rule. No new fieldlabels/BC/massconstraint or timewindow.
Earlier sixfocusedtests and45array legacygradientpreflight remain valid; no
production code/config/source/default/reference changed in this status session.

Historical resume closed: preserve fixedstage artifacts and failedfulltimegate;
do not restart/retry this experiment. Currentuser-directed adaptive owner is
listed above. Mainreceipt/index aliases now point to adaptive receipt;
completeduniform receipt remains channel-time40-validation.json. Broader
physicalshape accuracy remains unresolved.

### Requested unchunked40-time memory check — COMPLETE; keep accumulation

Compile-only actual full loss+gradient+SOAP graphs at B2/15633nodes/width128/
FP32. Final compiled8-time memory10,528,792,332bytes=9.80570GiB;40-time
37,952,225,164bytes=35.34576GiB. Physicalcard32607MiB=31.84277GiB.8-time actual
accumulation peak9.65003GiB is nearby, but is a different graph/flag runtime;
not an exact validation/bound on full40. No full40 update or allocation/OOM
experiment was executed, no speed measured. Diagnostic process23096 EXIT0 and
released its~592MiBCUDA context; the sole training run continues unchanged.

Diagnostic PREALLOCATE=false/GPUautotune_level=0 flags apply only to the separate
compile. Thus35.346GiB is a compiler estimate for that plan, not an assertion
that all possible40-time implementations/tuning settings cannot fit. The
intermediate rematerialization pass warned target22.31GiB/unreduced42.07GiB/
reduced25.20GiB; these are earlier-pass numbers, not final compiled heap usage.
Use final temporary+argument+output-alias stats above for the scoped comparison.

Source docs/tasks/channel_unchunked_memory.py; external result/sourcecopies:
/root/autodl-tmp/data/channel_nsch2d/unchunked-time-memory-20261101-01/.
Four executed source-copy byte comparisons, AST and independent byte-stat sum
checks pass. Compactreceipt docs/tasks/channel-unchunked-memory-validation.json;
method verified against https://docs.jax.dev/en/latest/201/memory-spaces.html#measuring.
No production numerical source/API/config/default/optimizer/reference/Git change.

Exact resume: diagnostic complete, do not rerun it or attempt the occupied-card
full40 executable. Keep existing cold5000/Nt40/micro8 owner989465/runner4868/
training4874 and owned endpoint/input/field/shape/full240node audits. Future
memory optimization needs its own isolated resource measurement. If causal is
later implemented with time blocks, compute the prefix over all40times, not
reset it at each8-time block. No causal/control/reference queue is added.

### Completed40-time uniform cold5k — matched causal baseline

Run20261101-5090-gmls-time40-online-width128-01 and all finite stages EXIT0.
At SAME5000updates versus original4time:

| RelativeL2 | Original4times | Uniform40times |
|---|---:|---:|
| pooledphi |6.70222%|5.36771%|
| terminalphi |9.44356%|7.17866%|
| velocity |5.04633%|4.48336%|
| pressure |14.69536%|9.21030%|

Current nominalFEM principalL2 all<10%, but only one reused developmentgeometry/
initialization, not unseenpopulation/stability or equalcompute evidence. Cold
wall3028.287s/warm.479180s, activepeak10,560,178,688bytes. The same-card memory
compilation diagnostic overlapped part of training, so timing is observed,
not an isolated repeated speed ranking. Original20k baseline remains retained.
Terminalminorityq20.8196%,interfacephi39.8440%,IoU.79666,contour10.7787%R0:
wholefieldL2 still understates the shape defect. Final diffusearea.153782
versusinitial.152055; openchannel mass includes boundaryflux, no constantmass
constraint introduced.

Canonical replay2.5034e-6,JVP9.5367e-7,flux1.4932e-7,wall9.4455e-8,
physicaldiv1.4901e-8 PASS. All5000actualsampler keys/time strata reconstructed,
99refreshes/198newinputs/16finalinputs rebuilt; no evolvedlabels. Complete
condition/draw-ID/amplitude prefix matches original4-time first5000updates;
time arrays intentionally differ because sample counts differ. All initial
model/fullSOAP bytes exact. Independently recomputed final saved-array metrics
maxgap6.94e-18. Source/rawfiles and same reference identity preserved.

Full240node audits nominal/trainingpool0 COMPLETE; mean scaledPDEs nominal
[.0004662035,.0005032962,.0004188924,.0003374024]. Firstnode t.025phase/chemical/
mx/my=[.00396507,.02430262,.01749137,.01525160]. Independent canonical auditgaps
1.83e-9nominal/4.66e-8training0. Thus earliest coupled residuals still exceed
average; no inverse-gradient arm's catastrophic1256spike. No causal benefit is
claimed from this noncausal control. Completed summaries, validation/input/
fulltime/shape/gallery direct paths in channel-time40-validation.json; old
finite owner989465/runner4868/child4874 have EXITED. Do not restart them.

### Completed denseBC comparison — retained accuracy and diagnosis evidence

All three endpoints are fixed20000updates, same nominal localFEM3840queries/
13times, same cold model/fullSOAP and actual40000condition/80000time draws.
Original GMLS baseline versus denseBC uniform versus denseBC inverse-gradient:

| Scheme | pooledphi | terminalphi | velocity | pressure |
|---|---:|---:|---:|---:|
| Original GMLS | 4.93310% | 7.16785% | 2.76924% | 9.96429% |
| DenseBC uniform | 5.00561% | 7.37444% | 4.74429% | 12.02435% |
| DenseBC inverse-gradient | 25.09485% | 26.69089% | 5.04438% | 29.71012% |

Same one initialization/reused development geometry, not population/stability
or a universal optimizer ranking. No new default or SOTA promoted.

Independent BCs really improve. At t6,512independent midpoint queries/side:
wallmu-normal RMS GMLS=.675666(original),.00819243(uniform),.00290345(adaptive);
physicalAD=.514678,.0371735,.00373492. Not exact classical Neumann certificates.
Original versusdenseuniform terminalqL2=20.4166/21.0104%, contour/R0=10.2449/
10.5825%, interface strainL2=25.3182/25.4029%, droplet strain88.9088/88.0983%.
Thus tighter BCs do not repair the main deformation error in this pair.
Adaptive terminalqL2=77.7911%, negativephaseIoU0, zero-contour missing; droplet/
interface strain137.661/37.1254%. Its late phase evolution is physically wrong.

New decisive initial-time evidence:
- Full240node mean scaledCH MSE original.000171195,uniform.000201256,
  adaptive5.237270. Adaptive firstnode t.025 accounts for99.9942%ofits CH defect;
  the rest-of-time mean is.000306893. Finaltrainingpool0 likewise has firstnode
  1232.286and99.9937%ofCH defect, excluding a nominal-only transfer explanation.
- Direct unchanged canonical per-equation loss at t.025 independently confirms
  uniformCH=.00914345495, adaptiveCH=1256.87121582. Frozen audit gave1256.87147528;
  relativegap~2.1e-7, consistent with FP32 versus independent FP64 accumulation.
  Code/source/result:
  /root/autodl-tmp/data/channel_nsch2d/densebc-initial-interval-check-20261101-01/
  check.py,result.json,sources/. Both actual checks EXIT0, CPU only, no newtraining.
- Actual saved sampling includes that firstinterval334times. Logged examples
  at2200/3400/7000have CH loss168.4/317.8/303.0; final smallbatchloss is not a
  full-time convergence certificate. This is not proof of a unique weighting/
  clipping/SOAP mechanism. HardIC at t0 did not enforce the next PDE time node.
- Diffuse minorityphase integral changes.152055->.0974394inadaptive, whereas
  uniform ends.152598andoriginal.152461. Open-domain mass has boundary fluxes;
  do not silently impose constant mass as an extra physical BC.

Run paths under /root/autodl-tmp/tf-logs/channel_nsch2d/:
- 20261101-5090-gmls-first-online-width128-01 (comparison accuracy baseline)
- 20261101-5090-gmls-densebc-uniform-online-width128-01
- 20261101-5090-gmls-densebc-balanced-online-width128-01
Each has complete model/SOAP/sampling/pool/source/summary/prediction and canonical
validation/shape/gallery outputs. No evolved training labels. OrdinaryFP32,
SOAP1e-3/betas.9, online16pool/B2/Nt4, GMLS PDE gradients/Laplacians/BDF2dt.025,
physical velocity still AD streamfunction curl. Encoder48midpoints/side is
preserved; optional BC192Gauss-Legendre points/side have arc-length weights.

Completion/evidence under /root/autodl-tmp/data/channel_nsch2d/:
- densebc-comparison-20261101-01/status.json: finite pair complete, exact cold
  model/fullSOAP and all actual condition/time arrays match original; all23
  initial known-input arrays identical between the newpair.
- densebc-final-audits-20261101-01/status.json: all6CPUstages EXIT0. Uniform/
  balanced -strain.json,-boundary/report.json,-residual/nominal-dense.json and
  final_training_pool_slot0-dense.json contain complete independent diagnostics.
- Full actual rolling audits both399refreshes/798fresh sampledinputs/16finalpool
  reconstructions. Canonical field replay2.86e-6/uniform,2.15e-6/adaptive; hard
  flux/wall/divergence pass. Full240node audits match actual four-time canonical
  losses2.64e-8/2.80e-8; additional firstnode replay independently passes above.
- All final reported field metrics independently recomputed from raw three-run
  arrays on identical reference/points/times, maxgap4.16e-17.
- Wall38.79min(uniform),44.62min(adaptive),original38.65min; warm75.33/93.01/
  76.20ms. All finite owners/workers EXITED; GPU idle. No queued follow-up.
- Compact final receipt docs/tasks/channel-densebc-optimization-validation.json;
  previous live receipt preserved externally at final-audits/live-receipt-before-final.json.
  All current main-receipt/index aliases updated, historical originalcontrols retained.

Prior source/API checks remain:11focusedtests,818prior BC-gradient cases,
production encoder/field parity0, strict23array payload and actual training
integration,15sourceAST/whitespace/direct-path copies. No unchanged expensive
suite rerun for the result report. README now records completed negative controls.
Physical PDE/BCs, original entry/defaults and old checkpoints remain unchanged.
No Git/history operation, cleanup, hidden numerical fallback or goal restoration.

Reference direct192x80/.025and/.0125 finite space/time/flux gates COMPLETE/PASS;
coarse27.12%Q flux failure retained. Finite tolerances are not a continuum error
bound. Do not restart or duplicate reference trajectories.

Exact resume: no process needs monitoring/restarting. If further optimization
is requested, investigate first-positive-time PDE convergence and the interaction
of stochastic time sampling, gradient clipping and inverse-gradient weighting.
One generic candidate is fixed coverage of the first BDF node plus explicitly
derived unbiased time quadrature for all other nodes, retaining one global
network. Derive/validate that sampling objective before implementation or a new
training run. Keep current physical BCs and comparisonbaseline; no timewindow,
constant open-domain mass, reference-supervised target or motion prior introduced.
Broader low-error droplet objective remains unresolved, so task stays active.

### Earlier session chronology (superseded by the final diagnosis above)

Latest clarification: user means replacing psi->velocity AD by GMLS, rather than
checking whether AD is buggy. Feasibility explained: u=Dy*psi,v=-Dx*psi; existing
saved10k audit has this postprocessing already, not a GMLS-curl training run.
Exact discrete divergence requires a compatible pair or explicit continuity
enforcement; generic commutator is nonzero, and finite-stencil wall/inlet
velocity BCs need explicit treatment. Current run remains AD-curl. No numerical
implementation or additional training was launched for this clarification.

Requested physical fields now use latest selected saved10k model (not an8000
prediction or a final20kclaim). CPU partial replay exits0; gallery paths below.
User steering asks whether AD streamfunction velocity causes the phase-shape
issue. A bounded CPU saved-checkpoint audit completed via new
docs/tasks/channel_streamfunction_audit.py (no new training/solve/physics change).
It checked FD/reverseAD/query independence and physical versus GMLS divergence,
including transport product defects and GMLS-curl commutator. No AD implementation
error observed on92points/fivetimes; learned-function and representation errors
remain distinct from derivative correctness. Existing finite
training and CPU reference jobs remain authorized and untouched.

Latest user-requested interim observation freezes the actual8000-step monitor for
matched comparison. Same nominal dense localFEM3840queries: GMLS/original phi
6.207212/6.470641%, terminalphi8.507374/8.911375%, u4.292635/3.488841%, p12.085583/
11.189176%. Phase slightly better, velocity/pressure worse; no overall accuracy
or shape conclusion. Actual5000--8000elapsed interval.0934073vs.1747506s/update,
1.87085x, with evaluation/rolling/host overhead, excluding initial compilation.
Independent full-input audit still only through1200;8000is scalar-log evidence.
CPUreference observed158/240,t3.95/6; finite validator waiting for fine, halfdt
not yet started. PIDs/screens remain live. Preserve20k/final-audit budget.

User explicitly asks to stop current training, try GMLS derivatives in PINO,
and asks whether GMLS is nonlocal. No old training existed: original20k+audits
already complete, GPUempty. Keep separate CPU reference jobs for this comparison.
Explained GMLS is neighborhood polynomial reconstruction of local differential
functionals, not the physical regional integral model. Neumann BCs remain
explicit losses. User confirms trying this route. No new goal tool/old queues.

Run `/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-gmls-first-online-width128-01`,
configtrain_operator_width128_gmls_first_online.py, seed42provenance.
Durablescreen858940.channel_gmls_first_online_20261101 (PPID1),runner858941,
train858968. Journal `/root/autodl-tmp/data/channel_nsch2d/20261101-5090-gmls-first-online-width128-01-process.json`;
sameprefix-training.log and-runner.log. Canonicaltrain.py, onecold20k then
validate_run/diagnose_phase/shape/gallery; no retries/default/FEM/old queues.

New derivatives={kind:gmls_first,boundary:gmls,velocity:unchanged_streamfunction_ad},
phase_transport={kind:gmls_conservative,quadrature:sbp4_mapped}. PDE/BC first
jets and div(phi*v) use cubic36neighborQR-GMLS. Existing GMLSLaplacian is
unchanged. Physicalvelocity remains SAME ADcurl/lifting to preserve hard
inlet/no-slip/flux/continuousdivergence; no new approximate continuity loss.
The GMLS conservative differential form has no exact SBP flux identity. All
other model/optimizer/physical/sampling/budget settings match original20k.
This does not isolate every nonlocal error or make the entire architecture
AD-free, nor exactly match a discrete primitive-variable numerical solver.

Fiveextra knownarrays store domain x/yweights and boundary neighbor/x/yweights,
21total keys. Precompute outside AD for every geometry; integerboundaryindices
preserved throughout poolcasting/audit. Scalarshape and rank/reproduction/range
checks fail loudly. Baseline configs/checkpoints retain their old arithmetic.
README/DEVELOPMENT explain explicit option and retained physicalAD/quadrature.
GMLS diagnostics are separate from the original SBP mass identity and physicalAD
constraint. No defaults/physical output or checkpoint format changed.

Ten unique focused tests pass: five new GMLS tests (cubicboundaryderivatives,
fluxsign, finite-difference lossparametergradient, actualcanonical3updateonline/
SOAP/fullinputaudit, identicalphysicaloutput/schema/config guards) plus five
operator/NS/online regressions. Firstfour41.304s, integration+regressions21.266s;
lastshapeguardrecheck3.473s. No full expensive unchanged suite rerun.
NewCPUpriorprobe docs/tasks/channel_gmls_first_preflight.py screens818boundary
stencils (16initial+802online/prefetch) in20.494s: maxcondition1.92978e6,
maxpolynomialerror2.3918e-13. Data-root gmls-first-pino-prior-20261101.json/.py.
Domain QRrank inherits identical oldLaplacian basis/neighbors/geometry screen;
every actual domain first-functional reconstruction also validates itself.
Original802screen alone is explicitly NOT a newboundarygradient certificate.

Actual newconfig equals old except derivative/flux/runname; all16sharedrawinitial
arrays, coldmodel and fullSOAPinitial bytes EXACT; all21keypayload shapes/dtypes/
ranges, separateeval and17reservedparameters checked.14numericalsource snapshots
exact. Run/gmls-first-initial-validation.json. InitialGPUfieldmonitor maxgap
3.52354e-9 versusoriginal. Newliveaudit run/gmls-first-live-validation.json checks
through1200loggedupdates:23FIFOrefreshes/46newinputs, exactstatelessparameters,
actualdomain/boundarygradientguardreports and no PDE/FEMlabels.
Nominal1000phi/terminalphi/u/p=10.288168/13.401917/5.934276/38.125358%; early
singledevelopmentgeometry, not converged/population/superiority evidence.
Effective500--1000interval.094917s/update includes monitors/rolling/host overhead.
Actualcoldcompile completes; CUDAautotunetimerwarning retained in log, no failure.

**Exact next action:** read livejournal/metrics/loss/rolling logs; leave prescribed
20kand finite saved-run audits running. Verify actual final sampling/inputs/model,
then compare field/terminalminority/contour/time to original20k using the same
FEMqueries. Newreceipt docs/tasks/channel-gmls-first-pino-validation.json;
optimizationreceipt/index retain originalcase separately under completed_original
and chronologicalonline_rolling_launch_20261101. Current aliases pointtonewrun.
CPUreference/refinement screens below remain independently live; don't restart,
kill or duplicate them. No externalblocker/Git/cleanup/default promotion.

### Preceding authoritative status — original online complete; direct CPU reference live

Run `/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-geometry-online-width128-01`
completed20000cold updates. Its durable finite runner completed all five stages:
training, validate_run, diagnose_phase, droplet_shape, phi_gallery. External
journal `/root/autodl-tmp/data/channel_nsch2d/20261101-5090-geometry-online-width128-01-process.json`
has statuscompleted_with_saved_run_audits. No train restart or oldgoal/queues.

Unchanged original residual/physical NSCH, streamfunction, GMLS3/36/weight16,
SBP conservative phase transport, BE-startedBDF2dt.025, linear-logtime,
width128/fourheads/64spatialfeatures/2condition/3decoder, FP32highestordinary,
SOAP1e-3beta1=beta2=.9refresh5.16slot/B2/Nt4/192x80,2CPUworkers,replace2every50,
prefetch2batches, independent upper/lower Uniform[.04,.14], fixedIC/driving.
No single-case frozen weight/tanh addition. Runtime seed42 is provenance.

Final nominal development3840query phi/terminalphi/minorityq/u/p/chemical =
5.323475/7.881282/14.906886/2.382571/9.902211/23.568772%. That logged minorityq
pools time; TERMINAL physical-area minorityq22.416449%, zero-contourmean11.399616%R0.
Shape remains too flat late. Wall4302.056s,warm.161472s/update. Actual399FIFO
refreshes/798installed and distinct sampled draws reconstructed,802submitted,
16finalpayloads rebuilt, immutableeval checked, no evolved labels. SavedGPU
replay3.0994e-6,JVPprimal4.8056e-7,flux1.8529e-7,wall9.5486e-8,div1.4901e-8.
Phase/SBP integral identitygap1.9168e-8 is not a complete accuracy certificate.
No unseenFEM population validation or SOTA/default promotion.

Actual complete final model, SOAPstate, sampling/pool inputs, summary, prediction,
validation.json,online-validation.json,phase-balance.json,droplet-shape and
phi-fields/phi-fields.png are preserved. Earlier10kCPU replay maxscalargap2.274e-8
and reservedmanifest0/5/11 prediction-only gallery also pass finite/IC/inlet/
wall/flux checks (<6.79e-7). Their plots have no per-geometry FEMaccuracy claim.
Exact partial receipt run/partial_010000-gallery-validation.json. Display-only
linear interpolation retains raw96x40queries and equal-scale contour zoom.

### Direct numerical nonlocal reference — current bounded work

User explicitly requests this new comparison. New task helpers:
- docs/tasks/channel_nonlocal_reference.py: all-GMLS FP64 CPU full NS--CH,
  current difference-form Laplacian, original physicalBC/traction, implicitBDF2,
  sparseNewton/current-matrixGMRES with explicit LUpreconditioner refresh.
- docs/tasks/channel_nonlocal_comparison.py: exact-identity/commonquery three-way
  originalfield,minority and contour metrics/plots; no reference promotion.
- docs/tasks/channel_nonlocal_reference_finalize.py: one finite refinement flow.
- tests/test_channel_nonlocal_reference.py: three meaningful independent checks.

Current direct Laplacian matches PINO to~3.64e-12 on96x40. No NN/AD/FEM is used
in numerical solves. First derivatives and conservative phase flux are GMLS,
whereas PINO uses AD first jets and mappedSBP flux, so this does NOT exactly
isolate only network optimization. Matched material/interface/IC/geometry and
pressure/traction convention are explicitly retained. Initial chemical potential
is eliminated from knownphi by the actual GMLS Laplacian; t0pressure is not scored.

Failed controls retained under data-root direct-gmls-smoke-20261101-* and
full-20261101-01. First guards exposed singular anisotropic toy stencils, fixed
by a fixture with declared physical spacing ratio; no relaxed condition limit.
Full-nodal pressure develops large oscillations, stopsGMRES at1.125 after44steps.
Coarse pressure can enforce sampleddivergence but miss fine-node divergence.
Single-phaseStokes fixture already shows layout problems. This is not proof of
one unique instability mechanism or evidence against FEM/nonlocal in general.
Interior-pressure reconstruction completes the following provisional control;
its boundary-flow error still requires refinement. No artificialcompressibility,
pressure diffusion, changed traction or hidden fallback was added.

Completed coarse: `/root/autodl-tmp/data/channel_nsch2d/direct-gmls-full-20261101-02`,
96x40,pressurestride1/interior,.025,T6,535.582s,240steps,allNewtonmax<1e-8.
Comparison `/root/autodl-tmp/data/channel_nsch2d/direct-gmls-coarse-comparison-20261101/`:
- GMLSvsFEM phi1.162559%,terminal1.137146%,q3.300466%,u.805082%,p6.448130%,
  terminalcontour1.857237%R0.
- OnlinePINO20kvsFEM phi5.323475%,terminal7.881282%,q22.416449%,u2.382571%,
  p9.902211%,terminalcontour11.399616%R0.
- PINOvscoarseGMLS phi4.928662%,terminal7.124818%,q20.351594%,u2.563909%,
  p11.262867%,terminalcontour10.119641%R0.
All3840points/13times/physicalidentities align and original savedFEM fields match.
Three-way PNG/PDF inspected: blueGMLS and blackFEM much closer than orangePINO
late; current result does not explain the entire shape gap by reference choice.

LIVE fine: `/root/autodl-tmp/data/channel_nsch2d/direct-gmls-fine-20261101-01/`.
Screen840158.channel_nonlocal_ref_20261101 (PPID1),solver840159,CPU4cores only,
192x80/interiorpressure/.025/T6,96x40query. Directsource snapshot exactly matches
current helper. Existing solver was preserved while ONLY its waiting launcher
parent ended; launcher143was reaped, then screenPPID1/solver confirmed alive.
No solve restarted. LaunchJSON sameprefix-launch.json; logfile sameprefix-screen.log.

LIVE finite validator: screen848608.channel_nonlocal_validation_20261101,
validator848609, status `/root/autodl-tmp/data/channel_nsch2d/direct-gmls-validation-20261101-01/status.json`.
It waits exactfinePID (finite10800s), then runs exactly one192x80/.0125/T6 CPU
solve at `/root/autodl-tmp/data/channel_nsch2d/direct-gmls-halfdt-20261101-01/`,
then frozenPINO20k/FEM comparisons and explicit96->192space/.025->.0125time flags.
Limits phi/terminalphi/u2.5%,pressure8%, maxsectionflux2.5%ofQ. Failed gates or
solver failures stay explicit, no retries or further experiments. Existing FEM
exports only; no newFEM/GPUtraining or old automaticgoal. Fine run has several
accepted real steps; runtime is CPU-solve dominated, not neural compilation.

**Exact resume:** read the finite validator status and fine/halfdt report.json and
steps.jsonl, preserve screens, do not duplicate live solves. On finite completion
inspect numerical space/time/flow flags, actual three-way plots and all raw
pressure/phase/velocity metrics; independently check arrays before treating it
as reference. Update docs/tasks/channel-nonlocal-reference-validation.json,
optimizationreceipt/index/PROGRESS. Current task remains incomplete with an
exact running continuation; no external blocker, Git/default or cleanup action.

### Preceding bounded droplet trial (completed before online launch)

**Bounded diagnosis and250step trial completed; shape issue remains unresolved.**
At the end of that preceding trial, execution sessions reaped0: training30735,validation10944,phasebalance
15781,dense11090,shape42504,phi gallery64734. GPU was empty before the new
user-authorized online run; no old scheduler/FEM queue was retained. Old automaticgoal and online/FEM queues remain cleared. No default/SOTA,
Git or cleanup action. No external blocker; current limitation is scientific.

### User's FEM correctness concern

`channel_fem_shape_audit.py` compares existing40x16dt.025/64x24dt.05/64x24dt.025
exports plus the user-viewed10kcheckpoint on the same3840physical queries.
Canonical `/root/autodl-tmp/data/channel_nsch2d/fem-shape-audit-v2-20261101/`
contains audit.json and inspectedcontours.png/pdf. Older non-v2report has a
time-key collision and is explicitly superseded; numerical values unchanged.
FEM also stretches/rebounds: t4.5fine/PINOwidthratio1.1394/1.5967, t6.9413/1.3204.
Terminal FEMspace/time minorityq1.3825/2.1907%,interfacephi2.6525/4.0189%,mean
contour.7230/1.2222%R0; PINO23.6174/45.3083/12.0955%. These are measured
refinement differences, not exact-error bounds or independent code verification.
Initial coarse/fine interface interpolation discrepancy11.43%is retained in rows;
chemical potential is outside the old acceptance gate and has~8.8/9.6%pooled
space/time changes. No new FEM solve was performed concurrently or afterward.

Weak/strong source trace found matching materialA/kappa, capillary +phi gradmu
residual, chemical signs, conservative CH transport, inlet/wall/outlet data and
original-pressure traction convention. This does not exclude all shared errors.
Known manufacturedGMLS composition consistency error14.8--16.2%is not a direct
nonlocal solution error or proof of shape causality. Do not infer pressure cause
from global pressure L2. Future work must separate reference, nonlocal and NN
errors rather than treating lower loss/FEMdistance as automatic physical proof.

### Completed bounded weighting trial

Run `/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-droplet-metric-coupled-01`
uses canonicaltrain.py/configtrain_single_lbfgs_droplet.py,seed42. Exact complete
15kSOAPparent `20261101-5090-time-refine-single-time-01`,same original250control
`20261101-5090-fulltime-lbfgs-single-01`. No missing10koptimizer reconstruction.
Newfrozenparentsoftminoritymetric:alpha24,tau.2,mean-one interior measure,
phase/chemical/momentum rows; BCs/PDEs/scales/operators unchanged. Negativeparent
phase3.65%receives40.83%weightbudget; all weightspositive,min/max.5504/12.3826.
Freeze all240BDFnodes BEFOREL-BFGS to retain a valid scalar line-search gradient;
no changing stopped-gradient weights, clipping, outputtanh or reference masks.

Actual model/oldSOAPstate/knowninputs/sampling bytes exactly match parent and
retained control; config differs only metric/logging. Independent formula matches
savedFP32weights after actualFP32measure cast. Fournewfocused tests74.356s plus
twoparent/time-accumulation regressions7.927s pass. Firstfullobjective670.741s;
250acceptedsteps/269calls,total1369.11s,final.00458002,22.32%ownlossreduction,
budgetexitstatus1(notconverged). FreshL-BFGShistory; no exactSciPyresume or newSOAP.

Final GPUreplaymaxgap1.43e-6,timeJVPprimalgap7.15e-7,fluxgap3.03e-7,no-slip9.52e-8,
div5.96e-8. Independently reconstructed parentphi at all240nodes exactly; weights
reconstructed exactly; no evolvedlabels. Phasebalance/SBP advectiongap1.74e-8
is an advection identity only, not completeCHaccuracy certification. Dense
all-timephi/u/p=6.3372/2.8550/10.9078%,terminalphi8.5165%,chemical31.0896%.

Same250control->weighted terminal: minorityq24.6510->24.4184%,interfacephi47.1877
->46.5863%,IoU77.4956->77.6295%,meancontour12.6603->12.3798%R0. Smallgain only,
visually nearly identical; not the desired shape improvement. Keep experimental
entry off by default. Single geometry, reused completed control, no robust
superiority/generalization claim.

Exact artifacts: run/{summary,refinement_summary,refinement-validation,validation,
dense-fem-evaluation,phase-balance,droplet-initial-validation}.json, model_final.eqx,
frozen_droplet_metric.npz/json and dense-fem-prediction.npz. Inspected phi gallery
run/dense-phi-figures/phi-fields.png. External data root `/root/autodl-tmp/data/channel_nsch2d/`
contains droplet-shape-{10k,parent15k,unweighted250,weighted250}-20261101 reports
and droplet-weighted-comparison-20261101/{comparison.json,comparison.png,pdf}.
Canonical FEMshapeaudit-v2as above. All generated arrays/media stay outside Git.

### Diff-level changes and verification

Added droplet_metric.py/config + strict problem validation, optional loss norm,
narrow parent whitelist/frozen payload/reconstruction in refinement.py. Absence
preserves original arithmetic and parameter checkpoints; unitmetricloss/gradient
identity and actual directional finite difference pass. New saved-field shape,
comparison and FEMshapeaudit scripts with common-query/physical identity guards;
contour self/concentric/missing tests and full comparisonselfidentity smoke pass.
README/DEVELOPMENT document the opt-in scheme and reference-shape limitations.
PROGRESS/DECISIONS/receipt/artifactindex updated. RelevantPythonAST/strictJSON/
whitespace including newuntrackedfiles and gitdiffcheck pass. Unchanged expensive
suites not rerun. No FEM/sourcephysics/default or Git history change.

**Exact next resume point:** Read receipt `channel-nsch-optimization-validation.json`
keys reference_droplet_shape_audit and droplet_weighted_comparison before new
experiments. The bounded request is done; no auto action. If user continues,
prioritize separating operator/reference/representation error in the capillary
balance, with shape-based convergence metrics. Do not keep increasing weights
from this negative result or resume the cleared onlinequeue. Original15kSOAP
parent is complete; the newL-BFGSfinal does NOT contain recoverable SOAP state or
SciPyhistory. Retain earlier model-only10k/artifacts and failedcontrols unchanged.

## Implementation and live resume

balance.py adds explicit phase/chemical/momentum/continuity(if soft)/boundary
groups, one shared VJP tape, sequential L1 norm probes to bound memory, stopped
inverse-norm/clip/90%new EMA, refreshed every100updates. Canonical train.py uses
the optional balance record; absence preserves uniform raw-loss arithmetic.
New direct/stream_gradnorm presets retain all other SOAP5000/width96/16geometry/
B2/Nt4 settings. Save optimizer_initial, raw terms, weights/norms, balance state
and actual input streams. No reference labels enter the PDE loss.

Two focused CPU tests pass51.833s: hand-derived weights/weighted gradient,
stop-gradient and held-weight branch; both direct and stream complete PDE
value/parameter-gradient paths, group conservation and dynamic refresh. Syntax
and whitespace pass. Independent frozen-checkpoint validation still required
after complete GPU runs. Existing reference gates reused, no new FEM worker.

Historical pre-swap driver docs/tasks/channel_gradnorm_experiment.py used exec session
92045; driver log /root/autodl-tmp/data/channel_nsch2d/gradnorm-pair-driver.log.
Direct run first:20261031-channel-direct-gradnorm-01; stream follows only on its
successful completion:20261031-channel-stream-gradnorm-01, both under existing
tf-logs/channel_nsch2d. Individual train-*-gradnorm-01.log files are in the data
root. Next monitor first-update compilation and complete both5000updates, then
validate_run.py on each, check actual inputs/time draws/common initialization,
compare fields/convergence and physical balances with the old uniform stream.
The two velocity liftings differ along with representation; this is a scheme
comparison, not an isolated theorem about hard versus soft continuity.

## Hardware interruption and current authoritative resume

User swapped GPU to RTX5090/32GB and explicitly requests completion. Old workers
are all gone. Old direct GradNorm run completed5000: phi43.0125%,u99.7210%,
p99.8068%,wall553.39s,warm27.02ms,peak2.392GiB. Old stream GradNorm stopped
during first training compilation, only step0evaluation/no trained checkpoint;
retain partial files/logs, do not synthesize completion. Old watcher also exited.

New5090FP32 JIT smoke passes. To keep hardware matched, restart all three controls
on5090, including uniform-stream baseline. Tag20261031-5090; driver script now
requires --tag and --modes. Exec session16400; logs
/root/autodl-tmp/data/channel_nsch2d/20261031-5090-controls-driver.log and
train-20261031-5090-{direct,stream,uniform}-01.log. New run directories are
20261031-5090-channel-direct-gradnorm-01,
20261031-5090-channel-stream-gradnorm-01,
20261031-5090-channel-stream-uniform-01. They cold-start with5000updates each.
Training records device identity; no precision/model/sampling reduction needed.
New serial validation watcher is launched for this driver; its log is
/root/autodl-tmp/data/channel_nsch2d/20261031-5090-validation-driver.log.
Use the PID marker20261031-5090-driver.pid for current driver identity.

User additionally asks whether first/second spatial derivatives can all be
nonlocal. Clarify current evidence favors sampled residual/integration failure,
not a demonstrated AD formula defect. Time JVP/parameter AD can stay. All-spatial
nonlocal needs a compatible gradient/divergence/Laplacian and actual mixed-BC
validation; a scalar regional Neumann Laplacian alone is not that scheme.
Continue this feasibility review alongside the already authorized comparison;
do not silently replace its derivatives or call GMLS the original pair kernel.

## Original regional-kernel consistency probe

Completed the pure NumPy/SciPy, single-CPU script
`docs/tasks/channel_nonlocal_probe.py`; outputs are
`/root/autodl-tmp/data/channel_nsch2d/nonlocal-channel-probe.{json,log}`.
No spatial AD, JAX worker or FEM solve is used by this probe. It constructs
the shape-tensor-corrected integral gradient, the original positive cubic
regional Laplacian, and the volume-weighted negative-adjoint divergence.
This is a consistency/boundary diagnostic, not a validated NS discretization.

On actual curved-channel grids48x20/96x40/128x64 with horizons.30/.20/.15,
linear-gradient reproduction gaps are1.02e-14/2.22e-14/6.31e-14;
pair mass and weighted-adjoint identities pass to roundoff. Full-horizon
smooth-field gradient errors decrease.803%/.357%/.201%, and Laplacian errors
.399%/.0931%/.0329%. These are operator probes, not solution-error estimates.

The bare regional kernel's straight-wall Poiseuille Laplacian errors in the
wall layer are257%/372%/490%: natural zero-flux closure is incompatible with
the no-slip parabola's nonzero normal velocity derivative. For v=(x,0), direct
gradient divergence integrates to3.853714(domain area), while the negative
adjoint without an open-boundary term integrates to roundoff zero. Thus neither
closed-domain adjoint nor scalar natural-flux closure can be substituted into
this channel unchanged. Mixed boundary closure and pressure stability remain
unsupplied/unvalidated; do not launch or advertise such a substitution as a
full-nonlocal NS scheme. The result does not establish an inherent failure of
nonlocal first derivatives or superiority of the current AD/GMLS method.

Primary-source follow-up: Du--Shi arXiv2203.14491 uses a prescribed no-slip
boundary layer and pressure relaxation for linear steady fully wall-bounded
Stokes; its open-channel/mixed-BC extension is not supplied by that paper.
Trask/Bochev/Perego arXiv1903.04621 constructs conservative point-cloud mimetic
divergence with virtual faces and GMLS moments. It is a possible distinct
discretization route, not evidence that our bare regional pair operator already
implements mixed NS conditions. Both are linked in the example README.

5090 direct GradNorm completed5000: phi42.9216%,terminalphi48.0599%,
u99.6562%,p99.9284%,cold620.63s,warm22.403ms,activepeak2.436GiB.
It is an accuracy failure, irrespective of loss reduction. Stream GradNorm
is next in the serial driver; final uniform stream and checkpoint/physics
audits remain pending. New5090 runs must be compared to each other, not the
previous card's timings. DriverPID4249, validation watcherPID10526.
Direct phi/speed field figures are generated and inspected: the droplet remains
near its initial location and velocity is almost zero through most of the
channel. Final comparisons will include independent section-flow and phase-area
plots, with separate FEM/network quadratures explicitly labelled.

## Phase-accuracy controls (latest authoritative continuation)

User subsequently notes velocity L2 improving while phi L2 worsens and asks for
staggering/phase preconditioning, emphasizing the minority droplet/interface.
The task remains active for these experiments; completing the first3controls
alone does not complete this extended request.

Implemented explicit opt-in presets train_stream_stagger.py and
train_stream_regions.py. Stagger uses25update phase/flow blocks for4000updates
then1000joint; stop cross-physics factors during each block and mask inactive
output-head updates after SOAP. Shared trunk/global SOAP state/LR remain shared;
no claim of fully frozen physical fields. Region metric retains.5full-domain
phase/chemical norm plus.5split equally over normalized current droplet/carrier/
interface regions. Clip phi only to construct stopped metric regions; no output/
PDE clipping. Minimum region fraction.001, positive floor and exact zero set;
no reference/future masks. Momentum/BC metrics unchanged. Do not combine the
two modifications until separate results justify it.

Focused CPU tests tests.test_channel_phase_controls pass4tests/147.631s:
independent region budgets/positivity/stopped derivative/empty phase, schedule
2000phase+2000flow+1000joint, inactive head SOAP-update mask, primal residual
identity/lagged cross-physics derivatives, and finite complete PDE gradients
for both extensions. Current ordinary baseline already loaded the prior source
before integration edits; no existing run is mutated.

New phase driverPID72734, exec49716, tag20261101-5090-phase, modes stagger/regions
is queued behind previous validationPID10526 and requires its exact-input/init
audit receipt. Driver log:
/root/autodl-tmp/data/channel_nsch2d/20261101-5090-phase-controls-driver.log.
Run dirs (tf-logs/channel_nsch2d):
20261101-5090-phase-channel-stream-stagger-01,
20261101-5090-phase-channel-stream-regions-01.
Both cold5000, same architecture/input/time/seed/SOAP and source snapshot policy.
New docs/tasks/channel_phase_validate.py watcher exec57498 waits for72734,
then serial GPU replay/flow/area audits, field plots and CPU five-control exact
sampling/input/init/stage audits. Log:
/root/autodl-tmp/data/channel_nsch2d/20261101-5090-phase-validation-driver.log.
Final5control results/figures go to
/root/autodl-tmp/data/channel_nsch2d/phase-comparison-20261101-5090-phase.
Do not start concurrent GPU workers while these queues are active.

At last check, first3control driver4249 is compiling uniform-stream (child48915).
Stream GradNorm has completed5000withphi30.7261%, having risen from28.41%at750;
full summary and independent audit remain to collect. New arms have not trained.
Next finish baseline/audits, inspect three-way curves; then monitor new arms,
compare full phi/dispersed L2, terminal phi, physical flow/area and runtime.
After results, update README/index/compact receipt/memory and run final syntax/
whitespace checks. No Git or deletion authorized.

Update: all first3trainings finished. On5090 uniform stream finalphi30.6001%,
terminalphi33.7253%,dispersed85.6832%,u4.8230%,p22.2422%,mu149.1728%,
training timer1152s. Stream GradNorm phi30.7261%,terminal34.2740%,
dispersed86.0360%,u4.3511%,p12.6868%,mu94.9155%,1261.45s,warm58.786ms,
activepeak7.774GiB(including compilation/autotuning workspace). Thus phase
accuracy did not improve, despite better velocity/pressure.

Direct GPU audit passes replay7.15e-7/primalJVP1.79e-7 but demonstrates a failed
physical solution: final flow.200000004at inlet->5.19654e-6at outlet,
max independent-probe divergence.447598,fluxgap.1999948. Its dispersed area
remains.1520..1526but droplet stays nearly stationary. A passed replay is not
an accuracy pass. Current validation watcher10526 is auditing stream next;
the queued phase driver72734 has not started GPU work.

Validator now concatenates independent saved/probe/section/boundary/domain
queries for one primal compilation, retaining a separate time/spatial-JVP probe;
actual direct replay verifies query-batch invariance. Fail on nonfinite values
or invalid shapes. This changes validation cost, not training or PDEs.
New phase watcherPID73148/exec57498; compare script now supports --phase-tag,
audits actual stage trace/counts/totals and all5input/init streams, and plots
phi/dispersed/u/p plus flow/area. Saved dispersed L2 is the affine-field
normalization counterpart of phi L2, not an independent regional-error measure;
flow/area and the actual fields supply separate physical diagnostics.

Baseline3way validation completed successfully. Result path
/root/autodl-tmp/data/channel_nsch2d/gradnorm-comparison-20261031-5090/results.json;
convergence.png/pdf and physics.png/pdf are generated and inspected. All actual
condition/time/known-input arrays match exactly,69common trunk arrays exact,
full initial stream-model bytes exact; both50refresh EMA/logged-total audits pass.
Uniform stream area ends.0562253,GradNorm stream.00819363,FEM.1520907;
both stream flux gaps<4e-7. Uniform warm45.993ms,activepeak2.990GiB;
direct22.403ms/2.436GiB;streamGradNorm58.786ms/7.774GiB. Timers include initial
compile/evaluation; CPU consistency/tests ran during compilation, not warm
training. No cross-hardware speed claim.

Queued phase driver72734 has now started stagger, child88833. It is still at
step0/first-update compilation at last check. Initial phi45.5579%,velocity10.6146%
exactly match both stream controls. Regions will follow only after successful
stagger completion. Watcher73148 remains waiting. Extra focused real-SOAP
momentum-switch test passes73.499s: after3phase calls, an unmasked flow update
moves inactive phase rows, while the applied mask zeros them. Initial test used
only SOAP's preconditioner-initialization call, which has no moment update, so
that test setup was corrected; no optimizer production code changed. All4other
phase-control tests remain valid; no unchanged expensive suite repeated.

Stagger has now completed5000: phi26.7727%,terminal28.6348%,dispersed74.9661%,
velocity4.7469%,pressure26.1134%,chemical242.4023%;about1139s,warm47.696ms,
activepeak2.989GiB. Actual stage counts2000/2000/1000. At4000(before joint)
phi30.0217%,velocity3.3553%,pressure24.2145%,chemical183.1744%; joint tail
improves phi while velocity degrades modestly. Saved ckpt_004000.eqx permits
later independent replay, without claiming full frozen-field blocks.
Stagger final independent GPU flow/area audit remains queued behind regions.
Early CPU field plotting exec9556 is running while regions compiles.
Regions is now the active GPU child of driver72734. New comparison-script
initial-optimizer byte assertions explicitly pass for the already started
stagger and both baseline stream variants. No initial optimizer/LR reset.

Stagger phi/speed/p field gallery generated; phi visually inspected. Its lower
L2 does not preserve a proper droplet: at saved384queries finalphi min.6780,
versus FEM-1.0347(uniform min.7451,GradNorm min.9291). It predicts a smeared
positive-phase deficit. Do not promote it as an accurate interface solution.
New frozen-model validator additionally reports full3977point min/max and
negative-phi core area for both new controls; no extra GPU evaluation is needed.
These are diagnostic quantities, not hard range bounds or conserved core area.

## 2026-11-01 actual coordinate-transform steering (current resume)

All five5090controls and independent replay/input/init audits are complete.
Five-way receipt:
/root/autodl-tmp/data/channel_nsch2d/phase-comparison-20261101-5090-phase/results.json.
Uniform phi30.6001%,stagger26.7727%,regions31.2075%; terminal diffuse area
.0562253/.1260099/.1212561 versus FEM.1520907. Stagger/regions full-grid final
minimum phi.6716858/.2124650: neither retains the negative-phase core.
Their replay/hard flux/divergence pass; these are failed phase solutions, no SOTA.
Correct dense prediction plot labels now explicitly say Prediction (the first
one-row render inherited an incorrect FEM label, corrected without data changes).

Stagger phase-balance diagnostic completed. At t=.5/1.5/3/4.5/6 weak residual
.01749/-.00386/-.00707/-.00495/.00268 versus integrated strong
.01636/-.00687/-.00599/-.00367/.00350. Advection quadrature gaps are ~.0008-.0031,
mobility-scaled Laplacian Gauss gaps<8e-5. This is the trained quadrature only,
not converged-integration evidence or a unique explanation of the remaining error.
All prior GPU workers and dense CPU rerenders exited successfully.

Latest user explicitly prefers actual tanh-output/Jacobian transformation to
region weighting. Next implement two matched cold5000controls: phi=1.25*tanh(s)
with existing exact IC/inlet lift, original residual versus phase transport
residual multiplied by1.25/J. Global saved FEM range[-1.1286841,1.0301374]
requires more margin than a=1.1; no universal CH maximum bound claimed.
Chemical relation and capillary force remain physical/untransformed, diffusion
mu divided by J explicitly. Add focused chain-rule/nonfactored-diffusion/J-guard/
full-PDE gradient tests, then train serially and audit against the same FEM gate.
Default stays uniform-stream experimental recipe. No blocker, Git or deletion.

Implementation/launch update: phase_coordinate.py, opt-in map-only/map+Jacobian
configs, network lift/Jacobian hooks, phase-only residual transformation and
per-evaluation J/range diagnostics are implemented. Four focused coordinate
tests pass61.558s on CPU, covering hard IC/inlet, chain-rule/full-J gradients,
nonfactored mu diffusion, stable J/fail guard and finite mixed-PDE gradients.
An unchanged-map hard-constraint/JVP regression is running on CPU (exec24444).
Serial cold5000driver156155 (exec30889), tag20261101-5090-tanh, has started
map-only on5090, then map+Jacobian. Watcher157980 (exec41982) will replay both
and compare against tag20261031-5090 on completion. Driver log:
/root/autodl-tmp/data/channel_nsch2d/20261101-5090-tanh-driver.log.
Same16geometry/B2/Nt4/W96/FP32/SOAP5000/seed42; no regional/stagger/GN combination.
First eval/step compilation remains active; no convergence result yet.

Overall goal now explicitly tracked: first achieve relative L2<10% for the
coupled channel phase field on a single geometry, then extend to an asynchronous
online/rolling pool of smooth geometries. Current matched controls still use
16 fixed training geometries but evaluate only one development FEM geometry;
this is not evidence that single-geometry accuracy or infinite-geometry
operator learning is solved. Complete the isolated transform comparison before
changing that condition-bank scope. The persistent goal remains active.

Session verification/resume: legacy hard IC/inlet/wall/query-JVP test passes
302.372s; coordinate4tests pass61.558s. AST/trailing-whitespace checks pass for
all channel Python/new helpers/tests, artifact index JSON parses, and
`git diff --check` passes. No unchanged expensive suite rerun. The source lift
refactor preserves legacy arithmetic and checkpoints; new optional map records
add no model leaves. Map-only step0phi45.3695%,terminal51.1817%,u10.6146%,
Jmin.3863064,inversefactor<=3.2358 at saved evaluation queries; these are
untrained values, not an improvement result. Both matched arms save actual
initial physical predictions for exact paired audit in addition to NN/optimizer
bytes and all real draws/known inputs. First update is still compiling; driver
and validator sessions remain30889/41982. Monitor these, then inspect completed
curves/dense fields, J/range and core/area before choosing a single-geometry
accuracy change. Do not mark the<10%/online-geometry goal complete.

Diff-level session changes: phase_coordinate.py and two opt-in configs; shared
network lift/Jacobian helper, phase-only exact residual scaling/config guards,
training reference-range/J/saturation/initial-prediction records; four focused
coordinate tests; serial driver/independent coordinate validator/comparison
extension. README records derivation/range/failure behavior. Memory updated:
this task, PROGRESS, DECISIONS and artifacts/index.yaml. New generated artifacts
are external, not in Git. Optional PDF extraction is CPU-only and does not
start FEM; no Git operations or file deletion performed. Leave task active.

Optional primary-source follow-up completed: attached4-ManuscriptFile.pdf
(Qiu et al.,20Apr2022,35pages) extracted to/tmp/channel-nsch-paper.txt with
base-environment pdfplumber (noFEM/JAX environment mixing). It predicts(u,v,p,C),
uses swish, original CH/NS residuals and adaptive time marching, not inverse-tanh
output residual preconditioning. Its reversed-vortex section explicitly notes
phase predictions can leave[-1,1], and uses70kAdam updates; it is not evidence
that5000global-time operator updates must suffice. Do not copy inconsistent
extracted equation symbols/potential definitions without independent checks.
Other checked primary sources: arxiv2203.07404 causal loss reformulation;
2302.13368 Phase-FieldDeepONet uses minimizing movement/one-step operators;
2404.18054v3 uses time subdivision/adaptive sampling/mass constraints. None is
a ready-made open-boundary global-time NS--CH recipe; constant-mass projection
would ignore this channel's boundary flux. No additional scheme queued yet.

Goal continuation classification: preceding turn made progress (implemented and
verified exact coordinate controls and launched actual training), not a blocker.
Live revalidation confirms driver156155, GPU child156158 and watcher157980 are
running; map-only first update compilation at~14.5min, no restart/retry.
Old final raw phase/chemical scaled losses are~1.11e-3/5.69e-4 with30.60%phi;
region losses~1.96e-3/1.35e-3 with31.21%phi. This discourages treating additional
loss decrease as solution accuracy; if maps fail, prioritize temporal/transport
consistency rather than simply scaling capacity/budget. Native-operator
validation and open-boundary conservation limitations remain explicit.

## Current goal continuation: map-only saturation failure, transformed arm live

Map-only last logged metric3000phi46.5587%,terminal47.6952%,u4.4564%,p29.5272%,
chemical1392.59%;12/13evaluation times saturate at1.25 and terminalcore0.
Last logged update3200has phase loss93117; next evaluation logging overflows
FP32 a/J to inf and JSON rejects it. Driver156155/validator157980 both exited1;
transformed arm had NOT started. Preserve all failed files, no completed5000
or actual-draw equivalence claim for map-only (sampling.npz never published).

Fixed host-only Jacobian diagnostic using FP64 division when J>0; exact zero
has explicit null/inverse_factor_status=jacobian_underflow_in_compute_dtype.
The training exact-J fail guard at1e-6 is unchanged. Focused host regression
passes (tiny positive J~1e-42, zero underflow, nonfinite error). No mathematical
or original coordinate-test result invalidated; no expensive rerun.

New driver181473, GPU child181476, exec46491, same tag20261101-5090-tanh,
starts only the previously unstarted tanh_preconditioned mode. Driver log
/root/autodl-tmp/data/channel_nsch2d/20261101-5090-tanh-preconditioned-driver.log;
training logtrain-20261101-5090-tanh-tanh_preconditioned-01.log. Initial model/
optimizer bytes, physical prediction arrays and every known-input array match
map-only exactly (independently checked). Step0 Jmin.3863064,phi45.3695%.
Watcher184166/exec84868 uses explicit--incomplete-map-only, will audit the
completed transform against complete old controls and retain truncated map-only
curves/scope. Do not rerun or overwrite the failed map-only run.

Independent CPU partial-checkpoint evaluator added. Saved2500model replay
passes logged metric gaps<=3.69e-8;phi46.5587%,terminal47.6952%,u4.6593%,p30.8630%.
All positive-time saved-query values are1.25 to FP32 roundoff,core0. Partial
phi/FEM/error gallery generated and inspected under the run's partial-2500-figures.
No optimizer recovery or full physical-balance pass claimed for this partial.
Compact receipt docs/tasks/channel-nsch-optimization-validation.json records
all five complete baselines, failed arm and current worker handles. README,
PROGRESS,DECISIONS,index updated. New media/checkpoints remain external.

Overall target is not phase-only: phase is the current bottleneck, but retain
separate velocity and pressure<10% requirements before overall completion;
chemical is reported separately with its weaker FEM convergence scope. After
these isolated16bankcontrols, fix exactly one(.12,.08)geometry for the next
accuracy experiment as user requests, before extending to online geometry pools.
Do not mistake one-FEM-geometry evaluation for single-geometry training.

## Session end: actual transform evidence collected, single BDF2 cold run launched

Map+Jacobian5000 completes1166.84s,warm47.482ms,activepeak3252196096bytes.
Final phi30.5730%,terminal33.5464%,dispersed85.6072%,u4.8590%,p20.8100%,
chemical121.6244%; Jmin.2786059,inversefactor4.4866,no near-bound saturation.
Independent GPU replay/hard constraints pass: savedgap9.54e-7,JVPgap4.77e-7,
fluxgap4.10e-7,div4.84e-8. Full3977grid terminalminphi.82106,corearea0,
diffusearea.0384520(FEM.1520907): no phase accuracy improvement or SOTA.
Actual5000condition/time draws and all known inputs match completed baseline
streams exactly; common69trunk arrays/full streammodel+optimizer bytes pass.
Map-only initial physical predictions also match; its missing sampling file
precludes a partial-arm actual-draw equivalence claim. Five-line convergence
(incomplete map-only dotted) and dense prediction gallery inspected. Receipt:
/root/autodl-tmp/data/channel_nsch2d/coordinate-comparison-20261101-5090-tanh/results.json.
All map drivers/audits have exited successfully except the preserved map-only
logging failure; no restart of that failed arm.

Implemented optional temporal.py BE/BDF2 physical rates and positive-grid time
checks, spatial-only derivative path, loss history evaluations, explicit uniform
interval sampling/logging and independent interval validator. Previous states
are recomputed by the SAME current model and differentiated, no frozen history.
Original JVP branch/checkpoint leaves remain unchanged. First control explicitly
rejects inverse-map/region/GradNorm/stagger combinations. Config
train_single_bdf2.py fixes onegeometry/B1/5000dt.05, additivephase,uniformloss.
Four temporal polynomial/initial-step/state-gradient/sampling/spatial-identity/
full mixed-PDE gradient tests pass64.173s onCPU. Original expensive suites not
repeated.36source AST/trailing-whitespace checks,2JSON schemas and diff--check
pass. README derivation/limitations and artifact index are current.

LIVE exact resume: driver212422, GPU child212431, exec45594; watcher215203,
exec59484. Tag20261101-5090-time, run
/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-time-channel-single-bdf2-01.
Driver log20261101-5090-time-driver.log and training
train-20261101-5090-time-single_bdf2-01.log under existing data root. Actual
known_inputs amplitude array is exactly[[.12,.08]], points(1,3977,2); B1 confirmed
from saved config. Step0phi45.5579%,terminal51.5217%,u10.6146%,p99.1436%.
First update still compiling at~4min; process is live, no terminal failure.
Watcher will run independentGPUreplay, continuous-JVP phase-balance diagnostic,
CPUfields and convergence against old16geometrycontinuous-time baseline with
bundled-change attribution clearly stated. All principal fields and terminal
phase must meet target before single-geometry success; full goal still requires
online rolling-geometry training and unseen-geometry validation. NoFEM worker,
Git operation, file deletion or default promotion performed. Goal stays active.

Diff-level changes this continuation: repaired host-only Jacobian diagnostic
and focused regression; partial-checkpoint evaluator and actual incomplete-arm
plot/audit scope; temporal module/config/network spatial helper/loss integration/
interval sampler+validator and4focused tests; single-run audit/plot watcher.
Changed memory files: this active task, docs/PROGRESS.md,docs/DECISIONS.md,
docs/tasks/channel-nsch-optimization-validation.json,artifacts/index.yaml.
Task incomplete; do not archive. Next poll existing BDF2handle/log, collect
full5000result and independent fields/balances, then select next accuracy step
from evidence. Do not change the online-generalization part of the objective.

## Session end: transport inconsistency measured, conservative control live

Previous goal turn classification: progress (implemented BDF2 and started an
actual single-geometry experiment). Current revalidation collected its terminal
result rather than restarting a live/slow job. BDF2 finishes1064.320s,warm34.324ms,
activepeak1697608960bytes;phi30.2276%,terminal33.7667%,u4.2158%,p21.8539%,
chemical153.5501%. Full replaygap1.19e-6,JVPgap4.77e-7,fluxgap2.33e-6,
div3.17e-8 pass; terminal diffusearea.0615874,minimumphi.720847,core0.
All120intervals visited127..198times; single condition5000visits. Actual
convergence and3977point dense prediction gallery inspected: it still leaves
an upstream diffuse patch rather than advecting a persistent droplet.
Independent continuous-JVP diagnostic at t1.5 has weak.1667526,strong-.00501594,
advection quadrature gap-.1716947 versus M-scaled Lap Gauss gap7.38e-5.
This supports spatial transport/integration inconsistency as a concern; it is
not a proof of a unique cause or a converged-integral direct native baseline.
BDF2driver212422/exec45594 and watcher215203/exec59484 exited0.

Online preparation: added input-only online.py host controller, no optimizer
integration or enabled train preset yet. Stateless unchanged independent uniform
two-amplitude draws, canonical FP64 geometry/GMLS (optional SBP) construction,
explicit casting/known-input schemas, bounded spawned CPUworkers/prefetch,
deterministic FIFO counters, fixed-shape slot updates, reserved-geometry errors,
input/state logs/direct IDs. Three focused tests pass0.556s; later schema
extension to16SBParrays passes the integration run below. Real2worker full3977
node probe installs8inputs,submits12including bounded4prefetch,in1.0761s with
.7894sconsumerwait including startup. NoJAXnumerical/FEM/PDEsolve in builders;
receipt /root/autodl-tmp/data/channel_nsch2d/online-input-probe-20261101/receipt.json.
No infinite-geometry training or generalization claim from these plumbing checks.

New transport.py uses the published Fisher--Carpenter NASA/TM2013-217971
AppendixA.1 SBP4-2 coefficients/positive norms. Mapped curl fluxes from the
SAME stream values give commuting discrete divergence; phase advective flux
integral equals actual open-boundary phase flux under the matching SBP norm.
Tests cover polynomial/SBP/positive quadrature, arbitrary-phase open-flux identity,
curved-wall closure and smooth physical advection refinement;3pass0.031s.
Frozen BDF2checkpoint CPU probe: potentialreplay1.67e-6,SBPvsADvelocity.267217%,
maximum integral-gap5.26e-17 usingFP64stencils/networkFP32. Probe receipt:
/root/autodl-tmp/data/channel_nsch2d/transport-probe-20261101-bdf2.json.
FP32 conservation + full mixed-PDE finite-gradient/phase-only replacement and
all3online tests pass29.161s. No unchanged expensive suite repeated.

Integrated opt-in train_single_conservative.py: onegeometry/B1/BDF2dt.05/
SOAP5000/W96, additivephase and original physical residuals. Only phase
advection and its associated PDE quadrature change; NS derivatives/GMLS
Laplacians/physicalBCs and encoder quadrature remain unchanged. Six extra
explicit known-input arrays are validated and stored, no checkpoint leaves or
physics identity change. Existing defaults remain experimental uniform recipe.
Training-dtype advection/boundary identity is additionally diagnosed; no claim
of exact full mass/energy conservation or original regional-kernel NS.

LIVE exact resume: driver247507,GPU247510,exec36089;watcher250602,exec15292.
Tag20261101-5090-transport, run
/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-transport-channel-single-sbp-01.
Training logtrain-20261101-5090-transport-single_sbp-01.log and driver/
validation logs20261101-5090-transport-{driver,validation}.log under data root.
First update still compiling (~7min at last log); process confirmed live, no
restart. Initial model+SOAP bytes and every10common input array exactly match
the BDF2baseline;6SBP arrays are the explicit new data. Initial scalar phi
45.5579%,u10.6146%,p99.1436% (tiny FP32 inference-rounding differences retained).
After completion watcher independently replays, checks physical/transport
balances and plots fields/curves versus the single BDF2baseline. Next inspect
its full5000result, full-grid core/area and independent FEM metrics; if needed
refine transport/operator resolution from evidence. Only after single-geometry
accuracy integrate rolling inputs into training and validate unseen geometries.

End verification: current38source AST/trailing whitespace checks, strictJSON
receipt/index parsing and gitdiff--check pass. No new binary in Git, cleanup,
FEM/GPU overlap, Git history operation or default/SOTA promotion. Changed memory:
this active task,PROGRESS,DECISIONS,DEVELOPMENT,channel-nsch-optimization-validation.json,
artifacts/index.yaml. Source/tests/helpers/README document the optional numerical
method and input-only prototype. Goal remains active; do not archive the task or
mark the<10%/online geometry objective complete.

## Continuous tanh cancellation and online integration preparation

The actual mixed-field map+Jacobian result remains30.5730%phi versus30.6001%
original; no promotion. Independent continuous eliminated-CH derivation shows a
more careful distinction. For phi=a*tanh(s), J=a*sech(s)^2, write

$$
H=-Aa\phi-\kappa\Delta s+\frac{2\kappa\phi}{a}|\nabla s|^2,
\qquad \mu=JH+A(a^2-1)\phi.
$$

Then the continuous diffusion term is exactly

$$
\frac{\Delta\mu}{J}=\Delta H-\frac{4\phi}{a}\nabla s\cdot\nabla H
+\left[-\frac{2\phi}{a}\Delta s+
\left(\frac{6\phi^2}{a^2}-2\right)|\nabla s|^2\right]H
+A(a^2-1)\left[\Delta s-\frac{2\phi}{a}|\nabla s|^2\right].
$$

No explicit singular division remains in this expression. The independent
2D AD check channel_tanh_elimination_check.py/JSON passes six FP64 amplitude/
latent-shift cases with relative gaps<=3.90e-14. This is not an exact discrete
GMLS chain rule and has not been trained. Current independent mu output does
not enforce this identity exactly. Standard a=1 still excludes overshoots and
requires infinite latent initial pure phases. Thus retain the failed mixed
experiment without claiming all CH transformations are ineffective.

Online preparation now includes canonical train.py integration, still disabled
in old/live presets. Cold opt-in train_operator_conservative_online.py is an
untrained192x80/W96/16slot/B2/20k research preset: two CPUworkers, replace2slots
every50updates, bounded2batch prefetch, no evolved training labels. Context
management closes workers on exceptions; donated replacement keeps shapes,
actual sampled draw IDs/amplitudes preserve geometry identities across reused
slots. Initial/final known inputs and frozen evaluation input snapshot are saved.
CPU audit reconstructs every actual sample/FIFO update and final canonical
operators; unsupported online continuation fails early.

Four focused online tests pass35.129s, including a six-update real tiny-network/
SOAP/full NS--CH/BDF2/SBP canonical training run with two replacements and
independent input reconstruction; synthetic labels are reporting-only fixtures,
not reference-solver accuracy evidence. Deliberate counter corruption fails.
The unchanged cold/full-state continuation canonical regression passes5.364s.
Existing full-resolution input-only two-worker throughput evidence is retained;
no production-resolution online training or unseen-geometry result yet.

## Session end: refined single geometry audited; dense FEM live; finite follow-up queued

192x80 global5000->10000 completes1251.405s(20.86min) stage, warm72.389ms,
activepeak5.5514GiB; cumulative training-stage time2345.276s. Finalphi7.4055%,
terminal10.0041%,u3.2137%,p10.3988%,mu33.9120%,dispersedfraction20.7360%.
The latter is the same physical phi discrepancy with the minority-fraction
normalization, not an independent regional metric. Do not round into success.
Independent replaygap1.67e-6, primal/JVP consistency, actual global sampling
keys/imported bytes, flux3.61e-7 and div4.10e-8 pass. Full15633point terminal
area.1497262 versus original FEM.1520907, negative core.1327981/minphi-1.05533.
The full-grid five-time phase gallery is inspected: droplet persists and
advects through the constriction. Single-validation.json reports targetfalse.
No missing-history or trained-generalization claim.

Dense-reference driver301347 is actually solving the coarse FEM export after
watcher285620 exits and the GPU is empty. It retains original physics, fixed
space/time gates and96x40midpoint queries; independent frozen-model evaluation
uses an explicit GPU subprocess only after FEM exits. No overlap or oldfile
overwrite. Driver log dense-reference-20261101-driver.log under data root.
Goal driver318238/exec29545 waits for it, then conditionally runs the explicit
strict train_single_conservative_long.py to20000. If still failing, it stops
with an exact failed-gate report, never claims completion. If passing, it runs
cold train_operator_conservative_online.py20k then audits actual inputs/physics.
Fresh unseen-geometry reference evaluation remains required afterward. Goal
progress/logs:20261101-5090-goal-{progress.json,driver.log} in existing data root.

Additional algebra: for constant mobility and any fixed linear L,
r_phi+M*L(r_mu)=phi_t+advection-M*A*L(phi^3-phi)+M*kappa*L(Lphi).
This block transformation cancels the independent mu in the phase row exactly;
FP64 independent linear-operator check gap2.22e-16. Unlike single-row division,
it incorporates the constitutive residual. It raises fourth-order/stiffness/BC
questions and is not implemented or claimed superior. The continuous tanh
factorization source/receipt now include this separate discrete linear identity.
No unsupported nonlinear discrete chain rule is invoked.

Diff-level changes this session: complete-state continuation contract and
refinement preset; global sampling audit; passive-advection resolution probe;
dense FEM/frozen-evaluation drivers; canonical opt-in rolling training, FIFO
reconstruction audit and real tiny-PDE tests; continuous eliminated-tanh and
linear block identity check; finite gated strict/online follow-up orchestration.
README/DEVELOPMENT record optional APIs and limitations. Four online tests pass
35.129s, one unchanged canonical cold/continuation regression5.364s; previous
three helper state tests9.734s retained. Current AST/strictJSON/whitespace checks
are required again after final memory edits. No binaries/media in Git, deletion,
Git history operation, default promotion or goal-completion claim.
Changed memory files: this task, PROGRESS.md,DECISIONS.md,DEVELOPMENT.md,
channel-nsch-optimization-validation.json,channel-tanh-elimination-check.json,
and artifacts/index.yaml. Leave the task active, not archived. Exact next action:
monitor the existing dense driver and finite goal driver; inspect dense target
and reference gates before choosing another numerical change. No restart or
new overlapping worker on an observation timeout.

Final session verification passes:59current Python AST/trailing-whitespace
checks,3strictJSON receipts/index and gitdiff--check. Completed training/audit/
test exec sessions are reaped with exit0. The only active task jobs are the
serial dense FEM driver301347 and finite gated follow-up318238; no parallel
JAX GPU/FEM solve was launched. Full goal remains active.

## Dense assessment recovered; strict continuation live; unseen cohort frozen

Previous goal turn is progress: real refinement/audits, canonical online input
integration and independent transform identities. This turn revalidated actual
processes rather than trusting wait markers. Dense FEM completed all three runs
and passes: maxspace/timephi1.5959%/.73775%,velocity1.4525%/.21522%,pressure
2.5835%/1.88790%; flux6.25e-14. The first frozen evaluation failed because
JAX_PLATFORMS=gpu tried an absent ROCm backend in this JAX build. No training
launched through that failed gate. Original failure logs are retained.

Changed only helper subprocess backend to explicit cuda, added evaluation-only
recovery reusing passed FEM, and actually completed it. Dense3840query final
metrics:phi7.34691%,terminal9.96925%,u3.25356%,p10.33462%,mu34.03863%,
dispersedfraction20.57294%. Pressure alone misses the prescribed pooled/terminal
single-geometry gate; individual later phi times also exceed10%, retained in
per-time data. Original all-time joint19.84% includes unconstrained initial
pressure versus FEMzero initialization and is not the acceptance metric.
Add positive-time joint reporting without removing that raw diagnostic.

Recovery driver357259/exec81159 exits0 after plots; failed original301347 and
finite driver318238 are terminal, not live waits. New finite driver358564/
exec84552 uses tag20261101-5090-goal-resume and actually starts strict child
359754, run /root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-goal-resume-single-long-01.
Global10000->20000 imports exact model+fullSOAP state and same original settings/
RNG/LR; initial physical metrics replay exactly. First update is compiling.
Log /root/autodl-tmp/data/channel_nsch2d/20261101-5090-goal-resume-single-long-train.log.
Do not restart on a monitoring timeout.

Frozen unseen16manifest is now actually generated BEFORE online training at
/root/autodl-tmp/data/channel_nsch2d/unseen-geometry16-20261101/manifest.json.
Independent fixed uniform-prior namespace271828/seed20261101, consecutive IDs,
full physical-family/direct-path metadata; not scored or selected by errors.
Initial and rolling inputs reject reserved amplitudes including FP32 matches.
Only known geometry parameters enter the trainer; no future field is loaded.
The exact actual reserved array is saved and independently reconstructed.

New cohort.py strict loader/CLI and cohort_evaluate.py prepare all16individual/
mean/p90/worst principal, minority and geometry-response diagnostics; independent
flux/wall/initial and3time/24query divergence/primal checks per geometry. Global
mean per-case four principal norms<10% is the predeclared population estimate;
worst cases remain explicit, with no uniform guarantee. These production-GPU
cohort computations are prepared, not yet executed. Three pureCPUcohort tests
pass1.312s; canonical tiny full-PDE/SOAP rolling test with a reserved manifest
passes in31.579s combined run. The first cohort return-value comparison failed
on tuple-versus-JSON-list metadata; canonicalizing prepare's returned family
fixes it and the two loader tests then pass.318s. Original failed log retained.

New channel_unseen_reference.py watcher366087/exec36004 waits for finite driver
358564, expects run20261101-5090-goal-resume-online-01. Only a completed actual
online model with physical/input replay can trigger four isolatedCPUFEMworkers,
each serially running coarse/time/fine with exact saved-config JSON and1BLAS
thread. No GPU overlap. Every case must pass original reference gates; none may
be dropped. Then frozenGPUcohort eval and blind predeclared case0/5/10plots run.
If online training was not completed, it reports that and exits without a false
accuracy claim. Source/config/README and DEVELOPENT describe this pipeline.

Frozen pressure gauge diagnostic: CPU/GPUprimalgap9.54e-7, mean traction
projection7.59e-19, pressure10.3346%->10.3929%. The reference was evaluation-only,
not used to choose the correction. No useful gain; no runtime gauge change or
extra pressure Dirichlet anchor is adopted. Probe at data root
pressure-gauge-probe-20261101.{json,log}, sourcechannel_pressure_gauge_probe.py.

## Session end — failed controls retained; temporal consistency isolated

The degree4operator continuation and cold degree4/dual-time control both finish
and pass independent physical replays, but miss the fixed accuracy target.
Dense results respectively(phi,terminal,u,p):
(7.53124%,10.62076%,2.97755%,11.06769%) and
(20.86073%,22.86350%,3.71903%,24.05188%). Operator-prior screening also actually
finishes all1040constructions; maximumcondition8.55e6, no reference solves or
stored operator arrays. Their previous drivers/watchers are terminal. No online
training or unseen FEM accuracy exists yet. Do not mistake prepared integration
or a construction proof for generalization evidence.

Current time-refinement run is active through canonical train.py, not complete.
Frozen best-network rate checks: dt.05 phi/velocity1.2453%/2.8490%;dt.025
.3231%/.7241%;dt.0125 .0816%/.1841% against its physical timeJVP. The new
kind=time_refine contract permits ONLY smaller integer BDF2 subdivision; no
model/map/physics/BC/operator/loss/LR/moment change. Actual production import
confirms model+SOAP bytes exact, parent physical replaygap0 and new/original
physical field gap0. Latest observed stage is first-update compilation; do not
restart or reinterpret compilation warnings as completed updates.

The off-grid cohort reuses identical16reserved amplitudes and physical prior;
101x43query grid has1coincidence versus3840/3840in the previous grid. Pipeline
now includes a separate nominal off-grid reference trio plus16unseen trios,
with alloriginal reference gates and no FEM/GPU overlap. Cohort explicit
query_grid API, tested draw equality, helpers and README/DEVELOPMENT are aligned.
No future validation fields are used in training or geometry selection.

Tanh interpretation remains precise: original quartic CH permits overshoot;
plain bounded tanh is incompatible with the sampled FEM range and pure-state
initial latent values. The tested amplitude1.25transport-row inverse-Jacobian
is algebraically valid but has no measured gain in the older matched16geometry
control (phi30.60%->30.57%). That comparison predates conservativeSBP/BDF2 and
must not be presented as a matched map ablation of the later7.35%model. Exact
continuous eliminated-CH factoring has independently passed; it is not a
nonlinear discrete GMLS chain rule or a trained-success result. Coupled Schur
preconditioning has primary precedent(Bosch/Kahle/Stoll,arXiv1610.03991), but
no new trained coupled-block/tanh elimination was implemented this session.
Region balancing is not adopted. No default promotion, cleanup or Git action.

Verification: new time-refinement guards/fullSOAP import2tests PASS8.269s;
previous combined degree4/dual-time full-PDE gradient PASS123.548s, construction
screen guard PASS.004s, offgridcohort4tests PASS.186s retained.78channel/helper/
testPythonAST+trailing-whitespace and3strictJSON receipts/index checks PASS;
final gitdiff--check required after this note. Diff-level changes: explicit
BDF2time-refinement import contract+two presets+test, frozen-rate probe, finite
follow-up suffix handling, corrected off-grid API/validation docs, actual
completed-control/prior/live-state memory/index. Shared package untouched.

Exact resume: driver477932/GPU477935/exec98359, run
/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-time-refine-single-time-01;
log /root/autodl-tmp/data/channel_nsch2d/20261101-5090-time-refine-single-long-train.log.
Poll existing handles through global15000and independent/dense validation.
Watcher478756/exec76107 expects actual20261101-5090-time-refine-online-01 and
writes unseen-geometry16-offgrid-20261101/time-refine-pipeline-progress.json.
If accuracy fails it must exit honestly without online/FEM. If it passes,
finish actual coldonline20kand all17reference trios/16case metrics/fields before
completion. Goal stays active; no external blocker. Changed memory files:
PROGRESS.md,DECISIONS.md,this task,channel-nsch-optimization-validation.json,
DEVELOPMENT.md,artifacts/index.yaml. Do not archive the unfinished task.

## Session end — temporal accuracy improves; pressure controls fail; coupled finisher live

This goal turn is progress: completed the actual time-refinement study and its
independent audits, two complete frozen pressure diagnostics, implemented and
checked a canonical full-time coupled optimizer stage, then launched it. Full
goal remains active and unachieved; no genuine external blocker.

Completed time-refined global15000 dense3840metrics(phi/terminal/u/p/mu):
6.33299/8.53774/2.85852/10.59792/29.53732%. Minority L2 remains17.7338%; do not
claim all interface/droplet errors<10%. Actual field gallery inspected: droplet
persists but late shape differs. Inventory.152055->.151051, core.131996. Physical
replay1.31e-6, JVPprimal7.15e-7, flux2.82e-7, divergence3.35e-8pass; actual global
draws reproduce. Stage1271.75s,warm72.77ms,activepeak5.5546GiB. New source/curve
compares the same10000parent and original dt.05extension over10000..15000;
no extrapolation. FinalPNG/PDF/JSON at data root time-refinement-comparison-
20261101-final.*, dense gallery under the time-refined run/dense-fem-figures.

Plain frozen pressure-head fit completes every240time/15633space node in~79s:
objective.000368637->.000347449, actual FP32 residual matches quadratic6.5e-11,
well-conditioned scaled Gram(.006996..16.535). Original p10.59792%->10.59316%,
other fields exactly unchanged; no useful target gain. Thermodynamic coordinate
Pi=p+phi*mu independently passes force/traction/JVP identities but its fixed
basis gives original physical p26.02817%, objective.25777, much worse. Do not
promote either. The optional pressure-coordinate extension preserves old
missing-record arithmetic/leaf shapes and is explicitly experimental/off;
physical p, not Pi, remains the reference/traction variable. General formulation
superiority is not inferred from a frozen existing basis. Both complete probe
source/config/model/fields/metric records stay in external data; first waiting
probe was stopped before any update with its interruption marker retained.

Canonical train.py now dispatches explicit train.refinement to refinement.py.
Original-variable/scaled residual and dt.025 remain unchanged. Full objective
and gradient average all240positive BDF nodes in8time batches, differentiating
one current network and all of its history states. FP32 neural derivatives,
FP64host accumulation/SciPy; this is neither time windows nor FP64NNtraining.
Imports complete audited SOAP parent. Inherited known_inputs/sampling/model/
SOAP bytes and initial physical fields actually match exactly. Parent SOAP
state is separate; SciPy final position/gradient is not exact-history recovery,
and no optimizer_final.eqx is emitted for a misleading SOAP continuation.
New full-time nodes/evaluation counts are audited before physical replay.
Accepted and objective-evaluation budgets/terminations remain explicit.

New tests: pressure coordinate/basis/quadratic3PASS12.717s; full-time synthetic
and ACTUALmixedNSCH monolithic-gradient comparison plus parentguards3PASS50.565s;
canonical original cold/continuation/global-RNG regression1PASS55.102s.86current
channel/helper/test Python AST+trailing-whitespace and3strictJSON receipts/index
PASS; gitdiff--check passes. README/DEVELOPMENT explain both opt-in APIs and
limitations. No shared-package/default/reference/physicalBC change, cleanup,
Git history operation or SOTA/goal-completion claim. All prior driver/probe/test
sessions reaped; expected stopped-waiter exit143 is not a training failure.

LIVE exact resume: driver526016/GPU526019/exec65360; first full-time gradient
compiling, no accepted-step result observed yet. Run
/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-fulltime-lbfgs-single-01;
log /root/autodl-tmp/data/channel_nsch2d/20261101-5090-fulltime-lbfgs-single-train.log.
Config examples/channel_nsch2d/configs/train_single_lbfgs.py;250accepted steps,
max1200SciPy evals/history20/gtol1e-6/ftol1e-9. Poll this handle; do not restart
because compilation/observation is slow. Verify actual gradient feasibility,
then independent inherited-input/node/physical/dense field checks and all four
unchanged10%criteria. Watcher528256/exec32759 expects actual audited
20261101-5090-fulltime-lbfgs-online-01; only a passing single geometry starts
cold20kcontinuous rolling-pool training. Then all17FEMtrios and frozen16unseen
metrics/response/fields remain required, with no FEM/GPU overlap. No production
online model or unseen FEM trajectories exist yet. No task archive.

Changed memory files: this task, PROGRESS.md,DECISIONS.md,DEVELOPMENT.md,
channel-nsch-optimization-validation.json,artifacts/index.yaml. New regression,
refiner/probe/controller/curve sources and explicit preset are recorded there.


## Session end — mapping applicability checked; trained spatial consistency isolated

Latest steering remains: do not adopt region balancing as the next remedy;
consider a genuine phase-coordinate/residual transformation. The mixed CH
rows are phi_t+v.grad(phi)-M*L(mu) and mu-A*(phi^3-phi)+kappa*L(phi). With
phi=a*tanh(s), dividing only transport by J does not cancel the independent
mu diffusion. Dividing chemical too leaves mu/J. Plain a=1excludes actual
quartic-CH overshoots and pure compact initial values have infinite latent
coordinates; a=1.25is a checked sample margin, not a universal bound. The old
matched16bank map+Jacobian control30.60%->30.57%predates SBP/BDF2; it is NOT
an ablation of the later6.33%model. Continuous eliminated-CH cancellation has
already independently passed, but neither a nonlinear discrete chain rule nor
a trained-success claim follows. Readme now states these limits and that the
region control stays optional/unpromoted. Primary sources reread:
https://ems.press/content/serial-article-files/15928
https://math.arizona.edu/~kglasner/research/precond.pdf
Glasner's tested examples are Allen--Cahn/solidification; CH/binary-fluid
extensions are proposed in its conclusions, not demonstrated CHNS results.

Actual time-extension250/276control finishes/audits, densephi6.35955%,terminal
8.59676%,u2.85636%,p10.70515%; finite budget, no target gain. Original full-time
control250/274and CPU chemical-time allocation are also completed. All old
handles81471/28698/31012/96274reaped exit0. No online or unseen FEM ran through
them; no old artifact was deleted, renamed or overwritten.

Two new frozen-model diagnostics complete: first full-grid GMLS3/GMLS4versus
AD on phi/mu plus1024velocity queries; second original FP32/promotedFP64 and
384x160G4at the SAME1024physical points with the original encoder frozen.
First physical/saved replays<=1.67e-6; second8.02e-6/9.54e-7. AD precision
changes<=2.14e-5relative, while grid refinement lowers e.g. t=1G4velocity-Lap
38.52%->7.00%,t6 37.65%->6.84%. These are derivative consistency on an NN,
not direct nonlocal/reference solution errors. FP32-named host stencil entry
uses rounded FP32 inputs with FP64NumPy summation, not a GPU reduction emulation.
Reports/source snapshots stay in the corresponding trained-operator-comparison-
20261101and trained-operator-grid-refinement-20261101external directories.

Diff-level additions: two diagnostic sources and one independently manufactured
second/third-derivative test; function-preserving time-transfer code/tests/presets
from the preceding work are now fully checked; two new spatial-only/online
presets use the EXISTING refinement API, no shared-package or core-PDE change.
README explains mapping range, coupled residual terms, discrete-chain-rule
limits and the explicit new trial. Receipt/index now link actual completed
metrics/audits and the correct physical-field folder.95channel/helper/testPython
AST+whitespace and4strictJSON checks pass; gitdiff--check passes. New derivative
unit test1PASS1.447s; time-transfer3PASS39.085s/shared-array1PASS6.317s retained.
Both new presets validate and the actual saved-parent continuation contract
passes. No expensive unchanged test suite is rerun.

LIVE exact resume: driver597710/GPU597711/exec8590, global15000->20000spatial-only
384x160/G3/Nt4/B1/FP32/W96control. Actual imported model+fullSOAP byte equality,
parent physical replay0and new-conditioning maxfieldchange2.10e-5pass. Step15000
sparse metrics agree; FIRST FULL TRAINING GRADIENT IS COMPILING. Do not claim
update speed/memory/L2gain yet. The peak~22GiBwas an estimate only. Existing
log/progress/run paths are listed in this task's authoritative header. Poll the
same job; retain a failed attempt if it actually cannot fit, do not silently
lower batch or change objective. Only an actual passing independent dense
single gate permits cold20konline with the explicit B1/Nt4counterpart. Watcher
598293/exec54693 then waits for audited spatial-fine-online-01before17FEMtrios/
16frozen unseen metrics/fields. No FEM/GPU overlap. Overall goal remains active,
unachieved, with no external blocker, no archive/default promotion/Git/cleanup.

Changed memory files: PROGRESS.md, DECISIONS.md, this active task,
channel-nsch-optimization-validation.json and artifacts/index.yaml. Example
README and explicit experimental sources/configs/tests carry implementation
and stable usage detail; no additional example STATUS file.


## Session end — spatial control complete; mixed-stress formulation validated and live

This goal turn is progress. The384x160spatial-only control actually completes
5000newSOAPupdates/global20000and all physical/input/dense gates. Final dense
phi/terminal/u/p/mu=6.04334/8.30515/2.86748/11.21032/28.14735%,minority16.92267%.
Pressure fails unchanged10%criterion, so no online/FEM follows. Actual training
2007.368s,warm.216910s,activepeak19.96504GiB. Parent fullstate/field replay pass;
first full gradient really fit32GB. Old driver597710/watcher598293and handles
8590/54693exit0/reaped. Fine-grid818known-input constructions and exact recipe
gate pass451.378s. Neither data bank is an evolved training/reference library.

Independent weak-momentum diagnostic first fails its declared quadrature gate;
retain failed-gate/source/log at momentum-closure-20261101, exec67875exit1.
Refined192/384/768volume plus48/96/192/384boundary query study passes unchanged
limits/savedGPUreplay3.28e-6, exec95474exit0. Its whole-domain GMLS/weak momentum
means are close, so NO extra global-momentum loss is added. A separate mapped
viscous SBPflux probe passes manufactured/GCL tests but is worse on the actual
frozen NN(e.g.t1 85.05%vsGMLS38.49%,fineflux19.54%vspriorfineGMLS4 7.00%).
Exec92296exit0and source/report retained; NO production flux operator/switch.

Implemented opt-in mixed Cauchy-stress NS based on Rao/Sun/Liu2020Eqs3--4.
One backbone/six raw outputs(phi,mu,psi,sxx,sxy,syy), physical p=-trace(sigma)/2;
original physical field API stays five channels. Two independent constitutive
rows in body/boundary, explicitly pressure-unit1scale, giving13loss components
instead of9. Original physical BCs and capillary coupling remain, no added
pressure/trace/continuity or duplicate traction residual. GMLS remains phi/mu;
stress gradients use AD, same psi-Hessian maximum derivative order. No all-
spatial nonlocal or learned accuracy claim. This is an equivalent physical
mixed formulation, not a tanh/region metric or copied paper outlet-pressure BC.

Function-preserving transfer adds194parameters. Actual new serialized model
independently verifies69nonoutput arrays and output signs/zero shear against
the learned parent. Actual GPU physical replay0; parent model and stored old
SOAP bytes exact. Initial-transfer-validation.json in the new run records this.
Old SOAP is provenance only; fresh deterministic L-BFGS history has no exact
shape-state restart claim. All240positiveBDFnodes are jointly differentiated,
including current network history states; no frozen previous-time windows.
Cold online preset preserves its own old initialization the same way.

Verification: four new equivalence/transfer/full-PDE accumulated-gradient/
parentguard tests PASS109.666s; old canonical-entry/globalRNG/time-extension
regressions3PASS23.768s. Added three actual PDE/SOAP updates plus parentguard
2PASS50.088s. Its first test attempt failed because a fixture assertion was
inserted in the wrong method; corrected before GPU launch and old log retained.
This is not an optimizer fault. Final static shape-guard identity1PASS.376s.
Global-momentum manufactured1PASS.292s and flux/GCL1PASS.901s.103currentchannel/
helper/testPythonAST+whitespace and12strictJSON checks pass; gitdiff--checkpasses.
A last train-summary label conditional is metadata-only: do not label a valid
non-SBPstress mode as SBP. Current trial has SBP and its numerics are unchanged.

Diff-level code: network output transfer/trace pressure/stress jets, pure stress
constitutive/divergence helper, optional mixed residual/body+boundary constraints,
config guards, full-time importer/reconstruction audit, two explicit presets,
focused tests, and truthful method metadata. Research-only CPU probes stay under
docs/tasks. README/DEVELOPMENT document exact formulations, added outputs/losses,
old checkpoint compatibility, initialization/state limitations and nonlocal scope.
No shared-package, physical-parameter/BC/reference, default/SOTA or Git/history
change; no cleanup/deletion/rename/large generated Git payload.

LIVE exact resume: driver639261/GPU639264/exec66513, run
/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-stress-single-01,
config examples/channel_nsch2d/configs/train_single_lbfgs_stress.py,
log /root/autodl-tmp/data/channel_nsch2d/20261101-5090-stress-single-train.log,
progress /root/autodl-tmp/data/channel_nsch2d/20261101-5090-stress-progress.json.
Initial physical metrics exactly parent; first full-time gradient is compiling,
zero complete objectives observed at last check. Poll same live process; do not
restart on observation/compiler warnings. Verify actual memory/250accepted steps/
termination and reconstructed-output/inherited-input/physical/dense audits.

Watcher641566/exec37363 expects actual stress-online-01. Only a passing single
geometry starts cold20kB2/Nt4/16slot/2replacementsevery50/2CPUworker rolling
geometry learning. New192x80full-prefix818inputscreen passes375.592s, exact
validate_construction_screen matches, maxcondition2.85663e6,poly3.58e-11; zero
solves/stored arrays. Exec58800exit0/reaped; no need rerun. Watcher then all17
FEMtrios/16predeclared unseen101x43metrics/response/fields with noGPU/FEMoverlap.
No actual online model or unseen FEM yet. Overall goal ACTIVE/UNACHIEVED, no
external blocker, no archive or claim that only a single-case solve is enough.

Changed memory: PROGRESS.md,DECISIONS.md,DEVELOPMENT.md,this active task,
channel-nsch-optimization-validation.json,artifacts/index.yaml. Exact header
state and linked actual reports take precedence over older chronological notes.

## Session end: scoped tanh answer and verified auxiliary-stress transfer

User steering is answered from actual mixed CH equations: no region balancing,
plain tanh excludes admissible overshoots; inverse-J magnifies small-J bulk and
independent chemical errors. Earlier30.60->30.57%experiment scope retained, not
an ablation of current6%case. Primary quartic-CH maximum-principle source rechecked:
https://ems.press/content/serial-article-files/15928, introduction.

Diff-level additions: physics-only auxiliary stress fit/test, explicit verified
initializer module/config/strict input and final-audit integration, focused
transfer/corruption test; README/DEVELOPMENT and PROGRESS/DECISIONS/this task/
validation JSON/artifact index updated. CPU diagnostic succeeds, old mixed
control actually completes/fails and is preserved; new separate finite control
launched only after real fit gates, strict actual parent reconstruction and
old driver exit. No labels/default/Git/deletion, no overall-goal completion.
Exact live resume in header, leave task active/unarchived.

Final verification:108PythonAST/whitespace and4current-task strictJSON files pass;
git diff --check passes. Actual GPU initial replay1.1921e-6and parent/oldSOAP/
model_initial/projectedmodel/copiedfit payload bytes independently checked;
run initial-transfer-validation.json records scope. New full gradient compiling,
not a trained-L2 claim; exact resume unchanged.

## Continuation: ordinary capacity fallback and physical-norm cross-check

Previous turn is progress: verified auxiliary solve/strict import/GPUinitial
replay and real finite control launch. Confirmed671008alive/CPUbusy through
14minutes, still first full gradient; keep same process, no restart/overlap.

Pressure-gauge/frozen-row/earlier time-feature controls already negative. Prepare
plain width128/fourheads/64spatial directions(128channels), original8linear-time
directions/maxfreq2, original4outputs/no extra metric,192x80/dt.025/GMLS3/SBP,
20kcoldSOAP with cap256covering256FFNaxes.567300trainable parameters plus128
stopped Fourier entries; maxarrayaxis768
is the existing temporal output projection, whose wide axis remains outside
SOAPcap as in96baseline. Config/schema/physical parity/single-online architecture
and existing818input-prefix contract pass. No actual full-step memory/L2 yet.

Queued driver685888/exec6101 waits for670990and completed failed dense gate,
otherwise skips if predecessor proceeds online. Tag20261101-5090-width128,
single config train_single_width128.py, online train_operator_width128_online.py.
Watcher686167/exec48210uses same16frozen unseen/offgrid manifest after actual
audited width128online completion, then isolated17FEMtrios/frozen metrics.
Generic finite driver now labels cold versus L-BFGS from actual selected config;
no PDE/optimizer/backend behavior changed. README/DECISIONS/receipt/index record
scope. Disk38GiBavailable, no cleanup.

Independent NumPy metric cross-check uses geometry Jacobian h(x) for physical
area on mapped query grid, holding temporal quadrature fixed. Pressure original
query norm->physical area norm:10k10.3346->10.2929%,15k10.5979->10.6093%,
spatialfine11.2103->11.2383%,isotropicstress28.2607->28.1258%. All still fail.
Retain original predeclared gate; no posthoc metric/threshold switch. Definition/
allphi/u/p values and direct paths in mapped_physical_norm_check receipt.

Actual first full gradient now completes;250budget training proceeds without
restart. First objective and latest sparse metrics saved in optimization receipt.
At25phi6.566%,terminal8.098%,u2.957%,pressure20.209%,chemical127.228%; no final
accuracy claim. Current CPU/GUI observations do not stop it. Verified prior
isotropic convergence curve inspected; its pressure spike is an actual result.
New capacity configs/model/schema/physical parity/input prefix checked;110Python
AST/whitespace pluscurrentJSON/diff checks pass. Existing PDE/gradient suites
unchanged and not redundantly rerun. No Git/default/cleanup/goal completion.

## Task-end for capacity-queue continuation

Progress, no external blocker. Actual full gradient and first GPU six-part
objective reproduce the CPU-fitted stress block; gap saved in receipt. Latest
actual sparse eval saved there (latest accepted-step evaluation in receipt), not a final dense gate.
671008is live; driver670990/exec93033still owns current250control and672045/44721
its conditional online/cohort follow-up. Keep it unchanged through completion.

Prepared/validated/queued plain width128single/online configs and generic driver
cold-stage label; no runtime PDE/loss/optimizer change. Queued685888/6101 and
watcher686167/48210only proceed after current failed final gate; otherwise skip.
Next resume: pollsame671008, complete250and independent strict initial-model/input/
physical/dense audit. Then actual queued width128production gradient/20k run if
needed; measure memory/speed/accuracy honestly. Do not silently lower shape/dtype.
Only passing single acceptance starts matching online rolling-pool training and
full16reserved unseen cohort; none exists yet. No overall-goal completion.

Diff-level changes: two capacity configs, generic research-driver label, stable
README explanation; PROGRESS/DECISIONS/task/optimizationreceipt/artifactindex
updated.110PythonAST/whitespace,4strictJSONandgitdiffcheck pass; config schemas/
physical parity/single-online architecture/actual818input screen contract/model
construction pass. No unnecessary unchanged expensive suite rerun, Git/default/
cleanup or archive. Task stays active with exact handles above.

## Frozen exact-stress GMLS first-derivative check: negative

CPU-only research diagnostic688888/exec13200completes, exit0. Independent
polynomial-gradient/viscous-divergence/flat-curved/smooth-refinement test passes
.288s. Actual parent replay3.2783e-6/primal0pass on the same1024queries/four
times, original encoder/known inputs fixed. At t1old velocity GMLS Laplacian
discrepancy38.49%; exact viscous stress differentiated by GMLS3/4gives65.58/55.59%,
fine GMLS3stress17.33%. This does not beat earlier fine velocity GMLS4at7.00%.
No production operator/config/loss option or training is added. Keep negative
result, no first-versus-second derivative general theorem claimed.

Direct source docs/tasks/channel_gmls_stress_probe.py; report/frozen_fields.npz/
source/progress/log in data root gmls-exact-stress-consistency-20261101. Saved
coarse/fine fields/stresses permit later host checks without re-evaluating NN.
Scope is host FP64 stencil sums with actual CPU neural fields, not GPU reduction,
native solution, new reference trajectory or trained L2 evidence. Existing
current250control and queued width128cold trial are unchanged.

## Task-end: stress route rejected after full audit; plain width128cold begins

This continuation is progress: completes the actual strict/dense stress gate,
checks a new derivative variant and rejects it from measured negative evidence,
validates/inspects paired actual physical-L2 curves, and confirms real cold next
training/fresh optimizer. No goal completion, seed selection or default promotion.
First training gradient of700795still compiling; exact handles/header above.

Diff-level changes this turn: research-only GMLS-gradient stress probe and
manufactured/refinement test; paired stress-curve plotting helper; DECISIONS/
PROGRESS/this active task/optimizationreceipt/artifactindex updated. No production
PDE/loss/network/BC/config change this turn.113PythonAST/whitespace,current4strict
JSONandgitdiffcheck pass; one focused numerical test.288s; real frozen replay/
source/input pairing/figure inspection/cold254leaf optimizer checks pass. No
unchanged expensive suite rerun, new FEM solve, Git operation, cleanup or archive.


## Session checkpoint — common-physical response validation and matched time control

Diff-level changes since the preceding checkpoint: optional common-query FEM
export and strict payload/protocol; identical-physical cohort response with
separate mapped response; both-frame refinement gates; pure saved-config
validation imports; explicit watcher protocol plus isolated FEM parity fixture;
focused common-query tests; convergence labels from actual point counts;
matched existing-time-coordinate presets and explicit baseline research-driver
path. Production PDE/BCs/defaults/reference acceptance stay unchanged.

Validation: nine focused checks PASS across three commands; real legacy FEM
refinement calculations equal stored accepted checks exactly. Actual short FEM
export parity0 is compatibility-only. Actual matched GPU initial model/SOAP
bytes and16known-input arrays are exact. 118related Python ASTs, four strictJSON
records, active-record aliases and git diff --check PASS. Earlier expensive
numerical/gradient suites were not rerun without an affected behavior.

Current task/goal remains ACTIVE. Authoritative header names the actual cold
linear/log20k driver743953/GPU743956/exec88813 and watcher745579/exec91559.
First gradient still compiles; no trained outcome or online/cohort exists.
Exact resume: inspect that log/process, finish20k, read driver independent
physical/input/dense assessment, and permit its already waiting online/FEM
pipeline only if all unchanged gates pass. Do not restart on observation
or compiler warnings. No Git history action, deletion, default/SOTA promotion,
FEM/GPU overlap, new supervision or completed-goal claim.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this active task,
docs/tasks/channel-nsch-optimization-validation.json,artifacts/index.yaml.

Final host-only reference-range cross-check: current dense fine FEM queries span
[-1.1366958069,1.0466548511] across all stored times; this is a sampled FEM range,
not a continuum bound. Earlier[-1.129,1.030]statement refers to the older384query
reference. Exact positive-time/three-reference ranges are stored in the receipt.
No reference solve, clipping, gate or training change follows from this check.


## Pressure spatial-mode diagnosis while the matched time-coordinate run trains

Previous goal turn is PROGRESS: completed capacity/audits and common-query
validation, exact matched initialization/new actual GPU training. This turn
observes the same743956 process through compilation and more than2000updates,
including SOAP refreshes. No timeout-based restart, FEM work or shape reduction.

New pure NumPy diagnostic docs/tasks/channel_pressure_modes.py reads completed
original15k and cold128linear20k dense predictions on exactly identical queries,
reference fields/times/amplitudes. Orthogonal constant/physical-x/physical-y/
remainder pressure SSE fractions are .661146/.180119/.000103/.158632 and
.721357/.204548/.000016/.074080 respectively. Thus the affine x plus constant
components account for84.13%/92.59% of pressure SSE. Error-budget closure9.10e-13,
independent direct least-squares coefficients agree7.50e-16; pressure L2 exactly
matches original dense final reports. No correction is applied or optimized.

Report/PNG/PDF/source are at data-root pressure-spatial-modes-20261101; the mean
cross-section profiles at.5/1/3/6s are inspected. This describes smooth global
error and changes the next diagnostic priority if the matched run fails; it does
not prove a unique cause, pressure gauge fix or trained benefit. Live20kbudget
and conditional online/FEM pipeline stay unchanged. Scope remains coupled
physical fields<10%, followed by actual online unseen validation.


## User steering: provisional pressure acceptance and concrete online follow-up

User permits pressure close to10%if the strict threshold remains difficult.
Working interpretation communicated to user: pressure<=11%is provisionally
acceptable, all other principal gates remain<10%; strict reports stay intact.
The current fixed20k training continues. At9500phi/terminal/u/p are5.4994/
8.0721/3.2103/10.3925%; intermediate only, final outcome still pending.

New research orchestration channel_near10_online.py is queued as actual
PID768053/exec84250after743953. It skips duplicate work if strict online already
proceeds. Otherwise it recomputes final saved dense metrics, requires complete
physical/input/reference audits and matching online recipe, and selects first
eligible named FINAL candidate: current width128linear/log, then original15k.
Never choose among intermediate steps/seeds. Fallback staged single vs cold
online is not a matched cold experiment. New online has fresh state and pool.
The helper records exact limits/strict flags/candidate decisions; historical
strict dense reports are never overwritten or reinterpreted as strict passes.

Actual CPUcandidate checks prove10.5979%fallback qualifies provisionally but
fails strict10%;13.2886%linear128case remains rejected. Four invalid tolerance/
nonfinite metric rejections pass. Dense scalar replay gap<=1.39e-17; online
physics/model/operator/optimizer parity passes. New online explicitly monitors
the qualifying3840-query dense gate; first policy receipt with historical384
monitorroot is retained and superseded by near10-online-policy-validation-dense-
20261101.json. No training was launched under that first monitoring record.

New watcherPID768069/exec56180after768053targets
/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-near10-online-01,
tag near10-common, unchanged offgrid16manifest/commonphysicalprotocol/4FEMworkers.
Data-root progress20261101-5090-near10-progress.json, driverlog-near10-driver.log;
watcherlog unseen-geometry16-offgrid-20261101-near10-common-driver.log;
watcherprogress in manifestfolder near10-common-pipeline-progress.json.
No actual online or unseen FEM exists yet. Goal remains unachieved.

Actual dual-time5k CPUcheckpoint metric replay PASSES max1.494e-8, command
exec3746terminal0. Phase gallery generated exec25930terminal0and inspected:
droplet persists/advects but late teardrop deformation remains. Its intermediate
CPUpredictions are not pointwiseGPUparity/full-optimizer/final-physical evidence.
Exact paths partial_005000.json/.npz/figures inside current run. Pressure-modes
report source/AST/strictJSON/exactcopy and gitdiff--check pass. No production
numerical/default change, evolved labels, cleanup, Git or goal-completion claim.


## Session end — active training and user-accepted pressure exception

Diff-level additions: saved-field pressure spatial-mode reporting helper; explicit
user-steered provisional-pressure online-only orchestrator, with named final
candidates, independent metric/audit/recipe checks, source copy, fresh cold pool
and no duplicate strict-path training. No physical loss/operator/model/default
or historical strict-result change. Current training stays on743956/exec88813;
strict driver743953/watcher745579, provisional driver768053/84250and watcher
768069/56180are confirmed live. Online runs and unseen FEM remain absent.

120relatedPython ASTs,4repo strictJSON,4external strictJSON, receipt/index aliases,
actual queued-driver source bytes and gitdiff--check PASS. Actual candidate checks
accept the10.5979%fallback only provisionally, reject13.2886%case and4invalid
limits/metrics. No unnecessary expensive unchanged suite rerun.5kCPUreplay and
field gallery pass/inspected as scoped above. Pressure-mode algebra/profiles pass.

Exact resume: finish the unchanged20k budget and actual strict driver audits.
If strict pipeline proceeds, provisional driver/watcher exit without duplicate
work. If strict single fails, provisional driver checks both named final cases,
uses explicitpressure<=11%/otherfields<10%, then cold20konline with rolling pool.
Its watcher requires actual completed input/physical audit before isolated17FEM
trios, both query-frame refinement gates and frozen16unseen physical-response
assessment. Keep strict and provisional acceptance separate at final audit.
Do not select intermediate checkpoints, restart on timeouts, overlap FEM/GPU,
claim full goal completion, delete files or alter Git. Task remains ACTIVE,
no external blocker, no archive or default/SOTA promotion.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
docs/tasks/channel-nsch-optimization-validation.json,artifacts/index.yaml.


## User cleared automatic goal — authoritative stop of follow-up work

get_goal returns null. Current-turn goal polling ended when the explicit clear
was recognized. Four exact scheduler command identities were verified before
SIGTERM, and743956training was retained; subsequent ps confirms only that GPU
training remains, parentPID1. No cancellation of weights/data, no GPU/FEM overlap,
no new online model or reference trajectory. All scheduler progress records now
preserve their prior state beneath an explicit user-cleared terminal stage.

This is cancellation of automatic pursuit, not objective achievement or a
numerical failure. Task remains incomplete; current physical training may finish
its already-started budget. Independent final validation, cold online20k and
frozen16cohort are no longer automatically scheduled. Only a new user request
can resume that work. Memory/task/index record this exact state and preserve
all previous results and commands; no archive or Git history action.

/task-end: changed memory PROGRESS.md,DECISIONS.md,this task,
channel-nsch-optimization-validation.json,artifacts/index.yaml. Documentation/
JSON consistency and git diff --check are the proportionate final checks; no
code tests rerun for the user-cleared scheduling record.


## Corrected cancellation outcome after execution-session reaping

All four parent execution handles return143. A final process assertion detects
that743956also disappeared after those handles were reaped; nvidia-smi confirms
GPUempty. It had remained live under parentPID1 immediately after SIGTERM, so
the intended retention did not persist. The user was promptly informed that
current training also ended and would not be automatically restarted. This
corrects the previous live-child cancellation note, not a successful20koutcome.

Preserve5k/10kmodel-only snapshots/logs;12100lastloss,12000lastfield evaluation.
No finalsummary/model/optimizer/sampling exists; no exact SOAP continuation or
online/FEM followed. Explicit progress/memory/index states now agree that all
work is stopped pending a new user request. No goal-status mutation needed:
get_goal is null. Final JSON/alias/absent-process/GPU and diff checks replace the
failed assumption that the child would stay live. No new code tests or restart.


## Requested physical-field gallery from latest saved10kmodel — complete

The user asks to see physical fields. This is a bounded evaluation request,
not resumption of the cancelled automatic goal. Latest actual saved checkpoint
ckpt_010000.eqx is independently loaded on CPU using exact saved config. Actual
replay command finishes exec76869exit0. Field L2phi/terminal/u/p=5.52192643/
8.25485886/3.10764881/10.45789545%. Maximum scalar GPUloggap2.987e-8; independent
NumPyrecalculation gap1.39e-17; exact reference fields/points/times agree with
accepted dense localFEM. This is not pointwiseGPUfield or optimizer-state audit.

Root /root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-width128-dualtime-single-01:
- partial_010000.json/.npz: partialmodel replay/fields, no finaltraining claim.
- partial_010000-figures/phi-fields.png and speed-fields.png(+PDF): three rows
  FEMreference/prediction/absoluteerror at0/1.5/3/4.5/6s.
- partial_010000-positive-pressure/pressure-fields.png(+PDF): same rows at
  .5/1.5/3/4.5/6s, with display scales restricted to those times.
- partial_010000-figures/pressure-fields.png retains the original all-time
  pressure gallery, including unprescribed t0; establishedpressureL2excludesit.
- partial_010000-gallery-validation.json: paths/scalars/displayprotocol/verification.
  Plot/evaluation source copies preserved beside galleries; no hashes.

Actual raw galleries exec71951, positivepressure98378and defaultparity88694all
exit0. Images inspected: droplet translates, late outline is more pointed than
FEM; velocity close, visible pressurestartup discrepancy. No evolvedfield
labels trained or new FEM solve. Linear display interpolation changes no metrics.

Plotting-only optional --field/--times selectors added to existing evaluate.py,
READMEusage updated. Missing selectors preserve old behavior; all3default PNGs
are pixel-identical to pre-edit output. Four absent/nonordered/duplicate/unknown
selection rejections pass. Selected five timestamps are checked exactly; scales
shared across times/firsttwo rows, pressureinitial data preserved separately.
Changedmodule AST/whitespace and JSON/diff checks pass, no expensive numerical
suite rerun for this reversible plotting change.

/task-end: bounded requested gallery delivered; original optimization remains
incomplete/stopped with automatic goal cleared, so keep active task unarchived.
Changed memory PROGRESS.md,DECISIONS.md,this task,optimizationvalidationJSON and
artifactindex. No training or scheduling was restarted, no cleanup/Git/default
promotion, no physical-residual modification or completed overall-goal claim.


## /task-end — requested online fields and direct numerical pilot

This session adds a CPU reserved-geometry gallery, directGMLS NS--CH prototype,
three-way saved-field comparison and finite reference-refinement finalizer, plus
threefocusednumerical tests. Production training/physics/defaults remain unchanged.
Online20k and finite saved-run audits completed; direct96control completed; fine
192and bounded half-dt/field validation remain running as described at top.
Partial10kCPUmetric replay, all galleryphysicalchecks, completed onlineactualinput/
model/flux/no-slip/divergence audits, common-frame metrics and inspected plots pass.
Three new numerical tests pass.375s; AST/strictJSON/alias/source and diff checks
are final proportionate verification. No expensive unchanged suite was rerun.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this active task,
docs/tasks/channel-nsch-optimization-validation.json,
docs/tasks/channel-nonlocal-reference-validation.json,artifacts/index.yaml.
Keep active task unarchived until the explicitly requested finite reference
validation finishes. No automaticgoal restoration, default/SOTA promotion,
Git history operation or cleanup.

Final verification: current-source3tests PASS.397s; finalizer CLIhelp, allfive
new/changedPythonAST/whitespace, strictJSON/currentreceipt-index aliases, source
snapshot equality and gitdiff--check PASS. Final20kphi gallery and three-way
contour figure visually inspected; mainfinite runner completed every stage0.


## /task-end — GMLS-first PINO residual experiment launched and live

Diff-level changes: opt-in GMLS domain/boundary gradients, strict21keyonline
payload/indices, PDE/BC/CHflux path, accurate summary/validation/balance
diagnostics, matchedresearchconfig, CPUpriorpreflight, focusedtests and docs.
Originalnetwork/physicallifting/Laplacian/defaults unchanged. Tenfocusedtests,
818boundarypriorchecks and realcoldstate/sharedinput/source/rollingupdate
audits pass. Training is live (1200witnessed), full20k and finalaccuracy audits
pending. Keep task ACTIVE with resume above; no finalprecision/advantage claim.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
docs/tasks/channel-nsch-optimization-validation.json,
docs/tasks/channel-gmls-first-pino-validation.json,artifacts/index.yaml.
DEVELOPMENT.md and exampleREADME document the new opt-in API. Final AST/JSON/
alias/source/whitespace checks complete the proportionate session verification.
No Git action, reference promotion, cleanup or automaticgoal restoration.

Finalcheck PASS:11changedPython ASTs/whitespace, strictJSON, currentversushistorical
receipt/indexaliases, unchanged baseline network/pointmodule/optimizer/runtime
source bytes and gitdiff--check. Later live monitor2500phi/terminal/u/p=
7.8671/11.5342/5.3928/23.7855%, lastloss2800 at observation; independently audited
rollingprefix remains1200 (23refreshes/46inputs), not a claimed full-run audit.
Finite20k run and separate CPUreference validation remain active.

## /task-end — derivative-alignment clarification

User asks whether GMLS-first PINO matches the direct GMLS numerical reference.
Checked current losses.py, network.fields(), matched research config and direct
Collocation/residual source. Same degree3/k36/weight16 GMLS reconstruction family,
scalar/velocity full-cloud Laplacian, CH div(phi*v) form and BDF2 dt.025. This
does not establish an identical discrete system:

- PINO velocity is the unchanged AD streamfunction curl, with continuous
  incompressibility hard constrained. Direct solve has primitive u/v and GMLS
  divergence equations at pressure points.
- Direct pressure gradients reconstruct from the interior pressure cloud;
  PINO uses pressure values over all domain nodes, so those stencils differ.
- PINO natural BCs are soft losses at off-node boundary midpoints; direct
  boundary rows strongly impose conditions at tensor-edge nodes.
- Global-network BDF histories differ from sequential solved-state histories.
  The displayed completed96x40 reference also has lower spatial resolution than
  current192x80 PINO; the separately launched fine reference is distinct.

Communicate partial method/residual alignment and retain discretization/BC and
training error as separate possibilities. No experiment or source behavior was
changed; existing finite jobs and exact resume above remain authoritative.
Memory changed: PROGRESS.md, DECISIONS.md and this task. Source inspection and
git diff --check are the proportionate verification; no code tests rerun for
this clarification. Keep this task active, with final finite-run audits pending.

## /task-end — requested interim result comparison at8000

Read actual journal, process identities, loss/metrics files and CPU steps log.
Matched metric/reference identities checked directly for original and GMLS8000;
timing uses identical5000--8000intervals. Receipt and current aliases updated;
chronological original run is preserved. No model, numerical config, runtime
process, default, Git history or external artifact was changed. No code tests
needed for a read-only status report and memory update. JSON/alias/diff checks
are proportionate verification. Task remains active with exact resume: allow
existing finite20k runner and independent reference validator to finish, inspect
saved-model physical/input/shape audits, then compare final metrics and contours.
Do not restart jobs or claim same-discretization/final/generalization success.

Changed memory: PROGRESS.md, DECISIONS.md, this task, GMLS-first receipt,
channel-nsch-optimization-validation.json and artifacts/index.yaml.

## Requested physical gallery at saved10k — generated and inspected

Canonical run20261101-5090-gmls-first-online-width128-01:
- partial_010000.json/.npz: actual CPU model replay and five-channel arrays.
- partial_010000-figures/phi-fields.png and speed-fields.png (+PDF), three rows
  localFEM/predicted/absoluteerror, times0/1.5/3/4.5/6.
- partial_010000-positive-pressure/pressure-fields.png (+PDF), same three rows
  at.5/1.5/3/4.5/6; unprescribed initial pressure omitted from display.
- partial_010000-shape/contours.png (+PDF) and metrics.json: equal physical zoom,
  zero-contour FEM black versus prediction orange, same phase times.
- partial_010000-gallery-validation.json: exact reference/query identity,
  independent scalar calculation, all four PNG/PDF sizes and source copies.

Replay phi/terminalphi/u/p=6.059923/8.554646/3.822430/10.484431%; maximum actual
logged scalar gap2.407e-8. Exact original13x3840x5reference arrays verified.
This is a one-development-geometry intermediate model, not pointwise GPU replay,
optimizer recovery, full-run audit or unseen generalization. Linear spatial
display interpolation changes no metrics. PNG total1,047,981bytes, all external
to Git. Sources preserved by direct path/byte equality, no hashes.

Images inspected: early phase close; late droplet too wide/flat. Terminal
minorityq24.6198%, interfacephi47.0233%, negative-phaseIoU.76130 and symmetric
contourmean12.8044%initialradius. Bulk globalphi8.55%does not imply shape success.
Velocity structure close with outlet discrepancies; startup/local pressure
differences remain. Only saved-field diagnostics were run, no FEM or training
restart. User's follow-up AD-curl audit continues under this same active task.

## /task-end — requested galleries and streamfunction AD diagnosis completed

New CPU-only research helper channel_streamfunction_audit.py loaded the same10k
checkpoint. Actual invocation exits0, report
run/partial_010000-streamfunction-audit.json; executed source copied beside it.
92querypoints (global+late/early interface), fivepositive times, both actualFP32
and promoted-coefficientFP64 diagnostics; full15633node compatibility check.
This does not train a new FP64 model or alter original checkpoint/settings.

FP32 JVPcurl versus pointwise reverseAD max8.941e-8; batchinvariance2.384e-7;
continuousdiv5.215e-8. FD h.003relative1.979e-5, fullfour-h table retained.
Promoted64 reversegap2.359e-16/div1.422e-16; FD quadratically converges to
relative1.994e-9 at h3e-5. Promoted64/32 velocityrelative1.008e-7. No AD
sign/axis/query coupling error observed. This only validates differentiation
of the learned function, not its physical accuracy or representation efficiency.

Compatibility diagnostic uses FP32 output/weights accumulated inFP64, not an
exact FP32 reduction replay. Terminal interiorGMLSdivRMS1.281e-4, interface
6.496e-6; interface phi*div/advectionRMS2.051e-6, conservativeGMLS minusAD
transport/advectionRMS.001332. Early.5s ratios.0001375/.01417. Fulltime bulk/
interface/productdefect/GMLScurl tables retained. Nonzero discrete curl
commutator confirms replacing AD by arbitraryGMLS curl alone is not a proof
of discrete incompressibility. No large interface divergence anomaly found;
no causal exclusion of accumulated discretization/learnedvelocity/coupled
optimization, and no direct-velocity ablation launched on a diagnostic question.

Gallery/reference/rawscalar/source/image verification and actual audit invocation
complete. New helper AST passes; no full expensive unchanged suite needed. Keep
active task incomplete: existing finite20ktraining/audits and reference
space/time validation continue at their same exact handles/resume points.
Changed memory: PROGRESS.md, DECISIONS.md, this task, GMLS-first receipt,
channel-nsch-optimization-validation.json and artifacts/index.yaml. No Git,
cleanup/default promotion, automaticgoal or new FEM/training queue.

## /task-end — GMLS streamfunction-to-velocity feasibility clarified

Corrected earlier interpretation: user asks about replacing the derivative
mechanism, not an AD implementation defect. Current source retains AD curl;
the prior saved-field audit already evaluates GMLS curl in postprocessing.
Direct derivation yields div_h(curl_h psi)=(DxDy-DyDx)psi; saved10k terminal
interiorGMLS-curl divergenceRMS8.731e-5, so do not promise hard incompressibility
or automatic no-slip/inlet from generic GMLS gradients. Pure spatial stencil
evaluation remains differentiable with respect to network parameters. No new
training/curl implementation/default/process/Git change. Read source/prior
actual diagnostic and checked documentation diff only; no code tests rerun.

Task remains active. Exact resume: existing AD-curl/GMLS-residual20k runner and
direct numerical reference validator retain their finite budgets and handles.
A new GMLS-curl variant must specify discrete continuity and velocity BCs and
be distinguished from the already running residual-only GMLS comparison.
Changed memory: PROGRESS.md, DECISIONS.md and this task.


## /task-end — reference versus PDE/BC diagnosis completed

User's bounded diagnosis is complete. Final GMLS-residual20k metrics against
nominal localFEM: pooledphi4.933101%,terminalphi7.167850%,velocity2.769241%,
pressure9.964289%,terminalminorityq20.416640%,contour10.244912%R0. Cold2318.832s,
warm.076195s/update. Canonical finite runner all five stages EXIT0. Fullactual
rolling audit399refreshes/798newinputs/16rebuiltfinalpayloads; GPU replay2.384e-6,
flux1.915e-7,wall9.609e-8,continuousdiv3.725e-9. Earlier live-prefix1200receipt
is retained as chronological evidence, superseded by final input audit.

Reference evidence:
- Fine192x80GMLS versuslocalFEM terminalphi.266802%,minorityq.769417%,velocity
  .103858%,pressure.562522%,contour.422011%R0. This is cross-method agreement,
  not an exact solution/error bound; halfdt still pending.
- Two actual curved-mesh manufactured nonzero Neumann/traction fixtures pass;
  normalflux errors~1e-12,traction~1e-14,pressureconstant correct. CH weak-form
  signs checked against official2019.1FEniCSdemo. No shared BC bug found.
- Native64x24/.025 FEM re-solve with read-only snapshot hook reproduces original
  entire stored field array EXACT0. Short None/hook parity also EXACT0. Natural
  boundary traces are weak-BC quantities, not pointwise zero certificates.
- Independent nativeGMLS boundary traces maxmu_n3.805e-11,traction6.98e-15,
  interiorcontinuity1.54e-14. Fine maxsectionflux.08666%Q. Coarse27.12%Q FAIL is
  retained, not used as an accepted hydrodynamic reference.

PDE/BC diagnostics:
- All240BDFtime nodes and15633domain nodes for nominal and finaltrainingpool0.
  Frozen preparation verified against canonicalfields9.54e-7; independently
  computed loss versus actual canonical32loss3.18e-8/7.67e-9. No heldout-geometry
  generalization claim from those two cases.
- Nominal mean scaledPDEMSE [1.71195e-4,4.57858e-4,1.62299e-4,1.09353e-4];
  trainingpool0 [1.73001e-4,4.06288e-4,1.54844e-4,9.96899e-5]. Chemical relation
  remains the largest scaledPDE term. 89.15%ofnominal residualenergy is away
  from the declared boundary strip; sampledobjective99.27%PDE.
- Sampledwallmu_nRMS.005616 versus denseGMLS.683376/physicalAD.568993. Original
  48sites/side exclude narrow outlet-corner zones. Independent512site/side
  traces verify97--98%ofwallmu_n squared error lies atx>3.95at3/6s. This is a
  real enforcement/validation defect, not evidence that all wall segments fail.
  Rolling geometry cannot fill fixed parametric boundary-site gaps.

Controlled CH reintegration (same96x40/.025 solver, analyticIC, no training):
- Frozen PINOvelocity + targetzeroNeumann: terminalphi6.485886%,q18.925096%.
- Samevelocity + actual continuous learnedNeumann: terminalphi6.485913%,
  q18.925177%; difference between solutions.002635%phiL2. Thus phase natural
  BC defect is not the main terminal droplet-shape cause in this control.
- FineGMLS referencevelocity interpolated from13times: terminalphi.975329%
  (linear) or.929772%(PCHIP),q2.845901/2.712971%. Velocity source, rather than
  simply reducing phase-equation or Neumann residual, controls this large gap.
  Temporal interpolation/coarse CH uncertainty stays explicit; not a fully
  coupled192grid ablation. All four CH nonlinear residualmaxima<5.5e-13.

Flow diagnosis and remaining limits:
- At6s velocityerror2.77%wholefield masks droplet shape-changing components.
  Droplet centeredvelocityrelativeerror127.4%,strain88.9%; interface35.4%/25.3%.
  Bothreferences on identical common-query derivative stencils differ in
  strain5.93%droplet/2.13%interface, versusPINO87.90%/25.44%againstFEM.
- Fine-grid homogeneous steadyStokes response correcting actual6soutletAD
  traction changes dropletvelocity by7.92e-7 (6.49e-5ofexistingvelocityerror),
  strain1.48e-5ofexistingstrainerror. Unittraction pressure-shift calibration
  and sparse residualpass~1e-10/1e-12. Staticlinear sensitivity does not exclude
  every transient/nonlinear NSCH boundary feedback. Main evidence points to
  chemical/NS coupling and deformation accuracy, not an AD implementation bug.
- Recommended next optimization is coupledPDE convergence with independent
  strain/local-velocity diagnostics, plus robust corner BC coverage. No new
  training or arbitrary loss/physical-model change was made in this session.

Files/verification:
- New research helpers: channel_frozen_ch.py,pde_bc_diagnosis.py,
  reference_velocity_ch.py,boundary_trace.py,velocity_shape_audit.py,
  outlet_stokes_sensitivity.py,bc_diagnosis_plot.py and fem_boundary_mms.py/
  fem_native_bc_audit.py (all with channel_ prefix in docs/tasks).
- solver.fem optional read-only native snapshot hook; absent-hook arithmetic
  unchanged, actual parity/reproduction0. README/DEVELOPMENT document it.
- tests/test_channel_frozen_ch.py: two meaningful polynomial/equilibrium and
  independent Jacobian-FD tests PASS.057s. Actual FEM/MMS/CH/Stokes controls pass.
  Eleven changedPython ASTs, strictJSON/aliases/sourcecopies, real raw-array
  identities, all three inspected PNG/PDFs and gitdiffcheck are final checks.
- Failed first frozen-audit attempt stopped at an unhashableConfigDict static
  check; v2 uses closures and is authoritative. No failed result promoted or
  numerical fallback. Newfigures/arrays/sourcecopies remain external, no weights
  copied/deleted, no Git/history/default/SOTA action or oldgoal restoration.

Main figures under diagnostic root/figures/controlled-contours.png(+PDF) and
pde-bc-convergence.png(+PDF); boundary-traces/mu-normal-traces.png(+PDF).
Compact diagnosisreceipt plus complete sourcecopies and raw240time audits
allow direct-path regeneration, no hashes. Current report aliases and final
20kstate agree; chronological originalrun remains preserved.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-reference-bc-diagnosis-validation.json,channel-gmls-first-pino-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml.
Keep task active solely for the existing independent reference halfdt validation
until its finite gates finish; exact resume at the authoritative header.


## /task-end — dense boundary and coupled-gradient controls implemented; LIVE

Diff-level changes: opt-in positive Gauss-Legendre boundary quadrature with a
separate unchanged encoder stream; strict23array online schema, target-gradient
shapes, replay/summary support; explicit paired research presets; reuse existing
inverse-gradient weighting on the validated joint GMLS/BDF2 path. Stagger remains
excluded; original residual functions and all physical PDE/BCs/defaults unchanged.
Three finite research helpers own input preflight, sequential controls and final
CPU saved-model audits. Existing diagnostics accept residual-only mode and report
actual BC counts. README and DEVELOPMENT describe API/semantics/limits.

11focusedtests pass (after correcting a tuple assertion only); actual3update
adaptive rolling integration and complete gradient identity pass. All818boundary
prior cases and production field parity0 pass. Actual coldmodel/fullSOAP bytes,
16shared input arrays and original encoder features match the completed control.
15changedPython AST/whitespace/sourcecopy and CLI checks pass; no unchanged
expensive suite rerun. Reference space/time/flux gates independently recomputed
from raw saved arrays and now pass, with original coarse failure retained.

First5000checkpoint independently reloaded on CPU, actual metric-log maxgap
9.72678551592e-09. All13x3840physical fields
and source are external direct-path artifacts. Not a pointwise GPU-field,
optimizer recovery or complete-run audit. At final session observation optimizer
step8200, latest monitor8000, actual165refreshes/
330freshinputs installed per producer/refresh logs. Complete sampling/
finalpool audit remains pending. Same5000error is close/slightly higher than
original, so no precision/shape improvement is claimed. Second arm not started.

The two20k controls and finite end diagnostics remain LIVE under the exact
owners/status files above; do not close/archive the task or duplicate jobs.
Next: inspect final fixed-budget errors, contours, minority phase, localstrain
and independent BC/fulltime residuals, then decide whether balancing helps.
No cleanup, Git history operation, SOTA promotion, automaticgoal restoration or
new FEM/reference trajectory. Changed memory files: PROGRESS.md, DECISIONS.md,
this task, channel-densebc-optimization-validation.json,
channel-nonlocal-reference-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml.


## /task-end — requested live result update

At17500updates denseBC uniform phi/terminal/u/p=5.12712/7.46589/5.02046/11.84104%; original matched=5.09497/7.36065/2.91173/9.95141%.
Same nominalFEMreference,3840querypoints,13storedtimes and matched optimizer
step checked from actual logs. The reported errors are wholefield, not final
localstrain or dropletshape metrics. Rawlosses cannot certify unsampled BCs.
Original20kresults are retained separately; no unequal-step accuracy ranking.
Per-run journal/process identities and GPU state confirm first control continues;
second automatically follows its canonical audits. Final independent CPU owner
remains waiting. No restart, cancellation, source/config or reference change.

Updated PROGRESS.md,this task,channel-densebc-optimization-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml. Strict JSON,
currentalias parity, retained historical control and gitdiffcheck are the
proportionate status-only checks; no numerical tests or solves rerun. Task stays
ACTIVE with the existing finite owners and exact resume above.


## /task-end — first final result and adaptive-control status

DenseBC uniform20k=5.00561/7.37444/4.74429/12.02435%vsoriginal20k=4.93310/7.16785/2.76924/9.96429%.
Terminalminority/contour also slightly worse. Canonical savedfield/hardflow/input
checks allPASS; original source/physical parameters/default retained. Balanced
monitor8500=25.52217/26.97420/4.96897/26.28229%, persistent phase failure so far.
Actual initial model/SOAP and23initial arrays are identical between the pair.
Weight trace may explain phase suppression but is not a causal/bug diagnosis.
Keep original20kbaseline and wait for the finite paired endpoint and independent
BC/strain checks. No job/cancel/parameter/reference/Git changes for status request.

Updated PROGRESS.md,this task,channel-densebc-optimization-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml. Read actual
summaries/shape/replay/input/weight/process records; strictJSON/alias/history/
whitespace checks suffice, no numerical test or trajectory rerun. Task remains
ACTIVE with exact existing owners and next actions above.


## /task-end — paired final comparison and independent initial-time diagnosis

Both prescribed20kcontrols and all per-run/final independent stages completed0.
No accuracy gain; original GMLS remains comparisonbaseline, defaults unchanged.
DenseBC sharply reduces independent BC errors but not phase/shape/strain error.
Inverse-gradient control fails phase, loses zero-contour and has a firstpositive
BDF-node CH spike, independently reproduced by unchanged canonical loss. That
node was actually sampled334times. Exact initial/full actual input parity and
raw final metric recomputation pass. Boundary versus dynamics attribution is
more concrete; unique optimization mechanism remains unproven. No newrun queued.

Diff-level changes this status session: completed-result README wording and
final compact validation/currentalias/index/task/dashboard updates; new read-only
first-node audit source/results are external. No production numerical change.
StrictJSON/alias/reference-frame/source/AST/whitespace checks are proportionate;
no unchanged expensive test suite or reference solve rerun. All workers exit,
GPUidle; no cleanup/Git/default/SOTA/automaticgoal action. Keep task active only
for unresolved broader droplet accuracy, with exact next research action above.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-densebc-optimization-validation.json,channel-nsch-optimization-validation.json,
and artifacts/index.yaml.


## /task-end — requested40-time sampling implemented; finite preflight/control LIVE

Diff-level changes: explicit5k/Nt40/8-time-accumulation research preset,
case-local pure scan gradient accumulator and strict schema/mode guards,
canonical training opt-in branch/summary. Original loss arithmetic and model/
PDE/BC/defaults preserved. Two finite research helpers measure real resource
use and conditionally own one cold5k trial; existing runner permits the declared
5k/Nt40 pair and adds final residual-only240node diagnostics. README and
DEVELOPMENT document sampling, one-update semantics and deferred causal control.

Focused numerical checks and corrected narrow rerun pass;8Python ASTs, CLIhelp,
strictJSON/currentalias/sourcecopy checks and gitdiffcheck are proportionate
verification.4-time resource probe complete;40-time resource compilation is live,
so training/error/memory acceptance remains pending. All numerical arrays and
weights remain external, direct paths/no hashes. No Git, cleanup, default/SOTA,
reference trajectory, hidden fallback, oldautomaticgoal or causal queue.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-time40-validation.json,channel-nsch-optimization-validation.json,
and artifacts/index.yaml. Keep task active at the exact finite owner above.


## /task-end — requested40-time result status, no trained accuracy yet

Read actual finite owner/canonical journals, both completed resource reports,
actualconfig, initialmodel/fullSOAP bytes, current physicalmetrics/loss logs and
live process/GPU state. Both probes PASS; cold5k has launched but firststep is
still compiling. No positive-step L2 comparison is available. Report measured
resource cost only; no accuracy improvement, convergence or SOTA claim.

This status session changes records only: PROGRESS.md,this task,
channel-time40-validation.json,channel-nsch-optimization-validation.json and
artifacts/index.yaml. StrictJSON/currentalias/history/source-path and whitespace
checks suffice; no code test, model/reference change, retry, Git or cleanup.
Keep the single task active at existing owner989465/runner4868/child4874 and the
same finite5000budget; next inspect the500step monitor and compare original
at the same step. Causal remains a separate unimplemented/unqueued control.


Final status refresh in the same session: initial compilation just finished,
loss.jsonl now contains the actualstep1record and the process is updating.
The earlier21:13compilation observation above is chronological. Still only
step0physicalmetrics, so no accuracy comparison is available. Currentreceipt/
PROGRESS aliases now say actual updates started; do not infer an exact live
step from a loss log written only every100updates. No process/source change.


Positive-step status during the requested memory investigation:500step monitor
now available. Same nominalFEM/reference times/queries: original4time versus
40time phi16.0392/8.8399%,terminal19.8146/12.7212%,u6.3346/6.7798%,p48.8803/
39.9311%. This is earlyphase/pressure improvement with slightlyworsevelocity,
not final/population evidence or a causal result. Original20kbaseline retained.
Separate compile-only unchunked8/40diagnostic remains live at PID23096; existing
training continues unchanged. Do not reuse preallocation numbers as activepeak.


## /task-end — unchunked40-time memory question answered with scoped compiler evidence

Added one compile-only research helper. Actual8/40full-gradient/SOAP GPU graphs
compiled/EXIT0 without running the updates:9.806/35.346GiB estimated. Full40
plan exceeds31.843GiBphysicalcard;8-time accumulation is retained for the live
control. Disabled-autotuning estimate and intermediate remat warning limits
are explicit; no runtime full40 speed/peak, universal fit or OOM claim. Actual
training continued, positive-step metrics remain separately recorded.

Sourcecopy equality4, helperAST, independent memory-stat sum, strictJSON/current
aliases/history and gitdiffcheck are proportionate verification. No numerical
production change or code test suite rerun; no training interruption/restart,
FEM trajectory, default/SOTA/Git/cleanup or new automatic queue. Research arrays
and generated source/report copies remain external/direct paths/no hashes.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-unchunked-memory-validation.json,channel-time40-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml. Keep task
active for the same finitecold5ktraining endpoint and broader droplet question.


## /task-end — requested live3000step comparison

Read actual finiteowner/canonicaljournals, physical/loss/refresh logs and live
process/GPU state.3250+updates completed, latestphysicalmonitor3000. Compared
SAMEstep/SAMEreference to original; allfourprincipal errors lower, pressure
still14.49%. Currentdrop is promising but not fixedendpoint/matchedcompute or
population evidence. Physical/model/reference config and actualcoldmodel/fullSOAP
bytes rechecked. Original20kbaseline and failedcontrols remain preserved.

This status-only session updates PROGRESS.md,this task,
channel-time40-validation.json,channel-nsch-optimization-validation.json and
artifacts/index.yaml. StrictJSON/currentalias/history/config/state/whitespace
checks suffice; no code/numerical tests, source/default/reference/Git/cleanup or
new queue change. Keep exactexistingowner989465/runner4868/child4874 and finite
5000endpoint/audits. Causal remains separate, unimplemented and unqueued.


Final snapshot refresh in this status session (2026-11-01 21:45China time): latest
physicalmonitor advanced to3500. SAMEstep original4time versus40time:
phi6.93737/6.13720%,terminalphi10.04398/8.27203%,velocity
5.20146/5.03522%,pressure18.63276/12.50463%. Actual
70refreshes/140freshinputs observed. Recent0.475256s/update gives about
11.9mintrainingleft plus owned finalaudits. Current receipts/dashboard aliases
use this newer3500snapshot; earlier3000numbers above remain chronology.
No runtime/source/default/reference/Git action. Keep existingfinitecold5kowner.


## /task-end — uniform40-time final validated; matched causal implemented and launched

Diff-level changes: case-local per-time loss API with preserved legacy reduction,
shared stopped prefix integration across the whole time batch, bounded two-pass
causal gradient accumulation, explicit schedule/schema/preset and separate
canonical compiled step. Physical model/network/templates/residuals unchanged.
Added trace/schedule/wholeobjective reconstruction, strict matched-config and
complete model/SOAP/bank/draw/rawmetric validation. Finite owner requires an
explicit control, checks initialstate/bank before firstupdate, and owns all
existing endpoint audits plus matched-control proof. README/DEVELOPMENT explain
weights, stage-budget choice, logs, scope and finite validation requirements.

Six focused tests pass with one corrected underflow-prone test fixture; actual
3update canonicalrolling/trace integration passes. Saved-source loss/allgradient
45array preflight gap0. Ten changedPython ASTs, actual CLI/config/coldstate/source
and strictJSON/alias/history/whitespace checks pass. No unchanged expensive suite,
FEM trajectory, sourcecheckpoint migration, Git/history/cleanup/default/SOTA
change or automaticgoal action. Generated states/arrays/figures/sourcecopies
remain external direct paths, no hashes. Uniformcontrol finalaudits complete;
new causalcontrol startup/model/fullSOAP/bank parity passed and firststep is
compiling. No positive-step causal accuracy claim. Exact live handles above.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-time40-validation.json,channel-causal-time40-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml.
Keep this task ACTIVE for the sole finitecausal5000 trial and broader shape
accuracy; no further training queued. Do not archive/duplicate existingowners.


Final session refresh: production firststep compiled successfully and actual
causalupdates are running. Live completed logs independently reconstruct global
40time weights and weighted objective; no reset at8time boundaries. Updated
receipt/dashboard aliases record latestlogged step, not an exact live counter.
All physicalmetrics still step0, so no causalaccuracy claim. Earlier startup
statements above are chronological; exact liveowner/finiteendpoint unchanged.


## /task-end — causal40 fixed5k endpoint fails; canonical checks complete

Read actual endpoint modelsummary/metrics/losses/causaltrace and finiteowner.
Training5000/EXIT0 and canonical replay/input/hardflow/causaltrace checks finish;
remaining owned phase/shape/gallery/full240node/paired proof continue. Same-step
uniform40 is better for all four principal fields. Regression follows forced
strength100 with zero late-time weights and growing unweightedphase defect;
do not present the tiny weighted objective as improvement or promote this pilot.
No unique optimizer cause or universal causal failure is asserted.

Status-only diff: current goal/status/exactresume, compactreceipt/activealiases
and observed failure rationale. Preserve old controls/histories. No production
source/config/default/reference/test/Git/cleanup or furtherqueue change.
Verification is strictJSON/alias/history/endpoint/source-of-metrics consistency
and gitdiffcheck; unchanged numerical tests are not rerun. Keep this task active.
Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-causal-time40-validation.json,channel-nsch-optimization-validation.json
and artifacts/index.yaml. No archive while finiteaudits/broader accuracy remain.


## /task-end — existing bidirectional causal controller integrated and cold5k started

Diff: new explicit adaptivepreset/schema, reuse sharedcontroller pergeometry,
pure dynamic epsilonstate in canonicalcompiledstep, everyactualupdate logging
and NumPy statechain validation. Historical fixedschedule/noncausal objective
and model/PDE/BC/optimizer unchanged. Runnerpaired proof now precedes independent
fulltime numericalgate. Stable API/usage docs match. Eightfocusedchecks and
sixAST/config/CLI/preexistingconstruction checks pass; no expensive unchanged
suite or reference trajectory rerun. Existingfailedfulltimegate is retained.

One new finitecold5k is detached in screen with exacthandles above. Startup/
firstactual-feedback and all endpointaccuracy tests remain pending; no new
bestresult/default or guaranteedimprovement claimed. Generated arrays/weights/
sourcecopies external at directpaths/nohash. No Git/history/cleanup action.

Changed memory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-adaptive-causal-time40-validation.json,channel-causal-time40-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml. Keep ACTIVE
for this finiteownedrun and unresolved broader dropletshapeaccuracy.

Session verification: eightfocusedtests PASS; sixchangedPython ASTs, strict
JSON/currentalias/history preservation, exactonlycausalconfigdifference and
tenactualsource/owner copies PASS. Actualcoldmodel/fullSOAP/initialbank parity
PASS. Positive-stepL2 pending; existingfiniteowner remains the exactresume.


## /task-end — requested adaptivecausal liveL2 comparison

Read actual owner/physical/loss/feedback/refresh logs. Startup is complete and
actual gradient/feedback updates run normally. Independently replay every
available used/next epsilon and periodic globalprefix/objective, compare same
physicalmonitorstep/reference against uniform and preservedfixedstage. Report
mixedpartial result rather than guaranteedcausal improvement. Let existing
finite5000 owner continue without any new training/source/default/reference/
Git/cleanup or automaticgoal action.

Status-only memorychanges: docs/PROGRESS.md,this task,
channel-adaptive-causal-time40-validation.json,channel-nsch-optimization-validation.json,
and artifacts/index.yaml. StrictJSON/alias/history/metric-source/whitespace
checks are proportionateverification; unchanged numericalsuite is not rerun.
Keep active for exactexistingowner endpoint/audits and unresolvedshapeaccuracy.


## /task-end — requested live3000step adaptivecausal status

Actualtrain/owner/physical/loss/feedback logs show continuedfinite progress.
Independent allavailablefeedback and periodicprefix reconstruction passes.
Same3000 comparison nowshows both pooled/terminalphase worse than uniform,
velocitysimilar and pressurebetter; earlierpartialterminal gain did not persist.
Report that mixedoutcome, not generalcausal superiority or an optimizer cause.
Existingrun remains unchanged, full5000/ownedfinalvalidation pending.

Changedmemory: docs/PROGRESS.md,this task,channel-adaptive-causal-time40-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml. StrictJSON,
activealias/history/savedmetric/whitespace checks suffice; no source/numerical/
reference/default/Git/cleanup/queue action or test rerun. Keep active with exact
existingowner resume above.


## /task-end — physicalfields delivered; requested timeJVP control implemented and live

Diff: savedarray galleryhelper/validation; physicaltimeJVP+GMLS branch and
continuousresampling preset/schema; exactsampler/temporalcomparison checks;
continuousunweighteddiagnostic; finiteowner/probe extensions and stable docs.
Nineunique focusedchecks pass with one testfixture correction explicitly
recorded, actualGPU20probe and13AST/14sourcecopies/config/CLI/diff checks pass.
OriginaladaptiveBDF5000/allfiniteaudits and samequerygallery complete; phase/
shape stillworse than uniform, so useractivegoal remains unresolved.

One coldJVP5000 is owned above; actualcoldstate/bank parity passes, firststep
compiling, no newL2claim. No extraGPU/FEMoverlap, reference solution/default/
Git/history/cleanup action, seedshopping or hiddenfuturequeue. Preserve all old
successes/failures; do not archive while newrun/shapegoal remains active.

Changedmemory: docs/PROGRESS.md,docs/DECISIONS.md,this task,
channel-adaptive-causal-time40-validation.json,channel-time-jvp-validation.json,
channel-nsch-optimization-validation.json and artifacts/index.yaml.
Generatedmedia/checkpoints/sourcecopies stay external/directpaths/nohash.


### Read-only frozen-model checks while timeJVP training continues

Actualsaved adaptiveBDF5000 network on CPU/3840referencequeries: physicalfield
replay maxgap2.861e-6. Compare physicaltimeJVP with BE-startedBDF2 atdt.025,
.0125,.00625 on t=.025,.05,.1,.5,1.5,3,4.5,6, allsamecurrentparameters.
At t.025 predictedinterfacephi_t relativegap7.4693%/.5421%/.1124%; original
firstpositiveBDFstep is BE while smallersteps useBDF2 there. Laterdt.025
interfacephi_t gaps~.344--.406%. Interfacevelocityrate absolutegap at t.025
.05710 versusJVP RMS.03205; do not oversell a large relativepercentage with
this small denominator. Rate gap is meaningful atinitialtime but does not prove
reference error, uniquephysicalcause or improvedtrainedsolution. NewJVPcontrol
remains necessary. ExactJSON:
/root/autodl-tmp/data/channel_nsch2d/frozen-bdf-time-gap-20261102-01.json.

Separate frozenGMLSdivergence check at0.5/1.5/3/4.5/6: physicalADcurl is exactly
incompressible, its GMLSdivergence is only approximate. But terminalinterface
phi*Ddivv RMS4.942e-6 versusactualCH residual.0138743, and subtracting this term
barely changes residual. Thus a freestream correction is not justified as the
mainlate-shape fix by this evidence. No transport/physics modified. ExactJSON:
/root/autodl-tmp/data/channel_nsch2d/frozen-gmls-free-stream-20261102-01.json.

If requestedJVP result stillmisfits, one generalconditioning candidate to
INSPECT(not yetchange/queue) is streamfunction correction's hardinlet envelope:
code stilluses(x/L)^2 while phase/mu use normalizedinletgate. At initialcenter
x=.8,L=4 multiplier=.04, changing strongly over thetrajectory. A bounded squared
inletgate could preserve value/firstderivative constraints; anyfuturecontrol
needs explicitderivation, hardBC/flux/parity checks and isolatedvalidation.
No evolvedshapeprior or extra regionweight is proposed from this observation.

Session refresh: actualJVPupdates/earlyL2 nowavailable, so earlierfirstcompile
notes are chronological. Updatedcurrentreceipt/main/index/dashboard with live
feedback/prefix reconstruction and frozenrate diagnostics. Ninefocusedtests
remainvalid; no unchangedsuite rerun. Goal ACTIVE, sameexistingowner/finite5k,
no reference/default/Git/cleanup or extraexperimentqueue action.

### New finite follow-up decision after JVP endpoint — normalized stream inlet envelope

JVP5000training/replay complete; phase/terminal/u/p6.77595/9.00161/4.36082/
8.20213%, warm.337752s, activepeak7.77479GiB. Saved actualcontours nearly
coincide with adaptiveBDF; visible late flattening persists. Existing runner
still owns its independent phase/shape/matched/fullcontinuous audits.
One opt-in normalized stream inlet envelope is next under activegoal; exact
hardBC derivation and rejected causal-attribution claims in DECISIONS20261102.
Before launch require focusedFP64 constraints/gradient/config/oldbranch checks,
production bounded resourceprobe and completed existing JVP auditedcontrol.
No physicalmodel/reference/default/Git/cleanup change. Newtrial must preserve
exacttime/condition streams but positive-time cold physicalvelocity differs.

### Session /task-end — requested JVP result and finite normalized-stream continuation

Diff-level changes: physicaltimeJVP/GMLS route, strictcontinuousfreshsampler and
canonicaltimeproof, independentcontinuousprobe diagnostic, matchedBDF/JVPaudit,
explicitJVPpreset, savedfieldgallery, and one optionalnormalizedstreamenvelope/
preset/matchingextension. No sharedoptimizer/PDE/reference/default changes.
JVP5000/allfiniteaudits complete; no materiallate dropletshapegain. Actual new
streamcold5k owner187439/training187445 LIVE, rawcoldstate/initialbank/14source
checks pass, actual firsttrainingupdate pending. No newbinaryGitpayload.

Verification:9unique JVP/causal checks plus3new streamchecks PASS. Initialnew
fluxcheck FAIL2.472e-6 at40Gauss sites; independent80/160quadrature converges
5.125e-10/2.134e-13, exactboundary streamflux gap0. Use160without loosening
2e-11gate; correctedconstraintcheck passes27.510s. Other2newchecks pass in
original82.575s suite. Failure/reason/source/logs preserved in externalprobe.
HistoricalBDF andJVP comparisonschema passes;7changedPythonAST passes.
Production20update normalizedstream probe finite, cold361.266s,warm.331074s,
activepeak7.829GiB; measured resourceonly on2fixedinputs, no probeweightsimport.

Memory files changed: PROGRESS, DECISIONS, thisactive task, completedBDF/current
JVP/newstream receipts, mainoptimizationreceipt and artifacts/index. Keepone
active task/goal (do not archive ormarkcomplete): liquidshape stillmisfits.
Exactresume/newrunpaths/stages in channel-stream-lifting-validation.json andits
external ownerjournal. No Git/history/push/default/cleanup/FEM action.

### Goal continuation — actual owner revalidated; frozen early residual evidence

Previous goalturn classification: PROGRESS (completed requestedJVP5k/allfinite
proof, implemented/checked normalizedstream ansatz and launched actualcold5k).
This continuation revalidated actuallive owner187439/training187445 via process
records, not merelyjournal. Preserveexistingfiniteendpoint, no duplicate run.

Read-only reduction of saved independent240continuous probes: nominal first.15s
(2.5%horizon,6actualprobes) carries47.8605%chemical,32.1666%momentumx,
38.1276%momentumy sampled temporal allocation, but only5.1680%CH. Allfour peak
at first probe t.0169749. This is a sampled integral estimate, not an exact
continuum integral/uniquecause. Other finalpool0case is recorded separately.
Report channel-jvp-initial-residual-validation.json. Earlycoupledinitialμ/flow
consistency remains a candidate ifcurrentcontrolfails; do not add an analytic
hardμ target without checking its actualGMLS/FEMdiscretization alignment.
Currentnormalizedstream experiment stays unchanged; no physics/default/Git/FEM.

### Frozen inlet-scale diagnostic completed (no production change)

CPUprobe source docs/tasks/channel_stream_envelope_probe.py and externalresult
/root/autodl-tmp/data/channel_nsch2d/frozen-stream-envelope-20261102-02/validation.json.
Actualsavedcold FP32parameter/operator values promotedCPUFP64, full15633nodes,
times.1/1/3, no fitting/evolvedlabels/FEM. Both originalquadratic and actualnarrow
envelope reproducecanonical physicalfields/timeJVP EXACTLY(maxgap0).
Independent scalar ADthirdderivatives match closedforms0/-48000/-6.

Cold momentummap(mean scaled mx+my MSE): original.4044734,narrow10.8621069,
geometryheight.5182754. Inletx<.2 contains4.9633%area but9.0594%/93.9787%/
7.0850%ofmomentumdefect. GlobalGMLSdiv RMS1.1642e-5/1.4464e-3/2.0853e-5;
inletRMS1.1680e-7/6.4910e-3/1.6776e-6. ChemicalMSE identical.1564260 inall3,
confirming this changes onlycoldvelocitycoordinate, not phase/chemical output.
This supports narrow-lifting stiffness/discretization mismatch as a reason for
poorinitialconditioning, not a proof of unique trained-error causality or a
successfulheightgate. No runtimeheightoption was added; live5k remainsunchanged.

Firstread-onlyCPUprobe01 stopped before firstcase after actualXLA constantfold
warnings ofcapturedf64dot/reduce-window inputs; source/log/interruptionmarker
preserved. Correctedexecution02 passes model/encoded/data explicitly throughJIT,
EXIT0 and exactcanonicalparity. No numericaltrainingpath changed or gate relaxed.
Current maintrial1500phi40.059%,terminal43.848%,u37.146%,p70.061%; gradient
updatescontinue. Retain fixed5000and allfiniteaudits before nexttrainingdecision.

### Dashboard memory compacted during verified wait

PROGRESS1245lines -> short current dashboard, with SAMEoneactive task/goal,
exactowner/failure/limits/nextaction and linked receipts. Full previouscontents
preserved in docs/tasks/archive/project-progress-through-20261102.md; relative
link destinations independently checked unchanged. No active task archive or
physics/default/runtime/Git action. Earlier results/gates remain traceable.

### Session /task-end — failed narrow endpoint verified; smoothheight cold5k live

Goalturn classification PROGRESS: completed actualnarrow5k/allfinitechecks and
rejected itsaccuracy; produced exactfrozen envelope/representativegradient/
phase-subspace diagnoses; implemented/checkedsmoothheightflag and launched one
actualcold5k with rawstate/input/source parity. GoalstillACTIVE/unachieved;
originalshape stillneeds improvement, no genuineexternalblocker.

Diff-level runtimechanges limitedtoexplicit normalized_height gate, strict
schema, opt-inpreset and itsphysicalcontracts. Oldbranchesarithmetic preserved;
sharedNN/optimizer/PDE/BC/reference/default untouched. Newread-onlyhelper sources
trace all frozenexperiments; no fittedprojectionweights/checkpoints emitted.
PROGRESS compacted1245->shortdashboard with fullhistoricalsnapshot and123rebased
linkdestinations verified unchanged, no missinglinks/active-task switch.

Validation: targeted5unittests PASS51.704s, including newheightconstraints/full
parameterFD andlegacyregressions; subsequenttestimportcleanup confirmed2entries
without changing numericalcontracts.7AST/strictmatchedconfig/diffcheck pass.
Frozenmapping parity0, scalarhthirdclosedformchecks pass. Bothfrozengradprobes
finite/group-partition pass (first8times early-biased,40strataauthoritative).
Phaseprojection reconstruction1.55e-15/GPUreplay4.26e-6/rank129 checks pass;
49920reference rows usedONLYforprojectiondiagnosis, no trainingtransfer.
GPU20update probe finite/timekeyidentity pass, warm.336673s/peak7.588GiB.

Exactresume currentowner232327/training232333, source/journal/currentreceipt
above. Firstfulltraining update pending; resource20updates do not implymain
5k hasupdated. Watchfirst500L2thenfixed5000/shape/fullcontinuousprobes. No new
FEM/default/SOTA/Git/history/cleanup orlater queue. Media/data remainexternal.
Memorychanged: PROGRESS,DECISIONS,thisactive task,completednarrow/currentheight
receipts,earlyresidualreceipt,mainoptimizationreceipt,index,and dateddashboard
snapshot. Active task is not archived; check actualprocess before futureaction.
