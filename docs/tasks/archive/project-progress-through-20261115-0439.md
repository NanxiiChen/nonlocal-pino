Historical dashboard preserved before compaction. See the [current dashboard](../../PROGRESS.md).

# Project Progress

## Active task

[Five-PINO three-method comparison suite](../pino-three-method-suite.md).
Complete derivative controls and component figures in each README/local website.
Four cases retain Nonlocal/Local AD/FEM PINO; on2026-11-15the user explicitly
omits Spinodal Local AD after measured cost, retaining Nonlocal/FEM PINO.
Completed the [Electropolish/Melting time-fusion controls](pino-time-fusion-comparison.md):
spatial attention + layerwise time FiLM versus joint(x,t)query without FiLM.
**Incomplete; no current Git or deployment authorization. Preserve SOTA defaults.**

## Current coverage

| PINO | Completed evidence | Remaining work |
|---|---|---|
| Ceramic corrosion | Three final20k methods;64local-FEM tests; component figures and local README/site verified against raw values | Complete; preserve selected assets |
| Coupled two-phase flow | Three final20k methods;3common reserved local-FEM geometries; all audits/figures, local publication and final suite review complete | Complete; preserve selected assets |
| Electropolish | Corrected three10k arms; exact pairing;64local-FD tests at all41records; figures/README/local site verified | Complete; temporal control also locally published |
| Melting3D | Three cold10k arms;64native tests; exact paired audit, figures/README/local site complete; separate localCG2case005 | Complete; temporal control also locally published |
| Spinodal | All8 local references and original resource/pairing audits pass; AD omitted by user | Fresh Nonlocal50k running; FEM50k then paired final audit queued |

[Direct-path inventory](../../../artifacts/pino_three_method_suite.json),
[metric/figure guide](../../PINO_COMPARISON_FIGURES.md).
Melting64native tests are not64independent local-FEM references.
Spinodal local PDE refinement/resource checks pass; final50k controls remain pending.

## Completed results and qualifications

Channel final20k means on3reserved cases:

| Method | Phase | Velocity | Pressure | Chemical potential |
|---|---:|---:|---:|---:|
| Nonlocal | 1.6435% | 1.5172% | 5.2840% | 13.9541% |
| Local AD | 2.1540% | 2.0866% | 8.5370% | 16.0556% |
| FEM PINO | 4.6465% | 4.3892% | 7.1489% | 43.7908% |

Phase includes13times; other fields use12positive times. Velocity uses a joint
vector norm. Only3reused geometries and one initialization; no universal ranking.
AD uses2048fresh interior queries plus192BC points,B8/Nt36/block9. Cold20k
process took10.15h; same-card warm probes are1.60sNL,1.74sAD,1.42sFEM.
Historical NL full-process timing used another physicalGPU.
Channel figures/README/local site pass nine-page build,342local links, raw-table,
byte-identity and PDF checks; no browser GUI or deployment is claimed.
Receipt: `artifacts/channel_three_method_publication_validation.json`.
Final requirement review at22:33UTC also checks all9actual field pairs/36norms,
complete41record histories and all four current source/repository/built-site
assets. Receipt:`artifacts/channel_three_method_suite_review.json`. Channel has
no remaining execution or publication work within the declared three-case scope.

Electro corrected final10k means on64local-FD cases:

| Method | Phase | Concentration | Channel mean | Cold process |
|---|---:|---:|---:|---:|
| Nonlocal |1.9713%|2.3563%|2.1638%|15.63min|
| Local AD |4.3904%|5.1374%|4.7639%|18.27min|
| FEM PINO |1.8486%|2.2072%|2.0279%|15.44min|

The strict audit passes identical cold model/Adam state, full recorded draws and
exact saved common-input witnesses. All41evaluations use the same64local-FD
cases/21uniform times. FEM is slightly more accurate in this one run; both
outperform AD here, without a cross-seed ranking claim. AD warm updates are
faster and use less active memory, but compile longer. Actual resource scopes
are separately recorded in `artifacts/electro_three_method_resources.json`.
Common prescribed inputs use the existing CPU FP64 SDF with one FP32cast;
production training/defaults stayFP32 and unchanged. Original attempt02failed
pairing remains retained as provisional historical evidence.

Four reviewed Electro PDF/PNG files total345,294bytes; the channel/Electro
collection totals855,948bytes. Example/rootREADME and local site are updated.
Nine-page build,350local links, all table values and source/repository/built-site
asset identity pass.192saved field pairs/384channel norms were independently
checked. No browserGUI or deployment claim. Receipt:
`artifacts/electro_three_method_publication_validation.json`.

Ceramic final current-tree review passed at19:43UTC: all six component means,
64sample channel-mean identities, direct model/reference paths and four
source/repository/built-site asset identities agree. Development4/weighted norms
and final64/equal-node norms remain distinct in both captions. Receipt:
`artifacts/ceramic_three_method_suite_review.json`. No training or figure changed.

## Live execution and next action

**User scope change2026-11-15: skip Spinodal Local AD.** No further AD
optimization/probe/long run is scheduled. The proposed512query-chunk probe was
cancelled before launch and its unused CLI edit reverted. The original three-way
queue and Taylor process are terminal; no abandoned AD trainer remains.

The completed actual-workload Taylor30probe reduces median update time from
8.207780to3.981214seconds; projected50k work113.997to55.295hours. Active peak
20.438GiB. All original initialmodel/SOAP/input arrays and finite state pass.
Independent receipt:artifacts/spinodal_taylor_gpu_resource_128.json. These are
resource/finite-state measurements, not final L2 accuracy. The expensive
fourth-order local derivatives arise from eliminating chemical potential; do
not generalize this cost to every AD formulation. Original/taylor evidence kept.

Latest verified process observation is2026-11-15T04:31:47.499556+00:00:
owner590811/start931260421 and Nonlocal child590819/start931260479 remain live;
CPU preview607224 waits. Nonlocal loss reaches32800; latest complete evaluation
32500 has phase31.7232%, chemical46.6139%, channel mean39.1685% against local FEM
(equal-node norms). Native volume-weighted phase is6.3256% (6.5654%at26500).
FEM50k is queued. Recent26500->32500inclusive rate is0.150678097s/update;
about0.732hremained for Nonlocal at that observation, excluding later
stalls/audits. This projection is not complete cold training time. No restart or
source/default change. Receipt:artifacts/spinodal_current_training_observation.json.

Read-only old/new comparison passes on the same8cases: at20500, native weighted
phase is old8.6213% versus new8.6880%; at10000,13.0099% versus13.0702%.
Old final50k native weighted phase4.8329% remains the historical result; new
final50k is pending. New control initial-PDE fraction is0 rather than0.2, with
hardIC preserved; different runtime/GPU also precludes a byte-identical replay.
Independent recomputation of the old final fields using the same equal-node
metric gives4.9174% against native references and32.2170% against local FEM.
Thus the large local-reference discrepancy already exists in the old model;
it is not evidence of a new training regression. Receipt:
artifacts/spinodal_sota_repeat_comparison.json. Final comparison should report
both references with equal-node norms; retain historical volume weighting as
explicit supplemental evidence. Neither numbers nor norms may be conflated.

- Journal:/root/autodl-tmp/tf-logs/spinodal-nonlocal-fem-20261115-01/journal.json
- Train log:/root/autodl-tmp/tf-logs/spinodal-nonlocal-fem-20261115-01/train-nonlocal.log
- Launch:artifacts/spinodal_nonlocal_fem_launch.json
- Qualification:artifacts/spinodal_nonlocal_fem_queue_qualification.json
- Queue/audit:docs/tasks/spinodal_nonlocal_fem_queue.py and spinodal_nonlocal_fem_audit.py

The queue reuses completed resource probes and8qualified references; it does
not rerun FEM solves. Before launch, exact numerical/reference source bytes,
normalized configs (including explicit CLIseed42), common cold state and raw
last20timing medians pass. Final-field audit body matches the retained original
helper exactly; only its selected method list changes. Original reference
helper/source snapshots remain untouched. New task-only scripts compile.

| Method | Stable inclusive seconds/update | Projected50k update-hours | PeakactiveGiB |
|---|---:|---:|---:|
| Nonlocal |0.124417|1.728|20.438|
| FEM PINO |0.098772|1.372|21.717|

Projections exclude compilation, evaluation/checkpointing and long-run rolling
stalls. All8reserved local-reference cases are evaluated every500updates and at
final50k. This is the entire declared Spinodal test cohort, not64geometries.
Keep native and local reference metrics separate. After final audit, inspect
the automatic two-method convergence/distribution previews and update Spinodal/root
README/local website. The old three-method figure owner stays stopped.

The independent CPU helper docs/tasks/spinodal_reference_norms.py is ready
for the eventual passed final.json. It recomputes both native/local equal-node
phase/chemical/channel means and positive-time chemical diagnostics; physical
area weighting is explicitly secondary. All32old-case phase norms, old weighted
mu and the independently checked reference gap match prior evidence. Invalid
budget, zero norm and incomplete method sets are rejected. Qualification:
artifacts/spinodal_reference_norm_qualification.json. This does not change live
training/render sources or establish completion of the new50k runs. Execute
the recorded command after the original final audit, before final tables.

Two-method plotting/loader/publisher/site support is ready, with explicit
spinodal_nonlocal_fem kind and nonlocal_fem filenames; keep NonlocalC0/FEMC2.
Other cases still require three methods. Fourdata-I/O tests pass, separately
verified plot dispatch rejects cross-case omission, and9-page localbuild passes.
No finalSpinodal image was generated or promoted. NewCPUpreview owner607224/
start931308501 is verifiedwaiting at2026-11-15T03:10:33.290141+00:00 on exacttrainerowner590811.
Its7render sources are snapshotted and now frozen; output after actualfinalaudit:
/root/autodl-tmp/tf-logs/spinodal-nonlocal-fem-figures-20261115-01/spinodal/provenance.json.
Receipt:artifacts/spinodal_two_method_figure_qualification.json; launch:
artifacts/spinodal_nonlocal_fem_figure_launch.json. Inspect actualnewfigures before
localpublication. This is implementation/queue progress, not a newL2 result.

Next routine check after2026-11-15T04:46:47.499556+00:00; use existing /tmp/pino-threeway-refresh-memory.py.
OneGPUtrainer, noFEM/JAXconcurrency, noGit/deployment. Preserve all other four
completed derivative studies, both temporal studies and allSOTAdefaults.

All8local references retain refinement/invariantqualification: maxphase meshgap
0.67923%, positivechemical0.74673%, phase timegap0.0010383%, mass3.49e-16,
noenergyincrease. Only1029required h=.025/.018; others .035/.025.
Bank:/root/autodl-tmp/data/confined_ch2d/operator_threeway_local_20261114_v2.
The native/local phase trajectory gap averages31.9633% (21.8253–43.5168%).
Full-horizon interior graph/continuum-nonlocal derivative gap averages0.2812%;
continuum nonlocal/local14.8231%, showing a finite-horizon contribution without
attributing the entire trajectory discrepancy. Receipts remain
artifacts/spinodal_local_reference_validation.json and
artifacts/spinodal_reference_gap_eight_cases.json. No numerical/default change.


Melting exact paired/source/10kboundary replay and192saved-field norms pass.
Final64native mean phase:0.2860/0.0850/0.1093%(NL/AD/FEM). Separate localCG2case005:
0.1928/0.1208/0.0671%. AD is best on the native cohort; FEM best on the one local
case. Native neural errors use24³queries×5times, not full128³/local-continuum L2;
localcase005retains its unpassed strict-refinement flag and practical acceptance.
Same-card cold training6.89/10.37/7.35min; warm inclusive medians15.81/25.17/16.16ms.
Four reviewed assets215,873bytes; channel/Electro/Meltingcollection1,071,821bytes.
README/root/local site and exact resources agree; nine-page build/358local links,
font/edit checks and source/repository/built-site identities pass.
Receipt:`artifacts/melting_three_method_publication_validation.json`.

Electro temporal final10k means FiLM/joint:phase1.9713/2.0577%,concentration
2.3563/2.4520%,channelmean2.1638/2.2549%. Actual shared-backbone/draw audits pass;
storedentries320706/264738 differ and are explicitly reported. Same-card cold
15.63/23.65min,warm-inclusive64.61/105.76ms,active12.03/32.45GiB. Accuracy is
close in this one realization; FiLM costs less training work here. Whole64monitor
medians0.412/0.445s include host error reductions, not isolated network inference.
Four reviewed temporal assets294,876bytes; nine-page build/367local links and
128saved-field pairs/256norms, tables and source/repo/site identities pass.
Receipt:`artifacts/electropolish2d_time_fusion_publication_validation.json`.
Melting temporal finalnative phase0.2860/0.0935%(FiLM/joint); separate accepted
localcase0050.1928/0.1302%. Cold6.89/14.99min,warm-inclusive15.81/60.44ms,
active1.45/9.36GiB. Joint is more accurate at this finalbudget but more expensive;
late-step variability, one initialization and unequal capacities remain explicit.
All128field pairs/128norms and shared-backbone/sampling audits pass. Four reviewed
Meltingassets184,109bytes; combinedtemporal478,985bytes/eightfiles. Nine-page
build/375local links,tables,exact arrays and source/repo/site identities pass.
Receipt:`artifacts/icemelting3d_time_fusion_publication_validation.json`.
The completed temporal child is archived; the parent Spinodal task stays active.

All-test monitoring:Electro/Melting64every250; Spinodal8every500. Existing
ceramic/channel scopes stay unchanged. No default change, Git or deployment.

## Complete training-cost report

User requires cold launch through training exit, including all setup/compilation,
updates/rolling waits, in-training evaluations and final writes. Verified14parent
training intervals: Electro15.63/18.27/15.44min; Melting6.89/10.37/7.35min;
Ceramic26.81/28.47/21.01min; channel approx9.18h/10.16h/8.03h (NL/AD/FEM).
Channel Nonlocal is historical on a different physical GPU and only approximate
external UTC timing with ten-second exit polling. Ceramic used another older
physical PRO6000 common to its three controls. Do not call all arms current-card.
Temporal FiLM/joint full-process costs: Electro15.63/23.65min, Melting6.89/14.99min.
Warm-only or internal-trainer times are not the primary total. Separate offline
banks, probes and post-training audits remain outside the training interval.

Corrected Ceramic/channel comparison text; added common tables/protocol to the
figure guide and website training chapter, linked from root/site overview.
All accuracy figures/weights and frozen live numerical sources unchanged.
Receipt:artifacts/pino_training_wall_costs.json; validation:
artifacts/pino_training_wall_cost_validation.json. Nine-page local build and
rendered tables/links pass. No deployment or Git action.

## Retained history

- [Channel child](channel-local-ad-comparison.md): complete.
- [Ceramic figures](ceramic-component-comparison-figures.md):
  NL/AD/FEM phase3.8175/32.2070/5.5535%,concentration4.6205/32.6060/6.5937%.
- [Completed two-method FEM study](fem-weak-pino-comparison-plan.md)
  and historical assets remain intact.
- [Latest authorized publication](channel-statistics-publication.md):
  mainf2b9127/Pages43556036346. Later local changes are not deployed.
- [Stopped sampling research](../pino-unified-rolling-core-halo.md) stays stopped;
  Spinodal retains its accepted full-field route.
- [Older dashboard](project-progress-through-20261114.md) and older
  task observations are historical, not current process status.

[Split/recovery guidance](../../migration/VALIDATION.md). Preserve external paths,
unrelated worktree changes and accepted assets. No Git operation this session.
