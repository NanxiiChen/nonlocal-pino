# Full-grid versus rolling core/halo particle corrosion

Date: 2026-11-06. Status: complete.

## Completed outcome and exact resume point

The requested full-grid cold20000 comparison and all final validation stages
completed. No process or required work remains. The existing rolling core/halo
default is unchanged: this task requested a comparison, not default promotion.
No Git operation, artifact cleanup, new FEM solve or follow-on training.

Same8reused localFEM geometries, arithmetic mean of per-geometry physical L2:

| Quantity | Rolling core/halo control | Full-grid endpoint |
|---|---:|---:|
| Phase pooled / terminal | 1.982720% / 3.120851% | 1.745104% / 2.865778% |
| Concentration pooled / terminal | 2.396368% / 3.740510% | 2.109341% / 3.448075% |
| Corroded field pooled / terminal | 5.900376% / 6.859836% | 5.195687% / 6.307766% |
| Terminal IoU | 96.2247% | 96.4870% |
| Held neural update median | 22.5625ms | 32.5267ms |
| Weight-refresh median | 57.8155ms | 95.0376ms |
| Total run, timing.json | 959.0387s | 1065.8039s |
| Active memory peak | 4.9282GiB | 7.1800GiB |
| Producer wait (already in total) | 158.8213s | 25.5436s |

Full phase pooled improves11.98%,terminal8.17%;8/8individual pooled errors and
7/8terminal errors improve. Total run+11.13%,held-update+44.16%. Full online
overhead32.431s includes25.544sproducer waiting; do not add it twice. Different
compute/producer overlap limits inference from total runtime alone. This is one
paired initialization and a previously inspected eight-case cohort, not a
population or cross-seed guarantee. Final dev4phi1.74023%/3.06340%,c2.11011%/
3.69554%. Whole-mesh results remain above the original1%aspiration.

Audits passed: all20000consumed identities,399events,316newgeometries,
1520spatialrefreshes,380distinctgeometries/1900spatialversions,64final arrays
exactly reconstructed. Separate128initial/final slots independently rebuild
mesh/CSR and check every physical coefficient, allfree rows, volumes and zero
mouth/padding PDE weights. Coldmodel/fulloptimizer,allslot/pool/time and actual
geometry/revision traces, installed geometry/IC seeds and initial/final12fixed
input arrays match the control exactly. Only sample.full_grid differs in config;
all shared physics/network/operator source bytes match. No labels or evaluation
overlap.16new focused tests plus26legacy regressions pass; full32step resource
qualification and final saved-field/metric replay complete. No need to rerun.

Final artifacts:
- Run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-full-grid-w192-cold20000-01
- Reused8evaluation: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-full-grid-w192-reused8-final-01
- Comparison/curve: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-full-grid-vs-core-halo-w192-01

Diff-level changes: opt-in full_grid payload/preset, canonical train.py control
and single-graph gather, online payload adaptation preserving the old branch,
initial/final input audit extension, independent full-grid pairing/CSR validator,
finite resource probe, saved-field comparator, focused tests and documentation/
artifact/memory records. Model, PDE, original loss and optimizer code unchanged;
no curated image modified. Exact next action: none; preserve both full/core runs.

## Goal and acceptance

User requests the next comparison using the complete mesh, without core/halo
subsampling. Keep the current rolling geometry prior, network, Adam20k, five
losses, references and time/geometry draws unchanged. Compute PDE residuals at
all non-mouth mesh nodes with their physical volume weights; include every
node needed by the full nonlocal operator. Initial/boundary loss sampling stays
matched, so this isolates PDE spatial sampling rather than also changing ICs.
No supervised targets, new FEM solves, default promotion, Git action or deletion
requested. Test meaningful operator/weight/masking parity, measure full-size GPU
memory/step cost, then cold-train and compare the fixed20k endpoints.

## Control and context

Rolling core/halo control:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-geometry-spatial-w192-cold20000-02.
Config train_online, width192/B4/Nt8/Adam20000; runtime seed1607243318.
Same64initial inputs at
/root/autodl-tmp/data/particle_corrosion2d/20261106-eight-input-bank64-01.
Development4: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-local-reference4-01.
Reused8: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-local-reserved8-01.
Baseline same8phase1.98272%pooled/3.12085%terminal,c2.39637%/3.74051%,
held22.5625ms,total959.0387s,activepeak4.9282GiB. All20kidentities and64final
slots audited;380unique geometries across the complete schedule.

Preflight mesh nodes range3977--4257,median4135 (current core count256 and
decoded halo padding3072). Full-grid nodes must use separate explicit padding;
do not silently truncate to existing max_halo4096. Geometry generation and
candidate rejection/retry remain identical. Different node counts require
fixed-shaped zero-weight padding without counting padded residuals.

## Original plan (completed)

1. Add opt-in full-grid payloads with physical-volume quadrature, no repeated
   full-graph copies for the eight irrelevant core-pool IDs.
2. Extend canonical train.py with a strictly checked paired rolling control;
   reconstruct initial full graphs from known geometry and preserve IC/BC arrays.
3. Focused parity/replay checks and complete-shape resource probe.
4. One cold20kfull-grid run, common-reference final8evaluation, timing/error
   comparison and records. Retain current rolling core/halo default pending
   actual evidence and any later user selection.

Blocker: none. All four planned stages completed. RTX5090 32GB is idle.

## Implementation and complete input-capacity qualification

Explicit opt-in train_full_grid adds only sample.full_grid={nodes:4608,
neighbors:64}. The original K_pool8 draws remain in the shared RNG trace but
select no distinct PDE sets; each full geometry stores one graph. Full graph
retains all mesh points, volume-weights all non-mouth PDE rows, zeros mouth/
padding PDE rows, and retains predicted mouth values as neighbors (softBC
unchanged). IC/BC/conditioning arrays and spatial refresh RNG remain matched.
Canonical train.py adds --full-grid-control for strict cold/actualdraw pairing.

All380actual geometry records passed full-row CPU capacity checks in90.499s
with4workers; no random resampling or PDE solve. Nodes3977..4257,freePDE rows
3968..4248;9mouth nodes each. Maximum full physical/nonmouth degree45, mouth9,
all positive volume measures/masks/operator checks pass. This qualifies the
actual full20k input stream, not an unbounded prior. Source bytes unchanged:
/root/autodl-tmp/data/particle_corrosion2d/20261106-full-grid-capacity380-01/report.json.
The full weighted loss is the discrete physical mean estimated by mass-based
core draws; adaptive gradient normalization does not make complete optimizer
updates identical or unbiased. Next: focused tests and production-shape probe.

Four new full-payload tests plus eight prior online tests pass;18prior
control/comparison tests also pass after runner integration. Three actual
8particle geometries (IDs15,92,251) independently match full versus sampled-row
operator actions exactly, retain soft mouth influences, and agree with native
CSR within FP32 coefficient precision. Full-row and quadrature parity report:
/root/autodl-tmp/data/particle_corrosion2d/20261106-full-grid-capacity380-01/parity_three_geometries.json.

Resource probe PID144048 is running32complete gradient updates at actual
width192/B4/Nt8, holding all64full graphs on GPU (160003072bytes). Output:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-full-grid-resource-01.
Console/PIDfiles use same prefix. Initial full-bank preparation25.54s;
PDE rows3992..4214in this initial64bank, with fixed4608row executable for all
future layouts. It is a resource probe, not a scientific training endpoint;
no production training has started yet. Next: wait for resource report and new
paired-audit tests, then start one canonical cold20kfull-grid run if memory fits.

## Cold production history (completed)

Resource probe completed32finite full-gradient updates including a warm weight
refresh: heldmedian32.2246ms,refresh94.5625ms,activepeak7.16569GiB,firstcompile
176.462s,total203.313s.32GBRTX5090 has sufficient headroom.12new strict full
control/audit tests pass, and independent runner/probe integration review finds
no blocker. Default remains rolling core/halo; no full-grid result exists yet.

Completed cold production former PID152331:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-full-grid-w192-cold20000-01.
Same seed1607243318,cold initial model/optimizer,geometry/IC/time draws as rolling
run02 through --full-grid-control. Console/PID/launch record beside the run.
Training-agent owns one finite finalizer: wait this PID and complete20k, then
online_validation --run FULL --workers4 (no --control),GPUfinal reused8eval,
then compare_full_grid (it performs the separate128slot full CSR/volume audit
once). Expected outputs:20261106-full-grid-w192-reused8-final-01 and
20261106-full-grid-vs-core-halo-w192-01 in same runroot. Do not start duplicate
training or rerun final audits unnecessarily. No source edits during production.
Next: monitor first compiled updates/replacements, complete paired20k and final
saved-field cost/error comparison; retain default unless later selected.

Finalizer execsession52772 owns /tmp/particle_full_grid_finish.py, with timeout
3600s and separate20261106-full-grid-final-{audit,evaluation,comparison}-01.console.log.
Production passed firstcompile193.84s and actual rolling replacements, at1900+
updates ordinary32--33ms and4.91scumulativeproducerwait. All current full-control
startup/source/config/cold/commontrace checks passed; do not duplicate this run.
