# PINO spatial sampling study

## Goal and acceptance

Find when the ceramic-particle pattern of bounded background input generation
and rolling geometry/spatial pools improves efficiency at comparable physical
error. Inventory the five featured PINOs. Compare fixed and rolling
core/halo sets on the current regular-domain recipes, and qualify full-grid,
core-count and boundary-density controls for parameterized geometries.

Compare matched cold initial weights/optimizer, condition/time draws, operator
stencils, boundary losses and evaluation cohorts within each case. Retain all
stencil support, keep original quadrature in expectation and distinguish active
GPU peak, synchronized step time, wall time, compilation and host overhead.
Select candidates by equal-step and equal-time error. No label solves, numerical
default change, Git history operation or cleanup has been requested.

## Current modes

| Case | Current spatial PDE sampling | Geometry or condition input |
|---|---|---|
| Melting | Fixed 32-entry core/halo pool, 256 cores | Fixed cube, online initial ellipsoids |
| Electropolish | Fixed 32-entry core/halo pool, 256 cores | Fixed rectangle, online initial interfaces |
| Spinodal | Full physical mesh, padded to 1536 nodes | Rolling geometries |
| Channel NS–CH | Full 193×81 mapped grid plus explicit boundary queries | Rolling geometries |
| Ceramic-particle corrosion | Rolling 8-set core/halo per slot, 256 volume draws per set | Rolling geometries |

The ceramic control should use the explicitly published six-particle
`train_physical_spaced_w256.py` preset. Its geometry and horizon differ from
the generic no-argument physical pilot. Retain the 350 s physical problem.

## Status

All queued training, resource probes and paired evaluations completed on the
same RTX 4090. Geometry queue finished at 2026-11-11 09:05 Asia/Shanghai.
GPU is idle. The confirmed channel/ceramic controls and regular-domain repeats
are complete. The optional Spinodal row-only pilot remains unrun. The user proceeds to
promote the successful Electropolish rolling recipe. This confirmed study
is complete; regional diagnostics and the row-only pilot are deferred.

## Completed regular-domain stage

Runner and audit scripts:
[pino_spatial_pool_runner.py](../pino_spatial_pool_runner.py) and
[pino_spatial_pool_audit.py](../pino_spatial_pool_audit.py). External run root is
`/root/autodl-tmp/tf-logs/pino-spatial-pools-20261111-01`.
`journal.json` is atomic and names the exact CLI, process and log for every arm.
Each training result is stored in the existing example-specific log directory.
The companion audit writes `paired-results.json` after each complete pair.

1. Electropolish width 96, 16 shared times, batch 8, Adam 10,000, 256 cores
   and identical safe padding. Fixed then rolling at seed 42; evaluate each
   endpoint on the same 64 native and local FD101 interfaces. Repeat at seed 43.
2. Melting batch 16, eight shared times, Adam 5,000, 256 cores and matched safe
   padding. Fixed then rolling at seed 43; compare the same eight held-out
   native diagnostic inputs, including pooled and terminal phase L2. The
   published seed-42 pair is an independent prior comparison on another GPU.
   Keep the existing fixed-chunk inference path and note the historical
   chunk-invariance issue. There is no new full-grid melting run.

Both recipes reuse their current width, physics, optimizer and training budget.
The explicit fixed arm shares safe padding and core IDs with its rolling
partner at initialization; it can differ from the canonical legacy fixed
path's padding. Do not mix the previous default's speed with the new rolling
speed. Existing hardware timings are not used to calculate new speedups.

For context, the old Electropolish width64/Nt4/5000 paired FD101 mean phase
pooled/terminal errors were 4.9640/3.8333% fixed and 4.8578/3.6507% rolling.
The old Melting seed42 native diagnostic pooled/terminal phase L2 was
0.1631/0.2218% fixed and 0.4217/0.2514% rolling. Neither comparison licenses a
universal default. The matched-run audit reproduces historical timing summaries
for both old pairs; its new timings come from the current device.

## New channel preflight

[pino_spatial_geometry_probe.py](../pino_spatial_geometry_probe.py) reads the 16
initial and 16 final geometry slots of the published 20k run. The external
`geometry-probe.json` has separate results for both banks. Per geometry, 32
draws are tested at each setting. Retain the union of all nonzero first/second
operator weights of the selected PDE rows, and of every explicit BC row. The
192 off-grid boundary queries are separate. The existing encoder is still full.
On the initial bank, BC rows alone require median 3260 graph nodes. The final
bank gives 3243. A sampled graph remap reproduces Laplacian and both gradients
bitwise on a nonconstant test field for each mode/count on the first draw in
each geometry. Every active boundary index also replays exactly.

| Nominal cores | Area-random median union | Half in wall band | Local tiles |
|---:|---:|---:|---:|
| 128 | 9040 / 57.8% | 7659 / 49.0% | 3672 / 23.5% |
| 256 | 12134 / 77.6% | 10361 / 66.3% | 4093 / 26.2% |
| 512 | 14638 / 93.6% | 13254 / 84.8% | 4855 / 31.1% |
| 1024 | 15555 / 99.5% | 15118 / 96.7% | 6256 / 40.0% |

The denominator is 15633 graph nodes, and there are 15089 free PDE targets.
The final-bank medians are within 0.3 percentage points. Counts omit fixed
padding and do not measure time, memory, gradient variance or PINO accuracy.
Tiles are 8×8 mapped index blocks, sampled with replacement by each block's
total PDE quadrature. Remainder tiles can contain fewer than 64 nodes.
Averaging rows by their original area within each tile and then averaging
tile draws preserves the full-domain loss in expectation. Tile overlap may
lower independent spatial coverage at a given nominal count.

For independent area draws, select each free node with probability proportional
to the exact volume weight. For wall stratification, select half of the cores
within one existing horizon (0.075) of the top or bottom physical wall, and
half elsewhere, using each stratum's normalized volume weights. Multiply
the stratum means by their actual mass fractions, approximately 0.16/0.84
for the measured geometries. This increases near-wall coverage while keeping
the area objective. Geometry-specific wall distance uses a dense wall sampling
only for the CPU support probe. Training must use a validated distance to the
declared wall and fixed-capacity masks. Boundary residual counts and norms are
unchanged across arms.

## Confirmed geometry experiments

The user confirmed 512-core scatter and local-tile controls against the full
channel grid, plus the ceramic near-wall experiment and same-device timing.

- Channel `train_spatial_control.py` preserves width192, batch8, SOAP, 36 physical
  time samples, all explicit BCs, and the full input encoder. The scatter arm
  has 512 area-proportional draws. The tile arm draws eight tiles by area and
  takes exactly 64 nodes per tile, with larger remainder tiles subsampled
  uniformly and corrected by inverse inclusion probability. Thus both have
  exactly 512 residual rows. Repeated random draws can reduce unique nodes.
  Compact graphs retain complete nonzero derivative and BC supports; masks
  are implicit in original zero stencil coefficients. Spatial batches roll
  at revision boundaries and remain tied to their geometry ID. Scatter
  reserves 15633 query slots to guarantee no overflow. Tile reserves 8192;
  overflow raises without truncation or a different random draw.
- On 768 draws of the production tile scheme across the saved 32 channel
  geometries, the halo union with boundary support was median 4962.5,
  maximum 5458. A per-geometry worst-tile-union upper bound reaches 6310 on
  that bank. The prospective rolling prior is broader; 8192 is the explicit
  buffer capacity. These counts describe compact support and actual padding
  must be reported separately from warm speed. Area and timing remain
  independent of the historical GPU throughput.
- Ceramic `train_physical_spaced_w256_wall_cores.py` selects exactly 128
  draws within `delta` of the real hole polygons and 128 from the rest.
  The normalized loss sums the stratum means with their true lumped-area
  fractions. The 64 paired initial geometries and all non-graph arrays are
  byte-identical. The initial near-wall area fractions span 10.625%–15.079%.
  The same 350 s physics, Adam20k, batch and time configuration, 256 residual
  rows and same padded graph shapes are used. The area arm uses the existing
  bank. Wall bank uses the same preparation seed 1307500965 and resides at
  `/root/autodl-tmp/data/particle_corrosion2d/20261111-spatial-wall-01/inputs64`.
  Both train with runtime seed 17617494 and evaluate the same four local and
  native cases, with an extra phase/concentration L2 restricted to the wall band.

## Owned queue

[PINO geometry queue](../pino_geometry_sampling_queue.py) and
[paired evaluator](../pino_geometry_sampling_results.py) own these stages, in order.

1. Wait for the current Electropolish/Melting queue to reach a terminal state
   and the same RTX 4090 to be idle. Each GPU child checks the captured GPU UUID
   and runs serially with the same preallocation fraction.
2. Channel full/scatter/tile 60-update resource probes. Start with time blocks of
   6 of the same 36 samples per update, and test 3 then 2 on a confirmed OOM.
   Accept only a common block across all three arms. Record first update,
   synchronized steady update, spatial prepare/transfer and active peak.
3. Cold ceramic pair to 20,000 each, local and native wall/whole-domain
   evaluation. Write `ceramic-paired.json` only after paired cold state,
   geometry/time/revision and fixed inputs pass.
4. Cold channel full/scatter/tile to 2,000 each with the accepted common block.
   Reconstruct online inputs and evaluate the existing three reserved
   independent FEM cases for each. Write `channel-paired.json` only after
   matching cold weights/SOAP and actual geometry/time draws.
5. Read the matched convergence histories at equal updates and wall time.
   Evaluate terminal droplet overlap and near-wall errors on the saved
   reserved fields before deciding whether a longer matched run is warranted.

Queue owner 49374 writes atomic stage and failure receipts under
`/root/autodl-tmp/tf-logs/pino-geometry-sampling-20261111-01/journal.json`.
Status at handoff is `waiting_for_regular_queue`. The exact child command,
logs and durations are added as each stage launches. The resource results
go in `channel-probes.json`. Spinodal's separate exact row-only pilot remains
deferred. Its reconstructed FP32 resolvent was dense on all eight meshes and
the hard mass projection uses full physical quadrature.

## Measured outcomes

All timings below come from the current RTX 4090. Pair audits pass for cold
model/optimizer bytes, geometry and physical time draws. Channel also passes
the full online-input reconstruction for every arm. Independent saved-field
norms reproduce all three channel reserved phase results and both ceramic
local metric sets. This is one matched geometry initialization and small
reused geometry cohorts; the channel experiment uses 2k updates.

### Channel cold 2000, three reserved local FEM cases

| Sampling | Mean phase L2 | Terminal phase L2 | Velocity L2 | Pressure L2 | Warm step median | Active peak |
|---|---:|---:|---:|---:|---:|---:|
| Full | 6.9872% | 8.1543% | 5.0462% | 10.1972% | 3.3055 s | 35.7145 GiB |
| Scatter512 | 8.6552% | 10.9580% | 5.3221% | 10.7980% | 2.7981 s | 35.3197 GiB |
| Tile512 | 21.0072% | 23.6907% | 6.9912% | 16.2469% | 1.3951 s | 20.7417 GiB |

Shared 36 random times, time block6, batch8, width192/SOAP. Tile is 2.37 times
as fast per warm update as full, 2.01 times as fast as scatter; phase
accuracy is worse at this common 2k budget. Every development phase curve
is still decreasing over 1500–2000; no converged-floor ranking is established.
Whole child-process walls, including construction/compilation/evaluation, are
116.18, 99.46, and 53.30 minutes. Producer waits for compact pools sum to
2.01 s scattered and 1.74 s tiled. Tile real halo median4958 executes a fixed
8192-node buffer; scatter median14626 executes15633. Full uses15633.

Direct result `channel-paired.json` under the geometry queue root. Reserved
fields and their reports reside in each `channel-2000-b6-<mode>/reserved/`.
Terminal interface overlap and channel wall-band error have not yet been
computed from those fields.

### Ceramic cold 20000, four local FEM development cases

| Sampling | Phase pooled/final | Concentration pooled/final | Wall-band phase pooled/final | Warm effective step median |
|---|---:|---:|---:|---:|
| Area256 | 3.4574 / 6.8475% | 4.2021 / 8.1976% | 5.6879 / 12.3905% | 84.6136 ms |
| Wall128+bulk128 | 32.1773 / 44.3496% | 34.1783 / 46.3174% | 31.2748 / 68.5786% | 84.7344 ms |

Native phase pooled/final is1.9777/3.8944% for area and31.4218/42.5043% for
wall. Active peaks20.4366/20.4044GiB; cold child walls33.93/33.98min.
Near-wall sampling failed to converge well in this paired trial. Similar
native and local failures identify a neural-training issue beyond
native/local discretization differences. The result does not by itself
identify its optimization mechanism. Near-wall area normalization and every
fixed conditioning/IC/BC input pass pairing; full installation replay of
the ceramic rolling operators remains an optional follow-up audit.

Direct result `ceramic-paired.json`; local/native/near scores and predictions
are saved in each run. Preserve both positive and negative runs.

### Regular-domain repeats

Electropolish uses width96/Nt16/Adam10k and the same64localFD101cases.
Seed42fixed/rolling phase pooled1.7470/1.5466%, terminal1.5265/1.3256%.
Seed43fixed/rolling pooled2.6325/2.4648%, terminal2.0501/1.6715%.
Warm median40.058/41.614ms and39.785/41.930ms respectively.
Rolling improves phase and concentration metrics in both pairs, at roughly
4–5% more step time. Cross-seed accuracy spread remains larger than the
within-pair pooled improvements.

Melting seed43/Adam5k uses the same8native diagnostic inputs, with phase
pooled0.1290/0.5449%, terminal0.2062/0.5199% fixed/rolling.
Warm median16.111/17.072ms; activepeak1.494GiB for both.
The earlier seed42pair likewise favored fixed on whole-field phase. The new
comparison supports retaining fixed in this case. The old query-chunk
invariance qualification still applies.

## Validation and exact resume

- Thirteen focused CPU tests passed for full-cover channel value/gradient,
  time accumulation, tile partition, ceramic weights and old graph paths.
  Config gates, changed-Python syntax and `git diff --check` passed.
- The existing regular-domain audit independently verifies all three completed
  cold pairs and records `paired-results.json`. Both queue journals have
  status `complete`, with all stages exited0. No pending queue child remains.
- Final input/metric checks are recorded under the geometry queue root in
  `independent-saved-field-check.json`. No new numerical solve was needed.
- Next optional work is channel interface-overlap/wall-region diagnostics,
  full ceramic operator replay, and the dense Spinodal row-only timing pilot.
  No new long training has been selected by the completed screening.
- Memory files updated: `docs/PROGRESS.md`, `docs/DECISIONS.md`, and this task.
