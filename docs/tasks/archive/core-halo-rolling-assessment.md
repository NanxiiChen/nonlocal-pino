# Rolling core+halo pools on fixed-domain physics-only operators

Date: 2026-10-31. Status: COMPLETE. No Git history authorization.

## Goal and acceptance

The user asks whether the asynchronous producer/bounded rolling-pool method
used for random CH geometries can also refresh core+halo sampling in the fixed
electropolish and melting domains. Inspect the actual canonical neural-operator
paths, distinguish condition sampling from residual-point sampling, quantify
fixed-pool core coverage, and identify fixed-shape/cache/reproducibility needs.
Provide a concrete reusable design and its measured or unmeasured limits.

Acceptance: trace current32entry cycling to source; establish core versus halo
coverage on the configured grids; verify which dependencies must be refreshed
together; explain preserved graph physics and physical loss sampling. No new
training, default changes, reference solves, external cleanup or Git actions.

## Context and status

No active experiment is running. The confined geometry-operator SOTA promotion
is complete. Both canonical melting/electropolish operator presets already use
continuous random initial conditions, but their spatial subgraphs are built
once via SparseD.build_subgraph_pool and cycled modulo32. SparseD uses a single
complete neighbor layer for these scalar residuals. Fixed global graphs need
not be rebuilt when only residual core nodes change.

## Source findings

- melting operator_training.build_pool calls SparseD.build_subgraph_pool once;
  the loop takes pool[k % len(pool)]. Canonical settings are128^3grid points,
  K_core256,K_pool32,B16/Nt8. make_sampler draws new ellipsoid parameters and
  computes image/point signed distances each update. Its latent AC loss masks
  all halo residual rows; each core needs all its one-hop graph neighbors.
- electropolish operator_training similarly builds32batches once and cycles
  iteration % len(batches). Canonical grid101x51/K_core256/B8/Nt4. It already
  draws continuous initial-profile coefficients each update. AC applies the
  graph to phi; CH applies it to algebraic q, so this model needs one halo layer.
  Free-node masks exclude top/bottom Dirichlet residual rows. IC/BC query blocks
  are appended separately and should remain unchanged in a first sampler control.
- electropolish online_nearest is cached per fixed batch. It is the curve-x
  lookup for the discrete SDF, not a learned feature. A refreshed entry must
  carry matching nearest indices; the fixed full-grid mapping can be computed
  once then indexed by the new node IDs, including the appended IC/BC nodes.
- SparseD.build_subgraph_pool uses exact observed maxima over its initial pool.
  _pad_entry can truncate if given undersized capacities, so it cannot be reused
  with the old observed limits for unbounded future draws. Complete core rows
  preserve the original nonlocal graph/BC behavior; dropping halo edges does not.

## Exact CPU coverage and illustrative fresh draws

Receipt: docs/tasks/core-halo-rolling-assessment.json. Reproduce current cores
with RandomState(i % (2**31-1)).choice(N,K,replace=False), i=0..K_pool-1.
This is the actual SparseD core draw; no graph/PDE/reference solve is required
to count unique core IDs.

| Canonical neural operator | Grid nodes | Static entries x cores | Distinct PDE cores |
|---|---:|---:|---:|
| electropolish |5151, of which4949free |32x256 |4112overall;3938free (79.5716% of free nodes) |
| melting |2097152 |32x256 |8172 (0.389671%) |

Coverage describes direct PDE residual centers only, not every coordinate that
influences parameter gradients. Halo/input-image/IC/BC contributions remain.
Initial-condition draws vary even when the spatial cores repeat. Network spatial
generalization/hard IC also help explain accurate melting fields despite the
small residual-center set; coverage alone does not prove an accuracy defect.

An independent CPU-only uniform-core illustration over5000fresh draws visits
all5151electropolish nodes and958107melting nodes (45.6861%). Expected uniform
coverage is1-(1-K/N)^M. These are not new training results or a future exact RNG
contract. They neither add off-grid coordinates nor change graph consistency.

Current RandomState core selection alone averages0.417ms/entry in2D and41.11ms
in3D on this machine. The fresh Generator illustration takes0.051/0.057s total,
but does not build halos, pack arrays, transfer data or overlap GPU work.
Current sub_sample also allocates a length-N global/local map. These findings
support profiling/efficient sparse subset selection; no end-to-end speed claim.

## Proposed reusable design

1. Precompute the full fixed graph/immutable neighbor tables and boundary data
   once. The producer constructs subgraphs, never Gmsh meshes/reference solutions.
2. A pure-NumPy CPU producer draws uniform fixed-count cores and gathers the
   complete needed halo, coefficients, local indices, coordinates and core mask.
   Initial trial can use a small bounded FIFO/pool, refilled after consumption;
   reuse/refresh cadence is an explicit configuration, selected after timing.
3. Use fixed capacities/dtypes and dynamic array arguments to the existing
   JIT step. Safe one-hop capacities are N_pad=min(N,K*(degree_max+1)) and
   E_pad=K*degree_max. Current melting caps are1792/1536, close to the1788nodes
   reported by the old measured pool. Electropolish should determine its bound
   and memory/activation overhead from the actual graph. Never crop edges or
   silently redraw a valid but oversized set. Deeper residual dependencies need
   their own complete closure; do not reuse one-hop assumptions for eliminated CH.
4. Replace all geometry-only entry data together; for electropolish derive SDF
   nearest indices from the same query IDs. Recompute initial fields from the
   fresh condition and recompute learned encoder/prepared states every update.
5. Keep separate deterministic condition/time/subgraph streams and scheduled
   draw IDs, independent of worker timing. Save actual draw identity and global
   step so a checkpoint can reconstruct the bounded pool. Preserve old presets.
6. Validate sampled core operator values against complete fixed-graph rows and
   audit masks/normalization/BCs before training. Time the entire producer and
   GPU transfer/wait, verify no per-refresh compilation, and compare trained
   L2 at matched architecture/budget/padding/reference/initial-condition streams.

Uniform refreshed cores target fuller sampling of the same discrete PDE norm.
Changing geometry priors expands operator-input coverage; changing cores expands
residual collocation coverage on a fixed finite grid. Neither alone removes
the nonlocal-versus-local discretization error. No accuracy improvement is yet
measured for the rolling-core proposal.

## Status / next action

COMPLETE, no blocker or required resume. Source/coverage/receipt consistency and
documentation/diff checks pass. No code tests or reference solves are required
for this read-only sampling investigation. No runtime/default/physics changes.
No implementation/training is queued by this discussion. A later experiment
should first profile the label-free subgraph producer and fixed padding, then
compare fixed/rolling pools under the same model and condition/time streams.

Task-end memory changes: this archived assessment, PROGRESS.md and DECISIONS.md.
Compact CPU evidence remains in docs/tasks/core-halo-rolling-assessment.json.
