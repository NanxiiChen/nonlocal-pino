# Fracture2D artifact locations

All paths below are ordinary directories, with no compatibility symlinks or
SHA-based lookup. Numerical results and TensorBoard event contents are retained.

| Role | Canonical directory |
| --- | --- |
| Accepted MLP PINN trajectory | `/root/autodl-tmp/tf-logs/fracture2d/pinn_mlp_baseline/` |
| Provisionally accepted history-conditioned PINN | `/root/autodl-tmp/tf-logs/fracture2d/pinn_history_conditioned_baseline/` |
| Historical trials, diagnostics and plots | `/root/autodl-tmp/tf-logs/fracture2d/diagnostics/` |
| Default outputs of new solver runs | `/root/autodl-tmp/tf-logs/fracture2d/runs/` |
| Current local FEM comparison | `/root/autodl-tmp/data/fracture2d/fem_reference/` |
| Current nonlocal damage direct reference | `/root/autodl-tmp/data/fracture2d/nonlocal_reference/` |
| Shared prepared mesh and nonlocal operator | `/root/autodl-tmp/data/fracture2d/geometry_assets/` |
| Older point-anchor FEM and nonlocal-mechanics references | `/root/autodl-tmp/data/fracture2d/legacy_references/` |

Each accepted PINN directory contains its own `events/`. Diagnostic event trees
are likewise attached to their diagnostic run, where an existing run could be
identified. Older pre-consolidation event trees are under
`diagnostics/legacy_events/`. No fracture2d events belong directly under the
parent `/root/autodl-tmp/tf-logs/` directory.

The current three-way comparison uses local P1 mechanics/local gradient AT2
for `fem_reference`, and local P1 mechanics/nonlocal AT2 direct damage for
`nonlocal_reference`. The PINNs use the latter model with neural damage.
`legacy_references/fem_point_anchor` is an older independent FEniCS mesh;
`legacy_references/nonlocal_mechanics` uses the older nonlocal mechanics route.
These older archives are not replacements for the current comparators.
The public `solver.py` remains the independent FEniCS entry and writes new
outputs under `data/fracture2d/runs/fem_point_anchor/`.

A `baseline` name records the user's provisional engineering acceptance. It
changes neither saved convergence flags nor the actual completed load range.
Run-specific limitations are in the baseline's `baseline.json`/`README.md`
and the [active task](../../docs/tasks/nonlocal-pinn-architecture.md).

The 2026-10-16 migration map, metadata backup and verification receipt live in
`tf-logs/fracture2d/_layout_migration_20261016/`. Current configs and checkpoint
path metadata use canonical locations. Old TensorBoard text/config payloads,
frozen source snapshots, and dated decision history retain their original
provenance; use the migration map to resolve their pre-cleanup paths. Historical
standalone scripts are not maintained entry points after relocation.

## Current FEM reference boundary and resolution

The saved `fem_reference/identity.json` declares engineering volume grips,
shared with the current nonlocal direct and PINN mechanics. Material occupies
0 <= y <= 1. Undamaged elastic grip strips extend to y=-0.048 and y=1.048,
with spacing0.004. Every added lower-grip node has uy=0; every added upper-grip
node has uy=U. Horizontal displacement is fixed only at material node0,
(x,y)=(0,0); other horizontal degrees of freedom are free. The material/grip
interfaces themselves are connected FEM nodes, not directly prescribed uy.
Other exposed boundaries, including notch faces, carry natural zero traction.
Local AT2 damage is solved only on material cells, with natural zero-flux
boundaries; the geometric notch is not a prescribed d=1 damage strip.

The actual material mesh is `cloud_h004_api.npz`: 71,307 nodes and 141,366
linear triangles, versus18,367 nodes in the older h=0.008 cloud. Its embedded
Gmsh metadata sets both h_far and h_near to0.004: this is a globally fine mesh,
not special crack-band refinement. Including grips gives77,331 nodes and
153,366 triangles. Physical l0=0.024 gives approximately six nominal mesh
spacings. All three current methods share the material nodes/triangles.
This is a finer discrete FEM comparator, not a demonstrated mesh-converged
continuum reference. This FEM path uses uniform load increments1e-4; the
nonlocal reference's additional critical-load steps are loading refinement,
not additional spatial refinement.
