# Rounded-notch fracture under horizontal shear

This example uses the same material mesh, rounded notch and hybrid plane-strain
AT2/history model as the accepted FEM-energy option in `fracture2d`. It removes
the artificial grip strips and prescribes displacement on the material edges.
The material domain has71,307 nodes and141,366 P1 triangles, with nominal
spacing0.004 and phase-field length0.024. No geometry change, symmetry penalty
or prescribed crack path is introduced.

| Boundary | Displacement |
| --- | --- |
| Entire top edge, y=1 | ux=U, uy=0 |
| Entire bottom edge, y=0 | ux=0, uy=0 |
| Other exposed edges, including notch | Natural zero traction |
| Damage on every exposed edge | Natural zero flux |

The reaction is the horizontal top-edge reaction, evaluated as the internal
force dotted with the prescribed horizontal displacement direction. It is not
the old vertical reaction. Direct edge clamps have different compliance from
the older appended-grip setup, so old force curves cannot serve as references.

## Numerical scheme

`train.py` runs local P1 direct mechanics and a history-conditioned neural
damage field. Nodal damage is interpolated by P1 shape functions and trained
through the lumped FE energy:

$$
J_H(d)=\sum_i V_i\left[\frac{G_c}{2l_0}d_i^2+H_i(1-d_i)^2\right]
       +\frac{G_cl_0}{2}\sum_T |T|\,|\nabla d_h|_T^2.
$$

The implementation uses the same detached positive normalization and omitted
H-only constant as `fracture2d/fem_damage.py`. Spatial derivatives come from
shape functions; parameter derivatives use AD. The history-conditioned
point-patch encoder/coordinate-query network, sigmoid output and full-domain
training are inherited from the accepted replay-free baseline. Replay training
is OFF by default. The bank has a fixed capacity512 for the two snapshots at
each of200 loads; collecting snapshots does not enable replay.

Each stagger iteration solves mechanics once, forms
H=max(previous-load endpoint H, current tensile driver), freezes/detaches H,
and warm-starts neural damage. Directly accept the new damage with omega=1.
Warm-start the network and Adam moments across iterations/loads. Each inner
fit has128Adam updates, except the one-time1024-update energy bootstrap.
Use the inherited damage RMS limit `(max(d)+0.01)*0.001`, displacement
RMS/load limit1e-4 and three consecutive passes. The progress-based rate
ladder1e-4→3e-5→1e-5 checks every128outers, at most four blocks per load.
Failure saves/stops rather than advancing. These are trial numerical settings,
not a claim of optimal shear convergence.

`solver.py` uses the same standard P1 FEM mechanics and directly solves the
scalar local AT2 system. Both displacement and damage use checked sparse CG;
the linear systems are independently checked against UFL assembly by
`validate_fem.py`. The reference uses strict field-change limits1e-5 for d,
1e-4 for u/load, three consecutive passes and a1024-outer cap. Neural stopping
and reference stopping intentionally differ, as in the preceding baseline.
Both use identical material parameters, mesh, BCs and physical stagger.
Reference data are used only for evaluation, never training targets.

The loading exploration is U=0→0.0200 with increments1e-4. This range does not
assert a fracture displacement; inspect the new reference to determine whether
it includes crack growth and postfracture behavior. The first direct sweep to
0.0100 still had local damage only, motivating the extended range. Material parameters remain
E=210000, nu=0.3, Gc=2.7, l0=0.024, residual stiffness1e-8. All fields useFP64.

## Entries and execution

The entries use ml_collections configuration. Shared numerical code stays in
`examples/fracture2d/`; this directory supplies the shear case and independent
validation, without importing `_legacy/`.

1. Prepare material-only mesh and direct edge constraints in the JAX environment:

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_shear.prepare
```

2. With no JAX GPU solver running, validate UFL mechanics/history/scalar operators
   and edge displacements/horizontal reaction in the FEniCS environment:

```bash
PYTHONPATH=. conda run --no-capture-output -n fenicsenv python \
  -m examples.fracture2d_shear.validate_fem
```

3. After FEniCS exits, run the direct reference, then the fresh neural trajectory:

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_shear.solver
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_shear.train
```

Pass `--config=examples/fracture2d_shear/configs/train_base.py` or
`configs/solver_base.py` explicitly to override settings. The defaults refuse
to overwrite outputs. Exact resume requires the same configuration and an
existing valid endpoint; an unconverged budget endpoint cannot be skipped.

To run an independent cold replay comparison using the same reference:

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_shear.train \
  --config=examples/fracture2d_shear/configs/train_replay.py
```

This preset samples five distinct H snapshots from all completed loads, using
the first-outer and endpoint snapshots already stored by the baseline. It
excludes the current load's snapshots. Every fourth Adam update uses total
old-task weight0.1 and current-task weight0.9; the last32updates of each inner
fit use only the current H. Early buffer shortages are masked in a fixed
current-plus-five batch, without replacement. Coordinate features/queries are
shared across histories; field encoders and cross-attention remain H-dependent.
The physical mesh, material, loading, network seed, optimizer and stopping
policy match the replay-free preset. This adds no reference targets or spatial
sampling; it trains on previous frozen-H energy objectives.

## Experimental current-driver irreversibility

The working default remains the maximum-history scheme described above.
The current-driver penalty presets below are retained for reproducing the
irreversibility comparison and require explicit selection.

An opt-in trial replaces maximum history in the damage energy by the current
tensile driver \(\psi^+(u)\), with the same projection/lumped quadrature:

$$
L=\frac{J_{\psi^+}(d)+\beta\frac{G_c}{l_0}
        \sum_i V_i\operatorname{ReLU}(d_i^n-d_i)}{C(\psi^+)}.
$$

Here \(J\) has the same omitted driver-only constant as above, and BOTH terms
use its original detached positive normalization \(C\). The dimensionless
\(\beta\) is explicit, initially1 for exploration, not a universal optimum.
The lower bound \(d^n\) is the preceding completed LOAD endpoint; it stays
fixed across all stagger/Adam iterations, and is detached together with psi.
Current psi is refreshed after every mechanics solve and fixed within each
inner fit. The network conditions on current psi rather than maximum H.
For storage compatibility the field key `H` holds current psi in this mode;
`identity.json` explicitly declares `history_floor=none_current_psi`.

This L1 penalty is a soft constraint. Field stopping also requires maximum
nodal healing <=1e-3. Logs separate physical energy, penalty, maximum healing,
volume-RMS healing and affected volume fraction; they do not demand a zero
unconstrained AT2 residual on an active inequality bound. Larger beta may
impede training. Compare several weights before accepting a new baseline.
No damage projection or per-outer monotone ratchet is applied.

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_shear.train \
  --config=examples/fracture2d_shear/configs/train_current_psi.py
```

Replay remains OFF: an old psi alone would not supply its path-dependent
damage bound. The original history-field reference, baseline and default
configuration remain unchanged. Reference differences now include a different
irreversibility treatment; identical solutions are not guaranteed. Mechanics
still uses the existing hybrid full degradation/tensile-driver scheme, not a
new jointly variational split. Penalty approaches are discussed in
[Gerasimov and De Lorenzis (2019)](https://arxiv.org/abs/1811.05334);
this L1 trial does not borrow their quadratic-penalty parameter formula.

For the screened beta10 candidate, run an independent cold trajectory:

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_shear.train \
  --config=examples/fracture2d_shear/configs/train_current_psi_cold.py
```

This preset starts with a fresh same-seed network/Adam/bank and zero physical
fields, loading0→0.0200 in1e-4 increments. It retains the inherited inner
updates, bootstrap, progress-based rate ladder and field stopping plus the
1e-3 healing gate; it does not load the warm-screen checkpoint. Results and
TensorBoard events use `pinn_current_psi_l1_beta10_cold/`. Beta10 is a trial
candidate selected for healing control, not an accepted replacement baseline.

## Artifacts

| Role | Default directory |
| --- | --- |
| Material mesh, explicit shear BCs, UFL receipts | `/root/autodl-tmp/data/fracture2d_shear/geometry_assets/` |
| Local FEM direct reference | `/root/autodl-tmp/data/fracture2d_shear/fem_reference/` |
| PINN results and TensorBoard | `/root/autodl-tmp/tf-logs/fracture2d_shear/pinn_history_conditioned/` |
| Random-five replay comparison | `/root/autodl-tmp/tf-logs/fracture2d_shear/pinn_history_conditioned_replay_random5/` |
| Reference TensorBoard | `/root/autodl-tmp/tf-logs/fracture2d_shear/fem_reference/events/` |

Every completed load records damage, ux/uy, history and load-reaction curves.
Displacement color limits use actual field extrema. No shear events are written
directly into the parent tf-logs directory. All paths are ordinary file paths;
no SHA lookup or compatibility symlinks. Run-specific status, validation and
resume instructions belong in the active architecture task, not a duplicate
example STATUS.md.
