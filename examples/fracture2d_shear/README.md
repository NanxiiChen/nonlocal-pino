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
is OFF. The bank has a fixed capacity512 for the two snapshots at each of200
loads; collecting snapshots does not enable replay.

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

## Artifacts

| Role | Default directory |
| --- | --- |
| Material mesh, explicit shear BCs, UFL receipts | `/root/autodl-tmp/data/fracture2d_shear/geometry_assets/` |
| Local FEM direct reference | `/root/autodl-tmp/data/fracture2d_shear/fem_reference/` |
| PINN results and TensorBoard | `/root/autodl-tmp/tf-logs/fracture2d_shear/pinn_history_conditioned/` |
| Reference TensorBoard | `/root/autodl-tmp/tf-logs/fracture2d_shear/fem_reference/events/` |

Every completed load records damage, ux/uy, history and load-reaction curves.
Displacement color limits use actual field extrema. No shear events are written
directly into the parent tf-logs directory. All paths are ordinary file paths;
no SHA lookup or compatibility symlinks. Run-specific status, validation and
resume instructions belong in the active architecture task, not a duplicate
example STATUS.md.
