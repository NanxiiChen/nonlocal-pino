# Quasi-static fracture of an L-shaped panel

This example tests a nonconvex material domain and crack nucleation without
an initial notch. It keeps the existing local P1 direct mechanics, hybrid
Amor history driver, lumped AT2 FEM damage energy and history-conditioned
neural network. The spatial derivative uses mesh shape functions. It is a
new single-geometry solve, not a demonstrated cross-geometry prediction.

The geometry and loading pattern follow the L-panel in
[Manav et al. (2024), section 4.3, Figure 5 and Appendix H](https://arxiv.org/pdf/2404.13154).
The domain is translated to a unit bounding square with its lower-right
quarter removed. This is an adaptation: material, regularization and
history-field irreversibility remain those of the existing examples, rather
than the paper's parameters and penalty-irreversible variational model.
The matched FEM reference solves this actual adapted model.

| Boundary | Condition |
| --- | --- |
| Bottom, y=0 and 0<=x<=0.5 | ux=uy=0 |
| Underside of right arm, y=0.5 and 0.94<=x<=1 | uy=U; ux free |
| All other material edges | Natural zero traction |
| Damage on all exposed edges | Natural zero flux |

The 270-degree reentrant corner is at (0.5,0.5). There is no initial crack,
symmetry constraint, prescribed future path, inertia or added grip strip.
The load is applied on a finite segment, not a mesh-dependent point load.
Geometry extents, corner and patch endpoints are explicit configuration fields.
Gmsh includes the patch endpoints as geometric vertices and creates material
triangles only; unused square quadrants are not part of the FE mesh.

## Model and solver

The damage objective is

$$
J_H(d)=\sum_i V_i\left[\frac{G_c}{2l_0}d_i^2+H_i(1-d_i)^2\right]
       +\frac{G_cl_0}{2}\sum_T |T|\,|\nabla d_h|_T^2.
$$

Plane-strain material parameters remain E=210000, nu=0.3, Gc=2.7, l0=0.024
and residual stiffness1e-8 in the same consistent unit system as fracture2d.
Nominal mesh size is0.004. q,d,H begin at zero, so this includes nucleation.
Each stagger solves mechanics once, freezes H=max(previous-load H,current
tensile driver) during damage fitting, and accepts damage without relaxation.

The PINN uses the same point-patch/coordinate-attention architecture, sigmoid
output, full-domain energy, Adam128 inner updates and initial1024 bootstrap.
Replay is OFF by default. The fixed bank1024 covers two H snapshots at each
of300loads. An opt-in replay preset uses five distinct snapshots sampled from
all completed loads, excluding the current load. Their total loss weight is
0.1 versus0.9 for the current history. Replay occurs every4Adam updates,
with the last32updates fitting only the current history. The fixed six-task
batch masks missing old snapshots early in loading. Coordinate features are
shared across tasks; history encoding and attention remain history-specific.
The rate ladder1e-4→3e-5→1e-5 observes128-outer progress blocks, maximum4blocks
per load. Stop on damage RMS<(max(d)+0.01)*0.001 and displacement RMS/U<1e-4
for three consecutive iterations. Direct damage uses sparse CG and strict
damage1e-5/displacement1e-4/three-pass stopping with1024outers. Budgets save
and stop; they never silently advance an unconverged state.

The initial exploratory range is U=0→0.0300 with du=1e-4. It is not a claim
of an already verified nucleation or failure displacement. Establish the new
direct reference first; old tension/shear force curves cannot be reused.

## Running

Use mutually exclusive environments and ordinary file paths. First prepare
the material cloud and edge constraints in the JAX environment:

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_lshape.prepare
```

With no GPU training/solver running, validate independent UFL stiffness,
history projection, scalar operator, edge constraints and vertical reaction:

```bash
PYTHONPATH=. conda run --no-capture-output -n fenicsenv python \
  -m examples.fracture2d_lshape.validate_fem
```

After FEniCS exits, run the reference and then fresh neural loading:

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_lshape.solver
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_lshape.train
```

For a separate cold-start replay comparison:

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d_lshape.train \
  --config=examples/fracture2d_lshape/configs/train_replay.py
```

All entries accept `--config=...` via ml_collections. Defaults refuse to
overwrite outputs; exact resume requires the same config and completed
endpoints. The numerical engine and material-edge UFL checks are shared with
the preceding examples; no legacy dependency or hash check is introduced.

| Artifact | Directory |
| --- | --- |
| Cloud and prepared geometry | `/root/autodl-tmp/data/fracture2d_lshape/` |
| Matched FEM reference | `/root/autodl-tmp/data/fracture2d_lshape/fem_reference/` |
| Reference events | `/root/autodl-tmp/tf-logs/fracture2d_lshape/fem_reference/events/` |
| Neural results/events | `/root/autodl-tmp/tf-logs/fracture2d_lshape/pinn_history_conditioned/` |
| Replay trial results/events | `/root/autodl-tmp/tf-logs/fracture2d_lshape/pinn_history_conditioned_replay_random5/` |

Compare fields at identical loads, the vertical patch reaction-displacement
curve, and iteration/wall cost. A successful fresh solve on this L-domain
does not establish transfer of existing weights across geometries. Run-specific
status and results belong in the active task, not this stable README.
