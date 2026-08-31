# PINN for Nonlocal Phase-Field Models

Physics-Informed Neural Networks with a **nonlocal peridynamic operator $D_\delta$**
that replaces the local Laplacian.  $D_\delta \approx -\nabla^2$ (as $\delta \to 0$)
with compact kernel support; Neumann BCs emerge automatically from interior-only
neighbour pairs — no BC loss terms needed.

## Examples

| Example | Physics | Highlights |
|---|---|---|
| `icemelting3d` | Allen-Cahn, ice sphere melting | Analytical solution, nonlocal vs local AD ablation |
| `electropolish2d` | KKS phase-field, electropolishing | 2-field AC+CH, rough interface, Neumann BC advantage |
| `fracture1d` | AT2 phase-field fracture | Analytical crack profile, staggered incremental loading |
| `fracture2d` | Rounded-notch AT2 phase-field fracture | Local FEM reference plus validated direct nonlocal fixed subproblems |
| `elasticity2d` | 2D elasticity | Nonlocal elasticity and boundary-aware operators |
| `pino_elasticity2d` | 2D elasticity PINN | Geometry-aware PINN reference for nonlocal elasticity |
| `corrosion1d/2d` | KKS pitting corrosion | Legacy; superseded by electropolish2d |

## Core idea

Standard PINN: $\nabla^2\phi$ via `jax.hessian` on every collocation point — expensive
and requires explicit BC penalty terms.

Nonlocal PINN: precompute a sparse $D$ matrix from neighbour pairs on a fixed grid,
then $D \cdot \phi$ is a cheap matvec.  Points on Neumann boundaries only interact
with interior neighbours; the zero-flux condition is built into the operator.

```
D @ φ ≈ -∇²φ    (kernel: cubic / quartic / Wendland)
```

## Key design decisions

- **Nonlocal $D_\delta$ over autograd Hessian**: avoids per-point Hessian overhead
  and handles Neumann BCs implicitly
- **GradNorm over fixed weights**: per-term adaptive loss weights from gradient norms,
  EMA-smoothed
- **L1 loss** (MAE) for PDE residuals: $\nabla |r|$ eliminates gradient imbalance
  between stiff spatial terms and small temporal terms
- **Grouped separate gradients** (regular step): PDE terms sharing `_compute_all`
  stay in one backward pass; BC and IC get their own.  Important for `tensor_split`
  performance (see `benchmarks/TENSOR_SPLIT_RTX_PRO_6000.md`)
- **Hard constraint** on output: enforces $\phi \in [0,1]$ and KKS mixture rule at
  the architecture level

## Scaling

Most JAX/nonlocal examples use non-dimensional characteristic length and time
scales. The `fracture2d` local FEM reference instead uses explicit consistent
material units with `L0=1`; see its example README and configuration.
The fracture2d direct nonlocal baseline keeps geometry, quadrature, graphs, and
operator inputs in float64 with JAX x64 enabled. A future float32 PINN must
rebuild and revalidate its strict-horizon graph in float32 rather than casting
the direct-baseline graph.

## Fracture2D engineering-grip diagnostic

The durable engineering-grip workflow is GPU-only for production solves and
uses independent CPU replay. The explicitly authorized unbounded warning
diagnostic is nonproduction evidence: it records finite damage-range and
healing violations without clipping or projection, but keeps non-finite,
mechanics/CG, incomplete, structural, and work-check failures hard. Every
unbounded output is `validated=false` and must not be interpreted as an
admissible phase-field solution or full-path validation.

Run it only from exact zero with the reviewed inputs and a new run stem:

```bash
source .venv/bin/activate
python -m examples.fracture2d.engineering_grip_workflow run \
  --source-cloud /root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz \
  --fem-archive /root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz \
  --output-directory /root/autodl-tmp/data/fracture2d/engineering_grip \
  --run-id engineering_grip_warning_unbounded_u7p5e3_s72 \
  --acceptance-mode warning --warning-safety-policy unbounded \
  --load-final 0.0075 --max-stagger-iterations 72 --gpu-device-index 0 \
  --delta 0.048 --h-g 0.02 --du-initial 1e-4 --du-min 5e-5 \
  --visibility-tolerance 1e-12 \
  --youngs-modulus 210000.0 --poisson-ratio 0.3 --k-res 1e-8 \
  --rank-rtol 1e-12 --Gc 2.7 --l0 0.024
```

The unbounded policy can commit any finite raw damage value outside `[0, 1]`
and any finite healing magnitude after the final stagger budget. Those values
are diagnostic evidence only, not a relaxed physical formulation.

## Environment

```bash
# FEniCS FEM solver
conda activate fenicsenv
mpirun --allow-run-as-root -np 12 python examples/icemelting3d/solver.py

# JAX PINN training
source .venv/bin/activate
python examples/icemelting3d/train.py --config=examples/icemelting3d/configs/train_base.py

# IMPORTANT: deactivate one env before activating the other
```

## Source structure

```
src/pinn/
  operators.py        # SparseD, DenseD, ExplicitVolumeSparseD, build_D()
  fd.py               # Finite-difference Laplacian (ablation)
  causal.py           # CausalWeightor (per-time-step weighting)
  optimizers.py       # SOAP / Adam wrapper
  arch.py              # MLP, ModifiedMLP, and FiLM PINN trunks
  embedding.py        # Learnable Fourier feature embeddings
  constraints.py      # Hard output and KKS constraints
  mesh.py             # Gmsh node reader
  utils.py             # Shared training and logging utilities

benchmarks/
  RESULTS.md                      # B200 measurements
  TENSOR_SPLIT_RTX_PRO_6000.md    # RTX PRO 6000: gradient grouping analysis
  bench_step.py, bench_tensor_split.py, bench_operator.py
  verify_equivalence.py           # Correctness gate for optimizations
```
