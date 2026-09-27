# Project Progress

> Dashboard, not changelog. Keep this short — detail lives in task files / git history.

## Current focus

- Active task: [`docs/tasks/fracture2d-lean-direct-solver.md`](tasks/fracture2d-lean-direct-solver.md).
  The adopted direct solver is now **one GPU-only file**,
  `examples/fracture2d/nonlocal_baseline.py` (`12,188` lines): the
  `src/pinn/nonlocal2d/` package was merged back with a behavior-preserving
  linker (`45,346 -> 16,259` lines), the CPU CG reference path was removed
  (the adapter is required, CPU replay evaluation stays authoritative), and the
  research variants were parked under `_legacy/` (ignored). Verification:
  `model_digest`/`static_digest` match the archived compact solution; entry
  tests `4/4`; GPU CLI smoke `0 -> 2e-4` returns `25.34683561371731`; full
  `verify`/`plot` pass. The Anderson / matched-load schedule / archive-budget dead code was then
  removed, and non-`fixed` policies are rejected by configuration.
- Next task (queued): [`docs/tasks/fracture2d-nonlocal-pinn.md`](tasks/fracture2d-nonlocal-pinn.md).
  The PINN must reuse the same graph, A2 energy, Amor history, and AT2 residual
  from `src/pinn/nonlocal2d/`.

## Direct nonlocal reference (adopted)

- Scheme: original BA/Amor/AT2 equations on the 3395-node `delta=0.048` cloud,
  `h_g=0.02`, `l0=0.024`, `Gc=2.7`, `E=210e3`, `nu=0.3`, `k_res=1e-8`, cubic
  kernel, fixed damped-Picard `omega=0.5`, `du=1e-4`, `du_min=6.25e-6`, at
  most 144 sweeps, `stabilized_warning` acceptance, load path `0 -> 0.0075`.
- Artifacts: fresh compact solution
  `.../engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz`
  (SHA256 `5f3bc12b...`); research archive SHA256 `9545bed7...`.
- Response: peak `641.34017` at `0.0061` (FEM `642.48573` at `0.0057`);
  terminal `69.15426` at `0.0075` (FEM `3.97718`); 76 states = 65 strict + 10
  warning; drift points `2 -> 31` from `0.0062` to `0.0075`.
- Accepted limitations: warning-only acceptance,
  `field_equations_validated=false`, terminal reaction mismatch, particle
  drift, and no completed strict `0 -> 0.0075` path. All are documented in
  `examples/fracture2d/README.md`.

## Current state

- `fracture1d` is an accepted engineering baseline; default PINN dtype is
  float32 with `spatial_emb_dim=32` and `spatial_scale=16`.
- The local FEM point-anchor warning-mode archive
  (`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz`)
  remains the separate continuum comparator.
- Parked and ignored: monolithic/arc/reduced research bundle, durable
  `engineering_grip_workflow.py`, and the stage-specific fracture2d test suites
  (moved to `_legacy/`). They remain in git history and on disk for reference.
- Memory: `docs/DECISIONS.md` is the append-only rationale log;
  `docs/tasks/archive/` holds the closed direct-stage records.
