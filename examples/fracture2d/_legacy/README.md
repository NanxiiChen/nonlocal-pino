# Archived fracture2d research

Historical nonlocal displacement/A2 solvers, one-off loading/restart drivers,
range/conditioning/coupling diagnostics, CPU auditors, experiments and their
configurations are retained here. The public route does not import this archive.
See [HISTORICAL_README.md](HISTORICAL_README.md) for the previous detailed
formulation/results record and the active task for exact run checkpoints.

The [experiment index](EXPERIMENT_INDEX.md) preserves the long historical
catalogue formerly in the repository README. Newer one-off sampling count,
resource, density, batch-hold, tolerance and speed-review sources are grouped
in [sampling_experiments_20261015](sampling_experiments_20261015/README.md).
`configs/train_core_halo_original.py` preserves the segment-only9/0 preset;
`configs/train_core_halo_tip.py` preserves the former separate tip preset.
The supported core+halo preset is now `../configs/train_core_halo.py`.

Module paths now start with `examples.fracture2d._legacy`; archived configs
live in `_legacy/configs/`. Existing imports and repository-root calculations
are relocated, not numerical formulas. Legacy regressions are under
`tests/_legacy/fracture2d/` and can be run explicitly:

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  .venv/bin/python -m unittest discover -s tests/_legacy/fracture2d
```

The pre-migration source is preserved at revision6ce5162 and in saved run
code snapshots. Use those for exact historical replay; the new public route
has a simpler file-path/configuration contract and does not silently resume
an old prefix/tail run. Saved external results and TensorBoard events remain
untouched. Do not launch archived jobs without a fresh explicit configuration
and output directory.
