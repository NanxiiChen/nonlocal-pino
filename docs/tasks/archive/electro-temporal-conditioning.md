# Electropolish temporal conditioning ablation

Goal: measure the field accuracy of alternative time injection on one
report case. Use the selected Electropolish 2D operator recipe and the
independent 64-interface local reference bank.

Acceptance: complete three matched 5,000-step cold runs, check training
draws and shared initialization, and report pooled and terminal relative
L2 for phase and concentration against the same local reference.

Arms: the existing three post-attention FiLM layers, a transfer of the first
FiLM modulation to the input of spatial cross-attention, and a joint
space-time query with no FiLM. Keep the encoder, width, attention, loss,
optimizer and physical sampling recipe. Report parameter counts explicitly.

Status: all three serial cold runs completed 5,000 updates, followed by the
common evaluation. Field errors are means of per-interface relative L2 norms
on 64 unseen inputs against independently time-refined local FD101. Pooled
norms use 21 times over the full interval and all 101 × 51 nodes.

| Arm | Phase pooled | Phase terminal | Concentration pooled | Concentration terminal | Stored entries | Warm nonrefresh median |
|---|---:|---:|---:|---:|---:|---:|
| Decoder FiLM | 3.98457% | 3.04288% | 4.72401% | 3.62814% | 146,242 | 10.39 ms |
| Query FiLM | 5.90800% | 4.78063% | 6.93101% | 5.63743% | 146,242 | 20.87 ms |
| Joint query | 8.08525% | 6.23716% | 9.32416% | 7.27765% | 121,218 | 19.77 ms |

The decoder arm has lower pooled phase and concentration errors on all 64
inputs than either alternative. The finer FD401 targets on four predetermined
inputs give pooled phase errors of 4.490%, 5.964%, and 7.979%, in the same order.
The 64 native graph targets give 3.910%, 5.871%, and 8.096%. The separate
in-training graph monitor uses a smaller cohort.

The relocation compares equal stored capacity and all 73 identical initial
arrays. The joint arm shares all 69 backbone arrays outside the removed time
MLP and introduces a bias-free time projection. A joint query uses the sum of
the spatial affine projection and the temporal linear projection, equivalent
to one affine projection on concatenated features. The difference in capacity
is part of this comparison. These results cover one seed, one case, and one
training budget. The timings are synchronized nonrefresh medians on the 48 GB
RTX 4090 with ordinary FP32 compilation.

Validation: the CPU check found exact historical forward values and common
initial leaves. Both new time JVPs matched the full time Jacobian and centered
differences. All six existing Electropolish operator regression tests pass.
All 40,000 sampled profiles, acceptance counts, and saved spatial, time, IC
and BC draws match across the runs. Same-device inference replay matches the
saved monitor fields exactly. Recomputed the errors of all 192 saved
predictions against both local and native references independently.

Artifacts: under `/root/autodl-tmp/tf-logs/electropolish2d`, use the prefix
`20261110-electro-time-injection-s42-` with `decoder`, `query_film`, or
`joint_query`. `.stdout.log` and `.eval.log` are adjacent to each run directory.
The variants use the explicit `train_operator_temporal_query.py` preset and
`--config.arch.temporal_query`, with `--seed=42` for all three arms.

Use `docs/tasks/electro_spatial_evaluate.py --run <run-directory>` with the
saved configuration and the existing evaluation banks. The paired audit is
stored as `20261110-electro-time-injection-s42-audit.json` alongside the run
directories. The configuration switch is opt-in. `git diff --check` passes.
