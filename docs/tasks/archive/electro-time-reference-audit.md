# Electropolish temporal sample count and reference audit

## Goal and acceptance

Check whether the 16-time operators need 10,000 updates and whether the
single-seed error difference at width 64 is sensitive to the training
execution or choice of numerical reference.

- Use the same training trajectories to compare 5,000 and 10,000 updates.
- Match width, initialization, spatial and condition draws for the temporal
  sample-count comparison. Also extend the previously best width-96 arm.
- Score the same 64 graph and local FD101 solutions, and compare FD101,
  FD201 and FD401 on the same four refinement cases. Check agreement of
  reference grids, times, initial fields and solver tolerances.
- Distinguish finite-reference sensitivity, repeatability and the effects
  of minimizing a finite-budget stochastic physics objective.

## Matched training budgets

The [prior budget comparison](electro-time-sampling-width.md) used separate
cold runs at four times. This comparison saves both checkpoints on each
seed-42 training trajectory and scores them against the same 64 local FD101
targets. Pooled errors use 21 snapshots and terminal errors use 5 s.

| Shared times | Width | Phase pooled 5k → 10k | Phase terminal 5k → 10k | Concentration pooled 5k → 10k | Concentration terminal 5k → 10k |
|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 4.220% → 2.483% | 3.408% → 2.138% | 4.995% → 2.963% | 4.039% → 2.550% |
| 16 | 64 | 4.286% → 2.555% | 3.028% → 1.760% | 5.056% → 3.049% | 3.613% → 2.105% |
| 16 | 96 | 3.934% → 1.740% | 3.234% → 1.540% | 4.658% → 2.072% | 3.859% → 1.843% |

All 64 cases improve in pooled phase error within each trajectory, against
both native and local references. The lowest pooled phase error is 1.740%
for 16 times and width 96 at 10,000 updates. It scores 1.537% on the same
64 native targets and 2.002% on four local FD401 targets. FD101 scores
2.043% on those same four cases. Width 96 at 10,000 updates is covered
by one seed, and the four-time width-96 control remains untested.

## Sample-count repeatability

At width 64, the new seed-42 pooled FD101 difference (16 times minus 4
times) is +0.066 percentage points at 5,000 and +0.072 at 10,000. At
seed 43 and 5,000 updates the scores are 6.625% and 6.588% respectively,
a difference of -0.037 points. On the 64 native targets, the corresponding
differences are +0.068, +0.095 and -0.011 points. The seed-43 four-case
subset has a positive difference on all local resolutions (+0.408 on
FD101, +0.429 on FD201 and +0.433 on FD401). Hold the sample set fixed
when interpreting the reference comparisons.

The previous four-time 10,000-step run scores 4.015%, compared with
2.483% in this run. Its initial model and all recorded input draws match
byte-for-byte. Two older four-time 5,000-step runs also share their
initial model and inputs and score 3.985% and 4.427%. This variation
limits attribution of the small equal-width differences to time count.
More uniform samples reduce variance of the raw temporal residual
average at fixed parameters, spatial nodes and conditions. The adaptive
inverse-gradient-norm weights, Adam trajectory and field evaluation
objective differ from that fixed-parameter calculation.

## Reference checks

The historical width-64 pooled penalty was +0.628 points on 64 FD101,
+0.658 on 64 native and +0.387 on the same four FD401 cases. The reference
switch preserves the direction. The code and saved metadata agree on
physical parameters, nondimensional scales, initial interface, snapshot
times and top/bottom Dirichlet values. The FD reference implements side
Neumann conditions with the mirror stencil. The native operator uses
in-domain graph bonds and differs in spatial discretization.

All 72 local and 64 native reference trajectories pass direct saved-field
coarse/fine time checks with maximum relative differences of 2.58e-8 and
4.60e-8 respectively. Local FD101 and native initial fields agree within
2.2e-19 at aligned nodes. The four-case mean pooled phase differences
FD101→FD201 and FD201→FD401 are 0.322% and 0.073%. Native→FD401 is 0.702%
on the same four cases. The local grid test covers four inputs.

## Protocol and validation

The canonical operator uses Adam, the original learning-rate schedule
starting at 1e-3, eight online conditions, 256 spatial cores with halo,
and the same fixed pool and space-attention/time-FiLM decoder. Per seed,
the runs share spatial and condition draws and the first four time
draws at each step. Equal-width initial models are byte-identical.
All eight checkpoints pass independent CPU recomputation of each stored
field metric against the aligned references. Maximum monitor prediction
replay difference is 4.42e-6 in absolute field value. All training source
snapshots match. The focused local mirror stencil and independent
reference Jacobian/solver agreement tests pass. GPU workloads overlapped,
so execution timings are excluded from the comparison.

The run root is `/root/autodl-tmp/tf-logs/electropolish2d`.

| Run directory | Seed | Times | Width | Updates |
|---|---:|---:|---:|---:|
| `20261111-electro-time-audit-s42-t4-w64` | 42 | 4 | 64 | 10,000 |
| `20261111-electro-time-audit-s42-t16-w64` | 42 | 16 | 64 | 10,000 |
| `20261111-electro-time-audit-s42-t16-w96` | 42 | 16 | 96 | 10,000 |
| `20261111-electro-time-audit-s43-t4-w64` | 43 | 4 | 64 | 5,000 |
| `20261111-electro-time-audit-s43-t16-w64` | 43 | 16 | 64 | 5,000 |

Use the settings in the table with the canonical operator preset.
Set `--config.log.checkpoint_step=5000` on the long runs, and
`--step=5000` in the [evaluator](../electro_spatial_evaluate.py) to score
their intermediate checkpoint. Saved evaluations are
`spatial_evaluation64_step_005000.{json,npz}` and
`spatial_evaluation64.{json,npz}`. The short runs use the final path.
The convergence plot shows the four-case monitor and the 64-case endpoint
comparisons on separate axes.

Complete. The intermediate write is opt-in. The experiments did not change
the training defaults at the time. All owned runs and evaluations exited
successfully.
