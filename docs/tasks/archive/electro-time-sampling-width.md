# Electropolish time sampling and width

## Goal

Test whether 16 sampled times improve the audited four-time Electropolish
operator and approach 2% pooled phase relative L2. If the error exceeds 2.2%,
test a modest width increase with time sampling held fixed.

## Acceptance

Complete the matched cold 5,000-update arm, score all fields against the same
64 local FD101 targets, and verify configuration, initialization and sampled
inputs. Separate time sampling and width effects and report the target gap.
Plot convergence on the four held-out graph-reference inputs and display
the independent FD101 final scores separately. Check late convergence and
extend the promising arm to 10,000 updates when its monitor is still falling.

## Result

The [time-injection ablation](electro-temporal-conditioning.md)
provides the completed width-64, four-time baseline and paired audit.
All four final checkpoints have been evaluated on the common local FD101
set. The 2% pooled phase target remains open.

| Times | Width | Updates | Phase pooled | Phase at 5 s | Concentration pooled | Concentration at 5 s | Warm update | Trainer wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 5,000 | 3.985% | 3.043% | 4.724% | 3.628% | 10.39 ms | 167.62 s |
| 16 | 64 | 5,000 | 4.612% | 3.425% | 5.436% | 4.067% | 21.89 ms | 236.92 s |
| 16 | 96 | 5,000 | 3.897% | 2.849% | 4.612% | 3.399% | 30.57 ms | 327.46 s |
| 4 | 64 | 10,000 | 4.015% | 3.238% | 4.760% | 3.846% | 10.36 ms | 238.42 s |

Increasing the time count at width 64 improved the pooled phase metric on
11 of 64 cases. Increasing width to 96 at 16 times improved 56 of 64 against
the width-64 arm at the same time count and 35 of 64 against the original
four-time control. The four-case FD401 check gives pooled phase errors of
4.490%, 4.877%, 4.737%, and 4.294% in table order. The wider arm ranks
slightly ahead of the four-time baseline on the 64-case FD101 set. The
four-time baseline ranks ahead on the four FD401 cases.

The 5,000-step four-time curve fell from 5.94% to 5.07% on the four-interface
monitor over its last 1,000 updates. That trend and the lower step cost
motivated the longer four-time trial. In its own history, the monitor
reached 4.73% at 8,000 updates and fluctuated between 4.21% and 5.21% over
the final 1,000 updates. The final local FD101 phase score is 4.015%,
compared with 3.985% in the separate 5,000-step run. The local score improved
on 29 of 64 cases. The plateau and the independent score support closing
this comparison at 10,000 updates and retaining the default.

## Protocol

Use the canonical `examples.electropolish2d.train` entry and the historical
`configs/train_operator_5k.py` recipe through `.venv`. All runs use Adam, ordinary
FP32 compilation, batch eight online initial interfaces, 256 spatial cores
with complete halo from the same fixed pool, spatial cross-attention and
three decoder time-FiLM layers. The scheduler is
`max(1e-5, 1e-3 * 0.95 ** (step / 1000))`. The host is an RTX 4090 with
48 GiB of memory. Warm timings are synchronized medians for updates without
a GradNorm refresh.

The convergence traces use the same four held-out graph-reference
interfaces every 250 steps. Local FD101 comparisons use 64 common unseen
interfaces and 21 saved times including the endpoint at 5 s. Metrics are
relative L2 per interface, averaged across cases. The 5,000-update result
from a separate cold run is compared with the long run's endpoint. The
long run's own convergence history is used for the within-run comparison.

## Validation

- The two width-64 5,000-step arms start from byte-identical model files.
  All three 5,000-step arms use identical spatial selections and 40,000
  accepted interface draws. Both 16-time arms use identical sampled times.
- The longer run matches the four-time control's initial model bytes and
  the spatial, temporal and interface draws through update 5,000. Its monitor
  trajectories vary despite matched inputs. The four-case pooled phase
  monitor at update 5,000 is 5.91% in the longer run and 5.07% in the
  separate short run. The maximum absolute prefix gap across the recorded
  monitor metrics is 6.17 percentage points. The budget comparison covers
  one seed and two cold executions.
- An independent CPU audit recomputed relative L2 from saved predictions
  and native/FD101 fields for all 64 cases in each arm. Final replay on
  the trainer's monitor fields agrees to within \(5.67 \times 10^{-7}\)
  in absolute field value.
- The [fixed-budget evaluator](../electro_spatial_evaluate.py) checks each
  finished run against the iterations declared in its configuration. The
  10,000-step artifact completed this path with an independent CPU audit.

## Run artifacts

The run root is `/root/autodl-tmp/tf-logs/electropolish2d`. The baseline and
each control contain `metrics.jsonl`, `spatial_evaluation64.json`,
`spatial_evaluation64.npz`, `model_final.eqx`, and `timing.json`.

| Run directory | Additional training flags |
|---|---|
| `20261110-electro-time-injection-s42-decoder` | Default |
| `20261110-electro-time16-s42-w64` | `--config.sample.Nt=16` |
| `20261110-electro-time16-s42-w96` | `--config.sample.Nt=16 --config.arch.width=96` |
| `20261110-electro-time4-s42-w64-10k` | `--config.train.iterations=10000` |

The convergence plot combines the logged four-case graph-reference curves
with the independent 64-case final scores from each run.

Status complete. No blocker. Training defaults and numerical code are
unchanged. The evaluation utility, README link, and repository task records
were updated.
