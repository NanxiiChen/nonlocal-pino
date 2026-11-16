The operator maps a random rough metal–electrolyte interface to coupled phase and concentration trajectories. The geometry, material coefficients and boundary drive stay fixed.

## Setup and boundary data

The dimensionless rectangle is \([0,1]\times[0,0.5]\). The upper boundary prescribes electrolyte \((\phi,c)=(0,0)\), the lower boundary prescribes metal \((1,1)\), and the side walls close transport. A six-mode random height function generates a polyline interface. The initial diffuse phase uses its signed segment distance, positive below the profile, and \(c_0=c_{Se}h(\phi_0)\). Initial endpoint slopes are preserved.

<figure>
  <a href="assets/electropolish2d-setup.png"><img src="assets/electropolish2d-setup.png" width="1181" height="685" alt="Gray metal and blue electrolyte separated by a randomly generated initial interface"></a>
  <figcaption>One prescribed initial interface. The rectangle and top, bottom and side boundary conditions are shared by all samples. <a href="assets/electropolish2d-setup.pdf">Editable PDF</a>.</figcaption>
</figure>

## Governing equations and scaling

With \(\Delta c=c_{Se}-c_{Le}\), set

$$
h(\phi)=3\phi^2-2\phi^3,\qquad
g(\phi)=\phi^2(1-\phi)^2,\qquad
q=c-\Delta c\,h(\phi)-c_{Le}.
$$

The chemical free energy \(Aq^2\) gives the phase driving force \(2A\Delta c\,q h'(\phi)\) at fixed \(c\). The gradient and double-well terms give \(\alpha_\phi\Delta\phi-\omega_\phi g'(\phi)\). With \(L_c=100\,\mu\mathrm m\), \(T_c=5\,\mathrm s\), \(\mathbf x^*=\mathbf x/L_c\) and \(t^*=t/T_c\), the local equations are

$$
\begin{aligned}
\phi_{t^*}&=N_3\Delta^*\phi-N_2g'(\phi)+N_1\Delta c\,q h'(\phi),\\
c_{t^*}&=N_{\mathrm{CH}}\Delta^*q,
\end{aligned}
$$

where \(N_1=2ALT_c\), \(N_2=L\omega_\phi T_c\), \(N_3=L\alpha_\phi T_c/L_c^2\), and \(N_{\mathrm{CH}}=DT_c/L_c^2\).

## Nonlocal residual and loss

The bounded phase and unrestricted liquid concentration are mapped to the total concentration using the algebraic KKS mixture. Apply the graph to the physical phase and to \(q\).

$$
\begin{aligned}
r_{\mathrm{AC}}&=\phi_{t^*}+N_3D_\delta\phi+N_2g'(\phi)
                       -N_1\Delta c\,q h'(\phi),\\
r_{\mathrm{CH}}&=c_{t^*}+N_{\mathrm{CH}}D_\delta q .
\end{aligned}
$$

Mean absolute PDE residuals and three physical MSE terms (initial, top and bottom) give five losses. Inverse parameter-gradient norms balance these terms. The side walls use regional no-exchange closure, including for input profiles with incompatible initial normal slopes. The local-AD control enforces its side-wall derivatives through an additional positive-time penalty.

## Field evolution

<figure>
  <a href="assets/electropolish2d.png"><img src="assets/electropolish2d.png" width="2007" height="775" alt="Best, middle and worst phase-error interfaces from the full 64-case test cohort across five times"></a>
  <figcaption>The original rolling-pool 10,000-step model: Best / Middle / Worst are inputs 30 / 5 / 56 from all 64 test interfaces. Selection uses equal-point, phase-only relative L² over the full 21-time space-time grid; Middle is the lower central actual sample. Columns show t* = 0, 0.25, 0.5, 0.75 and 1 (0–5 s). Colors show predicted phase; dashed curves are local-FD101 φ = 0.5 interfaces. Panel labels give instantaneous spatial phase relative L². <a href="assets/electropolish2d.pdf">Editable PDF</a> · <a href="https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_ranked_main_preview.json">Selection and reproduction</a>.</figcaption>
</figure>

## References and comparisons

The selected 10,000-step operator scores **1.547% pooled and 1.326% terminal phase error**, and **1.860% pooled and 1.580% terminal concentration error**. Evaluation uses an independent centered local FD101 reference with mirrored side-wall closure, 64 unseen interfaces, \(101\times51\) nodes and 21 saved times. The same checkpoint gives 1.637% pooled phase error on the four finer FD401 cases. The representative result uses the better observed final run of two paired seeds.

The current spatial study compares fixed and rolling pools with identical cold model and Adam state, initial profiles, times, IC/BC queries and safe padding. Both use width 96, 16 shared times and 10,000 updates.

| Runtime seed | Pool | Phase pooled / terminal | Concentration pooled / terminal | Warm step median |
| --- | --- | ---: | ---: | ---: |
| 42 | Fixed | 1.747% / 1.526% | 2.089% / 1.827% | 40.06 ms |
| 42 | Rolling | **1.547% / 1.326%** | **1.860% / 1.580%** | 41.61 ms |
| 43 | Fixed | 2.633% / 2.050% | 3.142% / 2.440% | 39.78 ms |
| 43 | Rolling | **2.465% / 1.672%** | **2.930% / 2.001%** | 41.93 ms |

All four arms ran serially on the same RTX 4090. Warm steps include synchronized condition preparation, spatial transfer and updates after the first 100 steps. Active peak memory is about 13.0 GiB. Rolling improves errors in both pairs with about 4–5% more step time. Seed-to-seed spread is larger than the pooled improvement within each pair.

The historical 5,000-step local-AD comparison used the same FD101 bank. Its two arms share the initial model and actual condition, time, core, IC and prescribed-boundary draws. The AD arm computes coordinate Hessian traces and adds the side-wall loss.

| Quantity | Nonlocal PINO, 5,000 steps | Local-AD PINO, 5,000 steps |
| --- | ---: | ---: |
| Phase \(L^2\), pooled / terminal | 4.427% / 3.176% | 8.595% / 5.498% |
| Concentration \(L^2\), pooled / terminal | 5.222% / 3.779% | 9.835% / 6.431% |
| Cold training process | 368.35 s | 693.03 s |
| First update, compilation plus call | 223.35 s | 533.87 s |
| Warm synchronized update mean | 16.71 ms | 19.35 ms |

The reported cold time includes setup and periodic evaluation. It excludes reference generation and the final 64-input test. On the direct nonlocal target, the historical 5,000-step graph model gives 4.392% pooled and 3.041% terminal phase error. The paired experiment measures this geometry family and one initialization.

## Three-method comparison

The cold 10k control compares **Nonlocal, Local AD and FEM PINO** with width 96,
batch 8 fresh initial profiles, 16 random shared times, Adam and the same rolling 256-core stream. Initial
model/Adam state, actual profile/time/core draws and saved common-input witnesses
pass exact matching. The comparison-only prescribed encoder/IC/Dirichlet data
use the existing SDF on CPU in FP64 with one FP32 cast; model training stays FP32.
Production defaults are unchanged.

FEM PINO uses the assembled P1 lumped weak operator \(M_L^{-1}K\) on the rectangular
mesh, with natural homogeneous side-wall Neumann flux and complete sampled rows.
It is a trained neural operator. The common **numerical reference** is an
independent local finite-difference solver. Nonlocal and FEM retain two L1 PDE
and three MSE IC/Dirichlet groups; Local AD adds a positive-time side-wall
Neumann penalty. The original random initial slopes are retained.

| Method | Phase | Concentration | Channel mean |
|---|---:|---:|---:|
| Nonlocal | 1.9713% | 2.3563% | 2.1638% |
| Local AD | 4.3904% | 5.1374% | 4.7639% |
| FEM PINO | 1.8486% | 2.2072% | 2.0279% |

<figure>
  <a href="assets/electropolish2d_three_method_convergence.png"><img src="assets/electropolish2d_three_method_convergence.png" width="2007" height="708" alt="Phase and concentration convergence on all 64 test cases for three PINO methods"></a>
  <figcaption>All 41 evaluations at steps 0, 250, …, 10,000 use the same 64 local-FD cases. Thin solid lines show raw sample means; thicker lines show log-LOWESS with span 0.22. Shading is a pointwise 95% empirical bootstrap interval from 2,000 resamples of complete case trajectories, paired across methods (seed 20261114). These are conditional intervals for a reused cohort and one training realization, not cross-seed uncertainty or guaranteed population/simultaneous coverage; smoothing and model-selection bias are not covered. The 401-point display grid adds no observations. <a href="assets/electropolish2d_three_method_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/electropolish2d_three_method_distribution.png"><img src="assets/electropolish2d_three_method_distribution.png" width="2007" height="708" alt="Final 10k phase and concentration error violins on 64 common local-FD test cases"></a>
  <figcaption>Final 10k weights, 64 errors per physical channel. Each error is the equal-node space-time norm on the 101×51 local-FD grid and 21 uniform saved times from 0 to 1, normalized per sample/channel before averaging samples. Extra early snapshots are excluded from this statistic. Log-error Gaussian KDE uses Scott bandwidth, three-bandwidth tails and internal quartiles; tails are density estimates, not extra samples. Axes show raw ratios without percentage scaling. <a href="assets/electropolish2d_three_method_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

FEM PINO is slightly more accurate than Nonlocal in this matched run, while
both have lower errors than Local AD. One initialization and a repeatedly
inspected test cohort do not establish a stable cross-seed ranking. These
comparison weights do not replace the selected SOTA/default or main field image.

| Method | Cold training process | First compile + update | Warm update mean | Active peak |
|---|---:|---:|---:|---:|
| Nonlocal | 15.63 min | 169.02 s | 32.78 ms | 12.03 GiB |
| Local AD | 18.27 min | 399.30 s | 23.32 ms | 5.41 GiB |
| FEM PINO | 15.44 min | 166.71 s | 32.32 ms | 12.02 GiB |

All three arms ran serially on the same RTX PRO 6000 Blackwell Server Edition
(UUID `GPU-85fe6412-636e-5d3c-d5f0-d1e9523c3faa`). Cold process time includes setup,
compilation, all in-training evaluation and input preparation; it excludes
reference generation and the separate final 64-case inference. Warm update time
includes synchronized differentiated loss/optimizer work, excluding condition
preparation, spatial transfer and evaluation. Mean condition preparation after
the first call is 33.22 / 35.98 / 32.99 ms (NL/AD/FEM), including the common CPU SDF
path; it is not isolated incremental host overhead. AD retains the shared graph
preparation for matching draws, so this is not an optimized graph-free pipeline.
AD uses less active memory and has faster warm updates here, but its longer
initial compilation increases total cold cost. Active peaks are JAX allocator
peaks, not the approximately 76 GiB preallocated pool. No universal speed claim follows.

[Exact checkpoint/configuration/figure record](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_three_method_figures.json)
· [resource measurements](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/electro_three_method_resources.json)
· [reproduction guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/docs/PINO_COMPARISON_FIGURES.md).
The earlier attempt with roundoff-sized input mismatches remains recorded;
the figures above use only the corrected fully paired run.

## Time-feature fusion comparison

This completed nonlocal comparison uses width 96, B8, Nt16, rolling 256 cores
and cold Adam 10k. Decoder FiLM attends only over spatial queries, then modulates
every pointwise hidden decoder layer with time. Joint queries project concatenated
spatial/time Fourier features into cross-attention and use unmodulated decoder
layers. Each condition is encoded once in both models; the joint version repeats
attention for each time. Time derivatives remain JVPs.

The shared encoder/attention/spatial-projection/decoder initialization and every
actual condition/time/core draw match exactly. Physical losses, optimizer recipe
and known prescribed inputs match; all 41 evaluations use the same 64 local-FD
inputs. The temporal modules and their optimizer-tree shapes differ: 320,706 versus
264,738 stored numeric entries, including frozen frequencies, with 263,106 shared
entries. This isolates the declared architecture change, not equal total capacity
or identical full initial networks. Final 10k weights are used without checkpoint
selection; positive-time reference fields never enter training.

| Temporal conditioning | Phase | Concentration | Channel mean |
|---|---:|---:|---:|
| Decoder FiLM | 1.9713% | 2.3563% | 2.1638% |
| Joint (x,t) query | 2.0577% | 2.4520% | 2.2549% |

<figure>
  <a href="assets/electropolish2d_time_fusion_convergence.png"><img src="assets/electropolish2d_time_fusion_convergence.png" width="2007" height="708" alt="Electropolish phase and concentration for decoder FiLM versus joint space-time query"></a>
  <figcaption>All 41 recorded sample means on 64 common local-FD inputs, evaluated every 250 steps including zero. Thin solid lines are raw means; thicker lines are log-LOWESS (span 0.22). Shading is a pointwise 95% empirical bootstrap interval from 2,000 resamples of complete sample trajectories, using identical resampling indices across methods (seed 20261114). The fitted display grid adds no observations. These intervals condition on a reused cohort and one training realization; they do not cover cross-seed uncertainty or smoothing/model-selection bias. The horizontal axis is elapsed wall time in seconds from cold process launch, including initialization, compilation and in-training evaluation. Intermediate times are reconstructed from the original launch-log creation, metric-file timestamps and recorded trainer counters; they are not direct parent timestamps and subsecond accuracy is not claimed. The existing final observation is placed at complete process exit, including final writes and teardown. No error is invented at time zero, and no step-rate extrapolation is used. LOWESS and its bootstrap interval are fitted on the elapsed-time coordinates. <a href="assets/electropolish2d_time_fusion_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/electropolish2d_time_fusion_distribution.png"><img src="assets/electropolish2d_time_fusion_distribution.png" width="2007" height="708" alt="Electropolish phase and concentration for decoder FiLM versus joint space-time query"></a>
  <figcaption>Final 10k phase/concentration relative norms on all 101×51 nodes and 21 uniform times from 0 to 1, with equal point weights and separate normalization for each sample/channel. Extra early diagnostic snapshots are excluded. Each violin uses all 64 sample errors, Gaussian KDE in log-error space, Scott bandwidth, three-bandwidth tails and quartiles. Tails are estimated density, not extra samples; axes show raw ratios, not percentages. <a href="assets/electropolish2d_time_fusion_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

Decoder FiLM has slightly lower final errors in this run, but the two models
have close accuracy and one initialization does not establish a stable accuracy
ranking. Joint queries cost more training time and active memory despite fewer
stored parameters. This is consistent with time-dependent attention processing
more queries; it does not imply that all inference workloads scale the same way.
Keep the current SOTA/default and older 5k/Nt4 ablation separately.

| Temporal conditioning | Cold training process | Warm inclusive median | Active peak | Stored entries |
|---|---:|---:|---:|---:|---:|
| Decoder FiLM | 15.63 min | 64.61 ms | 12.03 GiB | 320,706 |
| Joint (x,t) query | 23.65 min | 105.76 ms | 32.45 GiB | 264,738 |

Both runs use the same RTX PRO 6000 Blackwell Server Edition
(UUID `GPU-85fe6412-636e-5d3c-d5f0-d1e9523c3faa`). Cold time includes setup,
compilation and all in-training evaluations, excluding reference construction
and the separate final 64-case inference/audit. Warm inclusive time sums synchronized
loss/gradient/update, condition preparation and spatial transfer after the first
step, excluding evaluation. Active memory is allocator usage, not reservation.
The warmed complete 64-case monitor has a median cost of 0.412 s with FiLM
and 0.445 s with joint queries; this includes sequential prediction,
host transfer and error reductions, so it is not an isolated network benchmark.

[Direct checkpoint/configuration/figure provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_time_fusion_figures.json)
and [reproduction guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/docs/PINO_COMPARISON_FIGURES.md#nonlocal-time-feature-fusion-controls).

## Fixed initial-condition library size

Four cold runs use nested fixed libraries, the same initialization, 10,000 updates,
rolling spatial core/halo pools and the same 64 local-FD test inputs. Only the
number of distinct training profiles changes. Values below average separate
equal-point space-time phi/c relative norms over channels and test samples.

| Fixed profiles | Phase | Concentration | Channel mean ± sample SD |
|---:|---:|---:|---:|
| 64 | 12.16% | 13.59% | 12.87% ± 3.44% |
| 128 | 2.43% | 2.89% | 2.66% ± 1.09% |
| 256 | 2.62% | 3.13% | 2.88% ± 1.06% |
| 512 | 1.83% | 2.20% | 2.01% ± 0.80% |

<figure>
  <a href="assets/electropolish_ceramic_fixed_library_error.png"><img src="assets/electropolish_ceramic_fixed_library_error.png" width="2007" height="732" alt="Final error versus fixed training-library size for Electropolish and ceramic corrosion"></a>
  <figcaption>Electropolish is panel (a); panel (b) is the separate Ceramic 20k study. Each marker uses final weights and 64 test samples; shading is one sample standard deviation, not a confidence interval or seed variability. Axes show raw relative-error ratios. <a href="assets/electropolish_ceramic_fixed_library_error.pdf">Editable PDF</a>.</figcaption>
</figure>

The configuration-matched continuous-online baseline gives 2.16% channel-mean
error. The older selected online main model gives 1.70%, but uses the original
GPU FP32 prescribed-input computation; the matched controls share CPU FP64
encoder/IC data cast to FP32. Physics and network precision are unchanged. This
implementation difference has not been established as the cause of the error gap.
One initialization does not establish a monotonic library-size law.

The fixed-32 and repeated-online controls are now running to 15k with a 10k
checkpoint and the same 64-case evaluation. Their results are pending. Earlier
fixed-library and FiLM comparisons retain their original 10k budgets; defaults
remain unchanged. [Completed data and reproduction](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/fixed_condition_library_results.json).

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Condition | \(32\times64\) image, \(4\times8\) patches, 64 tokens |
| Model | Width 96, four heads, two encoder blocks, spatial cross-attention and three time-FiLM layers |
| Spatial work | \(101\times51\) graph, horizon 0.02, 256 cores plus full halo, 32 initial sets and a background rolling FIFO |
| Samples per update | 8 fresh initial profiles and 16 shared times |
| Background pool | One CPU worker, 32 prefetched sets, fixed 3,328-node / 3,072-edge capacities |
| Optimizer | Adam, \(10^{-3}\) initial rate, 10,000 updates, ordinary FP32 |

~~~bash
source .venv/bin/activate
python -m examples.electropolish2d.train \
  --config=examples/electropolish2d/configs/train_operator.py
~~~

The preset expects its prepared monitoring bank. The independent native/local evaluation banks and local-AD control are documented in the [example and reference preparation guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/electropolish2d/README.md).
