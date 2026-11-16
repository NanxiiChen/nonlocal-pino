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

The selected comparison uses two cold **15,000-update** nonlocal runs: spatial
cross-attention followed by time FiLM in every pointwise hidden decoder layer,
and joint spatial/time Fourier queries with unmodulated decoder layers. Both
encode each condition once; joint queries repeat attention for each sampled time.
Time differentiation uses JVP in both. Width 96, batch 8, 16 shared random times,
rolling 256-core/halo sampling, Adam and the original per-step learning-rate decay
are unchanged. Positive-time reference fields never enter training.

All 61 evaluations, including step zero, cover the same 64 local-FD test inputs.
The shared initial backbone and every actual condition/time/core draw match.
The temporal modules have different capacities: 320,706 versus 264,738 stored numeric
entries (including frozen frequencies), with 263,106 shared entries. This is
one paired initialization, not an equal-capacity or cross-seed accuracy study.
Only the final 15k weights enter the final comparison.

| Temporal conditioning | Phase | Concentration | Channel mean |
|---|---:|---:|---:|
| Decoder FiLM | 1.3960% | 1.6711% | 1.5335% |
| Joint (x,t) query | 1.5403% | 1.8522% | 1.6963% |

<figure>
  <a href="assets/electropolish2d_time_fusion_convergence.png"><img src="assets/electropolish2d_time_fusion_convergence.png" width="2007" height="708" alt="Electropolish 15k temporal comparison"></a>
  <figcaption>All 61 actual 64-case means, every 250 updates. Thin solid curves show raw means; thicker curves are log-LOWESS (span 0.22). Shading is the pointwise 95% whole-case-trajectory bootstrap interval (2,000 resamples, seed 20261114), conditional on this reused cohort and one training realization. The horizontal axis measures seconds from cold process launch, including setup, compilation, all training evaluations, writes and teardown. Intermediate observation times are reconstructed from original file timestamps and trainer counters; no subsecond accuracy or error at elapsed zero is claimed. The final existing observation is placed at complete process exit; no warm-step-rate extrapolation is used. <a href="assets/electropolish2d_time_fusion_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/electropolish2d_time_fusion_distribution.png"><img src="assets/electropolish2d_time_fusion_distribution.png" width="2007" height="708" alt="Electropolish 15k temporal comparison"></a>
  <figcaption>Final 15k weights, 64 test inputs, all 101×51 nodes and 21 uniform times in [0,1]. Each channel/sample uses its own equal-point full-space-time relative L2 norm. Violins show unsmoothed sample errors with log-error Gaussian KDE, Scott bandwidth, three-bandwidth tails and quartiles. Density tails do not add observations; plotted values are raw ratios. <a href="assets/electropolish2d_time_fusion_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

| Temporal conditioning | Complete cold process | Warm inclusive median | Active peak |
|---|---:|---:|---:|
| Decoder FiLM | 1283.698 s (21.39 min) | 63.41 ms | 12.03 GiB |
| Joint (x,t) query | 2146.986 s (35.78 min) | 113.63 ms | 31.59 GiB |

Accuracy is close in this realization; decoder FiLM uses about 40% less complete
cold training time. This does not establish a stable cross-seed accuracy ranking.

Both runs use the same NVIDIA RTX 6000D and driver, with CUDA command capture
explicitly disabled and no overlapping GPU training. Complete cold times exclude
offline reference construction and post-training independent inference/audits.
Warm inclusive time sums the update, condition preparation and spatial transfer
after the first update; it excludes evaluation. Active memory is allocator use,
not reserved memory. The production defaults and earlier derivative/5k ablations
remain separate from these 15k research controls.

[Weights, configurations, paired inputs, measured timings and figure reproduction](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_time_fusion_figures.json) · [Cold-clock evidence](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_time_fusion_wall_clock.json).

## Fixed initial-condition library size

Five cold 15k runs use nested fixed libraries of 32/64/128/256/512 initial profiles.
The continuous-online control also trains for 15k. All six share initialization,
physics, width 96/batch 8, 16 random times, Adam, rolling 256-core/halo sampling,
the original learning-rate schedule and 64 local-FD test cases. Only the number
of distinct training profiles changes; fixed libraries contain input coefficients,
not evolved reference fields. Online sampling draws fresh profiles each update.

For each test input and each physical channel (phase and concentration), compute
the equal-point relative L2 norm over the full 101×51 grid and 21 uniform times.
Average the two channel errors within each input, then average 64 inputs.
The table and graph use final 15k weights; spread is one across-input sample SD,
not a confidence interval or cross-seed variability.

| Training profiles | Phase | Concentration | Channel mean ± sample SD |
|---|---:|---:|---:|
| 32 | 15.6599% | 17.2311% | 16.4455% ± 3.6397% |
| 64 | 11.5526% | 12.9896% | 12.2711% ± 3.1282% |
| 128 | 5.0167% | 5.8947% | 5.4557% ± 2.2159% |
| 256 | 1.9316% | 2.3116% | 2.1216% ± 0.9882% |
| 512 | 1.8466% | 2.1998% | 2.0232% ± 0.7165% |
| Continuous online | 1.3960% | 1.6711% | 1.5335% ± 0.5809% |

<figure>
  <a href="assets/electropolish2d_library_size_error.png"><img src="assets/electropolish2d_library_size_error.png" width="1003" height="732" alt="Electropolish15k error versus fixed initial-condition count"></a>
  <figcaption>Electropolish alone, final 15k weights and the same 64 test inputs. Markers show fixed-library means; the horizontal online line is a separate continuous-input control. Shading is one sample SD. Both axes use the declared counts/raw error ratios; no percentages are applied to the plotted errors. <a href="assets/electropolish2d_library_size_error.pdf">Editable PDF</a>.</figcaption>
</figure>

Online sampling has the lowest final mean in this run. Increasing the fixed
library from 256 to 512 gives a smaller improvement than the earlier increases.

This is one initialization and one nested input library. Larger fixed libraries
and online sampling do not guarantee a monotonic accuracy improvement at a finite
budget. The earlier 10kstudy used a different GPU/runtime and remains historical;
its endpoints are not mixed into this 15k figure. The shared prescribed-input
path evaluates encoder/IC/Dirichlet data in CPU FP64 then casts once to FP32;
network training remains FP32. Defaults remain continuous online profiles.

The opt-in `operator.condition_library` explicitly specifies `path`, `count` and
`sampling='uniform_with_replacement'`. All final fields, coefficient banks,
weights and configs remain at the direct paths in the provenance record.

[Final 15k statistics, direct inputs and reproduction](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/electropolish_library15k_results.json) · [Historical 10k study](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/fixed_condition_library_results.json).

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
