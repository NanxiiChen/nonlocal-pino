A geometry-conditioned operator predicts the growth of a corrosion pit around six smooth inert ceramic inclusions. Particle shapes and poses vary while the physical coefficients, reservoir and initial pit stay fixed.

## Current SOTA default

The default `train.py` entrypoint selects the six-particle, width-256 nonlocal model with rolling geometry and core/halo inputs. A cold 20,000-step run through this entrypoint, without an explicit configuration override, gives the following final-weight errors against 64 held-out local-FEM solutions:

| Phase mean | Concentration mean | Channel mean ± sample SD |
| ---: | ---: | ---: |
| 3.6965% | 4.4839% | **4.0902% ± 0.5483 pp** |

Each channel's norm spans all space-time evaluation nodes. Channels are averaged within each sample, then samples are averaged equally. The standard deviation describes variation across these 64 samples, not repeated training seeds. Training took **42.72 min** on RTX PRO 6000 Blackwell; warm update median/mean were **102.36/114.84 ms**, with **20.32 GiB** peak active allocation. Process time includes compilation, input preparation and monitoring; the separate final test evaluation is excluded.

### Execution and reproducibility

The physical equations, forward nonlocal pair differences, five losses, Adam schedule and actual sampling stream retain the successful scheme. The default now uses exact fixed-linear field adjoints for parameter backpropagation: for \(y=Dv\), the field gradient is \(D^T\bar y\); repeated core extraction uses its corresponding selection-matrix transpose. Matrices are prepared only for the selected batch. This avoids the original atomic scatter accumulation in those reverse operations. Kernel autotuning is explicitly disabled; ordinary FP32 compilation and time JVP remain.

Repeated GPU updates exposed nondeterministic accumulation in the original path. Under controlled deterministic execution, current and historical sources produced identical 30-step model and optimizer states; no changed PDE or optimizer formula was found. A separate stale default import selected an older high-mobility 5 s case and has been corrected. The new default reproduces low error with identical cold states and complete sampling inputs. This is one paired initialization on the recorded environment, not a cross-seed/device guarantee, and the separate effects of autotuning and accumulation on final convergence were not quantified.

[Configuration, checkpoint, validation and execution provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/ceramic_execution_validation.json). The historical field image and three-method comparison below retain their original checkpoints, metrics and timings.

## Setup and boundary data

The reactive region is \([0,1]\times[0,0.6]\) minus the ceramic interiors, or \(50\times30\,\mu\mathrm m\) in physical dimensions. Depth increases downward. The initial semicircular pit has radius \(6\,\mu\mathrm m\), and the reservoir mouth is at \(y=0\), \(|x-0.5|\leq0.055\) in normalized coordinates. It prescribes \(\phi=c=0\). The remaining walls are neutral and impermeable.

Each particle uses a smooth radial perturbation with modes 3–5, area normalization, aspect stretch and rotation. Equivalent radii range from 2.25 to \(3.75\,\mu\mathrm m\). Clearance, curvature and near-pit requirements screen layouts using geometry alone. The reservoir can exchange concentration, so global concentration is free to change.

<figure>
  <a href="assets/particle_corrosion2d-setup.png"><img src="assets/particle_corrosion2d-setup.png" width="1181" height="885" alt="Gray metal, blue electrolyte pit and six smooth ivory ceramic particles with dark outlines"></a>
  <figcaption>Ivory ceramic regions are excluded from the reactive domain. Particle positions, shapes and orientations vary; the initial pit and narrow reservoir mouth are fixed. <a href="assets/particle_corrosion2d-setup.pdf">Editable PDF</a>.</figcaption>
</figure>

## Governing equations and scaling

The KKS mixture uses \(h(\phi)=3\phi^2-2\phi^3\), \(g(\phi)=\phi^2(1-\phi)^2\), \(\Delta c=c_{Se}-c_{Le}\), and \(q=c-\Delta c\,h(\phi)-c_{Le}\). The dimensional phase and concentration laws are

$$
\begin{aligned}
\partial_t\phi&=L\,[\alpha_\phi\Delta_x\phi-\omega_\phi g'(\phi)
                    +2A\Delta c\,q h'(\phi)],\\
\partial_t c&=D\Delta_x q .
\end{aligned}
$$

The displayed case sets \(L_c=50\,\mu\mathrm m\), \(T_c=350\,\mathrm s\), and \(L=2\times10^{-9}\,\mathrm{m^3/(J\,s)}\). After substituting \(\mathbf x^*=\mathbf x/L_c\) and \(t^*=t/T_c\), the coefficient groups are

$$
\begin{aligned}
N_1&=2ALT_c=74.9, & N_2&=L\omega_\phi T_c=12.32,\\
N_3&=L\alpha_\phi T_c/L_c^2=0.00721, &
N_{\mathrm{CH}}&=DT_c/L_c^2=119 .
\end{aligned}
$$

The initial phase uses a tanh of signed radial distance from the pit and \(c_0=c_{Se}h(\phi_0)\) away from the mouth. At the mouth the prescribed values take precedence, including at the initial time.

## Nonlocal residual and loss

The raw network channels \(z,s\) supply the physical fields through the algebraic KKS map.

$$
\phi=\tfrac12(1+\tanh z),\qquad
c=c_{Le}s+\Delta c\,h(\phi),\qquad
q=c_{Le}(s-1).
$$

The concentration channel \(s\) is unrestricted. Positive \(D_\delta\) uses visible, symmetric edges with physical volumes. The regional graph excludes segments crossing ceramic interiors. Evaluating \(D_\delta q=c_{Le}D_\delta s\) exploits constant cancellation and reduces FP32 loss of significance.

$$
\begin{aligned}
r_{\mathrm{AC}} &= \phi_{t^*}+N_3D_\delta\phi+N_2g'(\phi)
                     -N_1\Delta c\,q h'(\phi),\\
r_{\mathrm{CH}} &= c_{t^*}+N_{\mathrm{CH}}c_{Le}D_\delta s .
\end{aligned}
$$

The objective uses absolute AC and CH residuals with unit prescales. Mouth phase, mouth concentration and initial-field errors give three soft terms. Scale the concentration parts of prescribed-data errors by \(c_{Le}\) before squaring. Inverse-L1 gradient-norm weights refresh every 25 updates. AC and CH train jointly, with the original physical residual despite the bounded phase output.

## Field evolution

<figure>
  <a href="assets/particle_corrosion2d.png"><img src="assets/particle_corrosion2d.png" width="2007" height="824" alt="Best, middle and worst phase-error cases from 64 ceramic geometries, with warm-ivory particles and local-FEM contours"></a>
  <figcaption>The original width-256, 20,000-step field model: Best / Middle / Worst are inputs 50 / 53 / 30 from all 64 test geometries. Ranking uses equal-point, phase-only relative L² over all 11 saved times and physical mesh nodes; Middle is the lower central actual sample. Columns show t* = 0, 0.2, 0.5, 0.8 and 1 (0, 70, 175, 280 and 350 s). Colors show predicted phase; warm-ivory particles have dark outlines, and dashed curves are local-FEM φ = 0.5 interfaces. Panel labels give instantaneous spatial phase relative L², excluding ceramic interiors. Current-default metrics are reported separately. <a href="assets/particle_corrosion2d.pdf">Editable PDF</a> · <a href="https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_ranked_main_preview.json">Selection and reproduction</a>.</figcaption>
</figure>

## References and comparisons

These historical field-model comparisons use the same four development geometries, excluded from the rolling training stream. Pooled errors use spatial quadrature and trapezoidal time weights over 11 saved snapshots. The independent local P1 FEM solver uses weak-form stiffness, lumped mass and sparse BDF integration. A first-geometry mesh refinement from spacing 0.01 to 0.0075 changes the local reference phase by 0.1007% pooled. The current default gives 2.9046% pooled / 5.4357% terminal phase error on this same development metric; its 64-case test above uses a different norm and cohort.

| Reference | Phase pooled / terminal | Concentration pooled / terminal |
| --- | ---: | ---: |
| Independent local P1 FEM | 2.933% / 5.757% | 3.566% / 6.871% |
| Direct production nonlocal solver | 1.450% / 2.781% | 1.763% / 3.278% |

The two references differ by 1.72–1.80% pooled phase over these four geometries. Terminal corroded-region IoU is 94.99% against FEM and 98.10% against direct nonlocal integration.

| Same-stream cold 20k comparison | Width 192 | Width 256 |
| --- | ---: | ---: |
| Local phase pooled / terminal | 3.091% / 5.985% | 2.933% / 5.757% |
| Warm held-update median | 49.06 ms | 62.94 ms |
| Complete training process | 23.35 min | 28.47 min |
| Peak active allocation | 15.66 GiB | 20.69 GiB |

Both width and Fourier feature capacity change, and the parameter shapes initialize separately. The table uses the final 20k model on the already-inspected development cohort.

## Spatial derivative comparison

This three-method comparison uses final 20,000-step neural weights and 64 common held-out geometries. **FEM PINO** replaces spatial differentiation with the assembled mass-normalized P1 weak operator \(M_L^{-1}K\); natural homogeneous Neumann data enter through the weak form. **FEM numerical reference** is the independent refined P1/BDF solver used for evaluation. It is not a trained model, and its trajectories do not enter training.

For each sample and physical channel, relative \(L^2\) is computed over all space-time evaluation nodes. The component violins keep phase and concentration separate; the table also reports their per-sample channel mean. These equally weighted node norms differ from the four-case volume/time-weighted development curves below. The original three comparison checkpoints are retained.

| Method | Phase mean | Concentration mean | Channel mean ± sample SD | Cold process time |
| --- | ---: | ---: | ---: | ---: |
| Nonlocal PINO | 3.8175% | 4.6205% | 4.2190% ± 0.5719 pp | 26.81 min |
| Local AD PINO | 32.2070% | 32.6060% | 32.4065% ± 0.6047 pp | 28.47 min |
| FEM PINO | 5.5535% | 6.5937% | 6.0736% ± 0.8166 pp | 21.01 min |

<figure>
  <a href="assets/particle_corrosion2d_component_convergence.png"><img src="assets/particle_corrosion2d_component_convergence.png" width="2007" height="708" alt="Phase and concentration development convergence for Nonlocal, Local AD and FEM PINO"></a>
  <figcaption>All 21 actual records, every 1,000 updates, averaged over four development geometries with volume/time-weighted norms. Thin solid lines retain raw means; thick lines show log-error LOWESS (span 0.30, six neighbors). Shading is an approximate pointwise 95% bootstrap interval for the smoothed geometry mean, enumerating all 256 resamples of four complete trajectories. This preserves temporal dependence; the small-cohort intervals are conditional on one training trajectory and do not measure cross-seed uncertainty, guarantee population coverage or correct smoothing/model-selection bias. <a href="assets/particle_corrosion2d_component_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/particle_corrosion2d_component_distribution.png"><img src="assets/particle_corrosion2d_component_distribution.png" width="2007" height="708" alt="Separate phase and concentration error violins on64common FEM test geometries for three PINO methods"></a>
  <figcaption>Final 20k weights, all 64 test errors per physical channel, using unweighted full-space-time norms against the FEM numerical reference. Logarithmic axes show unscaled ratios; violins use log10-error Gaussian KDE, Scott bandwidth, three-bandwidth tails and dashed quartiles. The fitted tails are not additional samples. <a href="assets/particle_corrosion2d_component_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

The shared recipe uses width 256, Adam, rolling geometry/core-halo pools, and random times. These are seed-42 measurements on the same RTX PRO 6000 Blackwell Server Edition. Cold process time uses the parent timer from before process creation through exit, including imports, initialization, compilation, all updates, in-training evaluation, output writes and teardown. It excludes the separately launched final 64-sample evaluation. This replaces the earlier trainer-internal timings. [Exact timing provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_training_wall_costs.json). The displayed nonlocal model uses the historical source and launch settings. The later execution diagnosis and validated default are documented above; this figure remains the original three-method comparison. The unsuccessful execution is retained in the [experiment record](https://github.com/NanxiiChen/nonlocal-pino/blob/main/docs/tasks/fem-weak-pino-comparison-plan.md). This is a finite single-initialization comparison, not a general method ranking. [Exact model, configuration, metric and figure provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/ceramic_fem_comparison_publication.json).

## Time FiLM versus joint query

This comparison changes only how time enters the network. The default performs spatial cross-attention against both domain and wall tokens, then applies time FiLM in each of the three decoder MLP layers. The alternative adds projected time Fourier features to the spatial query before both cross-attention blocks and uses plain MLPs without FiLM. Spatial operators, soft initial/boundary losses, Adam, width 256, batch 4, 32 random times and rolling geometry/core-halo streams are shared.

Both cold runs complete 20,000 steps on the **same physical RTX PRO 6000 Blackwell Server Edition**, with identical shared initial parameters and actual sampled inputs. Every 500 updates, both evaluate all 64 common local-FEM geometries. Each physical channel uses an equal-point norm over all saved space-time points, followed by the arithmetic channel/sample mean; these curves do not use the historical quadrature-weighted development metric.

| Time injection | Phase mean | Concentration mean | Channel mean | Complete cold wall time |
| --- | ---: | ---: | ---: | ---: |
| Spatial attention + decoder FiLM | 3.6965% | 4.4839% | **4.0902%** | **2629.49 s** |
| Joint spatial/time query, no FiLM | 4.0578% | 4.9087% | 4.4832% | 12475.49 s |

FiLM finishes the fixed budget **4.74 times as fast**, with slightly lower error in this realization. Time is measured directly from before process creation through exit, including imports, preparation, compilation, updates, rolling waits, all in-training 64-case evaluations and final writes. Offline reference construction, qualification probes and later audits/plots are separate. This FiLM run reproduces all 64 historical default prediction arrays exactly; it adds dense 64-case histories and direct cold clocks. The historical default timing above uses its original monitoring schedule.

<figure>
  <a href="assets/particle_corrosion2d_time_fusion_convergence.png"><img src="assets/particle_corrosion2d_time_fusion_convergence.png" width="2007" height="708" alt="Phase, concentration and channel-mean convergence for FiLM and joint query versus complete cold wall-clock seconds"></a>
  <figcaption>All 41 actual observations, with equal-point space-time norms on the same 64 cases. Thin solid lines show raw means; thick solid lines show log-LOWESS (span 0.22). Shading is a pointwise 95% interval from 2,000 whole-case trajectory bootstrap draws; it is conditional on this monitored cohort and one initialization, not cross-seed uncertainty. The final point includes process exit and final writes. <a href="assets/particle_corrosion2d_time_fusion_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/particle_corrosion2d_time_fusion_distribution.png"><img src="assets/particle_corrosion2d_time_fusion_distribution.png" width="2007" height="708" alt="Final64case error distributions for phase, concentration and their per-sample channel mean, FiLM versus joint query"></a>
  <figcaption>Final 20k weights and all 64 actual samples. Channel mean averages the two separate relative norms within each sample. Log-error KDE uses Scott bandwidth, three-bandwidth tails, equal maximum widths and quartiles. Fitted tails do not add observations. <a href="assets/particle_corrosion2d_time_fusion_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

The temporal variants contain 2,248,450 versus 1,853,442 stored model entries. This single-initialization study does not establish an equal-capacity or cross-seed accuracy advantage. The 64 cases are excluded from training inputs but monitored during training. The default remains spatial attention followed by layerwise time FiLM. [Configuration, sources and full audit](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/ceramic_time_fusion_validation.json) · [Selected figures and reproduction](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_time_fusion_figures.json).

## Fixed geometry library and training stability

Four cold 20k runs keep the original physics, architecture and optimizer, with
nested fixed geometry banks and continuing spatial core/halo refresh. All use
the same initialization and 64 local-FEM evaluation geometries. Final equal-point
space-time channel means are:

| Fixed geometries | Phase | Concentration | Channel mean ± sample SD |
|---:|---:|---:|---:|
| 64 | 4.71% | 5.68% | 5.20% ± 0.74% |
| 128 | 31.97% | 33.61% | 32.79% ± 0.80% |
| 256 | 31.66% | 33.31% | 32.48% ± 0.79% |
| 512 | 31.97% | 33.61% | 32.79% ± 0.79% |

<figure>
  <a href="assets/electropolish_ceramic_fixed_library_error.png"><img src="assets/electropolish_ceramic_fixed_library_error.png" width="2007" height="732" alt="Fixed-library results retaining all poorly converged larger Ceramic runs"></a>
  <figcaption>Ceramic is panel (b); panel (a) is the separate Electropolish 10k study. Final weights, 64 test samples and one across-sample standard deviation. A passed input/metric audit does not indicate good solution accuracy. <a href="assets/electropolish_ceramic_fixed_library_error.pdf">Editable PDF</a>.</figcaption>
</figure>

The three larger banks plateaued near 32% and did not recover. Their cause is
unresolved; these results do not establish that more geometries generally harm
generalization. Training-stability work is deferred, and the successful rolling
geometry default is retained. [Research TODO](https://github.com/NanxiiChen/nonlocal-pino/blob/main/docs/TODO.md#ceramic-corrosion-training-stability-across-geometry-libraries)
· [Data, checks and direct run paths](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/fixed_condition_library_results.json).

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Network | Two point streams, width 256, four heads, two spatial cross-attention blocks, three time-FiLM layers |
| Spatial work | Target mesh spacing 0.015, horizon 0.026, 256 sampled cores, complete halo |
| Samples per update | 4 geometry draws and 32 freshly sampled times |
| Online pool | 64 geometry slots, four CPU workers, geometry and spatial payloads refreshed on separate schedules |
| Optimizer | Adam, 20,000 cold updates, initial \(10^{-3}\), halve every 3,000 updates, floor \(10^{-5}\), ordinary FP32 |
| Parameter gradients | Fixed-linear field adjoints; `nl.adjoint=dense_fixed`; `train.autotune_level=0` |

Prepare the initial known-input bank and independent local reference from the explicit six-particle recipe. The [repository run guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/particle_corrosion2d/README.md#run-the-documented-case) gives the preview, input preparation and local/native qualification commands. Then train from the repository root.

~~~bash
source .venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python -m examples.particle_corrosion2d.train \
  --inputs /path/to/input-bank --reference /path/to/local-reference \
  --output /path/to/new-run
~~~

Omitting `--config` selects the current SOTA through `configs/train_default.py`. The explicit preset is `configs/train_physical_spaced_w256_fixed_adjoint.py`; historical presets remain available for reproducing their saved runs. Omit `--seed` for a fresh initialization, or supply it deliberately for replay. New runs save execution settings and package/device information in `execution.json`.
