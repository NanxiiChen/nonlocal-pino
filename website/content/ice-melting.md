The operator predicts three-dimensional interface retreat for a family of initial ellipsoids. The phase field is near \(+1\) in ice and \(-1\) in water.

## Setup and boundary data

The cube \([-50,50]^3\) maps to \([0,1]^3\), and the physical horizon is 5 s. Normalized semiaxes are sampled independently from \([0.24,0.36]\), center coordinates from \([0.45,0.55]\), and the orientation is Haar-random. With the signed Euclidean distance \(d_E\) positive inside the ellipsoid, the initial profile is \(\tanh(d_E/(\sqrt{2}\epsilon^*))\). The native graph has no exterior exchange. The local FEM comparison uses homogeneous natural Neumann data.

<figure>
  <a href="assets/icemelting3d-setup.png"><img src="assets/icemelting3d-setup.png" width="1771" height="1470" alt="An ice-blue initial ellipsoid surrounded by pale water in the fixed cube"></a>
  <figcaption>Initial center, semiaxes and orientation vary; the cube and coefficients are fixed. Shading distinguishes ice and water, not temperature. <a href="assets/icemelting3d-setup.pdf">Editable PDF</a>.</figcaption>
</figure>

## Governing equations and scaling

Let \(F(\phi)=(1-\phi^2)^2/4\). Its derivative \(\phi^3-\phi\) gives the double-well term. The driving term uses \(\sqrt{2F}=(1-\phi^2)/\sqrt{2}\) on \([-1,1]\).

$$
\partial_t\phi=M\left[\Delta\phi-\frac{\phi^3-\phi}{\epsilon^2}\right]
-\lambda\,\frac{1-\phi^2}{\sqrt{2}\epsilon}.
$$

Set \(\mathbf x^*=(\mathbf x+50)/L_c\), \(t^*=t/T_c\), \(\epsilon^*=\epsilon/L_c\), and \(\lambda^*=\lambda T_c/L_c\). The coefficients are

$$
N_1=\frac{MT_c}{L_c^2},\qquad
N_2=\frac{N_1}{(\epsilon^*)^2},\qquad
N_3=\frac{\lambda^*}{\sqrt{2}\epsilon^*}.
$$

The selected setup uses \(L_c=100\), \(T_c=5\,\mathrm{s}\) and \(\epsilon^*=0.0225\).

## Nonlocal residual and loss

Use \(D_\delta\approx-\Delta^*\). The physical residual is \(r_\phi=\phi_{t^*}+N_1D_\delta\phi+N_2(\phi^3-\phi)+N_3(1-\phi^2)\). A hard latent lift supplies the initial condition for every sampled shape.

$$
u_\theta(\mathbf x,t^*)=\frac{d_E(\mathbf x)}{\sqrt{2}\epsilon^*}
    +t^*N_\theta(\mathbf x,t^*),\qquad
\phi_\theta=\tanh u_\theta .
$$

Since \(\partial_{t^*}\phi=(1-\phi^2)\partial_{t^*}u\), the transformed residual at a node has the exact factorization \(r_\phi=(1-\phi_i^2)r_u\) for finite \(u_i\).

$$
(r_u)_i=\partial_{t^*}u_i+
N_1\sum_j w_{ij}
\frac{\sinh(u_i-u_j)\cosh u_i}{\cosh u_j}
-N_2\tanh u_i+N_3 .
$$

The edge formula follows from the difference of two hyperbolic tangents divided by the output Jacobian at node \(i\). Log-domain arithmetic evaluates the cosh ratio without dividing by a saturated rounded phase value. The loss minimizes the transformed absolute residual on the core nodes. The hard lift supplies the exact initial phase without an IC penalty.

## Field evolution

<figure>
  <a href="assets/icemelting3d.png"><img src="assets/icemelting3d.png" width="2007" height="1960" alt="Three held-out ice ellipsoids across five times, each shown as predicted 3D surfaces and XY phase slices with reference interfaces"></a>
  <figcaption>The selected rolling-pool 10,000-step model on held-out inputs 5, 9 and 2 at t* = 0, 0.25, 0.5, 0.75 and 1 (0–5 s). Each sample pairs predicted φ = 0 surfaces from 128³ visualization queries with fixed XY slices; dashed slice contours are the native nonlocal reference. Each error label uses the full original 24³ volume at that time, not only the slice. <a href="assets/icemelting3d.pdf">Editable PDF</a>.</figcaption>
</figure>

## References and comparisons

The selected cold 10,000-step rolling model gives **0.1002% pooled / 0.1711% terminal phase L2** and **0.2908% / 8.4058% solid-indicator L2** on eight reference inputs. The native reference integrates the full cubic graph; neural error is measured on a 24³ diagnostic grid at five times.

| Pool, cold 10,000 updates | Pooled phase L2 | Terminal phase L2 | Warm effective median |
| --- | ---: | ---: | ---: |
| Fixed | 0.1203% | 0.1132% | 16.26 ms |
| Rolling, selected | 0.1002% | 0.1711% | 16.68 ms |

Both arms share runtime seed 43, initial model/Adam state, input shapes and time draws. Timings were measured serially on the same RTX 4090; active peaks are about 1.494 GiB. Late checkpoints fluctuate and their pooled medians favor fixed (0.1203% versus 0.1439%). The selected rolling recipe reflects the user's choice after this finite-budget comparison.

The historical 5,000-step optimizer comparison below uses the earlier fixed spatial pool.

The eight held-out shapes use time-refined direct nonlocal integration on the \(128^3\) graph. The field errors below pool five times on a \(24^3\) diagnostic grid. The optimizer control uses the same input draws and final budget.

| Mean relative error | Adam, historical 5k | SOAP, historical 5k |
| --- | ---: | ---: |
| Pooled phase \(L^2\) | 0.2266% | 0.0785% |
| Terminal phase \(L^2\) | 0.0922% | 0.1291% |
| Pooled solid-indicator \(L^2\) | 0.6425% | 0.2235% |
| Terminal solid-indicator \(L^2\) | 4.5017% | 6.5036% |
| Warm synchronized update | 22.43 ms | 33.66 ms |

In the separate one-shape local FEM check, the two references differ by 0.1474% pooled phase on the common nodes. Adam and SOAP give 0.1479% and 0.1861% against that FEM reference. The eight-shape table and the one-shape FEM check answer different questions. The approximate shrinking-sphere profile omits curvature effects and is unsuitable as the reference for arbitrary initial ellipsoids.

## Three-method comparison

These matched controls use the same width 48 backbone, 16 online initial
ellipsoids, eight shared random times, rolling 256 cores, hard latent initial
condition and Adam 10k recipe. Nonlocal uses the complete truncated graph rows.
FEM PINO uses sampled assembled P1 lumped weak rows from six Kuhn tetrahedra
per Cartesian cell, with natural homogeneous Neumann boundaries. Local AD uses
network coordinate derivatives and analytic derivatives of the prescribed
Euclidean initial distance, with 96 explicit physical no-flux face samples per
update. The original initial data are retained. Time derivatives remain JVPs.
FEM PINO is a trained neural operator, distinct from a numerical FEM reference.

The three cold 10k controls now pass identical initial model/Adam state,
all actual condition/time/core draws, frozen-source checks and the complete
10,000-step AD boundary replay. All 192 saved prediction/reference pairs are
independently checked. Final seed 42 phase errors are:

| Method | Native 64 phase mean | Native 64 sample SD | Separate local FEM case005 |
|---|---:|---:|---:|---:|
| Nonlocal | 0.2860% | 0.0572% | 0.1928% |
| Local AD | 0.0850% | 0.0172% | 0.1208% |
| FEM PINO | 0.1093% | 0.0225% | 0.0671% |

<figure>
  <a href="assets/icemelting3d_three_method_convergence.png"><img src="assets/icemelting3d_three_method_convergence.png" width="1003" height="708" alt="Melting final 10k three-method phase convergence on 64 native test ellipsoids"></a>
  <figcaption>All 41 actual evaluations at steps 0:250:10000 use the same 64 ellipsoids. Each sample contributes one equally weighted space-time phase norm on 24³ queries and five saved times (0, 0.25, 0.5, 0.75, 1). Thin solid lines are raw sample means; thicker lines are log-LOWESS trends (span 0.22). Shading shows pointwise 95% empirical bootstrap intervals from 2,000 complete-trajectory resamples, paired across methods (seed 20261114). The display grid creates no additional observations. These intervals condition on this training realization and reused cohort; they do not measure cross-seed uncertainty or account for smoothing/model-selection bias. <a href="assets/icemelting3d_three_method_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/icemelting3d_three_method_distribution.png"><img src="assets/icemelting3d_three_method_distribution.png" width="1003" height="708" alt="Melting final 10k three-method phase distribution on 64 native test ellipsoids"></a>
  <figcaption>Final 10k weights, 64 phase errors against the native nonlocal numerical reference. The norm pools all five times before normalization; it is not a mean of instantaneous errors or an error on every 128³ reference node. The log-error Gaussian KDE uses Scott bandwidth, three-bandwidth tails and internal quartiles; tails are density estimates rather than extra samples. The vertical axis shows raw relative ratios, without percentage scaling. <a href="assets/icemelting3d_three_method_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

Local AD has the lowest mean error on the native 64 cohort in this run.
FEM PINO has the lowest pooled error on the separate local CG2 case005. That
single diagnostic cannot establish the ordering on a local-reference population.
The previously accepted CG2 reference retains its unpassed strict-refinement
flag; it is a practical accuracy check, not a new continuum-convergence claim.
Native and local references differ by 0.1474% on case005. Do not combine this
single local sample with the 64 native violin or interpret the latter as a
64-case local-FEM comparison. One initialization and reused test inputs are
reported; the selected SOTA/default and main evolution figure remain unchanged.

| Method | Cold training process | Warm inclusive median | Active peak |
|---|---:|---:|---:|
| Nonlocal | 6.89 min | 15.81 ms | 1.45 GiB |
| Local AD | 10.37 min | 25.17 ms | 1.70 GiB |
| FEM PINO | 7.35 min | 16.16 ms | 1.31 GiB |

All three runs used the same RTX PRO 6000 Blackwell Server Edition
(UUID `GPU-85fe6412-636e-5d3c-d5f0-d1e9523c3faa`). Cold time includes setup,
compilation, preparation, all in-training evaluations and final native prediction
saving. It excludes reference generation and the separate final audit. The warm
inclusive median sums synchronized update, condition preparation and spatial
preparation/transfer after the first step; it excludes evaluation and is not
an isolated derivative-kernel timing. Active peaks are allocator arrays, not
the much larger reserved memory pool. AD retains common core/halo preparation
for exact input matching, though its PDE evaluates only 256 cores and 96 face
points. Nonlocal is faster than AD here; sampled FEM PINO has similar warm
cost to Nonlocal and does not need full-mesh network evaluation.

The 64 native reference bank preserves the original 10 files and adds 56 independently
drawn ellipsoids. Those full 128³ FP64 RK4 trajectories pass time-step halving
(0.005 versus 0.0025): the maximum saved-snapshot relative difference is 1.505e-7.
This qualifies native temporal integration, not continuum spatial accuracy.
[Exact final weights, configuration, data and figure provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_three_method_figures.json)
and [resource record](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/melting_three_method_resources.json)
keep the complete run paths and reference scopes.

## Time-feature fusion comparison

This completed comparison keeps nonlocal spatial derivatives, time JVP,
width 48, 16 fresh ellipsoids per update, eight shared random times, rolling
256 core queries, the hard latent initial condition and Adam 10k. Decoder FiLM
uses spatial-only cross-attention followed by time modulation at every hidden
decoder layer. Joint queries project concatenated spatial/time Fourier features
into attention, followed by unmodulated decoder layers. Both encode each
initial volume once; the joint model repeats attention across times.

The common initial encoder/attention/spatial-projection/decoder arrays and all
actual condition/time/core draws pass exact checks. Time modules differ:
85,825 versus 71,665 stored numeric entries, including frozen frequencies,
with 70,849 shared entries. Full initial/optimizer trees are therefore not
identical and this is not an equal-capacity study. Both use final 10k weights
and all 64 test ellipsoids every 250 updates, including zero. Reference
trajectories enter evaluation only.

| Temporal conditioning | Native 64 phase mean | Separate local FEM case005 |
|---|---:|---:|
| Decoder FiLM | 0.2860% | 0.1928% |
| Joint (x,t) query | 0.0935% | 0.1302% |

<figure>
  <a href="assets/icemelting3d_time_fusion_convergence.png"><img src="assets/icemelting3d_time_fusion_convergence.png" width="1003" height="708" alt="Melting phase error for decoder FiLM versus joint space-time query"></a>
  <figcaption>All 41 actual evaluations on the same 64 native-reference ellipsoids. Each sample's relative norm pools all 24³ diagnostic volume queries and five saved times (0, 0.25, 0.5, 0.75, 1). Thin solid lines are raw means; thicker lines are log-LOWESS trends (span 0.22). Shading shows pointwise 95% empirical bootstrap intervals from 2,000 complete-case-trajectory resamples, paired across methods (seed 20261114). The fitted display grid adds no checkpoints. These intervals condition on a reused cohort and one training realization, without cross-seed coverage or correction for smoothing/model-selection bias. The horizontal axis is elapsed wall time in seconds from cold process launch, including initialization, compilation and in-training evaluation. Intermediate times are reconstructed from the original launch-log creation, metric-file timestamps and recorded trainer counters; they are not direct parent timestamps and subsecond accuracy is not claimed. The existing final observation is placed at complete process exit, including final writes and teardown. No error is invented at time zero, and no step-rate extrapolation is used. LOWESS and its bootstrap interval are fitted on the elapsed-time coordinates. <a href="assets/icemelting3d_time_fusion_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/icemelting3d_time_fusion_distribution.png"><img src="assets/icemelting3d_time_fusion_distribution.png" width="1003" height="708" alt="Melting phase error for decoder FiLM versus joint space-time query"></a>
  <figcaption>Final 10k unsmoothed phase errors for all 64 native-reference inputs. The log-error Gaussian KDE uses Scott bandwidth, three-bandwidth tails and internal quartiles. Density tails do not represent extra samples. The axis shows raw relative ratios. These are errors on the saved diagnostic queries, not on every 128³ reference node and not against a 64-case local-FEM cohort. <a href="assets/icemelting3d_time_fusion_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

Joint queries improve the final native-cohort mean in this run, and also
reduce error on the separate local CG2 case005. Late-step values fluctuate, and
one initialization does not establish a stable cross-seed architecture ranking.
The local reference retains the earlier unpassed strict-refinement flag and
practical acceptance; one local case is not a reference population. Keep this
diagnostic separate from the native violin. The selected SOTA/default and main
field figure remain unchanged.

Together with Electropolish, this comparison shows an accuracy/cost tradeoff:
spatial-only attention plus decoder FiLM reduces measured training cost in both
cases, while its final accuracy advantage is case dependent. Do not describe
FiLM as universally more accurate or joint queries as uniformly ineffective.

| Temporal conditioning | Cold training process | Warm inclusive median | Active peak | Stored entries |
|---|---:|---:|---:|---:|---:|
| Decoder FiLM | 6.89 min | 15.81 ms | 1.45 GiB | 85,825 |
| Joint (x,t) query | 14.99 min | 60.44 ms | 9.36 GiB | 71,665 |

Both arms ran on the same RTX PRO 6000 Blackwell Server Edition
(UUID `GPU-85fe6412-636e-5d3c-d5f0-d1e9523c3faa`). Cold cost includes setup,
compilation, all in-training evaluations and final native-prediction saving,
excluding reference generation and the separate audit. Warm inclusive medians
sum synchronized loss/gradient/update, condition preparation and spatial transfer
after the first step; they exclude evaluation. Active peaks are allocator arrays,
not the reserved pool. The joint model uses fewer stored parameters yet processes
more attention queries; parameter count alone does not determine memory or cost.

[Direct checkpoint/configuration/figure provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_time_fusion_figures.json)
and [reproduction guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/docs/PINO_COMPARISON_FIGURES.md#nonlocal-time-feature-fusion-controls).

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Condition | \(32^3\) volume, \(4^3\) patches, 512 tokens |
| Model | Width 48, four heads, two encoder blocks, three time-FiLM layers |
| Spatial work | \(128^3\) graph, horizon 0.01, 256 cores plus full halo, 32 initial sets and a background rolling FIFO |
| Background pool | One CPU worker, 32 prefetched sets, 1,792 node slots / 1,536 edges |
| Samples per update | 16 fresh ellipsoids and 8 shared times |
| Optimizer | Adam, \(10^{-3}\) initial rate, 10,000 updates, ordinary FP32 |

~~~bash
source .venv/bin/activate
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m examples.icemelting3d.train \
  --config=examples/icemelting3d/configs/train_operator.py \
  --config.operator.action=prepare
python -m examples.icemelting3d.train \
  --config=examples/icemelting3d/configs/train_operator.py
~~~

Preparation creates a new reference bank at the configured path. See the [example source and full run notes](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/icemelting3d/README.md#pino-overview).
