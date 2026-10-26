# 2D KKS Phase-Field Corrosion — PINN + FEM Reference

## SOTA default

The canonical default uses **ordinary GPU compilation and FP64**, with the
**original phase residual**, stable raw-KKS CH edges, normalized concentration
BCs and all graph IC/BC nodes. Full5000-update/50s controls at seeds42/2026 give
terminal phi/c **0.002499/0.002944** and **0.006833/0.008077**. Worst-seed late
4000–5000 phi median is0.005004, versus0.009514 for the previous deterministic
FP32 default. This is the best supported speed/field-accuracy compromise among
the explored configurations; it does not establish a global optimum.

Late maxima do not improve for every seed; Dq transport errors remain high
(terminal free-node relative L2 about1.36–1.38), including large late spikes.
The historical individual bounded winner remains separate. Neither seed nor
an intermediate checkpoint is selected to promise low error. See the
[full default validation](../../docs/tasks/archive/ordinary-policy-defaults.md)
and [all convergence curves](../../docs/tasks/archive/ordinary-policy-defaults-corrosion2d-convergence.png).

Seed is not a strategy parameter. The canonical `get_config()` has no
`train.seed`; choose a run with `train.py --seed=2026`. With no override, the
common untuned run seed is42. Saved resolved config metadata still includes
`train.seed` for reproducibility and old checkpoint compatibility. The best
observed run below is provenance, not a seed to optimize or an accuracy promise.

The matched default requires the sparse graph/minibatches and does not accept
legacy causal/interface weighting, bias schedules or optional diagnostic
switches; unsupported overrides fail explicitly. Use a legacy config for those
controls. Changing physical/operator/grid settings requires a matching reference.

The canonical training entry is [train.py](train.py), loading
[configs/train_base.py](configs/train_base.py). It delegates to the validated
matched-graph trainer; changing the entry does not replace its loss/optimizer
loop. Run a fresh cold start from the repository root:

```bash
source .venv/bin/activate
python -m examples.corrosion2d.train
# Equivalent direct-script entry:
python examples/corrosion2d/train.py
# Explicit fresh run name:
python -m examples.corrosion2d.train --config.log.run_name=my-corrosion-run
```

Historical single-run SOTA denotes the **best observed final5000 terminal
phi+c relative L2** among40 run summaries at the earlier promotion. It does not denote
best pooled-time error, concentration transport, or cross-seed robustness.
That historical bounded checkpoint uses

$$
r_\eta=\frac{1+\eta}{4\phi(1-\phi)+\eta}r_\phi,\qquad \eta=0.1.
$$

Its phase weight is1 at the interface and at most11; it is not the full
inverse-Jacobian latent residual. The CH residual retains physical \(c_t\)
and uses stable raw-KKS edge differences for \(Dq\). KKS and AC/CH pre-scales
are retained. Initial and boundary conditions remain soft losses. Nodal ICs
use the prescribed analytic initial field and Dirichlet overrides at time0;
positive-time reference fields are evaluation targets only.

| Setting | Current default |
|---|---|
| Physical horizon / updates | 50 s /5000 |
| Network | FiLM,4 layers x64,gelu; FiLM conditioning2 layers x32 |
| Spatial / temporal embeddings | 64 /64 dimensions, scales2 /1 |
| Operator / grid | Sparse cubic graph,delta0.02,101x51 |
| Core / pool / times | 256 /100 /16; uniform t* in[0.001,1) |
| IC / BC sampling | All5151 graph IC nodes;9 pit +201 solid BC nodes |
| Data objective | `s_bc`: phi and c_L/c_Le at BC; physical phi,c at IC |
| Phase / CH objective | `phase` / `kks_raw_edges`,L1 |
| AC / CH scale | 1e-7 /1 |
| Optimizer / initial rate | SOAP /1e-3 |
| SOAP beta1 / beta2 / refresh | 0.95 /0.95 /5 updates |
| Rate schedule / gradient clip | Continuous0.95 per1000 updates,floor1e-5 /100 |
| GradNorm alpha / refresh / cap | 1 /25 updates /10000 |
| Stagger / causal / interface weight / time window | Off /off /0 /off |
| Late rate / CH-RMS continuation | Off /off |
| Floating arithmetic | FP64 (`train.compute_dtype="float64"`) |
| Matmul / GPU runtime | `highest`; ordinary (`train.gpu_policy="ordinary"`) |

The [local FP64 performance benchmark](../../docs/tasks/archive/fp64-performance.md)
measures the same-size SOAP/GradNorm workload. The canonical matched trainer now
accepts `--config.train.compute_dtype=float32` or `float64` (default `float64`). Both modes
generate initialization, targets, coordinates, stored graph coefficients and time
samples in FP32, then cast floating arrays to the selected arithmetic precision.
Model, PDE, gradients and SOAP states follow that precision; the existing Optax
scalar learning-rate schedule remains FP32. The global x64 flag alone is not a
substitute for this option. Each new run records dtype checks in `precision.json`.

The FP64 control preserves the stored FP32 graph diagonal, whose constant-field
defect is measured separately; it does not silently rebuild a different operator.
The [completed accuracy control](../../docs/tasks/archive/corrosion-fp64-accuracy.md)
uses one paired seed42,5000 updates and unchanged stored graph coefficients:

| Terminal50s metric | FP32 | FP64 |
|---|---:|---:|
| phi relative L2 | 0.00883538 | 0.00267672 |
| c relative L2 | 0.01042560 | 0.00320701 |
| q relative L2, free nodes | 0.01372465 | 0.00686141 |
| Dq relative L2, free nodes | 1.70256 | 1.22047 |

This final checkpoint improves, but late stability does not improve uniformly:
4000–5000 phi median0.007191->0.006069, maximum0.008835->0.014882;
Dq median1.706->1.852. FP32 already reaches phi0.002689 at4800. Frozen-model
cross-precision inference changes fields by less than1.7e-6, so the difference
comes from training trajectories. That earlier pair established neither a lower accuracy
floor nor cross-seed reliability; Dq remains inaccurate. Its FP32 default was
superseded by the full ordinary42/2026 validation above.
The new FP64 model is
`/root/autodl-tmp/tf-logs/corrosion2d/20261026-precision-float64-phase-seed42/ckpt_005000.eqx`.

The [ordinary-policy FP64 full control](../../docs/tasks/archive/ordinary-fp64-accuracy.md)
completes the precision/policy comparison with one additional seed42 cold start.
All four use the same graph/reference, exact common initial values, SOAP1e-3,
data/scales and5000 updates/50s. Independent CPU reloads give:

| Policy / precision | Terminal phi L2 | Terminal c L2 | Late phi median | Observed total minutes |
|---|---:|---:|---:|---:|
| Deterministic FP32 | 0.008835 | 0.010426 | 0.007191 | 19.66 |
| Ordinary FP32 | 0.020680 | 0.024565 | 0.012737 | 1.30 |
| Deterministic FP64 | 0.002677 | 0.003207 | 0.006069 | 24.91 |
| Ordinary FP64 | 0.002499 | 0.002944 | 0.004685 | 3.28 |

Ordinary FP64 is about6x faster end-to-end than the previous default in these observed runs,
with72% lower final phi error and35% lower late phi median. Compilation-cache
histories differ; these times include actual JIT/evaluation/save costs and are
not a fresh-cache compilation benchmark. Prior steady-update measurements give
~35ms ordinary FP64 versus~229ms deterministic FP32 independently.

This is not uniform stability improvement: ordinary FP64 late phi mean0.006661
is slightly worse than default0.006348, and maximum0.014885 exceeds0.008835.
Its final free q/Dq errors0.007739/1.357 improve on default0.013725/1.703, but
late Dq reaches18.665 at4600, versus default maximum1.915. Compared with
deterministic FP64, final phi/c improve slightly while final q/Dq worsen.
One paired seed establishes this outcome, not cross-seed robustness or a
guaranteed accuracy benefit. The subsequent independent2026 control supports
the current ordinary FP64 promotion, subject to the limitations above.

Replay this explicit research option through the same entry:

```bash
XLA_FLAGS='' python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_graph_ordinary.py --config.train.compute_dtype=float64 --config.log.run_name=my-ordinary-fp64
```

Checkpoint:
`/root/autodl-tmp/tf-logs/corrosion2d/20261026-full-graph_ordinary-float64-seed42/ckpt_005000.eqx`.
[Full field L2 curves](../../docs/tasks/archive/ordinary-fp64-accuracy-convergence.png),
[transport curves](../../docs/tasks/archive/ordinary-fp64-accuracy-transport.png),
[terminal fields](../../docs/tasks/archive/ordinary-fp64-accuracy-terminal-fields.png)
and [CPU audit](../../docs/tasks/archive/ordinary-fp64-accuracy-audit.json) retain
every evaluation, including spikes.

The [local-AD throughput study](../../docs/tasks/archive/local-ad-performance.md)
measures complete synchronized updates with the same model, IC/BC, SOAP/GradNorm
and3760–4032 active PDE pairs.50 warm-up and75 measured updates per method:

| Method | FP32 ms/update | FP64 ms/update |
|---|---:|---:|
| Current nonlocal, deterministic GPU policy | 229.31 | 266.11 |
| Local spatial AD, shared-time tensor sampling and time JVP | 11.98 | 31.32 |
| Local spatial AD, joint random space/time and time grad | 16.12 | 47.77 |
| Nonlocal, ordinary GPU policy diagnostic | 12.49 | 35.13 |

The large historical default-policy gap is chiefly associated with deterministic graph
aggregation: isolated actions cost~38ms versus~0.17ms under ordinary policy.
The ordinary full-step control is measured, not inferred from that kernel ratio.
Joint random sampling still permits time JVP; time-only grad/JVP controls agree.
That study measures throughput only and omits the explicit local no-flux loss.
The full-training controls below complete the local boundary-value problem;
their six-loss update costs are distinct from the earlier five-loss timings.
The canonical adapter now defaults to ordinary policy. These historical local
research presets explicitly retain FP32 deterministic behavior; the graph
ordinary research preset explicitly retains FP32.

### Full local-AD research controls

All variants use `train.py`, FP32,5000 updates and50 physical seconds:

```bash
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_local_tensor.py --config.log.run_name=my-local-tensor
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_local_joint.py --config.log.run_name=my-local-joint
# Requires XLA_FLAGS without --xla_gpu_deterministic_ops:
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_graph_ordinary.py --config.log.run_name=my-graph-ordinary
```

Local AD replaces the nonlocal action by the negative Laplacian of physical
phi and stable raw-KKS q. Tensor sampling keeps shared time JVP and the matched
free graph core coordinates; joint sampling draws independent space/time pairs
and differentiates time with scalar grad. Counts match3760–4032 active PDE
pairs/update; IC/Dirichlet data and SOAP/scales remain the same.

The local PDE additionally needs natural conditions on the90 bottom nodes
outside the open pit mouth and solid corners:

$$
\partial_n\phi=0,\qquad \partial_n q=0.
$$

`local_training.py` adds a sixth GradNorm loss for squared derivatives of
phi and \(s=c_L/c_{Le}\), so \(q_n/c_{Le}=s_n\). This boundary choice is
traceable to the original local FEM weak form in `solver.py`. The graph keeps
its existing truncated-domain operator and five-term objective.

The independent local reference is
`/root/autodl-tmp/data/corrosion2d/results_local_ad_20261026.npz`, with specification
and refinement records in the adjacent JSON. `local_reference.py` integrates
physical fields with SciPy BDF, centered finite differences, mirror no-flux
closure and an analytic Jacobian. Its401x201 solution is sampled onto the common
101x51 evaluation grid. The201-grid time-refinement error is below2.75e-9, but201->401
terminal phi/c differences are1.42%/1.68%; **sub-percent continuum accuracy is
not established**. Score graph and local training against their own reference,
and show cross-reference discrepancies separately. Ordinary GPU policy is an
explicit option and now the canonical policy; it is not a reproducibility guarantee.

The [full5000-update comparison](../../docs/tasks/archive/local-ad-full-training.md)
uses one paired seed42 cold start per method, identical initial parameters,
unchanged SOAP/learning rate/PDE scales and the matched local boundary above.
Independent CPU checkpoint reloads give terminal50s relative L2:

| Method | phi | c | Late4000–5000 phi median | Observed total minutes |
|---|---:|---:|---:|---:|
| Nonlocal, previous deterministic default | 0.008835 | 0.010426 | 0.007191 | 19.66 |
| Nonlocal, ordinary policy | 0.020680 | 0.024565 | 0.012737 | 1.30 |
| Local AD, tensor/time JVP | 0.346112 | 0.373221 | 0.360579 | 3.75 |
| Local AD, joint/time grad | 0.411813 | 0.425739 | 0.546657 | 3.72 |

Times include actual compilation, evaluation and saves, reuse existing compiler
caches and exclude reference generation. Thus they describe this run sequence,
not a fresh-cache compilation benchmark. Each method is scored against its own
reference; graph/local reference terminal phi/c discrepancies are1.28%/1.51%,
combining operator/boundary/discretization effects. Against the common local
reference, default/ordinary nonlocal phi errors are2.11%/3.28%, still much
smaller than these local training errors. The local runs are not separately
tuned local SOTA controls, and one seed does not establish intrinsic superiority
or stability. This historical comparison predates the ordinary FP64 promotion
above. The ordinary FP32
run has better terminal free-q/Dq errors (0.00934/1.238 versus0.01372/1.703),
so field accuracy and transport accuracy should remain separate criteria.

[Step/time convergence curves](../../docs/tasks/archive/local-ad-full-training-convergence.png),
[terminal phi fields](../../docs/tasks/archive/local-ad-full-training-terminal-phi.png)
and [independent audit](../../docs/tasks/archive/local-ad-full-training-audit.json)
retain all outcomes rather than selecting the best intermediate checkpoint.

GPU determinism specifies reproducibility, not a higher arithmetic precision or
guaranteed lower trained-model L2. The [policy diagnosis](../../docs/tasks/archive/compiler-policy-explanation.md)
finds that the current JAX0.10.1 deterministic graph kernel lowers3032 edge
updates into a3032-iteration while loop, while ordinary mode retains parallel
scatter. Fixed-input ordinary forward/input-gradient repeats vary at roughly
1e-8 relative; both policies agree with independent SciPy at that scale. Full
training amplifies small initial differences into different adaptive weights
and parameter paths. A single paired run cannot rank their expected accuracy.

Representative current-default run (the common untuned seed42):
`/root/autodl-tmp/tf-logs/corrosion2d/20261026-full-graph_ordinary-float64-seed42/ckpt_005000.eqx`.
Its resolved configuration is `config.json` in the same directory; it matches
current numerical defaults apart from explicit run metadata. Load this FP64
model with JAX x64 enabled and a promoted template, as described above.
The independent2026 checkpoint is
`/root/autodl-tmp/tf-logs/corrosion2d/20261026-defaultcheck-ordinary-float64-seed2026/ckpt_005000.eqx`.

`train.py` applies the explicit policy before backend initialization. Ordinary
mode requires no deterministic flag in `XLA_FLAGS`; a conflict raises an error.
Use `XLA_FLAGS='' python -m examples.corrosion2d.train` if necessary. For the old
precision/policy control, pass `--config.train.compute_dtype=float32` and
`--config.train.gpu_policy=deterministic`. Unrelated flags remain intact.
Deterministic compilation alone does not promise identical cross-hardware results.
Empty `log.run_name` creates a fresh timestamped directory; an existing explicit
name fails. Evaluation is every200 updates; checkpoints every1000 completed
updates, with initial/final models, metrics/loss JSONL, TensorBoard and source
snapshots. Load arrays/models by direct paths, with no file hashing.

### Historical best individual bounded checkpoint

| Verified final50 s metric | Historical checkpoint |
|---|---:|
| phi relative L2,all5151 nodes | 0.0014330496603163906 |
| c relative L2,all5151 nodes | 0.001718589539743543 |
| q relative L2,free nodes | 0.0178326 |
| Dq relative L2,free nodes | 1.50041 |

Checkpoint:
`/root/autodl-tmp/tf-logs/corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed42/ckpt_005000.eqx`.
Reference:
`/root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz`.
This independent SciPy integration checks training error against the same fixed
graph; its time-refinement discrepancy is3.64e-9, not a spatial-convergence claim.
The old local-FEM array is not the SOTA error reference.

**Limitations:** the same bounded setting at seed7 gives phi/c0.4220/0.4461;
seed42's late-window phi median is~0.0181, substantially above its final value.
The low final total-field errors coexist with Dq error~1.50. This was an
explicit best-observed-field selection, now superseded as the default by the
original-residual setting after cross-seed validation. It is not a robust
transport remedy. To replay it set `experiment.formulation=bounded`; the
research frozen variant is `bounded_frozen`. The optional original-residual
seed42 late-rate variant additionally uses `experiment.lr_tail_start=3000`;
it is not the current default.
See the [full audit](../../docs/tasks/archive/corrosion-transport-optimization.md)
and [earlier promotion](../../docs/tasks/archive/phasefield-sota-defaults.md).

Historical large-grid presets remain available through the same entry:
`--config=examples/corrosion2d/configs/train_1pit.py` or `train_2pit.py`.
Those use the legacy loop and their original sampling/objectives. The sections
below distinguish legacy usage and research controls from the SOTA table.


### Frozen bounded-weight research control

`--config.experiment.formulation=bounded_frozen` uses the same positive
multiplier and the same forward residual as `bounded`, with
`stop_gradient(w_eta(phi))`. For the L1 AC loss the update differentiates
only the physical residual (for nonzero residual; at zero use its L1
subgradient):

$$
g_{AC}=\mathbb E\left[w_\eta(\phi)\operatorname{sign}(r_\phi)
\nabla_\theta r_\phi\right].
$$

The differentiated `bounded` objective additionally contains
\(\mathbb E[|r_\phi|\nabla_\theta w_\eta]\). Thus the forward PDE zero set
and reported residual values agree, but the optimization directions differ.
Frozen weighting is a reweighted update, not the gradient of the original
state-weighted scalar objective. CH, KKS, data constraints and pre-scales
are unchanged. This is an experimental option; it is not the default.

### Matched cross-seed objective comparison

All entries are final5000 full-grid50 s **phi/c relative L2**, with the same
physical/reference model, soft nodal IC/BC, optimizer and sample budgets.
Same-seed initial parameters and temporal streams are held fixed.

| Phase objective | Seed42 | Seed7 | Seed2026 |
|---|---:|---:|---:|
| Original (`phase`, current default) | 0.00884 /0.01043 | 0.00422 /0.00501 | 0.01074 /0.01273 |
| Differentiated `bounded` | 0.00143 /0.00172 | 0.42200 /0.44614 | 0.40702 /0.43103 |
| `bounded_frozen` | 0.01457 /0.01727 | 0.01090 /0.01293 | 0.01062 /0.01262 |

Original also has the smallest worst-seed late phi median:0.009514 versus
bounded0.422158 and frozen0.014632. Frozen rescues bounded7/2026 but does not
consistently outperform original. Its final Dq errors improve7/2026 to~1.36/1.40,
but regress42 to~3.01 versus original~1.70. Original Dq remains~1.70–1.76:
accurate total concentration does not certify transport-gradient accuracy.
Original/frozen have one liquid component at all21 reference times; the failed
bounded models retain spurious bands. No sampling, hard-IC or pre-scale change
was needed for these interventions. Three observed seeds are a limited validation.

See the [complete controlled study](../../docs/tasks/archive/corrosion-robust-objective.md),
[raw convergence curves](../../docs/tasks/archive/corrosion-robust-objective-plot-convergence.png)
and [same-scale phi fields](../../docs/tasks/archive/corrosion-robust-objective-plot-fields.png).

### Earlier bounded independent seed check —2026-10-26

Keeping every strategy setting and the5000-update/50 s budget fixed, a new
predeclared run at `--seed=2026` gives terminal phi/c **0.407016/0.431025**,
versus the recorded seed42 best0.001433/0.001719. The seed7 counterpart had
already failed at0.422001/0.446140. Late4000..5000 phi median is0.404946,
range0.399760–0.408093, so the failure persists across the late window.
Independent CPU q/Dq free errors are0.310697/4.013898. At t*>=.6 the liquid
field has2 connected components instead of reference1; terminal liquid area
is9.92% of the domain, reference29.33%. This bounded strategy is **not stable
across the tested runs**, despite its strong best observed checkpoint.
Do not call it robust SOTA or choose seed42 to make it appear stable. That
fixed-control study preserved the then-default. The subsequent matched
objective comparison supersedes its default selection with original phase
residuals while preserving the physical/reference model.
See the [full validation](../../docs/tasks/archive/phasefield-seed-stability.md) and
[curves](../../docs/tasks/archive/phasefield-seed-stability-results-convergence.png).

## Governing Equations (Local / Classical)

2D KKS phase-field model for pit corrosion. Domain: $\Omega = [0,1] \times [0,0.5]$ (non-dimensional).

With $\Delta c_0 = c_{Se} - c_{Le}$ and
$q = c - h(\phi)\Delta c_0 - c_{Le}$, the free-energy density is

$$f = A q^2 + \omega_\phi g(\phi) + \frac{\alpha_\phi}{2}|\nabla\phi|^2.$$

**Allen-Cahn:**

$$
\frac{\partial \phi}{\partial t}
= L\left[\alpha_\phi \nabla^2\phi
- \omega_\phi g'(\phi)
+ 2Aq\Delta c_0 h'(\phi)\right]
$$

**Cahn-Hilliard:**

$$\frac{\partial c}{\partial t} = D\,\nabla^2\!\left[c - h(\phi)\Delta c_0 - c_{Le}\right]$$

The constant $c_{Le}$ is inert under the CH Laplacian, but remains part of
$q$ in the Allen-Cahn driving force.

**Interpolation functions:**

$$h(\phi) = -2\phi^3 + 3\phi^2,\quad g(\phi) = \phi^2(1-\phi)^2$$

### Geometry & Boundary Conditions

Pit(s) are semicircular notches on the bottom boundary ($y=0$), defined by centre $(cx, cy)$ and radius $R$.

| Boundary | Condition |
|---|---|
| Pit mouth ($y=0$, $\|x-cx\| < R/2$) | $\phi = 0,\; c = 0$ (active dissolution) |
| $y=0$, elsewhere | $\partial\phi/\partial n = 0,\; \partial c/\partial n = 0$ |
| $x=0,\; x=1,\; y=0.5$ | $\phi = 1,\; c = 1$ (solid electrode) |

**Initial condition** — sharp tanh profile at pit interface:

$$\phi_0(x) = \min_{\text{pits}}\left\{\frac12\left[1 + \tanh(k\cdot(d(x,\text{pit}) - R))\right]\right\},\quad c_0 = h(\phi_0)\cdot c_{Se}$$

where $k = \sqrt{\omega_\phi / (2\alpha_\phi)}\cdot L_c$.

For the current \(c_{Se}=1\) preset, the KKS output gives
\(c=c_L+(c_{Se}-c_{Le})h(\phi)\) and \(q=c_L-c_{Le}\). Thus the
pit mouth requires \(q=-c_{Le}\), outer solid boundaries require \(q=0\),
and the analytic initial profile requires
\(q_0=c_{Le}[h(\phi_0)-1]\). The transport-driving range is only
\(c_{Le}\approx0.0356643\); assess BC/IC concentration errors against this
scale, rather than the order-one total concentration alone. A steep phase
profile can coexist with nearly constant q and negligible CH transport.
The curved initial interface is IC data, not a permanent zero Dirichlet arc.

The finite tanh initial profile has nonzero tails at the pit mouth, whereas
the declared Dirichlet data are exactly zero. The matched graph reference
prescribes boundary nodes even at time zero; random interior PINN IC targets
use the analytic profile. This initial-boundary corner incompatibility should
be explicit when tightening both losses. Do not silently alter the physical
IC or impose zero values along the moving interface. A
[split boundary/initial-drive audit](../../docs/tasks/archive/corrosion-boundary-drive-audit.md)
records the measured effects and limitations.

---

## Non-Dimensionalisation

Characteristic length $L_c = 100~\mu\text{m}$ and time $T_c$ define:

$$\bm{x}^* = \frac{\bm{x}}{L_c},\qquad t^* = \frac{t}{T_c}$$

Four non-dimensional parameters:

$$N_{\text{CH}} = \frac{D T_c}{L_c^2},\quad N_{\text{AC,1}} = 2A L T_c,\quad N_{\text{AC,2}} = L\omega_\phi T_c,\quad N_{\text{AC,3}} = \frac{L\alpha_\phi T_c}{L_c^2}$$

### Non-dimensional Allen-Cahn

$$\frac{\partial\phi}{\partial t^*} - N_{\text{AC,1}}\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr](c_{Se}-c_{Le})h'(\phi) + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,3}}\,\nabla^{*2}\phi = 0$$

### Non-dimensional Cahn-Hilliard

$$\frac{\partial c}{\partial t^*} - N_{\text{CH}}\,\nabla^{*2}c + N_{\text{CH}}(c_{Se}-c_{Le})\,\nabla^{*2}h(\phi) = 0$$

equivalently:

$$\frac{\partial c}{\partial t^*} = N_{\text{CH}}\,\nabla^{*2}\!\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr]$$

---

## Nonlocal (Peridynamics-Inspired) Formulation

Replacing $\nabla^{*2}$ with the nonlocal operator $D_\delta$ (peridynamics style):

$$\frac{\partial\phi}{\partial t^*} + N_{\text{AC,3}}\,D_\delta\phi + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,1}}\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr]\Delta c_0\,h'(\phi) = 0$$

$$\frac{\partial c}{\partial t^*} + N_{\text{CH}}\,D_\delta\!\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr] = 0$$

where $\Delta c_0 = c_{Se} - c_{Le}$, and in the local limit $D_\delta \approx -\nabla^{*2}$.

The nonlocal operator is defined by a compactly-supported kernel $J(r)$:

$$(D_\delta u)(x_i) = \sum_{j \in N_\delta(i)} w_{ij}\,(u_i - u_j),\quad w_{ij} = -D_{ij}$$

Edge weights pre-computed from SparseD operator. Supported kernels: `cubic`, `quartic`, `wendland`.

---

## Physical Parameters

| Symbol | Value | Unit | Description |
|---|---|---|---|
| $D$ | $8.5 \times 10^{-10}$ | m²/s | Diffusion coefficient |
| $A$ | $5.35 \times 10^7$ | J/(m³·s) | Reaction constant |
| $L$ | 2.0 | m/(J·s) | Interface mobility |
| $\alpha_\phi$ | $1.03 \times 10^{-4}$ | J/m | Gradient energy coefficient |
| $\omega_\phi$ | $1.76 \times 10^7$ | J/(m³·s) | Barrier height |
| $c_{Se}$ | 1.0 | — | Equilibrium concentration (solid) |
| $c_{Le}$ | 5100 / 1.43e5 | — | Equilibrium concentration (liquid) |

### Non-Dimensionalisation

| Scale | 1-pit | 2-pit | Description |
|---|---|---|---|
| $L_c$ | $100~\mu\text{m}$ | $100~\mu\text{m}$ | Characteristic length |
| $T_c$ | 50 s | 15 s | Characteristic time |

---

## Project Structure

```
corrosion2d/
├── solver.py              # Shared FEM solver (pit geometry comes from config)
├── train.py               # PINN training (coordinate MLP / FiLM)
├── losses.py              # Loss functions (coordinate MLP)
├── residuals.py           # Shared phase/latent/bounded KKS objectives
├── residual_experiment.py # Shared matched training / reference engine; research CLI
├── transport_optimizer.py # Optional late SOAP learning-rate refinement
├── training_data.py       # Shared fixed analytic IC/BC samplers
├── training_entry.py      # Canonical matched-training adapter
├── configs/
│   ├── solver_1pit.py           # FEM solver config (1 pit)
│   ├── solver_2pit.py           # FEM solver config (2 pits)
│   ├── train_base.py            # Default observed-field SOTA (1 pit)
│   ├── train_1pit.py            # Historical PINN config (1 pit)
│   └── train_2pit.py             # PINN training config (2 pits)
├── results/               # FEM reference solutions (.npz)
└── plots/                 # Output plots
```

---

## Usage

### Run FEM Solver

```bash
# From the project root:
# 1 pit
python examples/corrosion2d/solver.py \
    --config=examples/corrosion2d/configs/solver_1pit.py

# 2 pits
python examples/corrosion2d/solver.py \
    --config=examples/corrosion2d/configs/solver_2pit.py

# Equivalent commands from this directory:
cd examples/corrosion2d
python solver.py --config=configs/solver_1pit.py
python solver.py --config=configs/solver_2pit.py
```

Output: `.npz` files with `X`, `Y` meshgrids and `phi`, `c` fields over time.

### Historical PINN presets (Coordinate FiLM MLP)

```bash
# 1-pit (from project root)
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_1pit.py

# 2-pit with spatial minibatch
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_2pit.py \
    --config.sample.K_core=1200
```

### Legacy-loop Config Options

```bash
# Architecture
--config.arch.type=film --config.arch.hidden_dim=64

# Nonlocal operator
--config.nl.delta=0.02 --config.nl.kernel=cubic

# Spatial minibatch (core+halo sub-sampling)
--config.sample.K_core=1200   # 0 = full grid
--config.sample.K_pool=200    # sub-graph pool size

# Staggered training
--config.loss.stagger_steps=10000 --config.loss.stagger_switch_every=25

# Temporal ramp (bias later times for merging problems)
--config.loss.t_ramp_steps=2000 --config.loss.t_ramp_early=0.3 --config.loss.t_ramp_late=3.0

# Causal training (per-t mode)
--config.loss.use_causal=True --config.causal.n_chunks=0
```

### Graph Model Status

The historical graph-model files (`train_graph.py`, `losses_graph.py`, and their
configs) are not present in the current checkout. The supported training path
is `train.py`, defaulting to `train_base.py`; `train_1pit.py` and
`train_2pit.py` explicitly retain the historical loop.


## Matched residual experiment

The experimental [runner](residual_experiment.py) supports `corrosion2d` and
`electropolish2d`. It retains each production architecture, physical parameters,
geometry and horizon (50 s corrosion, 5 s electropolish), using an explicit
101x51 cubic graph with delta 0.02, 256 core points, 16 times/update, SOAP,
lr 0.001, 5000 updates and seed 42. Causal/interface weighting is off, and
stagger defaults off unless explicitly selected;
the original AC/CH scales and five-component GradNorm/soft IC/BC losses remain.
Dirichlet nodes supply neighbouring values but do not enter the PDE core loss.
The preset is [configs/residual_pilot.py](configs/residual_pilot.py), separate
from the canonical SOTA defaults above.

Production `train.py` supports an initial period of alternating AC-only and
CH-only PDE losses, with IC/BC losses retained. The historical `train_1pit.py` preset alternates
every25 training updates for the first1000, then enables both PDE losses.
This alternates losses in training, rather than physical time steps or frozen
field solves: shared parameters and the KKS map still couple both outputs.
Both production and research runners preserve **unmasked, unbiased adaptive
weights** as persistent state. Activation masks and configured biases apply
only to the current gradient, so switching without a GradNorm refresh does
not erase the newly active equation's weight. Effective `weight/*` log values
may be zero; saved adaptive weights remain positive. Network checkpoints are
unchanged. Older masked-weight controller states are not the new state contract.
The earlier defect and concentration/transport audit are in the
[stagger audit](../../docs/tasks/archive/corrosion-concentration-stagger-audit.md).

```bash
.venv/bin/python -m examples.corrosion2d.residual_experiment train \
  --case corrosion2d --formulation latent \
  --stagger-steps 1000 --stagger-switch-every 25
# --formulation phase selects the matched original residual;
# --stagger-steps 0 selects joint training from the first update.
```

Stagger is measured in training updates; changing it does not march or shorten
physical time. Loss history records effective `weights`, positive
`adaptive_weights`, `active` bits and `stage`, including every stage transition.
The summary records exact AC-only/CH-only/joint update counts. GradNorm refresh
cadence is unchanged, and inactive PDE gradients are still measured when
refreshing adaptive weights. Terminal/pooled c_L/q errors and terminal free-node
graph Dq error are evaluation-only diagnostics, alongside phi/c L2. No reference
values enter the training loss or schedule.

The [corrected four-arm experiment](../../docs/tasks/archive/corrosion-stagger-experiment.md)
did not recover concentration with this warmup loss alternation. Its fresh
joint baseline also failed to reproduce the earlier accurate trajectory;
identical-input GPU gradient variation is measured in both old and new paths.
Keep concentration/transport diagnostics and establish a stable joint control
before interpreting a stagger or residual advantage. The tested configuration
is not promoted and the AC/CH pre-scales are retained.

### Concentration data-loss and initial-data controls

The research runner explicitly supports `--data-loss-mode`:

| Mode | BC data pair | IC data pair |
|---|---|---|
| `physical` (legacy default) | phi, total c | phi, total c |
| `s_bc` | phi, s | phi, total c |
| `s_bc_ic` | phi, s | phi, s |

Here \(s=c_L/c_{Le}=(1+\tanh\mathrm{raw}_1)/2\) for the current
\(c_{Se}=1\) KKS map. BC targets are0/1 for both components, and the normalized
IC target is \((\phi_0,h(\phi_0))\). The normalized modes validate the KKS map,
\(c_{Se}=1\), \(0<c_{Le}<1\), and the declared IC relation. They evaluate s
directly from the raw network output, avoiding cancellation in c-dc*h(phi).
These options change data-loss weighting and coupling; AC/CH equations,
pre-scales, physical output mapping and GradNorm rules are unchanged. Both
PDE residual formulations remain independently selectable. The legacy physical
`make_loss_vector` factory remains available to archived callers.

`--initial-sampling random_interior` preserves the legacy random IC samples.
`--initial-sampling matched_graph` instead uses every fixed graph node, with
analytic interior IC and prescribed Dirichlet values overriding boundary nodes
at time zero. This is exactly the existing discrete reference's initial state,
checked at startup. It does not invent a smooth compatible IC or an output
hard constraint; the finite graph represents the initial-boundary corner
jump between adjacent nodes. All new normalization controls use this same IC
sampling, including their physical-data baselines. Earlier random-IC results
are separate historical comparisons.

`--boundary-sampling random` retains legacy off-grid BC samples;
`--boundary-sampling matched_graph` uses every declared Dirichlet graph node.
Use both matched-graph switches together for the nodal comparison: off-grid
random zero-BC samples arbitrarily close to a non-Dirichlet mouth endpoint
otherwise demand a sharper initial spatial layer than the discrete reference.
The formal normalization cohort uses nodal BC and IC consistently. Three
preliminary graph-IC/random-BC runs were explicitly stopped and retained before
this alignment correction; they are not complete5000-update controls.

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true .venv/bin/python -m \
  examples.corrosion2d.residual_experiment train \
  --case corrosion2d --formulation phase \
  --data-loss-mode s_bc --initial-sampling matched_graph \
  --boundary-sampling matched_graph
```

The explicit GPU flag has passed an actual full-batch identical-input probe
on this host. Summaries record the flag, data-loss mode, BC/IC sampling and actual
BC/IC node counts. This does not imply an unmeasured full-trajectory guarantee.
Live evaluation additionally records separate pit/solid physical c errors,
q errors scaled by c_Le, and initial free-node q/Dq errors. Reference snapshots
are evaluation targets only; IC targets are computed from the prescribed IC
and boundary data. Normalized losses can still suffer tanh saturation and are
research options. The [completed normalized-data comparison](../../docs/tasks/archive/corrosion-normalized-data.md)
shows a two-seed field benefit for original residual + `s_bc`; strong IC
normalization and full latent mode do not improve the later phase solution.
Localized pit-mouth transport errors and late fluctuations remain, so this
candidate did not replace production defaults in that earlier study. The
current explicitly selected bounded SOTA above retains the pre-scales.

### Transport evaluation and CH loss norm

The research runner also declares `experiment.transport_evaluation`,
`experiment.ch_norm` and `experiment.ch_rms_floor` explicitly. Legacy settings
are `physical_difference`, `l1`, and `1e-12`. The new switches are independent:

- `--transport-evaluation kks_raw_edges` evaluates the same KKS chemical
  potential as \(q=c_{Le}(\tanh v-1)/2\), avoiding subtraction of the phase
  contribution from total c. It accumulates \(Dq_i=\sum_jw_{ij}(q_i-q_j)\)
  directly, so a constant q gives exact floating-point zero. The graph, c_t,
  physical outputs, AC residual and BC/IC remain unchanged. It requires the
  current \(c_{Se}=1\) KKS map. It changes evaluation roundoff, not the PDE.
- `--ch-norm rms` replaces the CH mean absolute residual by
  \(\sqrt{\operatorname{mean}(r_{CH}^2)+\epsilon^2}-\epsilon\), with explicit
  \(\epsilon=10^{-12}\). It retains residual units and a finite zero gradient
  at zero. Its gradient gives more weight to large residuals; this is an
  optimization-objective change. AC/data losses, scales and GradNorm stay fixed.

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true .venv/bin/python -m \
  examples.corrosion2d.residual_experiment train \
  --case corrosion2d --formulation phase --data-loss-mode s_bc \
  --initial-sampling matched_graph --boundary-sampling matched_graph \
  --transport-evaluation kks_raw_edges --ch-norm l1
```

The SOTA default uses raw edges with L1; RMS remains an ablation. The
legacy physical/normalized-data factories preserve their original L1/physical-
difference behavior for archived callers; `make_transport_loss_vector` reads
the new explicit settings. Training summaries and distinct run names record
both selections. Compare full-grid phi/c/q/Dq and CH residuals before selecting
a setting; low field L2 alone does not establish correct local transport.

Late learning-rate refinement is a separate SOAP control. Research fields
`experiment.lr_tail_start=0` and `experiment.lr_tail_end=1e-5` preserve the
legacy schedule by default. With `--lr-tail-start 3000 --lr-tail-end 1e-5`,
the original exponential schedule is unchanged through update index3000;
from that index it decays continuously to the declared end rate at the end
of the existing5000-update budget. The last actual update is index4999 and
uses a rate just above the endpoint. The target must be at least `lr_min`
and below the original rate at the switch. No optimizer moments, sampling,
losses or adaptive weights reset at the switch. It currently requires SOAP.
The new helper is saved alongside the research source snapshots; summaries
record the switch and actual first/last update rates. This is a separate
schedule experiment, not a physical-PDE or production-default change.

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true .venv/bin/python -m \
  examples.corrosion2d.residual_experiment train \
  --case corrosion2d --formulation phase --data-loss-mode s_bc \
  --initial-sampling matched_graph --boundary-sampling matched_graph \
  --transport-evaluation kks_raw_edges --ch-norm l1 \
  --lr-tail-start 3000 --lr-tail-end 1e-5
```

CH loss continuation is independent of the rate control:
`experiment.ch_rms_after=0` preserves a single norm. Setting
`--ch-rms-after 4000 --ch-norm l1` keeps the first4000 updates unchanged,
then uses the same RMS formula for the last1000 updates. The switch must
align with a GradNorm refresh and lie inside the update budget. Both compiled
losses use the same array shapes, optimizer state and model; BC/IC, AC loss,
physical fields and sampling remain fixed. No optimizer or adaptive state is
reinitialized. The summary records the switch and final CH norm; `stage` in
the loss history continues to describe AC/CH activation. This tests refinement
from an established interface rather than RMS from a cold initialization.

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true .venv/bin/python -m \
  examples.corrosion2d.residual_experiment train \
  --case corrosion2d --formulation phase --data-loss-mode s_bc \
  --initial-sampling matched_graph --boundary-sampling matched_graph \
  --transport-evaluation kks_raw_edges --ch-norm l1 --ch-rms-after 4000
```

The [completed transport study](../../docs/tasks/archive/corrosion-transport-optimization.md)
finds that direct KKS/edge evaluation improves phi/c in two seeds, while q/Dq
regresses in seed7. Late rate decay, bounded phase weighting and cold/late RMS
do not establish a two-seed improvement of both fields and transport. These
did not justify a robust joint-metric promotion in that study. The current
terminal-field SOTA explicitly selects bounded weighting while retaining KKS
and pre-scales. Evaluate driving concentration, global
transport and late-window variability alongside total-field L2.

With \(\phi=(1+\tanh z)/2\), the compared phase objectives are the original
\(r_\phi\) and its analytically evaluated transform

$$
\widehat r_z=\frac{r_\phi}{4\phi(1-\phi)}
=\frac12\left[z_t+N_3\sum_jw_{ij}
\sinh(z_i-z_j)\frac{\cosh z_i}{\cosh z_j}
+N_2(1-2\phi)-3N_1q\Delta c_0\right].
$$

The normalization matches the residual scale at \(\phi=0.5\). The CH equation
and KKS output map are identical in both arms; no hard IC, denominator floor or
reference supervision is added.

Corrosion inherits `ac_scale=1e-7`, `ch_scale=1`; electropolish inherits `1,1`.
These multiply the mean absolute PDE residuals before GradNorm. The phase
transformation introduces a state-dependent factor \(1/[4\phi(1-\phi)]\);
equal interface-centre scale does not preserve coupled gradient balance.
GradNorm's norm floor/ceiling and weight cap prevent exact scale compensation.
The KKS concentration branch can also saturate independently of the phase map.
Consequently this pilot tests transfer with the inherited scales; it does not
establish the best attainable L2 after formulation-specific scale tuning.

Historical local FEM arrays are not used as the graph accuracy target. The
runner regenerates an independent float64 SciPy BDF integration of the same
pair graph, with identical analytic IC and prescribed boundary nodes. It uses
an analytic sparse Jacobian, then tighter tolerances and a smaller maximum
time step to verify reference accuracy. Parameter, geometry, boundary, graph,
field-shape and physical-time metadata are checked before training. This tests
optimization at a fixed spatial discretisation; it does not establish
convergence to the continuous local PDE.

```bash
# From the repository root, using .venv; files are loaded directly by path.
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m examples.corrosion2d.residual_experiment reference --case corrosion2d
.venv/bin/python -m examples.corrosion2d.residual_experiment train --case corrosion2d --formulation phase
.venv/bin/python -m examples.corrosion2d.residual_experiment train --case corrosion2d --formulation latent
# Repeat with --case electropolish2d for the rough-surface problem.
```

The reference command refuses existing output files. Each training arm creates
a fresh named run and saves config/source paths, initial/final weights,
completed-update checkpoints, metrics JSONL, physical-field predictions and a
summary. Metrics include pooled and terminal full-grid phi/c relative L2,
terminal interface distance and boundary error. All reference fields are
evaluation-only. A single matched seed is a pilot, not universal evidence.
The [completed paired pilot](../../docs/tasks/archive/coupled-latent-residual-experiments.md)
records results and failure diagnostics; the transformation remains experimental.

For output-map/scale ablation the same research runner accepts explicit
training overrides, which create separate run directories:

```bash
.venv/bin/python -m examples.corrosion2d.residual_experiment train \
  --case corrosion2d --formulation latent --concentration-map direct \
  --ac-scale 1e-9 --ch-scale 1
# Replace direct by kks, and latent by phase, for the matched controls.
```

`--concentration-map kks` retains the bounded mixture. `direct` predicts
\(c=\mathrm{raw}_1\) independently and retains
\(\phi=(1+\tanh\mathrm{raw}_0)/2\). The research saved config records the
choice as `arch.use_hard_constraint`; false still retains the phase map in
this runner. The legacy-loop configs retain their existing output-map semantics. The PDE's
KKS free energy is unchanged. Direct concentration is not clipped: evaluation
records its range and fraction outside0..1. Equal raw initialization weights
do not imply equal initial physical fields across output maps. Each run also
records losses and actual adaptive weights in `loss_history.jsonl`; `pre_scale`
is an optimization setting and is distinct from physical nondimensionalization.
The [output-map/scale ablation](../../docs/tasks/archive/corrosion-kks-scale-ablation.md)
documents matched original/transformed controls. These overrides remain
experimental; the production KKS map is retained and the original residual
is an explicit alternative to the default bounded objective.

The runner also supports an explicit **bounded** phase preconditioner:

$$
r_\eta=\frac{1+\eta}{4\phi(1-\phi)+\eta}\,r_\phi,\qquad\eta>0.
$$

The weight is1 at the interface centre and at most \((1+\eta)/\eta\).
This preserves the physical PDE zeros while limiting pure-phase amplification;
it is a separate objective, not the exact latent residual or a silent numerical
fallback. It multiplies the original AC residual, including differentiation
of the state-dependent weight; CH, KKS and soft IC/BC are retained.

```bash
.venv/bin/python -m examples.corrosion2d.residual_experiment train \
  --case corrosion2d --formulation bounded --bounded-eta 0.1
# To test removing both equation pre_scales, explicitly add:
# --ac-scale 1 --ch-scale 1
```

The research preset explicitly declares `experiment.bounded_eta=0.1` (cap11);
the CLI rejects nonpositive/nonfinite values and use outside bounded training.
Eta1/0.1/0.01 give caps2/11/101. New run names record eta, map, scales and seed.
Bounded spatial weighting does not remove the AC/CH equation-scale imbalance;
omitting pre_scales requires its own physical relative-L2 validation. That
earlier pilot improved one seed and worsened the second, and both no-scale
controls lost the terminal interface. The current default instead selects the
later bounded/raw-edge/nodal-data seed42 field result, with this seed limitation
retained. See the
[bounded residual validation](../../docs/tasks/archive/corrosion-bounded-residual.md)
for the matched controls, checkpoint paths and physical-field metrics.

The corrected FEM solver now imposes solid left/right/top BCs as in training,
marks the bottom pit mouth by position on nodes, applies initial Dirichlet
traces and records actual save/final time increments. Old arrays are retained
unchanged. For a new local FEM comparison, choose a new output filename;
local FEM and finite-horizon graph solutions remain distinct models.

## Project split (2026-10-24)

This example belongs to `nonlocal-pinn`. Install the project and run from its
repository root; existing external data/output paths are retained. Shared
implementation paths changed during the split; see
[the source map](../../docs/migration/source-map.json) and
[migration notes](../../docs/migration/README.md).
