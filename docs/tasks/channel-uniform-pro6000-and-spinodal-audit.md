# Channel uniform core/halo on RTX PRO 6000 and Spinodal explanation

## Current status — resource-only completion

The latest user instruction supersedes the long-training plan below: complete
resource tests, record them, then switch to the ceramic FEM weak-residual
comparison. No channel long run was launched; the phase<2% goal was not tested.
The Spinodal part is settled: retain the successful original full-field scheme.
Historical instructions/results below are preserved as history, not restart
permission. Active work is [ceramic FEM comparison](fem-weak-pino-comparison-plan.md).

## Scope and acceptance

User authorizes a fresh background two-phase-flow run on the new RTX PRO 6000,
using uniform rolling core/halo, with larger cores/time blocks/batches or a longer
budget where useful. Aim for phase relative L2 below2% and less wall time than
the successful full-grid route. This is an accuracy target, not a promised result.
Measure new-card warm speed and active memory before choosing the long run.
Keep the successful physics, architecture and loss formulation. Record actual
configuration, sources, seed, sample streams, evaluation and process handles.

The second request is Spinodal, corrected explicitly by the user from ceramic
corrosion. Explain exactly why output parameterization/residual preconditioning
were changed, and whether original successful training can keep those choices
while changing only sampling. Do not start any Spinodal or ceramic training.
All old stopped queue owners stay stopped; this is a new finite channel workflow.
No Git operation or default promotion is authorized.

## Current evidence and plan

GPU verified: NVIDIA RTX PRO 6000 Blackwell Server Edition,97887MiB total,
UUID GPU-1e537971-8529-b7c1-1682-82073d2cae12, driver595.71.05, initially idle.
Available disk43,400,736,768bytes. Existing successful original channel recipe:
width192/B8/Nt36/SOAP(beta1=beta2=.9)/FP32, LR1e-3->1e-5 over20k,
fixed normalized original residuals, integral spatial operators and timeJVP.
Completed previous-card uniform512 core20k gives reserved3 phase3.295% and
velocity3.038%, versus historical full20k1.6435%/1.5172%. Do not reuse old timing
as a measurement on this GPU. Original full SOTA remains unchanged.

Plan: same-card short full and uniform512/1024/2048 resource probes with B8,
Nt36 and timeblock18, preserving complete supports and physical-area weighting.
Choose a larger-core candidate using measured time/memory, then cold-start a
longer run with an explicit lower learning-rate tail and model checkpoints.
Canonical training logs evaluate the fixed development FEM every500updates;
an external owner records progress periodically and evaluates reserved3 at end.
Only channel inference/training is authorized. No FEM solve is needed.

## Implemented channel workflow and verification

Research presets `train_uniform_pro6000.py` and `train_full_pro6000_probe.py`
retain the original architecture, physical equations, normalization, SOAP and
FP32 execution. Uniform sampling uses area-weighted 4x4 groups without wall
oversampling. B8/Nt36/timeblock18 replaces the previous compact timeblock6.
Candidates are 512/1024/2048 cores with query capacities10240/12288/15633;
capacity includes halo and boundary supports, not just loss rows. Pools and
geometry continue rolling; no default is promoted.

`channel_uniform_pro6000_queue.py` runs a support audit and four sequential60-step
resource probes, then selects one cold long run. Prefer the largest core>=1024
whose measured active peak is below94% of physical GPU memory and whose30k warm
cost fits within full20k warm cost. Otherwise choose the largest eligible faster
core for20k. If none qualifies, stop explicitly for review. This compares warm
resource estimates, not achieved equal-error training time. LR1e-3->5e-6 uses an
explicit decay over the selected budget. Evaluate development FEM every500,
save model every1000, record progress every30seconds, audit online inputs and
evaluate the same three reserved FEM geometries after completion.

Four focused selector tests passed. Both presets pass canonical config validation.
The support audit passes32 saved known-input geometries x3counts x4draws;
synthetic first/second operator rows and all boundary/encoder inputs match
exactly. Maximum actual query unions are7259/10229/13697 for512/1024/2048.
This finite audit is not a proof against all future random-geometry capacities;
runtime capacity validation remains strict, with no truncated neighborhoods.

First owner4353, bundle`channel-uniform-pro6000-20261113-01`, stopped before GPU
training: the audit mistakenly requested `condition` instead of actual channel
`domain` encoder fields. Corrected only the audit field names, reran it
successfully, and retained the failed journal/log. No PDE/sampler change.

New finite owner5245:
`/root/autodl-tmp/tf-logs/channel-uniform-pro6000-20261113-02/journal.json`.
Outer log:`/tmp/channel-uniform-pro6000-20261113-02.log`;
launch receipt:`/tmp/channel-uniform-pro6000-20261113-02-launch.json`.
At06:32:50UTC, support passed and full-grid60step probe5255 started.
No long-run accuracy is available yet. Inspect the journal before taking any
process action; do not start another owner or revive any old queue.

## Spinodal source diagnosis — no new training

The original accepted full-grid route remains intact. Its direct phase output is

$$
\phi=\phi_0+(t/T)(f_\theta-\langle f_\theta\rangle_V),
\qquad \mu=\beta(\phi^3-\phi)+\epsilon^2D\phi,
$$

where the conservative regional graph satisfies \(D\simeq-\Delta\) and
\(v^TD=0\). The original loss uses the dense, volume-self-adjoint
`linear_stable_resolvent` metric \(P\):

$$
r_\phi=T(\partial_t\phi+MD\mu),\qquad
\mathcal L=\sum_i\frac{v_i}{\sum_jv_j}(Pr_\phi)_i^2.
$$

Both exact discrete mass projection and dense metric need global field/residual
information. Selecting a local core alone cannot remove those dependencies.

The failed compact pilot changed TWO things to permit local query closure:
network output became an auxiliary conservative potential
\(\chi=(t/T)f_\theta\), with
\(\phi=\phi_0+\alpha D\chi\), \(\alpha=\epsilon^2/\beta\);
the dense metric was replaced by an identity-weighted potential residual
\(r_\chi=T(\partial_t\chi+M\mu/\alpha)\).
Although \(r_\phi=\alpha D r_\chi\) and physical trajectories admit the potential
representation, their squared residual losses and trainable field coordinates
are not identical. This was a formulation/conditioning experiment, not a clean
sampling-only ablation. Optional initial-rate lifting was not enabled in those
failed5k runs and its later queued training never started.

Same native8geometry phase metric: original full5k18.2051%, full50k4.8329%;
potential/identity full5k60.8780%, compact5k87.2623%. The modified full case also
fails, so core sampling alone is not a sufficient explanation. Effects of the
two changed choices have not been independently isolated; historical execution
is another confound. Source/previous raw-record audit references:
`operator_training.py`, `operator_metrics.py`, `operator_potential.py`, and
`pino-unified-rolling-core-halo.md` (raw-record recheck,2026-11-12).

Sampling-only control ALREADY exists but has not trained:
`train_operator_residual_rows.py` and `operator_residual_pool.py`. It preserves
the original output, global mass projection, eliminated chemical potential and
dense metric; selects only rows of \(Pr_\phi\) for the final weighted square sum.
The sampled quadrature has the original volume objective in expectation over
construction. Reusing a pool does not imply conditionally unbiased adaptive
gradients. All global neural queries and metric columns still remain, so this
is a useful clean control, not genuinely compact core+halo or a promised speedup.
Its previously queued owner was stopped before training and stays stopped.

The user has now explicitly decided to retain the full-field Spinodal route.
This resolves the investigation: keep original direct output, global mass
projection and stable-resolvent loss. Do not advance the row-sampling or compact
controls. Retain their evidence as history. No new Spinodal implementation,
reference solve or training this turn.

Status: new channel resource probes running; Spinodal explanation and decision complete.
Blocker:none. Next: inspect resource journal, selected profile and first long-run
update; retain exact background resume path and later report measured resources
and development/cohort errors separately. No Git operation.

## Session handoff

Latest inspected owner heartbeat06:36:20UTC: full probe5255 remains running,
initial FEM development evaluation completed, training compilation in progress,
no recorded optimizer update yet. CUDA autotuning emitted delay-kernel timing
warnings; there is no recorded fatal failure. The94,999MiB driver reading is
the97%allocator reservation, not an active-memory measurement. Use completed
probe `device_memory_bytes.peak_bytes_in_use` for selection/reporting.

Files added this session: two channel resource/training presets, finite queue
and its four-test selector suite, this task. Memory updated:PROGRESS,DECISIONS,
this task. Fixed-support real-input checks and configuration validations pass;
`git diff --check` passes. Existing numerical research edits are untouched.
Task remains active only for the channel background workflow. Exact resume:
read bundle02 journal/resources/selection and current child log before doing
anything else; if failed, inspect its explicit error, preserve outputs, and
address the authorized channel issue. Never restart a stopped Spinodal queue.

## FEM weak-residual comparison planning follow-up

The user asks to plan borrowing FEM's natural Neumann treatment for PINO.
The completed proposal is [fem-weak-pino-comparison-plan.md](fem-weak-pino-comparison-plan.md).
Source inspection finds ceramic corrosion already has conforming P1 stiffness,
lumped mass and an independent physical-field reference. The proposed third
arm uses this signed sparse action, retains the successful KKS output and L1
loss/scales, and avoids spatial AD and separate homogeneous-Neumann penalties.
Standard test functions are known FE shapes; no auxiliary test network is
needed. The proposal includes64sample/common-reference and same-hardware timing
controls, and distinguishes weak natural enforcement from pointwise NN traces.

Read-only background observation at06:44:35UTC: support/full resource probe
passed and core512 probe is running. No process/scope change follows from this
planning request. Spinodal remains full-field. No implementation or additional
training/reference solve started. Planning-session files: new proposal plus
PROGRESS,DECISIONS,this task. Verify Markdown links and git diff --check only;
no code tests are needed for these documentation-only edits. Exact channel
resume point remains the bundle02 journal and selected long-run record.

Related retained history: [paused unification study](pino-unified-rolling-core-halo.md).

## Deferred FEM TODO — user instruction

The user explicitly defers the planned FEM comparison until they confirm the
two-phase-flow experiment is finished. The proposal and PROGRESS now contain
an unchecked TODO with that release gate. It is not a dependency automatically
triggered by channel completion. No implementation/preparation/new experiment
or process action this session; the channel workflow remains the active task.
Task-end: updated the proposal,PROGRESS,DECISIONS,and this task; documentation
diff check only. Resume channel from its existing journal when requested;
resume FEM work only after the user's confirmation.

## Progress query at06:53:30UTC /14:53:30Asia-Shanghai

Read journal and resources directly: owner5245 running. Support/full/core512
passed; core1024 trainer40127 running; core2048 and selected long run pending.
No selection.json yet, hence no selected20k/30k training progress or accuracy.
Full/core512 warm inclusive medians1.6141362018/.8009981220seconds, a2.015x
throughput ratio in these short same-card probes. Active peaks are
89,558,744,320/55,167,894,272bytes. Whole child times614.667/594.803seconds
include cold preparation, compilation and evaluation. The ~95GBnvidia-smi
reading remains allocator reservation, not these measured active peaks.
Do not extrapolate equal-error convergence or reuse the512probe error as a
trained-model result. No process/config/default/Git change. Updated PROGRESS
and this task; diff-check only. Continue monitoring bundle02, retain FEM TODO
pending explicit user confirmation.

## Resource finalization after user scope change

At07:04:10UTC verified owner5245 was stopped while final child57170 continued.
Finalizer62692 waited for child exit0 and then terminated only the stopped old
owner. At07:13:04UTC, bundle02 journal became `completed_resources_only`, with
`long_training_cancelled_by_user:true`. Original journal and cancellation
receipt are retained; no selected long run or selection.json exists.

All use B8/Nt36/timeblock18,60updates on the same97887MiB PRO6000:

| Method | Warm step incl preparation (s) | Active peak bytes | Padded queries |
|---|---:|---:|---:|
| Full |1.6141362018|89558744320|15633|
| Uniform512 |0.8009981220|55167894272|10240|
| Uniform1024 |0.9439890683|79896918528|12288|
| Uniform2048 |1.4262898662|91070342400|15633|

These are resource measurements only. No equal-error speed or new long-run
accuracy follows. Driver reservation differs from active memory. Exact receipt:
`/root/autodl-tmp/tf-logs/channel-uniform-pro6000-20261113-02/resources.json`.
Channel work is stopped at this explicitly requested point; do not revive its
old scheduler. The newly authorized ceramic queue has a separate owner/task.
